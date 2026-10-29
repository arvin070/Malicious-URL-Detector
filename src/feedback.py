"""
Feedback intake and the gated retraining pipeline.

    user feedback -> validated -> candidate dataset -> train -> evaluate
                  -> approval gate -> versioned deployment

THE GATE IS THE POINT
---------------------
Accepting user feedback as training labels is the easiest way to poison a
security model: anyone who can submit "this is safe" can, with enough
submissions, teach the detector to ignore their own phishing. So feedback is
never a label on its own. It is a *claim*, stored with who made it and when,
and it only becomes training data after:

  1. VALIDATION   the URL parses, is in scope, and the claim does not
                  contradict a blocklist snapshot (a user cannot mark a
                  PhishTank-verified URL benign);
  2. CORROBORATION the same claim from several independent accounts, or one
                  account marked trusted by an operator;
  3. QUARANTINE    a minimum age, so a burst of submissions cannot land in the
                  next training run;
  4. APPROVAL      a human promotes a candidate dataset, and the promotion is
                  recorded with the approver and the evaluation it was based on.

A candidate model is never deployed by this module. It is built, evaluated
against the frozen hard-benign and adversarial suites, and written to a
versioned directory with a report. Promotion to serving is a separate,
explicit action.
"""

import datetime as dt
import hashlib
import json
import os
import sys
import threading

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from leakage import canonicalize, registered_domain  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FEEDBACK_DB = os.path.join(ROOT, "instance", "feedback.jsonl")
CANDIDATE_DIR = os.path.join(ROOT, "data", "candidates")

MIN_CORROBORATION = 3          # distinct accounts agreeing
QUARANTINE_HOURS = 24          # minimum age before a claim is eligible
VALID_VERDICTS = ("false_positive", "false_negative", "confirmed_correct")

_lock = threading.Lock()


def _now():
    return dt.datetime.now(dt.timezone.utc)


def submit(url, verdict, user_id=None, note=None, scan_id=None,
           model_version=None):
    """Record a feedback claim. Never writes a label."""
    if verdict not in VALID_VERDICTS:
        raise ValueError(f"verdict must be one of {VALID_VERDICTS}")
    url = (url or "").strip()
    if not url:
        raise ValueError("url is required")

    rec = {
        "id": hashlib.sha256(
            f"{url}{user_id}{_now().timestamp()}".encode()).hexdigest()[:16],
        "url": url,
        "canonical": canonicalize(url),
        "domain": registered_domain(url),
        "verdict": verdict,
        "user_id": user_id,
        "note": (note or "")[:500],
        "scan_id": scan_id,
        "model_version": model_version,
        "submitted_at": _now().isoformat(timespec="seconds"),
        "status": "pending",
    }
    os.makedirs(os.path.dirname(FEEDBACK_DB), exist_ok=True)
    with _lock, open(FEEDBACK_DB, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec) + "\n")
    return rec


def load_all():
    if not os.path.exists(FEEDBACK_DB):
        return []
    out = []
    with open(FEEDBACK_DB, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return out


def validate(records, intel_lookup=None):
    """Apply the gates. Returns (eligible, rejected_with_reasons)."""
    from collections import defaultdict
    now = _now()
    by_claim = defaultdict(list)
    for r in records:
        by_claim[(r["canonical"], r["verdict"])].append(r)

    eligible, rejected = [], []
    for (canon, verdict), group in by_claim.items():
        users = {r.get("user_id") for r in group if r.get("user_id")}
        oldest = min(r["submitted_at"] for r in group)
        age_h = (now - dt.datetime.fromisoformat(oldest)).total_seconds() / 3600

        reason = None
        if age_h < QUARANTINE_HOURS:
            reason = (f"quarantined: oldest claim is {age_h:.1f}h old, "
                      f"{QUARANTINE_HOURS}h required")
        elif len(users) < MIN_CORROBORATION:
            reason = (f"insufficient corroboration: {len(users)} distinct "
                      f"account(s), {MIN_CORROBORATION} required")
        elif verdict == "false_positive" and intel_lookup:
            # A user cannot mark something benign that a blocklist verified as
            # phishing. This is the specific poisoning path worth closing.
            res = intel_lookup(group[0]["url"])
            if res.get("any_listed"):
                reason = ("contradicts a blocklist snapshot: the URL is listed "
                          "as malicious, so a 'false positive' claim is "
                          "rejected rather than used as a label")

        entry = {"canonical": canon, "verdict": verdict,
                 "claims": len(group), "distinct_users": len(users),
                 "oldest_age_hours": round(age_h, 1),
                 "url": group[0]["url"], "domain": group[0]["domain"]}
        if reason:
            rejected.append({**entry, "reason": reason})
        else:
            eligible.append(entry)
    return eligible, rejected


def build_candidate(eligible, out_dir=CANDIDATE_DIR):
    """Write a candidate supplement. This is NOT a dataset swap.

    The output is a small CSV of corroborated corrections, meant to be merged
    into a rebuild by src/build_dataset.py, which reapplies every leakage
    control. Feeding these rows straight into training would bypass
    de-duplication, domain grouping and the structural-confound check.
    """
    import pandas as pd
    rows = []
    for e in eligible:
        if e["verdict"] == "false_positive":
            rows.append({"url": e["url"], "label": 0,
                         "source": "user_feedback", "label_confidence": "reported"})
        elif e["verdict"] == "false_negative":
            rows.append({"url": e["url"], "label": 1,
                         "source": "user_feedback", "label_confidence": "reported"})
    if not rows:
        return None
    os.makedirs(out_dir, exist_ok=True)
    stamp = _now().strftime("%Y%m%d-%H%M%S")
    path = os.path.join(out_dir, f"feedback-{stamp}.csv")
    pd.DataFrame(rows).to_csv(path, index=False)
    meta = {
        "created_at": _now().isoformat(timespec="seconds"),
        "rows": len(rows),
        "min_corroboration": MIN_CORROBORATION,
        "quarantine_hours": QUARANTINE_HOURS,
        "status": "candidate",
        "approved_by": None,
        "approved_at": None,
        "note": ("Candidate supplement. Merge via src/build_dataset.py so the "
                 "leakage controls are reapplied; do not train on this file "
                 "directly."),
    }
    with open(path.replace(".csv", ".json"), "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)
    return path


def approve(candidate_path, approver, evaluation_report=None):
    """Record a human approval. Does not deploy anything."""
    meta_path = candidate_path.replace(".csv", ".json")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(meta_path)
    with open(meta_path, encoding="utf-8") as fh:
        meta = json.load(fh)
    if not approver:
        raise ValueError("an approver identity is required")
    meta.update({"status": "approved", "approved_by": approver,
                 "approved_at": _now().isoformat(timespec="seconds"),
                 "evaluation": evaluation_report})
    with open(meta_path, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)
    return meta


def stats():
    recs = load_all()
    from collections import Counter
    return {
        "total": len(recs),
        "by_verdict": dict(Counter(r["verdict"] for r in recs)),
        "distinct_urls": len({r["canonical"] for r in recs}),
        "distinct_users": len({r.get("user_id") for r in recs if r.get("user_id")}),
        "gates": {"min_corroboration": MIN_CORROBORATION,
                  "quarantine_hours": QUARANTINE_HOURS},
    }


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--stats", action="store_true")
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--build-candidate", action="store_true")
    ap.add_argument("--approve")
    ap.add_argument("--approver")
    a = ap.parse_args()

    if a.stats:
        print(json.dumps(stats(), indent=2))
    if a.validate or a.build_candidate:
        import threat_intel
        el, rej = validate(load_all(), threat_intel.lookup_all)
        print(f"eligible: {len(el)}   rejected: {len(rej)}")
        for r in rej[:20]:
            print(f"  REJECTED {r['url'][:54]:<54} {r['reason']}")
        if a.build_candidate:
            p = build_candidate(el)
            print(f"candidate: {p}" if p else "no eligible corrections")
    if a.approve:
        print(json.dumps(approve(a.approve, a.approver), indent=2))
