"""
Run the hard-benign and adversarial suites against the URL model.

URL-only by design: these suites are about the layer that runs on every scan,
on every device, with no network. Fetching 60 pages would also make the result
depend on whether those sites happened to be up.

Nothing is fetched. Every case is a URL string.

Usage:
    python src/run_benchmarks.py
    python src/run_benchmarks.py --bundle models/v3/url_models.joblib
"""

import argparse
import json
import os
import sys
import time

import joblib
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from features import extract_features, FEATURE_NAMES  # noqa: E402
import benchmarks as B  # noqa: E402


def score_urls(model, feat_names, urls):
    X = pd.DataFrame([extract_features(u) for u in urls],
                     columns=FEATURE_NAMES)[feat_names]
    return model.predict_proba(X)[:, 1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", default="models/v3/url_models.joblib")
    ap.add_argument("--out", default="models/v3/benchmarks.json")
    args = ap.parse_args()

    b = joblib.load(args.bundle)
    model = b["models"][b["best_model_name"]]
    thr = float(b["threshold"])
    fn = b["feature_names"]

    print("=" * 78)
    print("URLGUARD BENCHMARKS  (URL layer)")
    print("=" * 78)
    print(f"  model     {b.get('model_version')}  threshold {thr}")

    # ---------------- hard benign ----------------
    urls = [u for u, _c in B.ALL_HARD_BENIGN]
    cats = [c for _u, c in B.ALL_HARD_BENIGN]
    t0 = time.perf_counter()
    scores = score_urls(model, fn, urls)
    latency_ms = (time.perf_counter() - t0) / len(urls) * 1000

    hb_rows = [{"url": u, "category": c, "score": int(round(s * 100)),
                "flagged": bool(s >= thr), "expect": "benign"}
               for u, c, s in zip(urls, cats, scores)]
    hb = B.summarise(hb_rows)
    n_flag = sum(r["flagged"] for r in hb_rows)

    print(f"\n[1] Hard-benign suite -- {len(hb_rows)} well-known sites")
    print(f"    false positives: {n_flag}/{len(hb_rows)} "
          f"({n_flag/len(hb_rows):.1%})")
    print(f"    {'category':<20}{'cases':>7}{'flagged':>9}{'rate':>8}"
          f"{'mean':>7}{'max':>6}")
    for cat in sorted(hb):
        d = hb[cat]
        print(f"    {cat:<20}{d['cases']:>7}{d['incorrect']:>9}"
              f"{d['error_rate']:>8.1%}{d['mean_score']:>7.0f}{d['max_score']:>6}")
    if n_flag:
        print("\n    flagged (these are false positives):")
        for r in sorted(hb_rows, key=lambda r: -r["score"]):
            if r["flagged"]:
                print(f"      {r['score']:>3}  {r['url']}")

    # ---------------- adversarial ----------------
    cases = B.default_adversarial()
    ascores = score_urls(model, fn, [c["url"] for c in cases])
    ad_rows = [{**c, "score": int(round(s * 100)), "flagged": bool(s >= thr)}
               for c, s in zip(cases, ascores)]
    ad = B.summarise(ad_rows)
    wrong = sum(1 for r in ad_rows
                if (r["expect"] == "benign") == r["flagged"])

    print(f"\n[2] Adversarial suite -- {len(ad_rows)} transformed URLs")
    print(f"    incorrect: {wrong}/{len(ad_rows)}")
    print(f"    {'family':<24}{'cases':>7}{'wrong':>7}{'rate':>8}{'mean':>7}")
    for fam in sorted(ad):
        d = ad[fam]
        print(f"    {fam:<24}{d['cases']:>7}{d['incorrect']:>7}"
              f"{d['error_rate']:>8.1%}{d['mean_score']:>7.0f}")

    bad = [r for r in ad_rows
           if (r["expect"] == "benign") == r["flagged"]]
    if bad:
        print("\n    incorrect cases:")
        for r in bad[:18]:
            verb = "flagged" if r["flagged"] else "missed"
            print(f"      {r['score']:>3} {verb:<8} [{r['family']}] {r['url'][:62]}")

    print(f"\n[3] Latency: {latency_ms:.2f} ms/URL (URL layer, single-row)")

    out = {
        "model_version": b.get("model_version"),
        "dataset_version": b.get("dataset_version"),
        "threshold": thr,
        "generated_at": pd.Timestamp.utcnow().isoformat(),
        "hard_benign": {
            "cases": len(hb_rows), "false_positives": int(n_flag),
            "fpr": round(n_flag / len(hb_rows), 4),
            "by_category": hb,
            "flagged": [r for r in hb_rows if r["flagged"]],
        },
        "adversarial": {
            "cases": len(ad_rows), "incorrect": int(wrong),
            "error_rate": round(wrong / len(ad_rows), 4),
            "by_family": ad,
            "incorrect_cases": bad,
        },
        "latency_ms_per_url": round(latency_ms, 3),
        "note": ("Hard-benign and adversarial cases are fixed probes. They are "
                 "never used for training or threshold selection -- doing so "
                 "would turn the benchmark into a fitting target."),
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2, default=str)
    print(f"\n    wrote {args.out}")
    print("=" * 78)


if __name__ == "__main__":
    main()
