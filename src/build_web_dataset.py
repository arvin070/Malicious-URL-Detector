"""
Build the website-content dataset by safely fetching pages for labelled URLs.

HOW THE LABELS WORK, AND WHY THAT MATTERS
-----------------------------------------
Labels are INHERITED from the URL dataset: a page fetched from a URL PhishTank
verified as phishing is labelled phishing. That is weak labelling and it is
stated plainly rather than implied to be page-level ground truth. Two
consequences are reported, not hidden:

  * A taken-down phishing URL often now serves a hosting-provider parking page
    or a 404. Fetching it yields benign-looking HTML under a malicious label.
    Rows whose final page is a parking/suspension notice are detected and
    dropped where recognisable, and the rest is a known source of label noise.

  * Benign URLs that have since gone dead are simply lost, which biases the
    benign side toward sites that are still up.

SPLITS ARE INHERITED, NOT RECOMPUTED
------------------------------------
Each row keeps the train/val/holdout assignment its URL already had in
data/urlguard_dataset.csv. Recomputing them here would risk putting a domain on
the URL model's train side and the website model's test side, which is exactly
the leak the whole pipeline exists to prevent.

SAFETY
------
Fetching goes through src/web_fetch.py: GET only, no JavaScript, no cookies, no
form submission, SSRF-guarded, size- and time-capped. Nothing on any fetched
page is executed.

Usage:
    python src/build_web_dataset.py --limit 4000
    python src/build_web_dataset.py --workers 16 --refresh
"""

import argparse
import datetime as dt
import hashlib
import json
import os
import sys
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from web_fetch import fetch, FetchError  # noqa: E402
from web_features import extract_web_features, WEB_FEATURE_NAMES  # noqa: E402
from js_analysis import JS_FEATURE_NAMES  # noqa: E402
from redirect_analysis import REDIRECT_FEATURE_NAMES  # noqa: E402
from page_layers import analyse_page  # noqa: E402

# Phrases that mean "this is not the page the label refers to any more".
# A suspended phishing site serving a registrar parking page would otherwise
# teach the model that phishing pages look like parking pages.
PARKED_MARKERS = [
    "this domain has been seized", "domain for sale", "buy this domain",
    "this account has been suspended", "account suspended",
    "website is parked", "parked domain", "domain parking",
    "coming soon", "under construction", "default web page",
    "apache2 ubuntu default page", "welcome to nginx",
    "index of /", "it works!", "future home of",
    "this site has been blocked", "site temporarily unavailable",
]

_print_lock = threading.Lock()


def looks_parked(html: str, text_len: int) -> bool:
    low = (html or "").lower()
    if any(m in low for m in PARKED_MARKERS):
        return True
    # Almost no content at all: nothing for a page model to learn from.
    return text_len < 40


def fetch_one(row, max_bytes):
    url = row["url"]
    rec = {
        "url": url, "label": int(row["label"]), "source": row["source"],
        "label_confidence": row["label_confidence"],
        "registered_domain": row["registered_domain"],
        "split": row["split"],
    }
    try:
        res = fetch(url, max_bytes=max_bytes)
    except FetchError as exc:
        rec.update({"ok": 0, "error_code": exc.code, "error": exc.reason})
        return rec
    except Exception as exc:                      # noqa: BLE001
        rec.update({"ok": 0, "error_code": "unexpected",
                    "error": f"{type(exc).__name__}: {exc}"})
        return rec

    try:
        feats = extract_web_features(res["html"], res["final_url"], res)
        # JS, redirect and structural fingerprints are computed HERE, while the
        # HTML is in hand. Storing the raw HTML instead would mean keeping
        # thousands of live phishing pages on disk; the derived features carry
        # what the models need without that.
        # Delegated to src/page_layers.py so the server computes these the
        # same way; a divergence here would feed the model different numbers
        # at serving time than it was trained on.
        layers = analyse_page(res)
        js, rd, fp = layers["js"], layers["redirects"], layers["fingerprint"]
    except Exception as exc:                      # noqa: BLE001
        rec.update({"ok": 0, "error_code": "parse_failed",
                    "error": f"{type(exc).__name__}: {exc}"})
        return rec

    rec.update({
        "ok": 1, "error_code": "", "error": "",
        "final_url": res["final_url"],
        "http_status": res["status"],
        "fetched_at": res["fetched_at"],
        "fetch_ms": res["elapsed_ms"],
        "parked": int(looks_parked(res["html"], feats["text_length"])),
        "tag_simhash": str(fp["tag_simhash"]),
        "form_sha": fp["form_sha"],
        **feats,
        **js["features"],
        **rd["features"],
    })
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--urls", default="data/urlguard_dataset.csv")
    ap.add_argument("--out", default="data/urlguard_web_dataset.csv")
    ap.add_argument("--manifest", default="data/web_dataset_manifest.json")
    ap.add_argument("--limit", type=int, default=6000,
                    help="URLs attempted (balanced across classes)")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--malicious-multiplier", type=float, default=4.0,
                    help="attempt this many malicious URLs per benign one. "
                         "Measured retrieval is ~75%% for benign and ~20%% for "
                         "malicious because phishing sites get taken down, so "
                         "an even split yields a badly skewed usable set.")
    ap.add_argument("--max-bytes", type=int, default=800_000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--refresh", action="store_true",
                    help="ignore an existing output file and refetch")
    args = ap.parse_args()

    if os.path.exists(args.out) and not args.refresh:
        print(f"{args.out} exists; use --refresh to refetch.")
        return

    urls = pd.read_csv(args.urls)

    # Malicious URLs are sampled NEWEST-FIRST where a date exists: a phishing
    # URL reported last week is far more likely to still serve its page than
    # one reported months ago. Benign URLs are sampled at random because they
    # do not decay the same way.
    mal = urls[urls.label == 1].copy()
    if "first_seen" in mal.columns:
        mal["_ts"] = pd.to_datetime(mal["first_seen"], utc=True, errors="coerce")
        mal = mal.sort_values("_ts", ascending=False, na_position="last")
    ben = urls[urls.label == 0].sample(frac=1, random_state=args.seed)

    m = args.malicious_multiplier
    n_ben = max(int(args.limit / (1 + m)), 1)
    n_mal = max(args.limit - n_ben, 1)
    sample = pd.concat([ben.head(n_ben), mal.head(n_mal)]) \
        .sample(frac=1, random_state=args.seed).reset_index(drop=True)

    print("=" * 78)
    print("URLGUARD WEBSITE DATASET BUILD")
    print("=" * 78)
    print(f"  source URLs    {args.urls}")
    print(f"  attempting     {len(sample):,} pages "
          f"({int((sample.label == 1).sum()):,} malicious / "
          f"{int((sample.label == 0).sum()):,} benign)")
    print(f"  workers        {args.workers}   max {args.max_bytes:,} bytes/page")
    print("  fetching is GET-only, no JavaScript, no cookies, no form "
          "submission,\n  SSRF-guarded. Nothing on any page is executed.\n")

    started = time.time()
    rows = []
    done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(fetch_one, r, args.max_bytes): i
                   for i, r in sample.iterrows()}
        for fut in as_completed(futures):
            rows.append(fut.result())
            done += 1
            if done % 250 == 0:
                ok = sum(r["ok"] for r in rows)
                with _print_lock:
                    print(f"    {done:>5,}/{len(sample):,}  "
                          f"retrieved {ok:,}  ({time.time()-started:.0f}s)")

    df = pd.DataFrame(rows)
    attempted = len(df)
    errors = Counter(df.loc[df.ok == 0, "error_code"])
    ok = df[df.ok == 1].copy()

    print(f"\n  attempted      {attempted:,}")
    print(f"  retrieved      {len(ok):,}  ({len(ok)/attempted:.1%})")
    print("  failures:")
    for code, n in errors.most_common():
        print(f"    {code:<22} {n:>6,}")

    parked = int(ok["parked"].sum())
    ok = ok[ok.parked == 0].reset_index(drop=True)
    print(f"\n  dropped as parked/empty  {parked:,}")
    print(f"  usable pages             {len(ok):,}")

    if len(ok) < 200:
        print("\n  Too few pages retrieved to train a website model.")
    by_label = ok.label.value_counts().to_dict()
    print(f"  by label       {by_label}")
    print(f"  by split       {ok['split'].value_counts().to_dict()}")

    # Retrieval rate per class is itself a finding: if phishing pages are far
    # harder to retrieve (taken down), the surviving sample is not a random
    # sample of phishing.
    rate = (df.groupby("label")["ok"].mean() * 100).round(1).to_dict()
    print(f"  retrieval rate by label (%)  {rate}")

    cols = (["url", "label", "source", "label_confidence", "registered_domain",
             "split", "final_url", "http_status", "fetched_at", "fetch_ms",
             "tag_simhash", "form_sha"]
            + WEB_FEATURE_NAMES + JS_FEATURE_NAMES + REDIRECT_FEATURE_NAMES)
    ok[cols].to_csv(args.out, index=False)

    manifest = {
        "dataset_version": f"URLGuard-Web-{dt.datetime.now(dt.timezone.utc):%Y-%m-%d}",
        "built_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "source_url_dataset": args.urls,
        "output": args.out,
        "sha256": hashlib.sha256(open(args.out, "rb").read()).hexdigest(),
        "n_features": len(WEB_FEATURE_NAMES),
        "n_js_features": len(JS_FEATURE_NAMES),
        "n_redirect_features": len(REDIRECT_FEATURE_NAMES),
        "attempted": attempted,
        "retrieved": int((df.ok == 1).sum()),
        "dropped_parked_or_empty": parked,
        "usable": len(ok),
        "by_label": {str(k): int(v) for k, v in by_label.items()},
        "by_split": {str(k): int(v) for k, v in ok["split"].value_counts().items()},
        "retrieval_rate_by_label_pct": {str(k): float(v) for k, v in rate.items()},
        "fetch_errors": dict(errors),
        "labelling": (
            "Labels are INHERITED from the URL dataset (weak labelling). A page "
            "is labelled phishing because its URL was, not because the page "
            "content was independently reviewed."),
        "known_biases": [
            "Taken-down phishing URLs often serve parking or error pages; "
            "recognisable ones are dropped, the rest is label noise.",
            "Benign URLs that have gone dead are lost, biasing the benign side "
            "toward sites still online.",
            "Retrieval rate differs by class, so the surviving sample is not a "
            "random sample of either class.",
            "Pages are parsed statically; JavaScript-rendered content is "
            "invisible to these features.",
        ],
        "safety": [
            "GET only; no form submitted and no credential ever sent.",
            "No JavaScript executed; HTML parsed as a static document.",
            "No cookies stored or returned.",
            "SSRF-guarded: private, loopback, link-local, multicast and "
            "reserved addresses refused, re-checked on every redirect.",
            f"Capped at {args.max_bytes:,} bytes and 5 redirects per page.",
        ],
        "splits_inherited_from": args.urls,
    }
    with open(args.manifest, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, default=str)

    print(f"\n  wrote {args.out}")
    print(f"  manifest -> {args.manifest}")
    print(f"  elapsed {time.time()-started:.0f}s")
    print("=" * 78)


if __name__ == "__main__":
    main()
