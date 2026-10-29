"""
Mine HARD NEGATIVES: legitimate URLs that structurally resemble phishing.

Why: error analysis of the baseline showed the dominant failure mode is benign
deep links being flagged (FPR 0.3215-0.4670 on unseen data).  The cause is the
training set's benign class, which comes from Alexa/Tranco top-site lists and
is therefore overwhelmingly *homepages* -- short, shallow, few digits, no query
string.  The models consequently learned "long / deep / query-heavy => phishing",
which describes deep links, not maliciousness.

Fix: add legitimate URLs that look phishing-like, from a THIRD corpus, so the
benign class covers the shape that was producing false positives.

Source: Mitake/PhishingURLsANDBenignURLs (CC-BY-SA-4.0), benign rows only.

LEAKAGE CONTROL (strict):
  * drop any URL already in the training corpus (canonical match)
  * drop any URL in the independent validation corpus (canonical match)
  * drop any URL whose REGISTERED DOMAIN appears in the validation corpus
    -- domain-level exclusion, so no validation domain is ever trained on
  * de-duplicate, and cap per-domain so a few domains cannot dominate

SAFETY: URLs are strings only.  Nothing is resolved, fetched or executed.

Usage:
    python src/build_hard_negatives.py                 # -> data/hard_negatives.csv
    python src/build_hard_negatives.py --n 3000
"""

import argparse
import io
import os
import re
import sys
import urllib.request
from collections import defaultdict

import pandas as pd
import tldextract

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

_extract = tldextract.TLDExtract(suffix_list_urls=())

SRC = ("https://huggingface.co/datasets/Mitake/PhishingURLsANDBenignURLs/"
       "resolve/refs%2Fconvert%2Fparquet/default/train/0000.parquet")
EXT_VAL = ("https://huggingface.co/datasets/shawhin/phishing-site-classification/"
           "resolve/refs%2Fconvert%2Fparquet/default")

_VALID = re.compile(r"^(?:https?://)?[^\s/?#@]+\.[^\s/?#]{2,}", re.IGNORECASE)


def canon(u: str) -> str:
    u = str(u).strip().lower()
    u = re.sub(r"^https?://", "", u)
    u = re.sub(r"^www\.", "", u)
    return u.rstrip("/")


def dom(u: str) -> str:
    return _extract(str(u)).top_domain_under_public_suffix or str(u)


def _get(url: str) -> pd.DataFrame:
    req = urllib.request.Request(url, headers={"User-Agent": "phishing-url-monitor"})
    with urllib.request.urlopen(req, timeout=300) as r:
        return pd.read_parquet(io.BytesIO(r.read()))


def phish_likeness(u: str) -> float:
    """How much does this benign URL look like the baseline's false positives?

    Built from the measured FP profile (long, deep, query-heavy, digit-dense),
    NOT from model scores -- so no model is used to select its own training data.
    """
    s = str(u)
    path = s.split("?", 1)[0]
    query = s.split("?", 1)[1] if "?" in s else ""
    depth = len([p for p in path.split("/")[1:] if p])
    digits = sum(c.isdigit() for c in s)
    return (
        min(len(s) / 100.0, 2.0) * 2.0
        + min(depth / 4.0, 2.0) * 1.5
        + min(len(query) / 50.0, 2.0) * 2.0
        + min(digits / 15.0, 2.0) * 1.0
        + min(s.count("-") / 5.0, 1.5)
        + min(s.count("%") / 3.0, 1.5)
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=3000, help="hard negatives to keep")
    ap.add_argument("--per-domain", type=int, default=3, help="cap per registered domain")
    ap.add_argument("--train-data", default="data/real_urls.csv")
    ap.add_argument("--out", default="data/hard_negatives.csv")
    args = ap.parse_args()

    print("Mining hard negatives (legitimate URLs that look phishing-like)")
    print("  source: Mitake/PhishingURLsANDBenignURLs (CC-BY-SA-4.0), benign rows only")

    df = _get(SRC)
    df = df[df["label"] == 0][["url"]].copy()
    n0 = len(df)

    df["url"] = df["url"].astype(str).str.strip()
    df = df[df["url"].str.len() > 0]
    df = df[df["url"].str.match(_VALID).fillna(False)]
    df = df.drop_duplicates(subset="url")
    n1 = len(df)

    # ---- leakage control --------------------------------------------------
    train = pd.read_csv(args.train_data)
    train_keys = set(train["url"].astype(str).map(canon))

    val = pd.concat([_get(f"{EXT_VAL}/{s}/0000.parquet")
                     for s in ("train", "validation", "test")], ignore_index=True)
    val = val.rename(columns={"text": "url"})
    val_keys = set(val["url"].astype(str).map(canon))
    val_domains = set(val["url"].astype(str).map(dom))

    df["_k"] = df["url"].map(canon)
    df = df[~df["_k"].isin(train_keys)]
    n2 = len(df)
    df = df[~df["_k"].isin(val_keys)]
    n3 = len(df)
    df["_d"] = df["url"].map(dom)
    df = df[~df["_d"].isin(val_domains)]
    n4 = len(df)

    # ---- rank by resemblance to the baseline's false positives ------------
    df["_score"] = df["url"].map(phish_likeness)
    df = df.sort_values("_score", ascending=False)

    kept, per = [], defaultdict(int)
    for _, r in df.iterrows():
        if per[r["_d"]] >= args.per_domain:
            continue
        per[r["_d"]] += 1
        kept.append(r["url"])
        if len(kept) >= args.n:
            break

    out = pd.DataFrame({"url": kept, "label": 0})
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    out.to_csv(args.out, index=False)

    print("\n--- selection ---")
    for name, n in [("benign rows in source", n0),
                    ("after cleaning + dedupe", n1),
                    ("after removing training URLs", n2),
                    ("after removing validation URLs", n3),
                    ("after removing validation DOMAINS", n4),
                    (f"kept (top phish-like, <={args.per_domain}/domain)", len(out))]:
        print(f"  {name:<44}{n:>9,}")
    print(f"\n  distinct domains kept: {out['url'].map(dom).nunique():,}")
    print(f"  mean URL length: {out['url'].str.len().mean():.1f} "
          f"(training benign: {train[train.label==0]['url'].str.len().mean():.1f})")
    print(f"\nWrote {len(out):,} hard negatives to {args.out}")
    print("Examples:")
    for u in out["url"].head(5):
        print(f"  {u[:100]}")


if __name__ == "__main__":
    main()
