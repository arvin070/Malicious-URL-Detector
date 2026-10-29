"""
Downloads a REAL phishing / legitimate URL dataset and writes data/real_urls.csv.

Source
------
Hannousse, A. & Yahiouche, S. (2021), "Towards benchmark datasets for machine
learning based website phishing detection: An experimental study",
Computers & Security 2021.  arXiv:2010.12847
Mirrored on the Hugging Face Hub as `pirocheto/phishing-url` (CC-BY-4.0).

The published corpus has 11,430 URLs with 87 pre-computed features.  We keep
ONLY the raw `url` string and the `status` label and recompute every feature
ourselves with src/features.py, so the project's own feature extractor stays
the single source of truth.

Provenance of the labels (per the paper):
  phishing    -- PhishTank and OpenPhish verified phishing feeds
  legitimate  -- Alexa / Tranco top-sites lists plus manual curation

SAFETY: URLs are handled purely as strings.  Nothing here (or anywhere else in
this project) resolves, requests, crawls or executes a dataset URL.

Usage:
    python src/download_real_dataset.py
    python src/download_real_dataset.py --out data/real_urls.csv
"""

import argparse
import io
import os
import re
import sys
import urllib.request

import pandas as pd

BASE = (
    "https://huggingface.co/datasets/pirocheto/phishing-url/"
    "resolve/refs%2Fconvert%2Fparquet/default"
)
PARQUET_FILES = [f"{BASE}/train/0000.parquet", f"{BASE}/test/0000.parquet"]

# A URL we are willing to featurise: has a host, no whitespace, http(s) or bare.
_VALID = re.compile(r"^(?:https?://)?[^\s/?#@]+\.[^\s/?#]{2,}", re.IGNORECASE)


def _fetch_parquet(url: str) -> pd.DataFrame:
    print(f"  fetching {url.rsplit('/', 3)[-2]}/{url.rsplit('/', 1)[-1]} ...")
    req = urllib.request.Request(url, headers={"User-Agent": "phishing-url-monitor"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        raw = resp.read()
    return pd.read_parquet(io.BytesIO(raw), columns=["url", "status"])


def load_raw() -> pd.DataFrame:
    """Parquet export first (no extra deps); fall back to the datasets library."""
    try:
        frames = [_fetch_parquet(u) for u in PARQUET_FILES]
        return pd.concat(frames, ignore_index=True)
    except Exception as exc:  # noqa: BLE001
        print(f"  parquet download failed ({exc}); trying `datasets` ...")
        try:
            from datasets import load_dataset
        except ImportError:
            raise SystemExit(
                "Could not download the dataset and `datasets` is not installed.\n"
                "Run:  pip install datasets   and try again."
            ) from exc
        ds = load_dataset("pirocheto/phishing-url")
        return pd.concat(
            [ds[s].to_pandas()[["url", "status"]] for s in ds.keys()],
            ignore_index=True,
        )


def clean(df: pd.DataFrame) -> pd.DataFrame:
    report = []
    n0 = len(df)
    report.append(("rows downloaded", n0))

    # --- normalise labels: phishing -> 1, legitimate -> 0 -------------------
    status = df["status"].astype(str).str.strip().str.lower()
    mapping = {
        "phishing": 1, "phish": 1, "malicious": 1, "bad": 1, "1": 1,
        "legitimate": 0, "legit": 0, "benign": 0, "good": 0, "0": 0,
    }
    unknown = sorted(set(status) - set(mapping))
    if unknown:
        print(f"  WARNING: dropping unrecognised label values: {unknown}")
    df = df.assign(label=status.map(mapping))
    df = df[df["label"].notna()]
    df["label"] = df["label"].astype(int)
    report.append(("after label normalisation", len(df)))

    # --- drop empty / whitespace-only / structurally invalid URLs -----------
    df["url"] = df["url"].astype(str).str.strip()
    df = df[df["url"].str.len() > 0]
    df = df[~df["url"].str.lower().isin({"nan", "none", "null"})]
    report.append(("after dropping empty URLs", len(df)))

    df = df[df["url"].str.match(_VALID).fillna(False)]
    report.append(("after dropping invalid URLs", len(df)))

    # --- exact duplicate URLs ----------------------------------------------
    df = df.drop_duplicates(subset="url", keep="first")
    report.append(("after dropping duplicate URLs", len(df)))

    # --- same URL carrying both labels: ambiguous, drop entirely ------------
    conflict = df.groupby("url")["label"].nunique()
    bad = set(conflict[conflict > 1].index)
    if bad:
        df = df[~df["url"].isin(bad)]
    report.append(("after dropping label-conflicting URLs", len(df)))

    print("\n--- preprocessing ---")
    prev = None
    for name, n in report:
        delta = "" if prev is None else f"   ({n - prev:+d})"
        print(f"  {name:<40} {n:>7,}{delta}")
        prev = n
    return df[["url", "label"]].reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/real_urls.csv")
    args = ap.parse_args()

    print("Downloading real phishing/legitimate URL dataset")
    print("  source: pirocheto/phishing-url (Hannousse & Yahiouche 2021, CC-BY-4.0)")
    df = clean(load_raw())

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    df.to_csv(args.out, index=False)

    n = len(df)
    n_phish = int(df["label"].sum())
    n_benign = n - n_phish
    print(f"\nWrote {n:,} rows to {args.out}")
    print(f"  benign   (0): {n_benign:,}  ({n_benign / n:.1%})")
    print(f"  phishing (1): {n_phish:,}  ({n_phish / n:.1%})")
    if n == 0:
        sys.exit("No usable rows survived cleaning -- aborting.")


if __name__ == "__main__":
    main()
