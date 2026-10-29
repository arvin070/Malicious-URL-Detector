"""
Re-derive the website dataset's train/val/holdout assignment.

WHY THIS EXISTS
---------------
The page corpus is expensive to collect (thousands of fetches, ~30% of which
succeed), so it is not rebuilt every time the URL dataset is. But its splits
were *inherited* from the URL dataset row by row, and that inheritance is what
keeps the fusion stacker honest: the stacker is fitted on validation rows and
tested on holdout rows, and it is only meaningful if the base URL and website
models never trained on those domains.

When the URL dataset is rebuilt the inheritance goes stale. After the
registered-domain grouping fix, only 748 of 2,704 page URLs still appeared in
the rebuilt URL dataset and 775 of the matchable rows disagreed about which
split they belonged to. Left alone, that would have put base-model training
domains into the stacker's test set.

WHAT IT DOES
------------
  1. recomputes the grouping key with the current registered_domain();
  2. takes the split from the URL dataset wherever that domain is known, so
     a domain the URL model trained on stays on the training side;
  3. assigns the remaining domains -- ones the URL model has never seen at
     all -- by a deterministic hash of the domain, so the assignment is
     reproducible and no domain can straddle two splits;
  4. asserts both properties afterwards rather than trusting them.

It never refetches anything and never changes a label.
"""

import argparse
import hashlib
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from leakage import registered_domain  # noqa: E402

# Proportions used only for domains the URL dataset has never seen. Weighted
# towards evaluation because unseen domains are worth more as test evidence
# than as additional training rows.
UNSEEN_WEIGHTS = [("train", 0.50), ("val", 0.20), ("holdout", 0.30)]


def _stable_bucket(domain: str) -> str:
    """Deterministic split for one domain.

    Uses sha256 rather than hash(): Python's hash() is salted per process, so
    the same domain would land in different splits on different runs and the
    dataset would stop being reproducible.
    """
    h = hashlib.sha256(domain.encode("utf-8")).digest()
    x = int.from_bytes(h[:8], "big") / float(1 << 64)
    acc = 0.0
    for name, w in UNSEEN_WEIGHTS:
        acc += w
        if x < acc:
            return name
    return UNSEEN_WEIGHTS[-1][0]


def resplit(web_path, urls_path, out_path=None):
    web = pd.read_csv(web_path)
    urls = pd.read_csv(urls_path, usecols=["url", "split"])

    urls["g"] = urls["url"].astype(str).map(registered_domain)
    web["registered_domain"] = web["url"].astype(str).map(registered_domain)

    # One split per domain. Where a domain appears several times in the URL
    # dataset it is already single-split (build_dataset guarantees it), so
    # taking the first is safe; assert it rather than assume.
    per_domain = urls.groupby("g")["split"].agg(["nunique", "first"])
    straddling = int((per_domain["nunique"] > 1).sum())
    if straddling:
        raise SystemExit(
            f"the URL dataset itself has {straddling} domains in more than "
            f"one split; rebuild it before re-splitting the page corpus")
    known = per_domain["first"].to_dict()

    before = web["split"].value_counts().to_dict()
    inherited = web["registered_domain"].map(known)
    n_inherited = int(inherited.notna().sum())
    web["split"] = inherited.where(
        inherited.notna(),
        web["registered_domain"].map(_stable_bucket))

    # ---- assertions, not assumptions -----------------------------------
    spread = web.groupby("registered_domain")["split"].nunique()
    assert int((spread > 1).sum()) == 0, "a domain ended up in two splits"

    url_train = set(urls.loc[urls["split"] == "train", "g"])
    evaluated = set(web.loc[web["split"].isin(["val", "holdout"]),
                            "registered_domain"])
    overlap = evaluated & url_train
    assert not overlap, (
        f"{len(overlap)} domains the URL model trained on landed in the "
        f"stacker's evaluation splits")

    out_path = out_path or web_path
    web.to_csv(out_path, index=False)

    after = web["split"].value_counts().to_dict()
    return {
        "rows": len(web),
        "inherited_from_url_dataset": n_inherited,
        "assigned_by_hash": len(web) - n_inherited,
        "before": before,
        "after": after,
        "domains": int(web["registered_domain"].nunique()),
        "url_train_domains_in_eval_splits": 0,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--web", default="data/urlguard_web_dataset.csv")
    ap.add_argument("--urls", default="data/urlguard_dataset.csv")
    ap.add_argument("--out")
    a = ap.parse_args()

    rep = resplit(a.web, a.urls, a.out)
    print("=" * 74)
    print("RE-SPLIT WEBSITE DATASET")
    print("=" * 74)
    print(f"  rows                      {rep['rows']:,}")
    print(f"  distinct domains          {rep['domains']:,}")
    print(f"  split inherited from URL  {rep['inherited_from_url_dataset']:,}")
    print(f"  assigned by domain hash   {rep['assigned_by_hash']:,}")
    print(f"  before                    {rep['before']}")
    print(f"  after                     {rep['after']}")
    print("  verified: no domain spans two splits, and no URL-model training")
    print("            domain appears in val or holdout")
    print("=" * 74)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
