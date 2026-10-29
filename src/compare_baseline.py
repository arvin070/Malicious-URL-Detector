"""
Baseline-vs-improved comparison, under ONE identical protocol.

WHY A SEPARATE SCRIPT
---------------------
models/baseline/ holds the v1 artifacts (21 features, Random Forest selected
from a single random 80/20 split). Those CSVs cannot be compared directly with
the new ones: they were produced on a different dataset snapshot, with
exact-string de-duplication, and with a different split rule. Putting the two
sets of numbers next to each other would compare three changes at once and
attribute all of it to the features.

This script instead re-evaluates each FEATURE SET on today's data, with the
same seeds, the same canonical de-duplication, the same domain groups and the
same folds. The only thing that differs between rows is the feature set, so
the difference between rows is attributable to the features.

  legacy     -- the original 21 features, unchanged
  structural -- legacy + the new lexical/structural features
  full       -- structural + hashed char/token n-grams

All three protocols are reported per feature set:
  random_8020      the original optimistic split, kept for comparability
  domain_disjoint  one GroupShuffleSplit holdout, no shared registered domain
  domain_cv_5fold  StratifiedGroupKFold, reported as mean +/- std over folds

The archived v1 numbers are emitted separately, as `models/baseline_archived.csv`,
clearly labelled as a different protocol -- preserved, never silently mixed in.

The independent unseen corpus is NOT touched here.

SAFETY: URLs are strings only. Nothing is resolved, fetched or executed.

Usage:
    python src/compare_baseline.py
    python src/compare_baseline.py --folds 5 --feature-sets legacy full
"""

import argparse
import os
import re
import sys

import numpy as np
import pandas as pd

from sklearn.model_selection import (train_test_split, GroupShuffleSplit,
                                     StratifiedGroupKFold)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from features import extract_features, FEATURE_NAMES, FEATURE_SETS  # noqa: E402
from leakage import (registered_domains, drop_exact_duplicates,  # noqa: E402
                     assert_no_domain_leakage, audit_folds)
from train import (make_models, metrics_at, fit_eval, mean_std_table,  # noqa: E402
                   ALL_METRICS, SEED)

_SCHEME = r"^[a-zA-Z][a-zA-Z0-9+.\-]*://"


def load_data(args):
    """Identical loading path to train.py, so the comparison is apples-to-apples."""
    df = pd.read_csv(args.data).dropna(subset=["url", "label"])
    df["label"] = df["label"].astype(int)
    df["url"] = df["url"].astype(str)

    if not args.no_hard_negatives and os.path.exists(args.hard_negatives):
        hn = pd.read_csv(args.hard_negatives).dropna(subset=["url"])
        hn["label"] = hn["label"].astype(int)
        hn["url"] = hn["url"].astype(str)
        hn = hn[~hn["url"].isin(set(df["url"]))].drop_duplicates(subset="url")
        df = pd.concat([df, hn], ignore_index=True)

    df, dup = drop_exact_duplicates(df)

    # same per-class scheme augmentation as train.py (breaks the
    # source/label confound between the two corpora)
    rng = np.random.default_rng(args.random_state)
    for lab in (0, 1):
        idx = df.index[df["label"] == lab]
        with_s = [i for i in idx if re.match(_SCHEME, df.at[i, "url"])]
        keep = int(round(args.scheme_rate * len(idx)))
        if keep < len(with_s):
            strip = rng.choice(with_s, size=len(with_s) - keep, replace=False)
            df.loc[strip, "url"] = df.loc[strip, "url"].str.replace(
                _SCHEME, "", regex=True)
    df, _ = drop_exact_duplicates(df)
    return df, dup


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/real_urls.csv")
    ap.add_argument("--hard-negatives", default="data/hard_negatives.csv")
    ap.add_argument("--no-hard-negatives", action="store_true")
    ap.add_argument("--model-dir", default="models")
    ap.add_argument("--baseline-dir", default="models/baseline")
    ap.add_argument("--feature-sets", nargs="+",
                    default=["legacy", "structural", "full"],
                    choices=sorted(FEATURE_SETS))
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument("--scheme-rate", type=float, default=0.40)
    ap.add_argument("--random-state", type=int, default=SEED)
    args = ap.parse_args()
    md = args.model_dir
    os.makedirs(md, exist_ok=True)

    df, dup = load_data(args)
    y = df["label"].to_numpy()
    groups = registered_domains(df["url"]).to_numpy()

    print("=" * 78)
    print("BASELINE vs IMPROVED -- one protocol, feature set is the only variable")
    print("=" * 78)
    print(f"  rows {len(df):,} ({dup['canonical_duplicates_removed']:,} canonical "
          f"duplicates removed) | {int(y.sum()):,} phishing / {len(y)-int(y.sum()):,} benign")
    print(f"  registered domains {pd.Series(groups).nunique():,}")
    print(f"  seed {args.random_state} | folds {args.folds}")

    # Extract once: every feature set is a column subset of the same matrix,
    # so extraction is not repeated per feature set.
    print("\n  extracting features (once, for all feature sets) ...")
    X_all = pd.DataFrame([extract_features(u) for u in df["url"].tolist()],
                         columns=FEATURE_NAMES)

    # one set of splits, shared by every feature set
    idx = np.arange(len(df))
    rtr, rte = train_test_split(idx, test_size=args.test_size,
                                random_state=args.random_state, stratify=y)
    gss = GroupShuffleSplit(n_splits=1, test_size=args.test_size,
                            random_state=args.random_state)
    gtr, gte = next(gss.split(idx, y, groups))
    assert_no_domain_leakage(groups, gtr, gte, "domain-disjoint holdout")
    sgkf = StratifiedGroupKFold(n_splits=args.folds, shuffle=True,
                                random_state=args.random_state)
    folds = list(sgkf.split(idx, y, groups))
    audit_folds(groups, folds, "StratifiedGroupKFold")
    print(f"  splits verified: domain-disjoint holdout and all {args.folds} "
          f"folds share 0 registered domains")

    rows = []
    for fs in args.feature_sets:
        names = FEATURE_SETS[fs]
        X = X_all[names]
        print(f"\n--- feature set '{fs}' ({len(names)} features) ---")

        for name, proto in make_models().items():
            # protocol 1: random 80/20
            _, p = fit_eval(proto, X.iloc[rtr], y[rtr], X.iloc[rte])
            m_rand = metrics_at(y[rte], p, 0.5)

            # protocol 2: domain-disjoint holdout
            _, p = fit_eval(proto, X.iloc[gtr], y[gtr], X.iloc[gte])
            m_dd = metrics_at(y[gte], p, 0.5)

            # protocol 3: domain-aware CV, per fold
            fold_rows = []
            for k, (tr, te) in enumerate(folds, 1):
                _, p = fit_eval(proto, X.iloc[tr], y[tr], X.iloc[te])
                fold_rows.append({"Algorithm": name, "fold": k,
                                  **metrics_at(y[te], p, 0.5)})
            ms = mean_std_table(pd.DataFrame(fold_rows)).loc[name]

            for metric in ALL_METRICS:
                rows.append({
                    "FeatureSet": fs,
                    "NumFeatures": len(names),
                    "Algorithm": name,
                    "Metric": metric,
                    "random_8020": round(m_rand[metric], 4),
                    "domain_disjoint": round(m_dd[metric], 4),
                    f"domain_cv_{args.folds}fold_mean": round(ms[f"{metric}_mean"], 4),
                    f"domain_cv_{args.folds}fold_std": round(ms[f"{metric}_std"], 4),
                })
            print(f"  {name:<20} CV ROC-AUC {ms['ROC-AUC_mean']:.4f}"
                  f"+/-{ms['ROC-AUC_std']:.4f}  "
                  f"F1 {ms['F1-Score_mean']:.4f}+/-{ms['F1-Score_std']:.4f}  "
                  f"FPR {ms['FPR_mean']:.4f}+/-{ms['FPR_std']:.4f}")

    out = pd.DataFrame(rows)
    out.to_csv(f"{md}/baseline_vs_improved.csv", index=False)
    print(f"\nWrote {md}/baseline_vs_improved.csv  ({len(out)} rows)")

    # ---- headline view: is the difference bigger than the fold noise? ------
    mean_col = f"domain_cv_{args.folds}fold_mean"
    std_col = f"domain_cv_{args.folds}fold_std"
    print("\n=== Domain-aware CV, mean +/- std by feature set ===")
    for metric in ("ROC-AUC", "F1-Score", "FPR"):
        sub = out[out.Metric == metric]
        piv = sub.pivot(index="Algorithm", columns="FeatureSet", values=mean_col)
        pivs = sub.pivot(index="Algorithm", columns="FeatureSet", values=std_col)
        print(f"\n  {metric}")
        for alg in piv.index:
            cells = "   ".join(
                f"{fs} {piv.loc[alg, fs]:.4f}+/-{pivs.loc[alg, fs]:.4f}"
                for fs in args.feature_sets)
            print(f"    {alg:<20} {cells}")
        if len(args.feature_sets) >= 2:
            a, b = args.feature_sets[0], args.feature_sets[-1]
            print(f"    delta ({b} - {a}), vs fold std of {b}:")
            for alg in piv.index:
                d = piv.loc[alg, b] - piv.loc[alg, a]
                sd = pivs.loc[alg, b]
                verdict = "within fold noise" if abs(d) < sd else "exceeds fold std"
                print(f"      {alg:<20} {d:+.4f}  (std {sd:.4f}) -- {verdict}")

    # ---- preserve the archived v1 numbers, clearly labelled ---------------
    arch = []
    for fn, protocol in [("algorithm_comparison.csv", "random_8020"),
                         ("algorithm_comparison_domain_disjoint.csv", "domain_disjoint"),
                         ("unseen_validation.csv", "independent_unseen")]:
        fp = os.path.join(args.baseline_dir, fn)
        if not os.path.exists(fp):
            continue
        a = pd.read_csv(fp)
        a.insert(0, "protocol", protocol)
        a.insert(0, "source_file", f"{args.baseline_dir}/{fn}")
        arch.append(a)
    if arch:
        ar = pd.concat(arch, ignore_index=True)
        ar.to_csv(f"{md}/baseline_archived.csv", index=False)
        print(f"\nArchived v1 numbers preserved -> {md}/baseline_archived.csv")
        print("  (different dataset snapshot and protocol -- NOT directly")
        print("   comparable to the table above; kept for the record only)")

    print("\n" + "=" * 78)


if __name__ == "__main__":
    main()
