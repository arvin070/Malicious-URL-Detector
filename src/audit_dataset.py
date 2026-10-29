"""
Dataset sanity / leakage audit -- run this BEFORE trusting any model metric.

Motivation: the project's original synthetic dataset was separable by a single
feature (`is_https` alone gave 100% accuracy), so every algorithm scored a
meaningless 1.000.  This script makes that failure mode impossible to miss by
reporting, for every feature, how well that ONE feature alone separates the
classes.  A real dataset should have no single feature anywhere near 1.00.

Checks performed:
  1. class balance
  2. single-feature discriminative power (AUC + best single-threshold accuracy)
  3. is_https breakdown by class            (the original leak)
  4. top TLDs by class + TLD-only accuracy  (source/collection bias)
  5. suspicious-keyword breakdown by class
  6. host overlap between classes and near-duplicate host concentration

LEAKAGE CONTROL (sections 7-10) -- run this before trusting any metric:
  7. canonicalization + exact/near-duplicate removal
  8. registered-domain structure and cross-class domain overlap
  9. split verification: a random 80/20 split vs a domain-grouped split,
     showing how many validation rows a random split lets the model memorise
 10. isolation of the independent unseen corpus (no URL and no domain of it
     may appear in training data)

Canonicalization is used ONLY to compare rows. It is never fed to the feature
extractor and never written back to the dataset -- the model must see each URL
exactly as a browser would.

URLs are treated strictly as strings; nothing is resolved or fetched.

Usage:
    python src/audit_dataset.py --data data/real_urls.csv
    python src/audit_dataset.py --unseen data/unseen_external_DO_NOT_TRAIN.csv
"""

import argparse
import os
import sys
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from features import extract_features, FEATURE_NAMES, count_suspicious_words  # noqa: E402
from leakage import (canonicalize, registered_domain, registered_domains,  # noqa: E402
                     drop_exact_duplicates, url_overlap, domain_overlap,
                     split_domain_leakage)

import tldextract  # noqa: E402

_extract = tldextract.TLDExtract(suffix_list_urls=())


def best_threshold_accuracy(x: np.ndarray, y: np.ndarray) -> float:
    """Highest accuracy achievable by thresholding this one feature."""
    vals = np.unique(x)
    if len(vals) > 256:  # cap work on continuous features
        vals = np.quantile(x, np.linspace(0, 1, 256))
        vals = np.unique(vals)
    best = 0.0
    n = len(y)
    for t in vals:
        pred = (x > t).astype(int)
        best = max(best, (pred == y).sum() / n, ((1 - pred) == y).sum() / n)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/real_urls.csv")
    ap.add_argument("--hard-negatives", default="data/hard_negatives.csv")
    ap.add_argument("--unseen", default="data/unseen_external_DO_NOT_TRAIN.csv",
                    help="cached independent corpus; written by validate_unseen.py. "
                         "Section 10 is skipped if absent.")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--random-state", type=int, default=42)
    ap.add_argument("--report", default="models/dataset_audit.csv")
    args = ap.parse_args()

    df = pd.read_csv(args.data).dropna(subset=["url", "label"])
    df["label"] = df["label"].astype(int)
    y = df["label"].to_numpy()
    urls = df["url"].astype(str).tolist()

    print("=" * 72)
    print(f"DATASET AUDIT -- {args.data}")
    print("=" * 72)

    # ---- 1. class balance -------------------------------------------------
    n, n_phish = len(df), int(y.sum())
    n_benign = n - n_phish
    print(f"\n[1] Class balance")
    print(f"    total     {n:>8,}")
    print(f"    benign  0 {n_benign:>8,}   {n_benign / n:6.2%}")
    print(f"    phishing1 {n_phish:>8,}   {n_phish / n:6.2%}")
    print(f"    majority-class baseline accuracy: {max(n_benign, n_phish) / n:.4f}")

    X = pd.DataFrame([extract_features(u) for u in urls], columns=FEATURE_NAMES)

    # ---- 2. single-feature discriminative power ---------------------------
    print(f"\n[2] Single-feature separability  (leak if AUC ~1.00 or acc ~1.00)")
    rows = []
    for f in FEATURE_NAMES:
        x = X[f].to_numpy(dtype=float)
        auc = roc_auc_score(y, x)
        rows.append({
            "feature": f,
            "AUC": max(auc, 1 - auc),          # direction-agnostic
            "best_1_thresh_acc": best_threshold_accuracy(x, y),
        })
    single = pd.DataFrame(rows).sort_values("AUC", ascending=False).reset_index(drop=True)
    print(single.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    worst = single.iloc[0]
    print(f"\n    strongest single feature: {worst['feature']} "
          f"(AUC {worst['AUC']:.4f}, acc {worst['best_1_thresh_acc']:.4f})")
    if worst["best_1_thresh_acc"] > 0.98:
        print("    *** LEAK: one feature alone almost fully separates the classes ***")
    else:
        print("    OK: no single feature separates the classes on its own.")

    # ---- 3. is_https (the original synthetic leak) -------------------------
    print(f"\n[3] is_https by class  (the leak in the old synthetic data)")
    ct = pd.crosstab(X["is_https"], y, normalize="columns")
    for v in sorted(ct.index):
        b = ct.loc[v, 0] if 0 in ct.columns else 0.0
        p = ct.loc[v, 1] if 1 in ct.columns else 0.0
        print(f"    is_https={v}   benign {b:6.2%}   phishing {p:6.2%}")
    print(f"    => AUC from is_https alone: "
          f"{single.set_index('feature').loc['is_https', 'AUC']:.4f}")

    # ---- 4. TLD distribution + TLD-only accuracy --------------------------
    tlds = [_extract(u).suffix or "<none>" for u in urls]
    tdf = pd.DataFrame({"tld": tlds, "label": y})
    print(f"\n[4] Top TLDs by class")
    top = [t for t, _ in Counter(tlds).most_common(12)]
    print(f"    {'tld':<14}{'benign':>9}{'phishing':>10}{'phish%':>9}")
    for t in top:
        sub = tdf[tdf["tld"] == t]
        b, p = int((sub["label"] == 0).sum()), int((sub["label"] == 1).sum())
        print(f"    {t:<14}{b:>9,}{p:>10,}{(p / (b + p)):>8.1%}")
    # how far does "memorise the TLD" alone get you?
    maj = tdf.groupby("tld")["label"].agg(lambda s: s.mode().iat[0])
    tld_acc = (tdf["tld"].map(maj) == tdf["label"]).mean()
    print(f"    accuracy from TLD alone (in-sample upper bound): {tld_acc:.4f}")
    print(f"    distinct TLDs: {tdf['tld'].nunique():,}")

    # ---- 5. suspicious keywords ------------------------------------------
    kw = np.array([count_suspicious_words(u) for u in urls])
    print(f"\n[5] Suspicious-keyword count by class")
    print(f"    benign   mean {kw[y == 0].mean():.3f}   has>=1: {(kw[y == 0] > 0).mean():6.2%}")
    print(f"    phishing mean {kw[y == 1].mean():.3f}   has>=1: {(kw[y == 1] > 0).mean():6.2%}")
    print(f"    => AUC from suspicious_word_count alone: "
          f"{single.set_index('feature').loc['suspicious_word_count', 'AUC']:.4f}")

    # ---- 6. host overlap / duplication -----------------------------------
    hosts = pd.Series([_extract(u).top_domain_under_public_suffix or u for u in urls])
    hdf = pd.DataFrame({"host": hosts, "label": y})
    per_host = hdf.groupby("host")["label"].nunique()
    both = int((per_host > 1).sum())
    print(f"\n[6] Host-level structure")
    print(f"    distinct registered domains: {hdf['host'].nunique():,}")
    print(f"    domains appearing in BOTH classes: {both:,}")
    print(f"    most frequent domains: "
          f"{', '.join(f'{h}({c})' for h, c in Counter(hosts).most_common(5))}")
    dup_frac = 1 - hdf["host"].nunique() / len(hdf)
    print(f"    rows sharing a domain with another row: {dup_frac:.1%}")
    if dup_frac > 0.5:
        print("    NOTE: heavy domain repetition -- a random split lets the model")
        print("          memorise domains seen in training (optimistic metrics).")

    # ======================================================================
    # LEAKAGE CONTROL
    # ======================================================================
    report = {}

    # ---- 7. canonicalization + duplicate removal --------------------------
    print(f"\n[7] Duplicate audit  (canonicalization used for COMPARISON only)")
    deduped, dup = drop_exact_duplicates(df)
    print(f"    rows as supplied                        {dup['rows_before']:>8,}")
    print(f"    exact string duplicates                 {dup['exact_string_duplicates']:>8,}")
    print(f"    duplicates after canonicalization       {dup['canonical_duplicates_removed']:>8,}")
    print(f"    rows after removing them                {dup['rows_after']:>8,}")
    print(f"    canonical URLs with CONFLICTING labels  {dup['label_conflicting_canonical_urls']:>8,}")
    if dup["label_conflicting_canonical_urls"]:
        print("    *** contradictory ground truth: same URL labelled both ways ***")
    if dup["canonical_duplicates_removed"]:
        print("    NOTE: canonical duplicates inflate metrics -- the same URL can")
        print("          land in both train and test under a random split.")
    else:
        print("    OK: no duplicate URLs remain.")
    report.update({f"dup_{k}": v for k, v in dup.items()})

    # the sections below audit the DEDUPED frame, which is what should be trained
    dy = deduped["label"].to_numpy()
    durls = deduped["url"].astype(str).tolist()
    groups = registered_domains(durls).to_numpy()

    # ---- 8. registered domains + cross-class overlap ----------------------
    print(f"\n[8] Registered-domain structure (the grouping key for validation)")
    n_dom = len(set(groups))
    print(f"    rows                                    {len(deduped):>8,}")
    print(f"    distinct registered domains             {n_dom:>8,}")
    print(f"    mean rows per domain                    {len(deduped)/n_dom:>8.2f}")
    print(f"    rows sharing a domain with another row  "
          f"{1 - n_dom/len(deduped):>8.1%}")
    cross = domain_overlap(deduped.loc[dy == 0, "url"], deduped.loc[dy == 1, "url"])
    print(f"    benign domains                          {cross['n_a']:>8,}")
    print(f"    phishing domains                        {cross['n_b']:>8,}")
    print(f"    domains in BOTH classes                 {cross['shared']:>8,}  "
          f"({cross['frac_of_b']:.1%} of phishing domains)")
    if cross["examples"]:
        print(f"      e.g. {', '.join(cross['examples'])}")
    print("    (shared domains are real -- compromised hosts and shorteners --")
    print("     but they mean a domain-grouped fold cannot be label-pure.)")
    report.update({"n_rows": len(deduped), "n_registered_domains": n_dom,
                   "domains_in_both_classes": cross["shared"]})

    if os.path.exists(args.hard_negatives):
        hn = pd.read_csv(args.hard_negatives).dropna(subset=["url"])
        hu = url_overlap(deduped["url"], hn["url"])
        hd = domain_overlap(deduped["url"], hn["url"])
        print(f"\n    hard negatives ({len(hn):,} rows) vs base corpus:")
        print(f"      shared canonical URLs                 {hu['shared']:>8,}")
        print(f"      shared registered domains             {hd['shared']:>8,}  "
              f"({hd['frac_of_b']:.1%} of hard-negative domains)")
        report.update({"hard_neg_shared_urls": hu["shared"],
                       "hard_neg_shared_domains": hd["shared"]})

    # ---- 9. split verification -------------------------------------------
    from sklearn.model_selection import (train_test_split, GroupShuffleSplit,
                                         StratifiedGroupKFold)
    print(f"\n[9] Split verification -- how much a split lets the model memorise")
    idx = np.arange(len(deduped))
    rtr, rte = train_test_split(idx, test_size=0.2,
                                random_state=args.random_state, stratify=dy)
    rrep = split_domain_leakage(groups, rtr, rte)
    print(f"    random 80/20 split:")
    print(f"      domains in both sides                 {rrep['shared_domains']:>8,}")
    print(f"      test rows whose domain was trained on {rrep['leaked_val_rows']:>8,}  "
          f"({rrep['leaked_val_frac']:.1%})")

    gss = GroupShuffleSplit(n_splits=1, test_size=0.2,
                            random_state=args.random_state)
    gtr, gte = next(gss.split(idx, dy, groups))
    grep = split_domain_leakage(groups, gtr, gte)
    print(f"    domain-disjoint 80/20 split (GroupShuffleSplit):")
    print(f"      domains in both sides                 {grep['shared_domains']:>8,}")
    print(f"      test rows whose domain was trained on {grep['leaked_val_rows']:>8,}  "
          f"({grep['leaked_val_frac']:.1%})")

    sgkf = StratifiedGroupKFold(n_splits=args.folds, shuffle=True,
                                random_state=args.random_state)
    worst = 0
    for i, (tr, va) in enumerate(sgkf.split(idx, dy, groups), 1):
        rep = split_domain_leakage(groups, tr, va)
        worst = max(worst, rep["shared_domains"])
    print(f"    StratifiedGroupKFold({args.folds}):")
    print(f"      worst fold shared domains             {worst:>8,}")
    if worst == 0 and grep["shared_domains"] == 0:
        print("    OK: domain-grouped splits are clean; the random split is not.")
    report.update({"random_split_leaked_test_rows": rrep["leaked_val_rows"],
                   "random_split_leaked_test_frac": round(rrep["leaked_val_frac"], 4),
                   "random_split_shared_domains": rrep["shared_domains"],
                   "domain_disjoint_shared_domains": grep["shared_domains"],
                   "groupkfold_worst_shared_domains": worst})

    # ---- 10. isolation of the independent unseen corpus -------------------
    print(f"\n[10] Independent unseen-corpus isolation")
    train_pool = list(deduped["url"].astype(str))
    if os.path.exists(args.hard_negatives):
        train_pool += list(pd.read_csv(args.hard_negatives)["url"].astype(str))
    if os.path.exists(args.unseen):
        ue = pd.read_csv(args.unseen).dropna(subset=["url"])
        uu = url_overlap(train_pool, ue["url"])
        ud = domain_overlap(train_pool, ue["url"])
        print(f"    cached corpus: {args.unseen}  ({len(ue):,} rows)")
        print(f"    unseen URLs also in training            {uu['shared']:>8,}  "
              f"({uu['frac_of_b']:.2%})")
        print(f"    unseen domains also in training         {ud['shared']:>8,}  "
              f"({ud['frac_of_b']:.1%} of {ud['n_b']:,})")
        if uu["shared"]:
            print("    *** URL-level leakage: these rows must be dropped before")
            print("        reporting unseen metrics (validate_unseen.py does this) ***")
            print(f"      e.g. {', '.join(uu['examples'])}")
        else:
            print("    OK: zero URL-level overlap -- the corpus is URL-isolated.")
        print("    Domain overlap is NOT removed by design: a detector must work on")
        print("    new pages of known hosts. It is reported so unseen numbers are")
        print("    read correctly, and hard-negative mining excludes these domains.")
        report.update({"unseen_rows": len(ue),
                       "unseen_shared_urls": uu["shared"],
                       "unseen_shared_domains": ud["shared"]})
    else:
        print(f"    SKIPPED: no cached corpus at {args.unseen}")
        print("    Run `python src/validate_unseen.py` once to fetch and cache it.")
        print("    The corpus is never used for training, feature or threshold")
        print("    decisions -- only for final reporting.")

    if args.report:
        os.makedirs(os.path.dirname(args.report) or ".", exist_ok=True)
        pd.DataFrame([report]).T.rename(columns={0: "value"}).to_csv(
            args.report, header=["value"])
        print(f"\n    machine-readable summary -> {args.report}")

    print("\n" + "=" * 72)


if __name__ == "__main__":
    main()
