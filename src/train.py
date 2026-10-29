"""
Train and COMPARE classification algorithms for phishing URL detection.

METHODOLOGY (v3 -- domain-aware, with fold-level variance)
----------------------------------------------------------
v3 changes on top of v2:
  * de-duplication is CANONICAL, not exact-string. 359 rows of
    data/real_urls.csv differ only by scheme / "www." / trailing slash, so
    exact-string dedupe left the same page in both train and test.
  * cross-validation reports mean +/- std ACROSS FOLDS, not one pooled
    out-of-fold number, so the fold-to-fold spread of each estimate is visible
    and model selection can be checked against it.
  * all three protocols (random 80/20, domain-disjoint, 5-fold domain-aware)
    are emitted side by side in validation_comparison.csv.
  * leakage checks live in src/leakage.py and are shared with the audit script,
    so the check used during training is the same one the audit reports.
  * --feature-set {legacy,structural,full} makes the feature blocks ablatable
    under one identical protocol.

METHODOLOGY (v2 -- domain-aware)
-------------------------------
The v1 pipeline selected a model from a single random 80/20 split.  That was
optimistic: 41.7% of URLs share a registered domain, so a random split lets a
model memorise domains, and it picked Random Forest even though XGBoost
generalised better to an independent corpus.

v2 therefore:
  1. groups every URL by registered domain and uses StratifiedGroupKFold 5-fold
     CV, so no domain is ever in both train and validation within a fold;
  2. selects the model on out-of-fold ROC-AUC (threshold-free ranking quality),
     not on one random split;
  3. selects the deployment threshold on out-of-fold predictions using a
     pre-declared rule -- maximise F1 subject to FPR <= --max-fpr (default 0.05)
     -- because the measured failure mode is false positives, not misses.  The
     ceiling is a product requirement (a blocking extension must not cry wolf),
     chosen from out-of-fold data and the deployment context, never from the
     independent corpus;
  4. still reports the random 80/20 split so v1 numbers stay comparable.

The independent corpus (src/validate_unseen.py) is NOT touched here: no model,
no threshold and no feature decision is made with reference to it.

URLs are treated strictly as strings; nothing is resolved or fetched.

Usage:
    python src/train.py
    python src/train.py --no-hard-negatives      # ablation
    python src/train.py --data data/sample_urls.csv --no-hard-negatives
"""

import argparse
import os
import re
import sys

import numpy as np
import pandas as pd
import joblib

from sklearn.base import clone
from sklearn.model_selection import (train_test_split, GroupShuffleSplit,
                                     StratifiedGroupKFold)
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.metrics import (
    classification_report, roc_auc_score, confusion_matrix, accuracy_score,
    precision_score, recall_score, f1_score, average_precision_score,
    RocCurveDisplay, PrecisionRecallDisplay, ConfusionMatrixDisplay, roc_curve
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from features import extract_features, FEATURE_NAMES, FEATURE_SETS  # noqa: E402
from leakage import (registered_domain, registered_domains,  # noqa: E402
                     drop_exact_duplicates, assert_no_domain_leakage,
                     audit_folds)

SEED = 42
METRIC_COLS = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]
ALL_METRICS = METRIC_COLS + ["FPR", "FNR"]


def build_feature_matrix(urls, feature_names=None):
    names = feature_names or FEATURE_NAMES
    rows = [extract_features(u) for u in urls]
    return pd.DataFrame(rows, columns=FEATURE_NAMES)[names]


def hostname_of(u: str) -> str:
    """Strip a URL down to what a DNS query would reveal: the hostname.

    No scheme, no port, no path, no query, no fragment, no userinfo.
    """
    s = str(u).strip()
    s = re.sub(r"^[a-zA-Z][a-zA-Z0-9+.\-]*://", "", s)
    s = s.split("/", 1)[0].split("?", 1)[0].split("#", 1)[0]
    s = s.rsplit("@", 1)[-1]          # drop user:pass@
    s = s.split(":", 1)[0]            # drop :port
    return s.strip().lower()


def mean_std_table(per_fold: pd.DataFrame) -> pd.DataFrame:
    """Collapse a per-fold metric table into mean/std columns.

    ddof=1 (sample std): the folds are a sample of possible domain partitions,
    not the population, and reporting the smaller population std would
    understate how much the estimate moves with the partition.
    """
    g = per_fold.groupby("Algorithm")[ALL_METRICS]
    out = pd.concat({"mean": g.mean(), "std": g.std(ddof=1)}, axis=1)
    out.columns = [f"{m}_{stat}" for stat, m in out.columns]
    return out[[f"{m}_{s}" for m in ALL_METRICS for s in ("mean", "std")]]


def make_models():
    """Fresh, identically-seeded estimators."""
    return {
        "Decision Tree": DecisionTreeClassifier(
            max_depth=12, min_samples_leaf=3, class_weight="balanced",
            random_state=SEED),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, min_samples_leaf=2, class_weight="balanced",
            random_state=SEED, n_jobs=-1),
        "XGBoost": XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.1,
            eval_metric="logloss", random_state=SEED, n_jobs=-1),
        "Logistic Regression": Pipeline([
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000, class_weight="balanced",
                                       random_state=SEED)),
        ]),
    }


MODELS = make_models()  # kept as a module-level name for tests/back-compat


def metrics_at(y, proba, thr=0.5):
    pred = (proba >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "Accuracy": accuracy_score(y, pred),
        "Precision": precision_score(y, pred, zero_division=0),
        "Recall": recall_score(y, pred, zero_division=0),
        "F1-Score": f1_score(y, pred, zero_division=0),
        "ROC-AUC": roc_auc_score(y, proba),
        "FPR": fp / (fp + tn) if (fp + tn) else 0.0,
        "FNR": fn / (fn + tp) if (fn + tp) else 0.0,
    }


def freeze_for_prediction(model):
    """Make predict_proba bit-reproducible by serialising its reduction.

    Fitting is already deterministic at a fixed random_state regardless of
    n_jobs: each tree's seed is drawn before any work is handed to a worker.
    PREDICTION is not. RandomForest.predict_proba accumulates 300 per-tree
    probability arrays with `out += prediction` across threads, and
    floating-point addition is not associative, so the thread interleaving
    changes the last 1-2 bits (measured: up to 4.4e-16).

    That is invisible until a sample's probability lands exactly on the
    decision threshold -- then 1 ULP flips its class, and the 4th decimal of
    Precision/F1/FPR moves between two otherwise identical runs. Measured on
    this dataset: Random Forest F1 0.8415 vs 0.8414 across consecutive runs.

    Setting n_jobs=1 for prediction removes it entirely (delta exactly 0.0).
    Fitting keeps n_jobs=-1, so the cost is nil -- and serving scores one URL
    at a time, where thread fan-out was never a win anyway.
    """
    for obj in (model, getattr(model, "_final_estimator", None)):
        if obj is not None and hasattr(obj, "n_jobs"):
            try:
                obj.n_jobs = 1
            except (AttributeError, ValueError):
                pass
    return model


def fit_eval(proto, Xtr, ytr, Xte):
    m = clone(proto)              # clone() handles Pipelines; type(p)(**params) does not
    m.fit(Xtr, ytr)               # parallel fit: deterministic at fixed seed
    freeze_for_prediction(m)      # serial predict: bit-reproducible
    return m, m.predict_proba(Xte)[:, 1]


# ----------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/real_urls.csv")
    ap.add_argument("--hard-negatives", default="data/hard_negatives.csv")
    ap.add_argument("--no-hard-negatives", action="store_true")
    ap.add_argument("--no-scheme-augment", action="store_true",
                    help="disable scheme augmentation (ablation)")
    ap.add_argument("--scheme-rate", type=float, default=0.40,
                    help="fraction of EACH class that retains its URL scheme")
    ap.add_argument("--model-dir", default="models")
    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument("--random-state", type=int, default=SEED)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--feature-set", default="full", choices=sorted(FEATURE_SETS),
                    help="legacy = the original 21; structural = + new lexical "
                         "features; full = + hashed char/token n-grams")
    ap.add_argument("--url-transform", default="full", choices=("full", "hostname"),
                    help="'hostname' reduces every URL to its hostname before "
                         "feature extraction, to train the model the Android VPN "
                         "layer needs: a DNS query carries no path, query or "
                         "scheme, so a model served there must be trained on "
                         "what it will actually see. Implies --no-scheme-augment.")
    ap.add_argument("--max-fpr", type=float, default=0.05,
                    help="FPR ceiling for the deployment-threshold rule. 0.05 "
                         "reflects the deployment context: the extension BLOCKS "
                         "pages, and a false block costs far more than a miss "
                         "(users learn to click through frequent false warnings). "
                         "A looser 0.10 ceiling selects 0.45, whose FPR (0.0921) "
                         "is WORSE than the 0.50 default -- F1-maximisation and "
                         "FPR-reduction pull in opposite directions here.")
    args = ap.parse_args()
    os.makedirs(args.model_dir, exist_ok=True)
    md = args.model_dir

    # ---------------------------------------------------------- load data
    df = pd.read_csv(args.data).dropna(subset=["url", "label"])
    df["label"] = df["label"].astype(int)
    df["url"] = df["url"].astype(str)
    df["source"] = "corpus"
    n_base = len(df)

    if args.url_transform == "hostname":
        # A DNS query exposes the hostname and nothing else: no scheme, no
        # path, no query string. Training on full URLs and serving on hostnames
        # would be a train/serve skew -- the model would lean on path_depth,
        # query_entropy and friends, all of which are identically zero at
        # serving time. So reduce the inputs FIRST and let the rest of the
        # pipeline (canonical dedup, domain grouping, CV) run normally.
        #
        # Dedup after this collapses many URLs onto one hostname. That is the
        # point: 50 phishing paths on one host are ONE example to a hostname
        # model, and counting them 50 times would badly inflate every metric.
        args.no_scheme_augment = True   # there are no schemes left to augment
        print("URL transform: hostname-only (for the Android DNS layer)")

    n_hard = 0
    if not args.no_hard_negatives and os.path.exists(args.hard_negatives):
        hn = pd.read_csv(args.hard_negatives).dropna(subset=["url"])
        hn["label"] = hn["label"].astype(int)
        hn["url"] = hn["url"].astype(str)
        hn["source"] = "hard_negative"
        before = len(hn)
        hn = hn[~hn["url"].isin(set(df["url"]))]          # no duplicates
        hn = hn.drop_duplicates(subset="url")
        df = pd.concat([df, hn], ignore_index=True)
        n_hard = len(hn)
        print(f"Hard negatives: {before:,} supplied, {n_hard:,} added "
              f"({before - n_hard:,} dropped as duplicates)")

    if args.url_transform == "hostname":
        before_h = len(df)
        df["url"] = df["url"].map(hostname_of)
        df = df[df["url"].str.len() > 0].reset_index(drop=True)
        print(f"  reduced {before_h:,} URLs to hostnames "
              f"({df['url'].nunique():,} distinct)")

    # Canonical de-duplication, not merely exact-string.
    # The audit found 359 rows in data/real_urls.csv that differ only by
    # scheme / "www." / trailing slash / case. Exact-string dedupe keeps them,
    # and a random split then puts the same page in train AND test.
    df, dup = drop_exact_duplicates(df)
    print(f"De-duplication: {dup['rows_before']:,} -> {dup['rows_after']:,} rows "
          f"({dup['exact_string_duplicates']:,} exact, "
          f"{dup['canonical_duplicates_removed']:,} canonical duplicates removed)")
    if dup["label_conflicting_canonical_urls"]:
        print(f"  WARNING: {dup['label_conflicting_canonical_urls']:,} canonical "
              f"URLs carry conflicting labels")

    # ---- scheme augmentation (breaks a source/label confound) ------------
    # The base corpus stores 100% of URLs WITH a scheme; the hard-negative
    # corpus stores 0% with a scheme, and every hard negative is benign.
    # Left alone, "no scheme" is a perfect benign shortcut and the model
    # collapses on any scheme-less corpus (measured: recall 0.835 -> 0.015).
    #
    # A uniform random strip is not enough: hard negatives have no scheme to
    # begin with, so benign would still end up with a lower scheme rate than
    # phishing. Instead strip PER CLASS so both classes end at the same
    # scheme rate, making scheme presence carry zero label information while
    # `is_https` stays meaningful on the rows that retain it.
    _SCHEME = r"^[a-zA-Z][a-zA-Z0-9+.\-]*://"
    if not args.no_scheme_augment:
        rng = np.random.default_rng(args.random_state)
        target = args.scheme_rate
        for lab in (0, 1):
            idx = df.index[df["label"] == lab]
            with_s = [i for i in idx if re.match(_SCHEME, df.at[i, "url"])]
            keep = int(round(target * len(idx)))
            if keep < len(with_s):
                strip = rng.choice(with_s, size=len(with_s) - keep, replace=False)
                df.loc[strip, "url"] = df.loc[strip, "url"].str.replace(
                    _SCHEME, "", regex=True)
        df, _ = drop_exact_duplicates(df)
        hs = df["url"].str.match(_SCHEME)
        r0 = hs[df.label == 0].mean()
        r1 = hs[df.label == 1].mean()
        print(f"Scheme augmentation (target {target:.0%} per class): "
              f"benign {r0:.1%} / phishing {r1:.1%} retain a scheme "
              f"-> |gap| {abs(r0 - r1):.3%}")

    n, n_pos = len(df), int(df["label"].sum())
    print(f"\nDataset: {args.data}"
          + (f" + {args.hard_negatives}" if n_hard else ""))
    print(f"  {n:,} rows -- {n_pos:,} phishing / {n - n_pos:,} benign "
          f"({n_pos / n:.1%} positive)")
    print(f"  base corpus {n_base:,} | hard negatives {n_hard:,}")
    print(f"  features available: {len(FEATURE_NAMES)}")

    feat_names = FEATURE_SETS[args.feature_set]
    X = build_feature_matrix(df["url"].tolist(), feat_names)
    y = df["label"].to_numpy()
    groups = registered_domains(df["url"]).to_numpy()
    print(f"  feature set: {args.feature_set} ({len(feat_names)} of "
          f"{len(FEATURE_NAMES)} available)")
    print(f"  distinct registered domains: {pd.Series(groups).nunique():,}")

    # ============================================================
    # 1. DOMAIN-GROUPED CROSS-VALIDATION  (model selection)
    # ============================================================
    print(f"\n=== Domain-grouped {args.folds}-fold CV (no domain in two folds) ===")
    sgkf = StratifiedGroupKFold(n_splits=args.folds, shuffle=True,
                                random_state=args.random_state)
    folds = list(sgkf.split(X, y, groups))
    fold_audit = audit_folds(groups, folds, "StratifiedGroupKFold")
    fold_audit.to_csv(f"{md}/cv_fold_audit.csv", index=False)
    print(f"  verified: 0 shared domains across all {args.folds} folds")
    print(f"  fold sizes: "
          f"{', '.join(str(int(v)) for v in fold_audit['val_rows'])}")

    # Two complementary readings of the same folds:
    #   * per-fold metrics -> mean +/- std, which shows how much the estimate
    #     moves when the domain partition changes. This is the number to quote.
    #   * pooled out-of-fold predictions -> one metric over every row, used for
    #     threshold selection because a threshold needs one score distribution,
    #     not five.
    oof = {name: np.zeros(len(y)) for name in MODELS}
    fold_rows = []
    for name, proto in MODELS.items():
        for k, (tr, te) in enumerate(folds, 1):
            _, p = fit_eval(proto, X.iloc[tr], y[tr], X.iloc[te])
            oof[name][te] = p
            fold_rows.append({"Algorithm": name, "fold": k,
                              **metrics_at(y[te], p, 0.5)})
        m = metrics_at(y, oof[name], 0.5)
        print(f"  {name:<20} pooled OOF  acc {m['Accuracy']:.4f}  "
              f"F1 {m['F1-Score']:.4f}  AUC {m['ROC-AUC']:.4f}  "
              f"FPR {m['FPR']:.4f}")

    per_fold = pd.DataFrame(fold_rows)
    per_fold.round(4).to_csv(f"{md}/cv_folds.csv", index=False)

    cv_ms = mean_std_table(per_fold)
    cv_ms.round(4).to_csv(f"{md}/cv_results.csv")

    print(f"\n  {args.folds}-fold domain-aware CV, mean +/- std across folds:")
    for name in MODELS:
        r = cv_ms.loc[name]
        print(f"    {name:<20} " + "  ".join(
            f"{m} {r[m+'_mean']:.4f}+/-{r[m+'_std']:.4f}"
            for m in ("Accuracy", "F1-Score", "ROC-AUC", "FPR")))

    # pooled OOF table kept separately: it is what the threshold sweep uses
    cv_df = pd.DataFrame(
        {name: metrics_at(y, p, 0.5) for name, p in oof.items()}).T
    cv_df.index.name = "Algorithm"
    cv_df.round(4).to_csv(f"{md}/cv_oof_pooled.csv")

    # ---- domain-aware curves, from the out-of-fold predictions -------------
    # The ROC/PR/confusion artifacts further down are on the RANDOM holdout and
    # are kept so v1 figures stay comparable. These are the same three views
    # under the domain-grouped protocol -- i.e. the honest ones.
    figr, axr = plt.subplots(figsize=(6.5, 6))
    figp, axp = plt.subplots(figsize=(6.5, 6))
    for name, p in oof.items():
        RocCurveDisplay.from_predictions(y, p, ax=axr, name=name)
        PrecisionRecallDisplay.from_predictions(y, p, ax=axp, name=name)
    axr.plot([0, 1], [0, 1], ls="--", c="gray", lw=.8)
    axr.set_title(f"ROC -- out-of-fold, domain-grouped {args.folds}-fold CV")
    figr.tight_layout(); figr.savefig(f"{md}/roc_domain_cv.png", dpi=150,
                                      bbox_inches="tight"); plt.close(figr)
    axp.set_title(f"Precision-Recall -- out-of-fold, domain-grouped {args.folds}-fold CV")
    figp.tight_layout(); figp.savefig(f"{md}/pr_domain_cv.png", dpi=150,
                                      bbox_inches="tight"); plt.close(figp)

    fig_cm, axes_cm = plt.subplots(1, len(oof), figsize=(4.0 * len(oof), 3.9))
    cv_cm_rows = []
    for axx, (name, p) in zip(np.atleast_1d(axes_cm), oof.items()):
        cm = confusion_matrix(y, (p >= 0.5).astype(int), labels=[0, 1])
        ConfusionMatrixDisplay(cm, display_labels=["benign", "phishing"]).plot(
            ax=axx, colorbar=False, cmap="Blues", values_format="d")
        axx.set_title(f"{name}\n(out-of-fold)", fontsize=9)
        tn, fp, fn, tp = cm.ravel()
        cv_cm_rows.append({"Algorithm": name, "TN": tn, "FP": fp, "FN": fn,
                           "TP": tp, "FPR": round(fp / (fp + tn), 4),
                           "FNR": round(fn / (fn + tp), 4)})
    plt.tight_layout(); plt.savefig(f"{md}/confusion_matrices_domain_cv.png",
                                    dpi=150, bbox_inches="tight"); plt.close(fig_cm)
    pd.DataFrame(cv_cm_rows).set_index("Algorithm").to_csv(
        f"{md}/confusion_matrices_domain_cv.csv")

    # ---- model selection ---------------------------------------------------
    # Criterion: highest MEAN out-of-fold ROC-AUC across the domain-grouped
    # folds. Threshold-free, so it measures ranking quality rather than one
    # arbitrary cut; and averaged over folds, so it is not a single partition's
    # luck. Random Forest is not privileged -- whatever wins here is served.
    best_name = cv_ms["ROC-AUC_mean"].idxmax()
    runner = cv_ms["ROC-AUC_mean"].drop(index=best_name).idxmax()
    gap = cv_ms.loc[best_name, "ROC-AUC_mean"] - cv_ms.loc[runner, "ROC-AUC_mean"]
    pooled_std = cv_ms.loc[best_name, "ROC-AUC_std"]
    print(f"\n  selected by mean domain-aware CV ROC-AUC: {best_name} "
          f"({cv_ms.loc[best_name, 'ROC-AUC_mean']:.4f} "
          f"+/- {pooled_std:.4f})")
    print(f"    runner-up {runner} "
          f"({cv_ms.loc[runner, 'ROC-AUC_mean']:.4f}); gap {gap:.4f}")
    if gap < pooled_std:
        print(f"    NOTE: the gap is smaller than the fold-to-fold std, so the "
              f"ordering of {best_name} and {runner} is not resolved by this "
              f"evidence alone.")

    # ============================================================
    # 2. THRESHOLD SELECTION  (on out-of-fold predictions only)
    # ============================================================
    print(f"\n=== Threshold sweep 0.10-0.95 on out-of-fold predictions ===")
    rows = []
    for name, p in oof.items():
        for t in np.round(np.arange(0.10, 0.96, 0.05), 2):
            rows.append({"Algorithm": name, "Threshold": t, **metrics_at(y, p, t)})
    sweep = pd.DataFrame(rows)
    sweep.round(4).to_csv(f"{md}/threshold_sweep.csv", index=False)

    best_sweep = sweep[sweep.Algorithm == best_name].copy()
    ok = best_sweep[best_sweep["FPR"] <= args.max_fpr]
    if len(ok):
        chosen = float(ok.loc[ok["F1-Score"].idxmax(), "Threshold"])
        rule = f"max F1 subject to OOF FPR <= {args.max_fpr}"
    else:
        chosen = float(best_sweep.loc[best_sweep["FPR"].idxmin(), "Threshold"])
        rule = f"min FPR (no threshold reached FPR <= {args.max_fpr})"
    print(f"  {best_name} sweep (every 0.05):")
    print(best_sweep[["Threshold"] + METRIC_COLS[:4] + ["FPR", "FNR"]]
          .to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"\n  deployment threshold = {chosen}  [rule: {rule}]")
    print(f"  at 0.50 -> FPR {metrics_at(y, oof[best_name], .5)['FPR']:.4f}; "
          f"at {chosen} -> FPR {metrics_at(y, oof[best_name], chosen)['FPR']:.4f}")

    # threshold / FPR / recall curve
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    bs = best_sweep.sort_values("Threshold")
    for col, c, ls in [("Recall", "tab:blue", "-"), ("FPR", "tab:red", "-"),
                       ("Precision", "tab:green", "--"), ("F1-Score", "tab:purple", "--")]:
        ax.plot(bs["Threshold"], bs[col], label=col, color=c, linestyle=ls)
    ax.axvline(chosen, color="black", lw=1, ls=":")
    ax.annotate(f"deploy @ {chosen}", (chosen, 1.02), fontsize=9, ha="center")
    ax.axhline(args.max_fpr, color="tab:red", lw=.7, alpha=.4)
    ax.set_xlabel("decision threshold"); ax.set_ylabel("rate")
    ax.set_title(f"Threshold trade-off -- {best_name} (out-of-fold, domain-grouped CV)")
    ax.set_ylim(0, 1.08); ax.legend(fontsize=8); ax.grid(alpha=.25)
    plt.tight_layout(); plt.savefig(f"{md}/threshold_curve.png", dpi=150,
                                    bbox_inches="tight"); plt.close(fig)

    # ============================================================
    # 3. HOLDOUTS: random 80/20 (comparable to v1) + domain-disjoint
    # ============================================================
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=args.test_size, random_state=args.random_state, stratify=y)
    print(f"\n=== Random stratified {int((1-args.test_size)*100)}/"
          f"{int(args.test_size*100)} holdout (for comparison with v1) ===")
    print(f"{len(Xtr):,} train / {len(Xte):,} test")

    results, trained, confusions, probas = [], {}, {}, {}
    figr, axr = plt.subplots(figsize=(6.5, 6))
    figp, axp = plt.subplots(figsize=(6.5, 6))
    for name, proto in MODELS.items():
        m, p = fit_eval(proto, Xtr, ytr, Xte)
        probas[name] = p
        trained[name] = m
        sc = metrics_at(yte, p, 0.5)
        results.append({"Algorithm": name, **{k: sc[k] for k in METRIC_COLS}})
        cm = confusion_matrix(yte, (p >= .5).astype(int))
        confusions[name] = cm
        tn, fp, fn, tp = cm.ravel()
        print(f"\n--- {name} ---")
        print(classification_report(yte, (p >= .5).astype(int),
                                    target_names=["benign", "phishing"],
                                    zero_division=0))
        print(f"  TN={tn} FP={fp} FN={fn} TP={tp}  "
              f"FPR={sc['FPR']:.4f} FNR={sc['FNR']:.4f}  AUC={sc['ROC-AUC']:.4f}")
        RocCurveDisplay.from_predictions(yte, p, ax=axr, name=name)
        PrecisionRecallDisplay.from_predictions(yte, p, ax=axp, name=name)

    axr.plot([0, 1], [0, 1], ls="--", c="gray", lw=.8)
    axr.set_title("ROC Curve Comparison (random 80/20 holdout)")
    figr.tight_layout(); figr.savefig(f"{md}/roc_comparison.png", dpi=150,
                                      bbox_inches="tight"); plt.close(figr)
    axp.set_title("Precision-Recall Comparison (random 80/20 holdout)")
    figp.tight_layout(); figp.savefig(f"{md}/pr_comparison.png", dpi=150,
                                      bbox_inches="tight"); plt.close(figp)

    res = pd.DataFrame(results).set_index("Algorithm").round(4)
    print("\n=== Algorithm Comparison (random holdout, threshold 0.5) ===")
    print(res)
    res.to_csv(f"{md}/algorithm_comparison.csv")

    fig2, ax2 = plt.subplots(figsize=(9, 5))
    res[METRIC_COLS].plot(kind="bar", ax=ax2)
    ax2.set_ylim(0, 1.05); ax2.set_ylabel("Score")
    ax2.set_title("Algorithm Comparison Across Metrics (random holdout)")
    plt.xticks(rotation=12, ha="right"); plt.tight_layout()
    plt.savefig(f"{md}/metric_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig2)

    k = len(confusions)
    fig3, axes = plt.subplots(1, k, figsize=(4.0 * k, 3.9))
    cm_rows = []
    for axx, (name, cm) in zip(np.atleast_1d(axes), confusions.items()):
        ConfusionMatrixDisplay(cm, display_labels=["benign", "phishing"]).plot(
            ax=axx, colorbar=False, cmap="Blues", values_format="d")
        axx.set_title(name, fontsize=9)
        tn, fp, fn, tp = cm.ravel()
        cm_rows.append({"Algorithm": name, "TN": tn, "FP": fp, "FN": fn, "TP": tp,
                        "FPR": round(fp / (fp + tn), 4),
                        "FNR": round(fn / (fn + tp), 4)})
    plt.tight_layout(); plt.savefig(f"{md}/confusion_matrices.png", dpi=150,
                                    bbox_inches="tight"); plt.close(fig3)
    cm_df = pd.DataFrame(cm_rows).set_index("Algorithm")
    cm_df.to_csv(f"{md}/confusion_matrices.csv")

    # ---- domain-disjoint holdout ----
    gss = GroupShuffleSplit(n_splits=1, test_size=args.test_size,
                            random_state=args.random_state)
    gi, gj = next(gss.split(X, y, groups))
    grep = assert_no_domain_leakage(groups, gi, gj, "domain-disjoint holdout")
    print(f"\n=== Domain-disjoint holdout: {len(gi):,} train / {len(gj):,} test ===")
    print(f"  {grep['train_domains']:,} train domains / "
          f"{grep['val_domains']:,} test domains, 0 shared")
    grows = []
    for name, proto in MODELS.items():
        _, p = fit_eval(proto, X.iloc[gi], y[gi], X.iloc[gj])
        sc = metrics_at(y[gj], p, 0.5)
        grows.append({"Algorithm": name, **{c: sc[c] for c in ALL_METRICS}})
    gdf = pd.DataFrame(grows).set_index("Algorithm").round(4)
    print(gdf)
    gdf.to_csv(f"{md}/algorithm_comparison_domain_disjoint.csv")
    gdf.to_csv(f"{md}/domain_disjoint_results.csv")   # requested filename

    # ============================================================
    # 3b. THE THREE PROTOCOLS SIDE BY SIDE
    # ============================================================
    # Same data, same seeds, same estimators -- only the split rule differs,
    # so the difference between rows IS the leakage the random split allows.
    rnd = pd.DataFrame(
        {name: metrics_at(yte, probas[name], 0.5) for name in MODELS}).T[ALL_METRICS]
    cmp_rows = []
    for name in MODELS:
        for m in ALL_METRICS:
            cmp_rows.append({
                "Algorithm": name, "Metric": m,
                "random_8020": round(rnd.loc[name, m], 4),
                "domain_disjoint": round(gdf.loc[name, m], 4),
                f"domain_cv_{args.folds}fold_mean": round(cv_ms.loc[name, f"{m}_mean"], 4),
                f"domain_cv_{args.folds}fold_std": round(cv_ms.loc[name, f"{m}_std"], 4),
            })
    vc = pd.DataFrame(cmp_rows)
    vc.to_csv(f"{md}/validation_comparison.csv", index=False)
    print("\n=== Validation protocol comparison (ROC-AUC / F1 / FPR) ===")
    show = vc[vc.Metric.isin(["ROC-AUC", "F1-Score", "FPR"])]
    print(show.to_string(index=False))
    print(f"\n  leakage cost (random 80/20 minus domain-disjoint ROC-AUC):")
    for name in MODELS:
        d = rnd.loc[name, "ROC-AUC"] - gdf.loc[name, "ROC-AUC"]
        print(f"    {name:<20} {d:+.4f}")

    # ============================================================
    # 4. FINAL MODELS: refit on ALL data, save bundle
    # ============================================================
    print("\n=== Refitting final models on the full dataset ===")
    final = {}
    for name, proto in MODELS.items():
        m = clone(proto)
        m.fit(X, y)
        freeze_for_prediction(m)   # the served models must score reproducibly
        final[name] = m
        print(f"  fitted {name}")

    imp = pd.DataFrame({
        name: pd.Series(m.feature_importances_, index=feat_names)
        for name, m in final.items() if hasattr(m, "feature_importances_")})
    if not imp.empty:
        ref = best_name if best_name in imp.columns else imp.columns[0]
        imp = imp.sort_values(ref, ascending=False)
        imp.round(5).to_csv(f"{md}/feature_importance.csv")
        print(f"\nTop 15 features ({ref}):")
        print(imp[ref].head(15).round(4).to_string())
        fig4, ax4 = plt.subplots(figsize=(8.5, 8))
        imp.head(20).iloc[::-1].plot(kind="barh", ax=ax4)
        ax4.set_title(f"Feature importance (top 20, ordered by {ref})")
        ax4.set_xlabel("importance")
        plt.tight_layout(); plt.savefig(f"{md}/feature_importance.png", dpi=150,
                                        bbox_inches="tight"); plt.close(fig4)

    joblib.dump({
        # --- contract relied on by webapp/app.py, monitor.py, the extension ---
        "models": final,
        "best_model_name": best_name,
        "feature_names": list(feat_names),
        "comparison": res,
        # --- v2 additions (optional for consumers) ---
        "threshold": chosen,
        "threshold_rule": rule,
        "confusion": cm_df,
        "cv_results": cv_ms.round(4),
        "cv_results_pooled_oof": cv_df.round(4),
        "cv_folds": per_fold.round(4),
        "validation_comparison": vc,
        "feature_set": args.feature_set,
        "dedup": dup,
        "comparison_domain_disjoint": gdf,
        "dataset": os.path.basename(args.data),
        "n_rows": n,
        "n_hard_negatives": n_hard,
        "n_features": len(feat_names),
        "selection": {"criterion": f"mean ROC-AUC over {args.folds} "
                                   f"domain-grouped folds",
                      "cv": f"StratifiedGroupKFold({args.folds}) on registered domain"},
        "split": {"test_size": args.test_size, "random_state": args.random_state,
                  "stratified": True},
        "version": 2,
    }, f"{md}/url_models.joblib")

    print(f"\nSaved bundle -> {md}/url_models.joblib")
    print(f"  best model        : {best_name}")
    print(f"  deployment thresh : {chosen}")
    print(f"  features          : {len(feat_names)} ({args.feature_set})")
    print("Artifacts: cv_results.csv (mean+/-std), cv_folds.csv, cv_oof_pooled.csv,")
    print("           roc_domain_cv.png, pr_domain_cv.png, confusion_matrices_domain_cv.*,")
    print("           cv_fold_audit.csv, validation_comparison.csv,")
    print("           domain_disjoint_results.csv, threshold_sweep.csv, threshold_curve.png,")
    print("           algorithm_comparison.csv, algorithm_comparison_domain_disjoint.csv,")
    print("           roc_comparison.png, pr_comparison.png, confusion_matrices.*,")
    print("           feature_importance.*, metric_comparison.png")


if __name__ == "__main__":
    main()
