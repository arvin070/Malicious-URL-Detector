"""
Train and select the website-pattern model.

This is the second model in the stack. It sees ONLY page features -- no URL
features at all. That separation is deliberate: if it were given the URL it
would mostly relearn the URL model, and the risk engine would be blending a
model with a copy of itself rather than combining two kinds of evidence.

Splits are INHERITED from the URL dataset, so a registered domain the URL model
trained on can never appear in this model's validation or holdout.

The honest caveats live in data/web_dataset_manifest.json and are repeated in
the model card: labels are inherited from the URL (weak labelling), phishing
pages are much harder to retrieve than benign ones, and JavaScript-rendered
content is invisible to a static parser.

Usage:
    python src/train_web.py
"""

import argparse
import json
import os
import platform
import sys
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from web_features import WEB_FEATURE_NAMES, FEATURE_GROUPS  # noqa: E402
from leakage import assert_no_domain_leakage  # noqa: E402
from train import freeze_for_prediction, mean_std_table, ALL_METRICS  # noqa: E402
from train_v3 import full_metrics, METRIC_ORDER, measure_inference, choose_threshold  # noqa: E402

SEED = 42
MODEL_FAMILY_VERSION = "1.0"


def build_models(seed=SEED):
    """Imputation is in every pipeline: a page that failed to yield one feature
    must not take the whole row out of training."""
    def pipe(est):
        return Pipeline([("impute", SimpleImputer(strategy="median")),
                         ("scale", StandardScaler()), ("clf", est)])

    return {
        "Logistic Regression": pipe(LogisticRegression(
            max_iter=3000, class_weight="balanced", random_state=seed)),
        "Decision Tree": Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("clf", DecisionTreeClassifier(max_depth=10, min_samples_leaf=5,
                                           class_weight="balanced",
                                           random_state=seed))]),
        "Random Forest": Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("clf", RandomForestClassifier(
                n_estimators=300, min_samples_leaf=2, class_weight="balanced",
                random_state=seed, n_jobs=-1))]),
        "XGBoost": XGBClassifier(
            n_estimators=300, max_depth=5, learning_rate=0.08,
            subsample=0.9, colsample_bytree=0.9, eval_metric="logloss",
            random_state=seed, n_jobs=-1),
        "Neural Net (MLP)": pipe(MLPClassifier(
            hidden_layer_sizes=(48, 24), max_iter=500, early_stopping=True,
            n_iter_no_change=15, random_state=seed)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/urlguard_web_dataset.csv")
    ap.add_argument("--dataset-manifest", default="data/web_dataset_manifest.json")
    ap.add_argument("--model-dir", default="models/web")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--max-fpr", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()
    md = args.model_dir
    os.makedirs(md, exist_ok=True)

    df = pd.read_csv(args.data)
    with open(args.dataset_manifest, encoding="utf-8") as fh:
        dsmeta = json.load(fh)

    print("=" * 78)
    print("URLGUARD WEBSITE-PATTERN MODEL")
    print("=" * 78)
    print(f"  dataset  {args.data}  ({dsmeta['dataset_version']})")
    print(f"  pages    {len(df):,}  "
          f"({int(df.label.sum()):,} malicious / {len(df)-int(df.label.sum()):,} benign)")
    print(f"  domains  {df.registered_domain.nunique():,}")
    print(f"  features {len(WEB_FEATURE_NAMES)} (page only -- no URL features)")
    rr = dsmeta.get("retrieval_rate_by_label_pct", {})
    if rr:
        print(f"  retrieval rate by label: {rr}  <- phishing pages are harder "
              f"to retrieve;\n    the surviving sample is not a random sample "
              f"of either class.")

    X = df[WEB_FEATURE_NAMES].astype(float)
    y = df["label"].to_numpy()
    groups = df["registered_domain"].to_numpy()

    tr = (df["split"] == "train").to_numpy()
    va = (df["split"] == "val").to_numpy()
    ho = (df["split"] == "holdout").to_numpy()
    if va.sum() < 20 or ho.sum() < 20:
        # Too few pages survived to hold three slices apart. Merging val into
        # train and reporting on holdout alone would be the usual fix, but it
        # would make threshold selection and final reporting share data.
        print(f"\n  Not enough pages for three splits "
              f"(train {tr.sum()}, val {va.sum()}, holdout {ho.sum()}).")
        print("  Refusing to train a model whose numbers could not be trusted.")
        return 1

    Xtr, ytr = X[tr], y[tr]
    Xva, yva = X[va], y[va]
    Xho, yho = X[ho], y[ho]
    print(f"  split    train {tr.sum():,} | val {va.sum():,} | holdout {ho.sum():,}")
    assert not (set(groups[tr]) & set(groups[va])), "train/val domain leak"
    assert not (set(groups[tr]) & set(groups[ho])), "train/holdout domain leak"
    print("  verified: 0 registered domains shared between splits")

    # ---------------- model comparison ----------------
    print("\n[1] Model comparison (train -> val)")
    rows, fitted, probas, timings = [], {}, {}, {}
    for name, proto in build_models(args.seed).items():
        t0 = time.perf_counter()
        m = clone(proto)
        m.fit(Xtr, ytr)
        fit_s = time.perf_counter() - t0
        freeze_for_prediction(m)
        p = m.predict_proba(Xva)[:, 1]
        fitted[name], probas[name] = m, p
        timings[name] = measure_inference(m, Xva)
        timings[name]["fit_s"] = round(fit_s, 1)
        met = full_metrics(yva, p, 0.5)
        rows.append({"Model": name, **{k: met[k] for k in METRIC_ORDER},
                     **{k: met[k] for k in ("TN", "FP", "FN", "TP")},
                     "single_url_ms": timings[name]["single_url_ms"]})
        print(f"    {name:<22} AUC {met['ROC-AUC']:.4f}  PR-AUC {met['PR-AUC']:.4f}  "
              f"F1 {met['F1-Score']:.4f}  FPR {met['FPR']:.4f}  "
              f"{timings[name]['single_url_ms']:.3f} ms")
    comp = pd.DataFrame(rows).set_index("Model").round(4)
    comp.to_csv(f"{md}/model_comparison.csv")

    # ---------------- grouped CV ----------------
    print(f"\n[2] Domain-grouped {args.folds}-fold CV on train")
    n_groups = len(set(groups[tr]))
    folds_n = min(args.folds, max(2, n_groups // 10))
    sgkf = StratifiedGroupKFold(n_splits=folds_n, shuffle=True,
                                random_state=args.seed)
    fold_rows = []
    for name, proto in build_models(args.seed).items():
        for k, (a, b) in enumerate(sgkf.split(Xtr, ytr, groups[tr]), 1):
            if len(set(ytr[b])) < 2:
                continue
            m = clone(proto)
            m.fit(Xtr.iloc[a], ytr[a])
            freeze_for_prediction(m)
            pp = m.predict_proba(Xtr.iloc[b])[:, 1]
            fold_rows.append({"Algorithm": name, "fold": k,
                              **{kk: vv for kk, vv in
                                 full_metrics(ytr[b], pp, 0.5).items()
                                 if kk in ALL_METRICS}})
    cv_ms = None
    if fold_rows:
        per_fold = pd.DataFrame(fold_rows)
        per_fold.round(4).to_csv(f"{md}/cv_folds.csv", index=False)
        cv_ms = mean_std_table(per_fold)
        cv_ms.round(4).to_csv(f"{md}/cv_results.csv")
        for name in build_models().keys():
            if name in cv_ms.index:
                r = cv_ms.loc[name]
                print(f"    {name:<22} AUC {r['ROC-AUC_mean']:.4f}"
                      f"+/-{r['ROC-AUC_std']:.4f}  F1 {r['F1-Score_mean']:.4f}"
                      f"+/-{r['F1-Score_std']:.4f}")

    # ---------------- selection ----------------
    print("\n[3] Selection (validation AUC within an FPR ceiling; "
          "ties on inference cost)")
    cand = []
    for name in fitted:
        thr, sweep, rule = choose_threshold(yva, probas[name], args.max_fpr)
        m_at = full_metrics(yva, probas[name], thr)
        cand.append({"name": name, "threshold": thr, "rule": rule,
                     "val_auc": comp.loc[name, "ROC-AUC"],
                     "val_fpr_at_thr": m_at["FPR"],
                     "val_recall_at_thr": m_at["Recall"],
                     "ms": timings[name]["single_url_ms"],
                     "meets_ceiling": m_at["FPR"] <= args.max_fpr})
        sweep.round(4).to_csv(
            f"{md}/threshold_sweep_{name.replace(' ', '_')}.csv", index=False)
    el = pd.DataFrame(cand).sort_values(["meets_ceiling", "val_auc"],
                                        ascending=[False, False])
    print(el.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    el.round(4).to_csv(f"{md}/selection.csv", index=False)

    pool = (el[el.meets_ceiling] if el.meets_ceiling.any() else el) \
        .reset_index(drop=True)
    leader = pool.iloc[0]
    best_name = leader["name"]
    note = "highest validation ROC-AUC within the FPR ceiling"
    if cv_ms is not None and best_name in cv_ms.index:
        std = float(cv_ms.loc[best_name, "ROC-AUC_std"])
        tied = pool[pool.val_auc >= leader.val_auc - std]
        if len(tied) > 1:
            cheapest = tied.sort_values("ms").iloc[0]
            if cheapest["name"] != best_name:
                print(f"\n    {len(tied)} models within one fold-std ({std:.4f}); "
                      f"selecting the cheapest: {cheapest['name']}")
                best_name = cheapest["name"]
                note = f"within one fold-std ({std:.4f}); selected on cost"
            leader = tied[tied.name == best_name].iloc[0]
    best_thr = float(leader["threshold"])
    print(f"\n    selected: {best_name}  threshold {best_thr}  ({note})")

    # ---------------- holdout, once ----------------
    print("\n[4] Holdout (scored once)")
    final = clone(build_models(args.seed)[best_name])
    final.fit(pd.concat([Xtr, Xva]), np.concatenate([ytr, yva]))
    freeze_for_prediction(final)
    p_ho = final.predict_proba(Xho)[:, 1]
    ho_met = full_metrics(yho, p_ho, best_thr)
    pd.DataFrame([{"threshold": best_thr, **ho_met}]).round(4).to_csv(
        f"{md}/holdout_results.csv", index=False)
    for k in METRIC_ORDER:
        print(f"      {k:<12} {ho_met[k]:.4f}")
    print(f"      confusion    TN={ho_met['TN']} FP={ho_met['FP']} "
          f"FN={ho_met['FN']} TP={ho_met['TP']}")

    # ---------------- curves + importance ----------------
    from sklearn.metrics import RocCurveDisplay, PrecisionRecallDisplay
    figr, axr = plt.subplots(figsize=(6.5, 6))
    figp, axp = plt.subplots(figsize=(6.5, 6))
    for name, p in probas.items():
        RocCurveDisplay.from_predictions(yva, p, ax=axr, name=name)
        PrecisionRecallDisplay.from_predictions(yva, p, ax=axp, name=name)
    axr.plot([0, 1], [0, 1], ls="--", c="gray", lw=.8)
    axr.set_title("Website model -- ROC (validation)")
    figr.tight_layout(); figr.savefig(f"{md}/roc_validation.png", dpi=150,
                                      bbox_inches="tight"); plt.close(figr)
    axp.set_title("Website model -- Precision-Recall (validation)")
    figp.tight_layout(); figp.savefig(f"{md}/pr_validation.png", dpi=150,
                                      bbox_inches="tight"); plt.close(figp)

    imp_source = final[-1] if isinstance(final, Pipeline) else final
    group_importance = {}
    if hasattr(imp_source, "feature_importances_"):
        imp = pd.Series(imp_source.feature_importances_, index=WEB_FEATURE_NAMES)
        imp.sort_values(ascending=False).round(5).to_csv(
            f"{md}/feature_importance.csv", header=["importance"])
        print("\n    top page features:")
        for n, v in imp.sort_values(ascending=False).head(10).items():
            print(f"      {n:<32} {v:.4f}")
        for g, names in FEATURE_GROUPS.items():
            group_importance[g] = round(
                float(imp[[n for n in names if n in imp.index]].sum()), 4)
        print(f"\n    importance by group: {group_importance}")

    # ---------------- bundle ----------------
    model_version = (f"URLGuard-WEB-{best_name.split()[0][:3].upper()}-"
                     f"v{MODEL_FAMILY_VERSION}")
    bundle = {
        "models": {best_name: final},
        "best_model_name": best_name,
        "feature_names": WEB_FEATURE_NAMES,
        "threshold": best_thr,
        "threshold_rule": leader["rule"],
        "selection_note": note,
        "model_version": model_version,
        "dataset_version": dsmeta["dataset_version"],
        "scope": "website",
        "n_features": len(WEB_FEATURE_NAMES),
        "n_pages": len(df),
        "validation": comp.to_dict(orient="index"),
        "holdout": {"threshold": best_thr, **ho_met},
        "inference": timings[best_name],
        "group_importance": group_importance,
        "trained_at": pd.Timestamp.utcnow().isoformat(),
        "limitations": dsmeta.get("known_biases", []),
        "environment": {"python": platform.python_version(),
                        "platform": platform.platform()},
        "version": 1,
    }
    joblib.dump(bundle, f"{md}/web_model.joblib")
    with open(f"{md}/model_card.json", "w", encoding="utf-8") as fh:
        json.dump({k: v for k, v in bundle.items() if k != "models"},
                  fh, indent=2, default=str)
    print(f"\n[5] Saved {md}/web_model.joblib")
    print(f"    model version {model_version}")
    print(f"    threshold     {best_thr}")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())
