"""
Train and select a model on the versioned multi-source dataset.

WHAT MAKES THIS DIFFERENT FROM train.py
---------------------------------------
`src/train.py` is kept unchanged: it builds its own splits from the original
corpus and the previously published numbers reproduce from it exactly. This
script consumes the *pre-split* multi-source dataset instead, so the splits
that were engineered for leakage safety in build_dataset.py are the splits
actually used -- rather than being recomputed, and possibly differently, here.

SELECTION IS NOT BY ACCURACY
----------------------------
A URL blocker that is 99% accurate and wrong about 1 benign site in 100 is
unusable: users meet the false positive far more often than the catch. Models
are therefore ranked on validation ROC-AUC but chosen against an explicit
false-positive ceiling, with inference cost as a tie-breaker because the model
has to run on a phone.

THE HOLDOUT IS TOUCHED ONCE
---------------------------
Model choice, threshold choice and every hyperparameter come from the training
and validation splits. The holdout is scored exactly once, at the end, as a
final readout.

SAFETY: URLs are strings. Nothing is fetched or resolved.

Usage:
    python src/train_v3.py
    python src/train_v3.py --scope hostname
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
from sklearn.metrics import (accuracy_score, average_precision_score,
                             confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score)
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
from features import extract_features, FEATURE_NAMES  # noqa: E402
from leakage import assert_no_domain_leakage, audit_folds  # noqa: E402
from train import freeze_for_prediction, mean_std_table, ALL_METRICS  # noqa: E402

SEED = 42
MODEL_FAMILY_VERSION = "3.0"


def build_models(seed=SEED):
    return {
        "Logistic Regression": Pipeline([
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(max_iter=3000, class_weight="balanced",
                                       random_state=seed)),
        ]),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=14, min_samples_leaf=3, class_weight="balanced",
            random_state=seed),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, min_samples_leaf=2, class_weight="balanced",
            random_state=seed, n_jobs=-1),
        "XGBoost": XGBClassifier(
            n_estimators=400, max_depth=6, learning_rate=0.08,
            subsample=0.9, colsample_bytree=0.9,
            eval_metric="logloss", random_state=seed, n_jobs=-1),
        # Lightweight net: two small hidden layers. Included because the brief
        # asks whether a NN helps; kept small because it has to be a candidate
        # for on-device inference, not just a leaderboard entry.
        "Neural Net (MLP)": Pipeline([
            ("scale", StandardScaler()),
            ("clf", MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=400,
                                  early_stopping=True, n_iter_no_change=12,
                                  random_state=seed)),
        ]),
    }


def full_metrics(y, proba, thr=0.5):
    pred = (proba >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "Accuracy": accuracy_score(y, pred),
        "Precision": precision_score(y, pred, zero_division=0),
        "Recall": recall_score(y, pred, zero_division=0),
        "F1-Score": f1_score(y, pred, zero_division=0),
        "ROC-AUC": roc_auc_score(y, proba),
        "PR-AUC": average_precision_score(y, proba),
        "FPR": fp / (fp + tn) if (fp + tn) else 0.0,
        "FNR": fn / (fn + tp) if (fn + tp) else 0.0,
        "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
    }


METRIC_ORDER = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC",
                "PR-AUC", "FPR", "FNR"]


def measure_inference(model, X, repeats=3):
    """Per-URL inference cost. Single-row latency is what the phone pays."""
    single = X.iloc[[0]]
    t0 = time.perf_counter()
    n = 200
    for _ in range(n):
        model.predict_proba(single)
    single_ms = (time.perf_counter() - t0) / n * 1000

    batch = X.iloc[:min(len(X), 5000)]
    best = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter()
        model.predict_proba(batch)
        best = min(best, time.perf_counter() - t0)
    return {
        "single_url_ms": round(single_ms, 4),
        "batch_rows": len(batch),
        "batch_ms": round(best * 1000, 1),
        "throughput_per_s": round(len(batch) / best, 0),
    }


def choose_threshold(y, proba, max_fpr):
    """Lowest threshold whose FPR stays within the ceiling, maximising F1.

    Chosen on VALIDATION only. The ceiling is a product decision, not a fitted
    quantity: the extension and the app interrupt the user, and an interruption
    the user learns to dismiss protects nobody.
    """
    best = (0.5, -1.0)
    rows = []
    for t in np.round(np.arange(0.05, 0.96, 0.01), 2):
        m = full_metrics(y, proba, t)
        rows.append({"threshold": t, **{k: m[k] for k in METRIC_ORDER}})
        if m["FPR"] <= max_fpr and m["F1-Score"] > best[1]:
            best = (float(t), m["F1-Score"])
    if best[1] < 0:                     # ceiling unreachable
        sweep = pd.DataFrame(rows)
        t = float(sweep.loc[sweep["FPR"].idxmin(), "threshold"])
        return t, pd.DataFrame(rows), f"min FPR (no threshold met FPR<={max_fpr})"
    return best[0], pd.DataFrame(rows), f"max F1 subject to val FPR<={max_fpr}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scope", default="url", choices=("url", "hostname"))
    ap.add_argument("--data", default=None)
    ap.add_argument("--dataset-manifest", default=None)
    ap.add_argument("--model-dir", default=None)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--max-fpr", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()

    if args.data is None:
        args.data = ("data/urlguard_dataset.csv" if args.scope == "url"
                     else "data/urlguard_dataset_hostname.csv")
    if args.dataset_manifest is None:
        args.dataset_manifest = ("data/dataset_manifest.json" if args.scope == "url"
                                 else "data/dataset_manifest_hostname.json")
    if args.model_dir is None:
        args.model_dir = ("models/v3" if args.scope == "url"
                          else "models/v3_hostname")
    md = args.model_dir
    os.makedirs(md, exist_ok=True)

    with open(args.dataset_manifest, encoding="utf-8") as fh:
        dsmeta = json.load(fh)
    dataset_version = dsmeta["dataset_version"]

    df = pd.read_csv(args.data)
    print("=" * 78)
    print(f"URLGUARD TRAINING  --  scope={args.scope}")
    print("=" * 78)
    print(f"  dataset  {args.data}")
    print(f"  version  {dataset_version}")
    print(f"  rows     {len(df):,}  "
          f"({int(df.label.sum()):,} malicious / "
          f"{len(df) - int(df.label.sum()):,} benign)")
    print(f"  domains  {df.registered_domain.nunique():,}")

    print("\n[1] Feature extraction")
    X_all = pd.DataFrame([extract_features(u) for u in df["url"]],
                         columns=FEATURE_NAMES)
    y_all = df["label"].to_numpy()
    groups = df["registered_domain"].to_numpy()
    print(f"    {len(FEATURE_NAMES)} features")

    tr = (df["split"] == "train").to_numpy()
    va = (df["split"] == "val").to_numpy()
    ho = (df["split"] == "holdout").to_numpy()
    Xtr, ytr = X_all[tr], y_all[tr]
    Xva, yva = X_all[va], y_all[va]
    Xho, yho = X_all[ho], y_all[ho]
    print(f"    train {tr.sum():,} | val {va.sum():,} | holdout {ho.sum():,}")

    # The splits came from build_dataset.py; verify rather than assume.
    assert not (set(groups[tr]) & set(groups[va])), "train/val domain leak"
    assert not (set(groups[tr]) & set(groups[ho])), "train/holdout domain leak"
    assert not (set(groups[va]) & set(groups[ho])), "val/holdout domain leak"
    print("    verified: 0 registered domains shared between any two splits")

    # ================= 2. compare models on validation =================
    print("\n[2] Model comparison (trained on train, scored on val)")
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
                     "single_url_ms": timings[name]["single_url_ms"],
                     "fit_s": timings[name]["fit_s"]})
        print(f"    {name:<22} AUC {met['ROC-AUC']:.4f}  PR-AUC {met['PR-AUC']:.4f}  "
              f"F1 {met['F1-Score']:.4f}  FPR {met['FPR']:.4f}  "
              f"{timings[name]['single_url_ms']:.3f} ms/url")

    comp = pd.DataFrame(rows).set_index("Model").round(4)
    comp.to_csv(f"{md}/model_comparison.csv")

    # ================= 3. domain-grouped CV on train =================
    print(f"\n[3] Domain-grouped {args.folds}-fold CV on the training split")
    sgkf = StratifiedGroupKFold(n_splits=args.folds, shuffle=True,
                                random_state=args.seed)
    folds = list(sgkf.split(Xtr, ytr, groups[tr]))
    audit_folds(groups[tr], folds, "cv fold").to_csv(
        f"{md}/cv_fold_audit.csv", index=False)
    fold_rows = []
    for name, proto in build_models(args.seed).items():
        for k, (a, b) in enumerate(folds, 1):
            m = clone(proto)
            m.fit(Xtr.iloc[a], ytr[a])
            freeze_for_prediction(m)
            pp = m.predict_proba(Xtr.iloc[b])[:, 1]
            fold_rows.append({"Algorithm": name, "fold": k,
                              **{kk: vv for kk, vv in
                                 full_metrics(ytr[b], pp, 0.5).items()
                                 if kk in ALL_METRICS}})
    per_fold = pd.DataFrame(fold_rows)
    per_fold.round(4).to_csv(f"{md}/cv_folds.csv", index=False)
    cv_ms = mean_std_table(per_fold)
    cv_ms.round(4).to_csv(f"{md}/cv_results.csv")
    for name in build_models().keys():
        r = cv_ms.loc[name]
        print(f"    {name:<22} AUC {r['ROC-AUC_mean']:.4f}+/-{r['ROC-AUC_std']:.4f}"
              f"  F1 {r['F1-Score_mean']:.4f}+/-{r['F1-Score_std']:.4f}"
              f"  FPR {r['FPR_mean']:.4f}+/-{r['FPR_std']:.4f}")

    # ================= 4. selection =================
    print("\n[4] Selection")
    print("    Criterion: highest validation ROC-AUC among models that can "
          "reach")
    print(f"    FPR <= {args.max_fpr} at some threshold; ties broken on "
          f"inference cost.")
    eligible = []
    for name in fitted:
        thr, sweep, rule = choose_threshold(yva, probas[name], args.max_fpr)
        m_at = full_metrics(yva, probas[name], thr)
        eligible.append({
            "name": name, "threshold": thr, "rule": rule,
            "val_auc": comp.loc[name, "ROC-AUC"],
            "val_fpr_at_thr": m_at["FPR"], "val_recall_at_thr": m_at["Recall"],
            "ms": timings[name]["single_url_ms"],
            "meets_ceiling": m_at["FPR"] <= args.max_fpr,
        })
        sweep.round(4).to_csv(f"{md}/threshold_sweep_{name.replace(' ', '_')}.csv",
                              index=False)
    el = pd.DataFrame(eligible).sort_values(
        ["meets_ceiling", "val_auc"], ascending=[False, False])
    print(el.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    el.round(4).to_csv(f"{md}/selection.csv", index=False)

    ok = el[el.meets_ceiling]
    pool = (ok if len(ok) else el).reset_index(drop=True)
    leader = pool.iloc[0]
    best_name = leader["name"]
    selection_note = "highest validation ROC-AUC within the FPR ceiling"

    # Tie-break on cost. Models whose validation AUC is within one
    # fold-to-fold standard deviation of the leader are statistically
    # indistinguishable here, so ranking them by AUC is reading noise. Among
    # those, the cheapest to run wins -- this model ships to a phone, where a
    # 10x slower estimator is a real cost and a 0.0009 AUC difference is not.
    std = float(cv_ms.loc[best_name, "ROC-AUC_std"])
    tied = pool[pool["val_auc"] >= leader["val_auc"] - std]
    if len(tied) > 1:
        cheapest = tied.sort_values("ms").iloc[0]
        if cheapest["name"] != best_name:
            print(f"\n    {len(tied)} models are within one fold-std "
                  f"({std:.4f}) of {best_name}:")
            for _, r in tied.iterrows():
                print(f"      {r['name']:<22} AUC {r['val_auc']:.4f}  "
                      f"{r['ms']:.2f} ms/url")
            print(f"    Their AUC ordering is not resolved by this evidence, so "
                  f"the\n    cheapest to run is selected instead: "
                  f"{cheapest['name']} "
                  f"({cheapest['ms']:.2f} ms vs {leader['ms']:.2f} ms).")
            best_name = cheapest["name"]
            selection_note = (f"within one fold-std ({std:.4f}) of the AUC "
                              f"leader; selected on inference cost")
        leader = tied[tied.name == best_name].iloc[0]

    best_thr = float(leader["threshold"])
    runner = pool[pool.name != best_name].iloc[0]["name"] if len(pool) > 1 else None
    print(f"\n    selected: {best_name}  threshold {best_thr}")
    print(f"    rationale: {selection_note}")

    # ================= 5. temporal validation =================
    print("\n[5] Temporal validation (train on older, test on newer)")
    temporal = {"available": False}
    if "temporal_split" in df.columns:
        t_tr = (df["temporal_split"] == "train").to_numpy()
        t_te = (df["temporal_split"] == "test").to_numpy()
        if t_tr.sum() >= 500 and t_te.sum() >= 200:
            assert not (set(groups[t_tr]) & set(groups[t_te])), "temporal leak"
            trows = []
            for name, proto in build_models(args.seed).items():
                m = clone(proto)
                m.fit(X_all[t_tr], y_all[t_tr])
                freeze_for_prediction(m)
                pp = m.predict_proba(X_all[t_te])[:, 1]
                met = full_metrics(y_all[t_te], pp, 0.5)
                trows.append({"Model": name,
                              **{k: met[k] for k in METRIC_ORDER}})
            tdf = pd.DataFrame(trows).set_index("Model").round(4)
            tdf.to_csv(f"{md}/temporal_results.csv")
            print(f"    cutoff {dsmeta['counts']['temporal']['cutoff']}")
            print(f"    train {t_tr.sum():,} (older) / test {t_te.sum():,} (newer)")
            print(tdf[["ROC-AUC", "PR-AUC", "F1-Score", "FPR", "Recall"]]
                  .to_string())
            delta = {}
            for name in tdf.index:
                delta[name] = round(
                    float(tdf.loc[name, "ROC-AUC"] - comp.loc[name, "ROC-AUC"]), 4)
            print("\n    change vs the random domain-disjoint validation split:")
            for k, v in delta.items():
                print(f"      {k:<22} {v:+.4f} ROC-AUC")
            temporal = {"available": True,
                        "cutoff": dsmeta["counts"]["temporal"]["cutoff"],
                        "n_train": int(t_tr.sum()), "n_test": int(t_te.sum()),
                        "results": tdf.to_dict(orient="index"),
                        "delta_vs_val_auc": delta}
        else:
            temporal = {"available": False,
                        "reason": f"too few dated rows "
                                  f"(train {int(t_tr.sum())}, test {int(t_te.sum())})"}
            print(f"    unavailable: {temporal['reason']}")

    # ================= 6. holdout, scored once =================
    print("\n[6] Final holdout (scored once, never used for any decision)")
    X_fit = pd.concat([Xtr, Xva])
    y_fit = np.concatenate([ytr, yva])
    # Every candidate is refitted, not just the winner: the product shows a
    # per-model breakdown and a model-agreement count, and those have to come
    # from real estimators rather than a single model quoted five times.
    refit = {}
    for name, proto in build_models(args.seed).items():
        m = clone(proto)
        m.fit(X_fit, y_fit)
        freeze_for_prediction(m)
        refit[name] = m
    final = refit[best_name]
    p_ho = final.predict_proba(Xho)[:, 1]
    ho_met = full_metrics(yho, p_ho, best_thr)
    ho_met_50 = full_metrics(yho, p_ho, 0.5)
    pd.DataFrame([{"threshold": best_thr, **ho_met},
                  {"threshold": 0.5, **ho_met_50}]).round(4).to_csv(
        f"{md}/holdout_results.csv", index=False)
    print(f"    at the deployment threshold {best_thr}:")
    for k in METRIC_ORDER:
        print(f"      {k:<12} {ho_met[k]:.4f}")
    print(f"      confusion    TN={ho_met['TN']} FP={ho_met['FP']} "
          f"FN={ho_met['FN']} TP={ho_met['TP']}")

    # ================= 7. curves =================
    from sklearn.metrics import (RocCurveDisplay, PrecisionRecallDisplay,
                                 ConfusionMatrixDisplay)
    figr, axr = plt.subplots(figsize=(6.5, 6))
    figp, axp = plt.subplots(figsize=(6.5, 6))
    for name, p in probas.items():
        RocCurveDisplay.from_predictions(yva, p, ax=axr, name=name)
        PrecisionRecallDisplay.from_predictions(yva, p, ax=axp, name=name)
    axr.plot([0, 1], [0, 1], ls="--", c="gray", lw=.8)
    axr.set_title("ROC -- validation (domain-disjoint)")
    figr.tight_layout(); figr.savefig(f"{md}/roc_validation.png", dpi=150,
                                      bbox_inches="tight"); plt.close(figr)
    axp.set_title("Precision-Recall -- validation (domain-disjoint)")
    figp.tight_layout(); figp.savefig(f"{md}/pr_validation.png", dpi=150,
                                      bbox_inches="tight"); plt.close(figp)

    figc, axc = plt.subplots(1, 2, figsize=(9, 4))
    for ax, (ttl, mm) in zip(axc, [(f"holdout @ {best_thr}", ho_met),
                                   ("holdout @ 0.50", ho_met_50)]):
        ConfusionMatrixDisplay(
            np.array([[mm["TN"], mm["FP"]], [mm["FN"], mm["TP"]]]),
            display_labels=["benign", "malicious"]).plot(
                ax=ax, colorbar=False, cmap="Blues", values_format="d")
        ax.set_title(f"{best_name} -- {ttl}", fontsize=9)
    plt.tight_layout(); plt.savefig(f"{md}/confusion_holdout.png", dpi=150,
                                    bbox_inches="tight"); plt.close(figc)

    if hasattr(final, "feature_importances_"):
        imp = pd.Series(final.feature_importances_, index=FEATURE_NAMES)
        imp.sort_values(ascending=False).round(5).to_csv(
            f"{md}/feature_importance.csv", header=["importance"])

    # ================= 8. bundle =================
    model_version = (f"URLGuard-{best_name.split()[0][:3].upper()}-"
                     f"v{MODEL_FAMILY_VERSION}")
    bundle = {
        "models": refit,
        "best_model_name": best_name,
        "feature_names": FEATURE_NAMES,
        "threshold": best_thr,
        "threshold_rule": leader["rule"],
        "selection_note": selection_note,
        "model_version": model_version,
        "dataset_version": dataset_version,
        "scope": args.scope,
        "n_features": len(FEATURE_NAMES),
        "n_rows": len(df),
        "trained_at": pd.Timestamp.utcnow().isoformat(),
        "selection": {
            "criterion": f"validation ROC-AUC subject to FPR<={args.max_fpr}",
            "candidates": el.to_dict(orient="records"),
        },
        "validation": comp.to_dict(orient="index"),
        "cv_results": cv_ms.round(4),
        "holdout": {"threshold": best_thr, **ho_met},
        "temporal": temporal,
        "inference": timings[best_name],
        "environment": {"python": platform.python_version(),
                        "platform": platform.platform()},
        "version": 3,
    }
    joblib.dump(bundle, f"{md}/url_models.joblib")
    with open(f"{md}/model_card.json", "w", encoding="utf-8") as fh:
        json.dump({k: v for k, v in bundle.items()
                   if k not in ("models", "cv_results")},
                  fh, indent=2, default=str)

    print(f"\n[7] Saved {md}/url_models.joblib")
    print(f"    model version   {model_version}")
    print(f"    dataset version {dataset_version}")
    print(f"    threshold       {best_thr}  [{leader['rule']}]")
    print(f"    inference       {timings[best_name]['single_url_ms']:.3f} ms/url")
    print("=" * 78)


if __name__ == "__main__":
    main()
