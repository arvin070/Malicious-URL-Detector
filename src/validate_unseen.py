"""
Real-world validation on UNSEEN URLs from an INDEPENDENT corpus.

This corpus is held out of EVERYTHING: no model, feature, hyperparameter,
hard-negative or threshold decision is made with reference to it.  Hard-negative
mining additionally excludes every URL *and every registered domain* that
appears here, so the validation domains were never trained on.

Evaluated at two thresholds:
  * 0.50  -- the original default, kept so v1 numbers stay comparable
  * the bundle's calibrated threshold, chosen on out-of-fold domain-grouped CV

External corpus: `shawhin/phishing-site-classification` (ODbL), a curated subset
of the Kaggle "Phishing Site URLs" corpus (taruntiwarihp), built from PhishTank
submissions plus benign crawls -- independent of the training corpus.

KNOWN CONFOUND (reported, not corrected): this corpus stores URLs with the
scheme stripped, so `is_https` is 0 for every row.  Section [6] quantifies it.

SAFETY: URLs are strings only.  Nothing is resolved, fetched or executed.

Usage:
    python src/validate_unseen.py
"""

import argparse
import io
import os
import re
import sys
import urllib.request

import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, RocCurveDisplay, PrecisionRecallDisplay
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from features import extract_features  # noqa: E402
from leakage import canonicalize as canon, registered_domain  # noqa: E402
import tldextract  # noqa: E402

_extract = tldextract.TLDExtract(suffix_list_urls=())

BASE = ("https://huggingface.co/datasets/shawhin/phishing-site-classification/"
        "resolve/refs%2Fconvert%2Fparquet/default")
SPLITS = ["train", "validation", "test"]
_VALID = re.compile(r"^(?:https?://)?[^\s/?#@]+\.[^\s/?#]{2,}", re.IGNORECASE)


def strip_scheme(u: str) -> str:
    return re.sub(r"^https?://", "", str(u).strip())


def fetch_external() -> pd.DataFrame:
    frames = []
    for s in SPLITS:
        print(f"  fetching external split '{s}' ...")
        rq = urllib.request.Request(f"{BASE}/{s}/0000.parquet",
                                    headers={"User-Agent": "phishing-url-monitor"})
        with urllib.request.urlopen(rq, timeout=180) as r:
            frames.append(pd.read_parquet(io.BytesIO(r.read())))
    df = pd.concat(frames, ignore_index=True).rename(
        columns={"text": "url", "labels": "label"})
    return df[["url", "label"]]


def metrics(y, pred, proba=None):
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "Accuracy": accuracy_score(y, pred),
        "Precision": precision_score(y, pred, zero_division=0),
        "Recall": recall_score(y, pred, zero_division=0),
        "F1-Score": f1_score(y, pred, zero_division=0),
        "ROC-AUC": roc_auc_score(y, proba) if proba is not None else np.nan,
        "FPR": fp / (fp + tn) if (fp + tn) else 0.0,
        "FNR": fn / (fn + tp) if (fn + tp) else 0.0,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", default="models/url_models.joblib")
    ap.add_argument("--train-data", default="data/real_urls.csv")
    ap.add_argument("--hard-negatives", default="data/hard_negatives.csv")
    ap.add_argument("--model-dir", default="models")
    ap.add_argument("--baseline", default="models/baseline/unseen_validation.csv")
    ap.add_argument("--cache", default="data/unseen_external_DO_NOT_TRAIN.csv",
                    help="where the fetched corpus is cached so that "
                         "src/audit_dataset.py can verify its isolation OFFLINE. "
                         "Never read by train.py.")
    ap.add_argument("--refresh", action="store_true",
                    help="re-download even if the cache exists")
    args = ap.parse_args()
    md = args.model_dir

    b = joblib.load(args.model_path)
    models = b["models"]
    best = b["best_model_name"]
    thr = float(b.get("threshold", 0.5))

    print("=" * 78)
    print("INDEPENDENT UNSEEN VALIDATION (no retraining, no tuning on this data)")
    print("=" * 78)
    print(f"  bundle version    : v{b.get('version', 1)}")
    print(f"  trained on        : {b.get('dataset')} "
          f"({b.get('n_rows', 0):,} rows, {b.get('n_hard_negatives', 0):,} hard negatives)")
    print(f"  features          : {b.get('n_features', len(b['feature_names']))}")
    print(f"  selected model    : {best}  [{b.get('selection', {}).get('criterion', 'n/a')}]")
    print(f"  deployment thresh : {thr}  [{b.get('threshold_rule', 'n/a')}]")

    # ---- [1] fetch + clean + prove disjointness ---------------------------
    print("\n[1] External corpus: shawhin/phishing-site-classification (ODbL)")
    from_cache = bool(args.cache) and os.path.exists(args.cache) and not args.refresh
    if from_cache:
        ext = pd.read_csv(args.cache)[["url", "label"]]
        print(f"  using cached copy: {args.cache} ({len(ext):,} rows) "
              f"[--refresh to re-download]")
    else:
        ext = fetch_external()
    n0 = len(ext)
    ext["url"] = ext["url"].astype(str).str.strip()
    ext = ext[ext["url"].str.len() > 0]
    ext = ext[~ext["url"].str.lower().isin({"nan", "none", "null"})]
    ext = ext[ext["url"].str.match(_VALID).fillna(False)]
    ext["label"] = pd.to_numeric(ext["label"], errors="coerce")
    ext = ext[ext["label"].isin([0, 1])]
    ext["label"] = ext["label"].astype(int)
    ext = ext.drop_duplicates(subset="url")
    n1 = len(ext)

    train_urls = [pd.read_csv(args.train_data)["url"].astype(str)]
    if os.path.exists(args.hard_negatives):
        train_urls.append(pd.read_csv(args.hard_negatives)["url"].astype(str))
    all_train = pd.concat(train_urls, ignore_index=True)
    tr_keys = set(all_train.map(canon))
    tr_doms = set(all_train.map(registered_domain))

    ext["_k"] = ext["url"].map(canon)
    overlap = int(ext["_k"].isin(tr_keys).sum())
    ext = ext[~ext["_k"].isin(tr_keys)].drop(columns="_k")
    n2 = len(ext)
    print(f"    rows downloaded                      {n0:>7,}")
    print(f"    after cleaning + dedupe              {n1:>7,}")
    print(f"    after removing URLs seen in training {n2:>7,}  (removed {overlap})")

    if args.cache and not from_cache:
        os.makedirs(os.path.dirname(args.cache) or ".", exist_ok=True)
        ext[["url", "label"]].to_csv(args.cache, index=False)
        print(f"    cached -> {args.cache}")
        print("    This cache is the POST-ISOLATION set: cleaned, de-duplicated")
        print("    and with every URL seen in training already removed, so")
        print("    src/audit_dataset.py can verify isolation offline and the")
        print("    test suite can assert it. Nothing reads it for training.")

    urls = ext["url"].tolist()
    y = ext["label"].to_numpy()
    ext_doms = {registered_domain(u) for u in urls}
    shared = ext_doms & tr_doms
    print(f"    evaluating {len(ext):,} unseen URLs: {(y==0).sum():,} benign / "
          f"{int(y.sum()):,} phishing ({y.mean():.1%} positive)")
    print(f"    registered domains shared with training: {len(shared):,} of "
          f"{len(ext_doms):,} ({len(shared)/len(ext_doms):.1%}) -- kept, realistic")

    # ---- [2] score ---------------------------------------------------------
    X = pd.DataFrame([extract_features(u) for u in urls], columns=b["feature_names"])
    probas = {name: m.predict_proba(X)[:, 1] for name, m in models.items()}

    figr, axr = plt.subplots(figsize=(6.5, 6))
    figp, axp = plt.subplots(figsize=(6.5, 6))
    for name, p in probas.items():
        RocCurveDisplay.from_predictions(y, p, ax=axr, name=name)
        PrecisionRecallDisplay.from_predictions(y, p, ax=axp, name=name)
    axr.plot([0, 1], [0, 1], ls="--", c="gray", lw=.8)
    axr.set_title("ROC on UNSEEN independent corpus")
    figr.tight_layout(); figr.savefig(f"{md}/unseen_roc.png", dpi=150,
                                      bbox_inches="tight"); plt.close(figr)
    axp.set_title("Precision-Recall on UNSEEN independent corpus")
    figp.tight_layout(); figp.savefig(f"{md}/unseen_pr.png", dpi=150,
                                      bbox_inches="tight"); plt.close(figp)

    rows, cm_rows = [], []
    for label, t in [("0.50 (default)", 0.5), (f"{thr:g} (deployed)", thr)]:
        preds = {n: (p >= t).astype(int) for n, p in probas.items()}
        for name in models:
            rows.append({"Threshold": label, "Model": name,
                         **metrics(y, preds[name], probas[name])})
            tn, fp, fn, tp = confusion_matrix(y, preds[name], labels=[0, 1]).ravel()
            cm_rows.append({"Threshold": label, "Model": name,
                            "TN": tn, "FP": fp, "FN": fn, "TP": tp,
                            "FPR": round(fp/(fp+tn), 4), "FNR": round(fn/(fn+tp), 4)})
        votes = np.sum([preds[n] for n in models], axis=0)
        cons = (votes >= np.ceil(len(models)/2)).astype(int)
        rows.append({"Threshold": label, "Model": "Consensus (majority)",
                     **metrics(y, cons, np.mean(list(probas.values()), axis=0))})
        tn, fp, fn, tp = confusion_matrix(y, cons, labels=[0, 1]).ravel()
        cm_rows.append({"Threshold": label, "Model": "Consensus (majority)",
                        "TN": tn, "FP": fp, "FN": fn, "TP": tp,
                        "FPR": round(fp/(fp+tn), 4), "FNR": round(fn/(fn+tp), 4)})

    res = pd.DataFrame(rows).set_index(["Threshold", "Model"]).round(4)
    cmdf = pd.DataFrame(cm_rows).set_index(["Threshold", "Model"])
    print("\n[2] Performance on unseen URLs")
    print(res.to_string())
    res.to_csv(f"{md}/unseen_validation.csv")
    cmdf.to_csv(f"{md}/unseen_confusion.csv")
    print("\n[3] Confusion matrices")
    print(cmdf.to_string())

    # ---- [4] old vs new ---------------------------------------------------
    if os.path.exists(args.baseline):
        old = pd.read_csv(args.baseline).set_index("Model")
        new_dep = res.xs(f"{thr:g} (deployed)", level="Threshold")
        new_def = res.xs("0.50 (default)", level="Threshold")
        comp = []
        for m in ["Decision Tree", "Random Forest", "XGBoost"]:
            if m not in old.index:
                continue
            comp.append({
                "Model": m,
                "old_Acc": old.loc[m, "Accuracy"], "new_Acc@0.5": new_def.loc[m, "Accuracy"],
                "new_Acc@dep": new_dep.loc[m, "Accuracy"],
                "old_F1": old.loc[m, "F1-Score"], "new_F1@0.5": new_def.loc[m, "F1-Score"],
                "new_F1@dep": new_dep.loc[m, "F1-Score"],
                "old_AUC": old.loc[m, "ROC-AUC"], "new_AUC": new_dep.loc[m, "ROC-AUC"],
            })
        cdf = pd.DataFrame(comp).set_index("Model").round(4)
        print("\n[4] OLD (v1 baseline) vs NEW -- independent corpus")
        print(cdf.to_string())
        cdf.to_csv(f"{md}/old_vs_new_unseen.csv")

        obest = old.loc["XGBoost"]
        nbest = new_dep.loc[best]
        print(f"\n    v1 deployed = Random Forest @0.5 | v2 deployed = {best} @{thr:g}")
        v1dep = old.loc["Random Forest"]
        print(f"    v1: acc {v1dep['Accuracy']:.4f}  F1 {v1dep['F1-Score']:.4f}  "
              f"AUC {v1dep['ROC-AUC']:.4f}")
        print(f"    v2: acc {nbest['Accuracy']:.4f}  F1 {nbest['F1-Score']:.4f}  "
              f"AUC {nbest['ROC-AUC']:.4f}  FPR {nbest['FPR']:.4f}")

    # ---- [5] error analysis ----------------------------------------------
    print(f"\n[5] Error analysis -- {best} @ {thr:g}")
    pred = (probas[best] >= thr).astype(int)
    err = pd.DataFrame({"url": urls, "true_label": y, "pred": pred,
                        **{f"score_{n}": np.round(p, 4) for n, p in probas.items()}})
    fps = err[(err.true_label == 0) & (err.pred == 1)]
    fns = err[(err.true_label == 1) & (err.pred == 0)]
    print(f"    false positives: {len(fps):,} ({len(fps)/max((y==0).sum(),1):.1%} of benign)")
    print(f"    false negatives: {len(fns):,} ({len(fns)/max((y==1).sum(),1):.1%} of phishing)")
    err[err.true_label != err.pred].to_csv(f"{md}/unseen_errors.csv", index=False)
    print("    worst false positives:")
    for _, r in fps.nlargest(5, f"score_{best}").iterrows():
        print(f"      {r[f'score_{best}']:.3f}  {str(r.url)[:76]}")
    print("    worst false negatives:")
    for _, r in fns.nsmallest(5, f"score_{best}").iterrows():
        print(f"      {r[f'score_{best}']:.3f}  {str(r.url)[:76]}")

    # ---- [6] scheme confound ---------------------------------------------
    print("\n[6] Confound check -- cost of the missing URL scheme")
    tr = pd.read_csv(args.train_data).sample(n=4000, random_state=42)
    yy = tr["label"].to_numpy()
    bm = models[best]
    for lab, fn_ in [("with scheme (as trained)", lambda s: s),
                     ("scheme stripped", strip_scheme)]:
        Xs = pd.DataFrame([extract_features(fn_(u)) for u in tr["url"].astype(str)],
                          columns=b["feature_names"])
        pr = bm.predict_proba(Xs)[:, 1]
        m = metrics(yy, (pr >= thr).astype(int), pr)
        print(f"      {lab:<26} acc={m['Accuracy']:.4f}  F1={m['F1-Score']:.4f}  "
              f"AUC={m['ROC-AUC']:.4f}")
    print("    (gap = cost of the missing scheme; the rest is genuine corpus shift;")
    print("     these two rows are in-sample, so only their difference is meaningful)")

    print("\nSaved: unseen_validation.csv, unseen_confusion.csv, unseen_errors.csv,")
    print("       unseen_roc.png, unseen_pr.png, old_vs_new_unseen.csv")
    print("=" * 78)


if __name__ == "__main__":
    main()
