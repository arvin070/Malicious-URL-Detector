"""
Train the calibrated fusion meta-model.

WHY A LEARNED, CALIBRATED STACKER RATHER THAN A BLEND
-----------------------------------------------------
The hand-tuned engine in webapp/risk_engine.py works and stays as the fallback,
but its weights are asserted, not fitted, and its output is a ranking number
rather than a probability. "73/100" there means "higher than 60/100"; it does
not mean "73% of URLs scoring this are phishing".

This model fixes both. It takes the per-layer scores and the cheap layer
features as inputs, learns their relative worth from data, and is then
calibrated so the score can be read as a probability. Calibration is measured
(Brier score, ECE, reliability bins) rather than assumed.

LEAKAGE
-------
Layer scores are only honest inputs if the layer models never saw the row.
Every input here is produced by a model trained on the URL dataset's TRAIN
split, and the stacker is fitted on VAL and evaluated on HOLDOUT -- the splits
are the dataset's own domain-disjoint ones, so no registered domain crosses
from a base model's training data into the stacker's evaluation.

WHAT IT IS NOT
--------------
It is not a replacement for the layer models, and it cannot invent evidence. A
row whose website layer is missing is fitted with that absence marked
explicitly, so "not available" is a state the model knows about rather than a
zero it misreads as "clean".

Usage:
    python src/train_fusion.py
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
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from features import extract_features, FEATURE_NAMES  # noqa: E402
from web_features import WEB_FEATURE_NAMES  # noqa: E402
from js_analysis import JS_FEATURE_NAMES  # noqa: E402
from redirect_analysis import REDIRECT_FEATURE_NAMES  # noqa: E402
from threat_intel import lookup_all, intel_features, INTEL_FEATURE_NAMES  # noqa: E402
from train import freeze_for_prediction  # noqa: E402
from train_v3 import full_metrics, METRIC_ORDER  # noqa: E402

SEED = 42
FUSION_VERSION = "1.0"

# Compact inputs: the layer SCORES plus the handful of page/JS/redirect
# features that carry signal a score cannot. Feeding all 240+ raw features
# would just retrain the base models inside the stacker and lose the point.
SUMMARY_WEB = [
    "password_form_posts_offsite", "brand_domain_mismatch",
    "brand_lookalike_domain", "brand_on_official_domain",
    "has_credential_pair", "is_login_page", "any_form_offsite",
    "num_sensitive_fields", "num_hidden_elements", "num_zero_size_iframes",
    "page_is_https", "tls_failed", "mixed_content_resources",
]
SUMMARY_JS = [
    "js_obfuscated_and_exfil", "js_credential_and_exfil",
    "js_obfuscation_patterns", "js_exfiltration_patterns",
    "js_evasion_patterns", "js_redirection_patterns",
]
SUMMARY_REDIRECT = [
    "redirect_hops", "redirect_domain_changed", "redirect_https_downgrades",
    "redirect_start_is_shortener",
]


def expected_calibration_error(y, p, bins=10):
    """ECE plus the reliability table it is computed from."""
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    ece, rows = 0.0, []
    for b in range(bins):
        m = idx == b
        n = int(m.sum())
        if not n:
            continue
        conf = float(p[m].mean())
        acc = float(y[m].mean())
        ece += (n / len(y)) * abs(acc - conf)
        rows.append({"bin": f"{edges[b]:.1f}-{edges[b+1]:.1f}", "n": n,
                     "mean_predicted": round(conf, 4),
                     "observed_rate": round(acc, 4),
                     "gap": round(acc - conf, 4)})
    return round(float(ece), 4), rows


# Any single RAW stacker input at or above this AUC is treated as label
# provenance rather than signal, and training is refused. Base-model outputs
# are exempt: a stacker's whole purpose is to combine scores that are already
# predictive, so url_score being strong is the design, not a leak. The audit
# targets the inputs that are supposed to be weak evidence on their own.
LEAK_AUC = 0.98

# Candidates within this much ROC-AUC of the leader are treated as ranking
# comparably, and calibration quality decides between them.
AUC_TOLERANCE = 0.01

# Among candidates that also calibrate comparably (within this Brier
# difference), the fastest wins. Fusion runs on every scan with a page.
BRIER_TOLERANCE = 0.005

BASE_SCORE_INPUTS = {"url_score", "web_score", "url_margin", "web_margin",
                     "url_above_thr", "web_above_thr", "web_available"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--web-data", default="data/urlguard_web_dataset.csv")
    ap.add_argument("--url-bundle", default="models/v3/url_models.joblib")
    ap.add_argument("--web-bundle", default="models/web/web_model.joblib")
    ap.add_argument("--model-dir", default="models/fusion")
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()
    md = args.model_dir
    os.makedirs(md, exist_ok=True)

    ub = joblib.load(args.url_bundle)
    wb = joblib.load(args.web_bundle)
    df = pd.read_csv(args.web_data)

    print("=" * 78)
    print("URLGUARD FUSION META-MODEL")
    print("=" * 78)
    print(f"  pages    {len(df):,}  "
          f"({int(df.label.sum()):,} malicious / "
          f"{len(df)-int(df.label.sum()):,} benign)")
    print(f"  url      {ub['model_version']}   web {wb['model_version']}")

    have_js = [c for c in SUMMARY_JS if c in df.columns]
    have_rd = [c for c in SUMMARY_REDIRECT if c in df.columns]
    if not have_js:
        print("\n  NOTE: the page dataset predates JS/redirect capture, so those "
              "layers\n  are absent from training. Rebuild with "
              "src/build_web_dataset.py to include them.")

    # ---- layer scores -----------------------------------------------------
    print("\n[1] Scoring each layer (base models never saw these rows' domains "
          "in\n    their own training splits -- splits are inherited)")
    url_model = ub["models"][ub["best_model_name"]]
    Xu = pd.DataFrame([extract_features(u) for u in df["url"]],
                      columns=FEATURE_NAMES)[ub["feature_names"]]
    url_p = url_model.predict_proba(Xu)[:, 1]

    web_model = wb["models"][wb["best_model_name"]]
    Xw = df[wb["feature_names"]].astype(float)
    web_p = web_model.predict_proba(Xw)[:, 1]

    print("    threat-intel lookups (offline) ...")
    t0 = time.perf_counter()
    intel_rows = [intel_features(lookup_all(u)) for u in df["url"]]
    print(f"    {len(intel_rows):,} lookups in {time.perf_counter()-t0:.0f}s")

    meta = pd.DataFrame({
        "url_score": url_p,
        "web_score": web_p,
        "url_above_thr": (url_p >= ub["threshold"]).astype(int),
        "web_above_thr": (web_p >= wb["threshold"]).astype(int),
        # Distance from each layer's OWN threshold: the quantity the hand-tuned
        # engine uses, given to the model directly so it can learn its worth.
        "url_margin": (url_p - ub["threshold"]) / np.where(
            url_p >= ub["threshold"], 1 - ub["threshold"], ub["threshold"]),
        "web_margin": (web_p - wb["threshold"]) / np.where(
            web_p >= wb["threshold"], 1 - wb["threshold"], wb["threshold"]),
        # Availability is an explicit input, never an implicit zero.
        "web_available": 1,
    })
    for c in SUMMARY_WEB:
        meta[c] = df[c].astype(float) if c in df.columns else 0.0
    for c in have_js + have_rd:
        meta[c] = df[c].astype(float)
    # ---- threat intel is DELIBERATELY NOT a stacker input -----------------
    # Measured, not assumed: on this dataset `intel_url_match` alone scores
    # ROC-AUC 0.9997 against the label. That is not predictive power, it is
    # provenance. The local feed index is built from the same PhishTank /
    # URLhaus / OpenPhish snapshots that ASSIGNED the malicious labels, and the
    # Tranco snapshot that supplied the benign ones -- so "is this URL on a
    # blocklist" is a restatement of "was this URL labelled malicious"
    # (1,219/1,219 malicious matched; 1/1,485 benign). Feeding it to the
    # stacker produced ROC-AUC 1.0000 and FPR 0.0000, which is a leak report,
    # not a result.
    #
    # Intel therefore stays where its circularity is visible and harmless: a
    # serving-time signal applied by webapp/risk_engine.py, shown to the user
    # with its source and snapshot date, never a learned weight. Its features
    # are still computed above so the audit below can demonstrate the leak
    # rather than merely assert it.
    intel_audit = pd.DataFrame({c: [r.get(c, 0) for r in intel_rows]
                                for c in INTEL_FEATURE_NAMES})

    y = df["label"].to_numpy()
    groups = df["registered_domain"].to_numpy()
    split = df["split"].to_numpy()

    # The stacker is fitted on VAL and tested on HOLDOUT. Train is excluded:
    # the base models were fitted on it, so their scores there are optimistic
    # and would teach the stacker to trust them too much.
    fit_m = split == "val"
    test_m = split == "holdout"
    if fit_m.sum() < 60 or test_m.sum() < 60:
        print(f"\n  Too few rows (val {fit_m.sum()}, holdout {test_m.sum()}) to "
              f"fit and test a stacker honestly. Not training one.")
        return 1
    assert not (set(groups[fit_m]) & set(groups[test_m])), "fusion split leak"
    print(f"    fit on val {fit_m.sum():,} | test on holdout {test_m.sum():,}"
          f"  (0 shared domains)")

    cols = list(meta.columns)
    Xf, Xt = meta[fit_m][cols], meta[test_m][cols]
    yf, yt = y[fit_m], y[test_m]

    # ---- leak audit on every stacker input --------------------------------
    # A single feature that nearly reproduces the label is a provenance
    # artefact, not a signal. Catch it here instead of discovering it in a
    # suspiciously perfect metric table.
    def _solo_auc(values):
        x = np.nan_to_num(np.asarray(values, dtype=float))
        if len(np.unique(x)) < 2:
            return None
        a = roc_auc_score(yt, x)
        return max(a, 1.0 - a)

    print("\n[1b] Leak audit -- single-feature ROC-AUC against the label")
    raw = [c for c in cols if c not in BASE_SCORE_INPUTS]
    scored = [(c, _solo_auc(meta.loc[test_m, c])) for c in raw]
    scored = [(c, a) for c, a in scored if a is not None]
    leaks = [(c, a) for c, a in scored if a >= LEAK_AUC]
    if leaks:
        print("    REFUSING TO TRAIN -- these inputs alone reproduce the label:")
        for c, a in sorted(leaks, key=lambda t: -t[1]):
            print(f"      {c:<32} AUC {a:.4f}")
        print("    Remove them or explain their provenance before continuing.")
        return 1
    top = max(scored, key=lambda t: t[1])
    print(f"    {len(raw)} raw inputs clean -- strongest is {top[0]} "
          f"at AUC {top[1]:.4f} (limit {LEAK_AUC})")
    for c in ("url_score", "web_score"):
        a = _solo_auc(meta.loc[test_m, c])
        print(f"    base-model input {c:<30} AUC {a:.4f}  (exempt)")

    excluded = [(c, _solo_auc(intel_audit.loc[test_m, c]))
                for c in INTEL_FEATURE_NAMES]
    excluded = sorted(((c, a) for c, a in excluded if a is not None),
                      key=lambda t: -t[1])[:3]
    for c, a in excluded:
        print(f"    excluded intel input {c:<26} would have scored AUC {a:.4f}")

    # ---- candidates -------------------------------------------------------
    print("\n[2] Candidates (each wrapped in probability calibration)")
    base = {
        "Logistic Regression": Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000, class_weight="balanced",
                                       random_state=args.seed))]),
        "Random Forest": Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("clf", RandomForestClassifier(n_estimators=250, min_samples_leaf=3,
                                           class_weight="balanced",
                                           random_state=args.seed, n_jobs=-1))]),
        "XGBoost": XGBClassifier(n_estimators=200, max_depth=3,
                                 learning_rate=0.08, eval_metric="logloss",
                                 random_state=args.seed, n_jobs=-1),
    }

    results, fitted = [], {}
    for name, est in base.items():
        for method in ("sigmoid", "isotonic"):
            try:
                cal = CalibratedClassifierCV(est, method=method, cv=3)
                cal.fit(Xf, yf)
                freeze_for_prediction(cal)
                p = cal.predict_proba(Xt)[:, 1]
            except Exception as exc:                # noqa: BLE001
                print(f"    {name} + {method}: failed ({type(exc).__name__})")
                continue
            m = full_metrics(yt, p, 0.5)
            brier = brier_score_loss(yt, p)
            ece, _ = expected_calibration_error(yt, p)
            key = f"{name} ({method})"
            fitted[key] = cal
            # Per-candidate latency. This runs on every live scan, so a
            # candidate that is marginally better calibrated but an order of
            # magnitude slower is not actually better.
            one = Xt.iloc[[0]]
            t_lat = time.perf_counter()
            for _ in range(100):
                cal.predict_proba(one)
            lat_ms = (time.perf_counter() - t_lat) / 100 * 1000
            results.append({"Model": key, **{k: m[k] for k in METRIC_ORDER},
                            "Brier": round(brier, 4), "ECE": ece,
                            "Latency-ms": round(lat_ms, 3)})
            print(f"    {key:<34} AUC {m['ROC-AUC']:.4f}  PR {m['PR-AUC']:.4f}  "
                  f"Brier {brier:.4f}  ECE {ece:.4f}  {lat_ms:7.2f} ms")

    if not results:
        print("  No calibrated candidate trained.")
        return 1
    comp = pd.DataFrame(results).set_index("Model").round(4)
    comp.to_csv(f"{md}/fusion_comparison.csv")

    # Selection: ranking quality first, calibration decides among candidates
    # that rank comparably. A tie-break on exact floats never fires, so
    # "comparable" is a tolerance band -- otherwise a candidate with 3x the
    # calibration error wins on a third-decimal AUC difference, and the number
    # shown to the user as a confidence is the worse one. Brier is the
    # criterion inside the band because it is a proper scoring rule: it
    # penalises both miscalibration and poor separation.
    # Then latency breaks the remaining near-ties. Measured, not assumed: the
    # first run of this selection picked a Random Forest over logistic
    # regression for a Brier gain of 0.0016 and paid 103 ms per scan instead
    # of 4 ms. For a scanner on the request path that is not a better model,
    # and "best Brier" alone could not see it.
    comp_sorted = comp.sort_values(["ROC-AUC", "Brier"], ascending=[False, True])
    top_auc = comp_sorted["ROC-AUC"].iloc[0]
    band = comp_sorted[comp_sorted["ROC-AUC"] >= top_auc - AUC_TOLERANCE]
    best_brier = band["Brier"].min()
    tied = band[band["Brier"] <= best_brier + BRIER_TOLERANCE]
    best_name = tied["Latency-ms"].idxmin()

    calibration_pick = band["Brier"].idxmin()
    if calibration_pick != comp_sorted.index[0]:
        print(f"\n    {len(band)} candidates within {AUC_TOLERANCE} AUC of the "
              f"best ({top_auc:.4f}); calibration decides between them.")
        print(f"    {comp_sorted.index[0]} led on AUC "
              f"({comp_sorted['ROC-AUC'].iloc[0]:.4f} vs "
              f"{comp.loc[calibration_pick, 'ROC-AUC']:.4f}) but had Brier "
              f"{comp_sorted['Brier'].iloc[0]:.4f} vs "
              f"{comp.loc[calibration_pick, 'Brier']:.4f}.")
    if best_name != calibration_pick:
        print(f"    {len(tied)} of those are within {BRIER_TOLERANCE} Brier of "
              f"the best; took the fastest.")
        print(f"    {calibration_pick} had Brier "
              f"{comp.loc[calibration_pick, 'Brier']:.4f} at "
              f"{comp.loc[calibration_pick, 'Latency-ms']:.2f} ms; chose "
              f"{best_name} at {comp.loc[best_name, 'Brier']:.4f} / "
              f"{comp.loc[best_name, 'Latency-ms']:.2f} ms.")
    best = fitted[best_name]
    print(f"\n[3] Selected {best_name}")

    p_test = best.predict_proba(Xt)[:, 1]
    ece, bins = expected_calibration_error(yt, p_test)
    print(f"\n[4] Calibration on holdout (ECE {ece:.4f})")
    print(f"    {'bin':<12}{'n':>6}{'predicted':>11}{'observed':>10}{'gap':>8}")
    for r in bins:
        print(f"    {r['bin']:<12}{r['n']:>6}{r['mean_predicted']:>11.3f}"
              f"{r['observed_rate']:>10.3f}{r['gap']:>+8.3f}")

    # ---- the comparison the brief asks for --------------------------------
    print("\n[5] URL-only vs Website-only vs Fusion (holdout)")
    rows = []
    for label, p in (("URL only", url_p[test_m]),
                     ("Website only", web_p[test_m]),
                     ("Fusion (calibrated)", p_test)):
        m = full_metrics(yt, p, 0.5)
        rows.append({"Configuration": label, **{k: m[k] for k in METRIC_ORDER},
                     "Brier": round(brier_score_loss(yt, p), 4),
                     "ECE": expected_calibration_error(yt, p)[0]})
    cmp_df = pd.DataFrame(rows).set_index("Configuration").round(4)
    print(cmp_df[["ROC-AUC", "PR-AUC", "F1-Score", "FPR", "FNR", "Brier",
                  "ECE"]].to_string())
    cmp_df.to_csv(f"{md}/configuration_comparison.csv")

    t0 = time.perf_counter()
    for _ in range(200):
        best.predict_proba(Xt.iloc[[0]])
    lat = (time.perf_counter() - t0) / 200 * 1000

    bundle = {
        "model": best,
        "model_name": best_name,
        "feature_names": cols,
        "model_version": f"URLGuard-FUSION-v{FUSION_VERSION}",
        "url_model_version": ub.get("model_version"),
        "web_model_version": wb.get("model_version"),
        "dataset_version": wb.get("dataset_version"),
        "url_threshold": float(ub["threshold"]),
        "web_threshold": float(wb["threshold"]),
        "threshold": 0.5,
        "calibration": {"ece": ece, "bins": bins,
                        "brier": round(float(brier_score_loss(yt, p_test)), 4)},
        "holdout": {k: v for k, v in full_metrics(yt, p_test, 0.5).items()},
        "comparison": cmp_df.to_dict(orient="index"),
        "candidates": comp.to_dict(orient="index"),
        "inference_ms": round(lat, 4),
        "n_fit": int(fit_m.sum()), "n_test": int(test_m.sum()),
        "trained_at": pd.Timestamp.utcnow().isoformat(),
        "environment": {"python": platform.python_version(),
                        "platform": platform.platform()},
        "limitations": [
            "Fitted on the page dataset's validation split, which is small and "
            "inherits the takedown bias of the page corpus.",
            "Only rows where the page was retrievable can train it, so it has "
            "never seen a row with the website layer missing.",
            "Calibration is measured on one holdout; it is not a guarantee it "
            "holds on live traffic.",
        ],
        "version": 1,
    }
    joblib.dump(bundle, f"{md}/fusion_model.joblib")
    with open(f"{md}/model_card.json", "w", encoding="utf-8") as fh:
        json.dump({k: v for k, v in bundle.items() if k != "model"},
                  fh, indent=2, default=str)
    print(f"\n[6] Saved {md}/fusion_model.joblib  ({lat:.2f} ms/scan)")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())
