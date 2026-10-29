"""
Serving side of the calibrated fusion meta-model.

WHAT THIS IS
------------
src/train_fusion.py fits a calibrated stacker whose inputs are the URL model's
probability, the website model's probability, each one's distance from its own
threshold, and a selected set of raw page, JavaScript and redirect features. It
is a learned combination, not an average: the weights come from measured
holdout performance, and the output is a calibrated probability rather than a
rescaled sum of parts.

This module loads that model and applies it, with two rules that matter more
than the model itself.

RULE 1 -- THRESHOLD INTEL IS NOT AN INPUT
On this dataset, "is this URL on a blocklist" reproduces the label at ROC-AUC
0.9997, because the blocklist snapshots are where the malicious labels came
from. Those features are therefore excluded from the stacker (see the leak
audit in src/train_fusion.py) and threat intel stays a serving-time signal
applied by risk_engine, where its provenance is shown to the user.

RULE 2 -- MISSING EVIDENCE IS STATED, NOT IMPUTED AS INNOCENCE
The stacker was fitted on rows where a page was successfully retrieved. When no
page is available the fused score is not produced at all: the URL-only
assessment stands, and the result says the webpage layer was unavailable. A
model asked to score a row of zeros would answer confidently about evidence it
never saw.
"""

import os

import joblib
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FUSION_MODEL_PATH = os.path.join(ROOT, "models", "fusion", "fusion_model.joblib")

# A fused probability this close to the decision threshold is reported as
# undecided rather than rounded into a band. Taken from the holdout calibration
# table, where the middle bins hold few rows and show the largest gaps between
# predicted and observed rates -- exactly the region where the number should
# not be presented as an answer.
UNDECIDED_MARGIN = 0.15

_bundle = None
_tried = False


def get_bundle():
    """The fusion bundle, or None when no fusion model has been trained."""
    global _bundle, _tried
    if _bundle is None and not _tried:
        _tried = True
        if os.path.exists(FUSION_MODEL_PATH):
            try:
                _bundle = joblib.load(FUSION_MODEL_PATH)
            except Exception:                       # noqa: BLE001
                _bundle = None
    return _bundle


def available():
    return get_bundle() is not None


def _margin(proba, threshold):
    """Signed distance from a threshold, normalised so +1 is maximal
    confidence above it and -1 maximal confidence below."""
    if proba >= threshold:
        span = 1.0 - threshold
        return (proba - threshold) / span if span > 0 else 0.0
    span = threshold if threshold > 0 else 1.0
    return (proba - threshold) / span


def build_row(url_proba, url_threshold, web_proba, web_threshold, page_feats,
              layer_feats):
    """Assemble the stacker's input row from the live layers.

    Any column the page did not supply is left as a missing value, which the
    model's imputer handles the same way it did during training -- rather than
    being silently set to zero, which would assert a measurement that was
    never taken.
    """
    row = {
        "url_score": float(url_proba),
        "web_score": float(web_proba),
        "url_above_thr": int(url_proba >= url_threshold),
        "web_above_thr": int(web_proba >= web_threshold),
        "url_margin": _margin(float(url_proba), float(url_threshold)),
        "web_margin": _margin(float(web_proba), float(web_threshold)),
        "web_available": 1,
    }
    src = dict(page_feats or {})
    src.update(layer_feats or {})
    bundle = get_bundle()
    for col in bundle["feature_names"]:
        if col in row:
            continue
        v = src.get(col)
        row[col] = float(v) if v is not None else float("nan")
    return row


def score(url_proba, url_threshold, web_layer, page_feats, layer_feats):
    """Apply the fusion model. Returns None when it must not be applied.

    None means "no fused score", which is a different statement from a low
    score, and callers are expected to fall back to the URL-only assessment
    rather than treating the absence as reassurance.
    """
    bundle = get_bundle()
    if bundle is None:
        return None
    if not (web_layer or {}).get("available"):
        return None
    web_proba = web_layer.get("raw_probability")
    if web_proba is None:
        return None

    try:
        row = build_row(url_proba, url_threshold, web_proba,
                        bundle.get("web_threshold", 0.5), page_feats,
                        layer_feats)
        X = pd.DataFrame([row], columns=bundle["feature_names"])
        proba = float(bundle["model"].predict_proba(X)[0, 1])
    except Exception:                               # noqa: BLE001
        # Fail open: a fusion failure must leave the URL assessment intact,
        # never block the scan.
        return None

    thr = float(bundle.get("threshold", 0.5))
    undecided = abs(proba - thr) <= UNDECIDED_MARGIN
    present = [c for c in bundle["feature_names"]
               if not pd.isna(row.get(c, float("nan")))]
    missing = [c for c in bundle["feature_names"] if c not in present]

    return {
        "available": True,
        "method": "calibrated_meta_model",
        "probability": round(proba, 6),
        "score": min(99, max(1, int(round(proba * 100)))),
        "threshold": thr,
        "threshold_score": int(round(thr * 100)),
        "above_threshold": proba >= thr,
        "undecided": undecided,
        "undecided_margin": UNDECIDED_MARGIN,
        "model_name": bundle.get("model_name"),
        "model_version": bundle.get("model_version"),
        "url_model_version": bundle.get("url_model_version"),
        "web_model_version": bundle.get("web_model_version"),
        "calibration": {
            "brier": (bundle.get("holdout") or {}).get("brier"),
            "ece": (bundle.get("calibration") or {}).get("ece"),
            "fitted_on_rows": bundle.get("n_fit"),
            "tested_on_rows": bundle.get("n_test"),
        },
        "inputs_present": len(present),
        "inputs_total": len(bundle["feature_names"]),
        "inputs_missing": missing,
        "basis": ("Calibrated meta-model over the URL score, the page score "
                  "and selected page, JavaScript and redirect features. "
                  "Layer scores are combined by learned weights, not averaged."),
        "limitation": ("Fitted on {n} validation pages and tested on {t} "
                       "held-out pages, which is a small sample; the "
                       "probability is calibrated on that sample, not on live "
                       "traffic.").format(n=bundle.get("n_fit"),
                                          t=bundle.get("n_test")),
    }


def model_info():
    """Summary for /api/meta. None when no fusion model is installed."""
    b = get_bundle()
    if b is None:
        return None
    return {
        "model_name": b.get("model_name"),
        "model_version": b.get("model_version"),
        "trained_at": b.get("trained_at"),
        "dataset_version": b.get("dataset_version"),
        "url_model_version": b.get("url_model_version"),
        "web_model_version": b.get("web_model_version"),
        "threshold": b.get("threshold"),
        "inputs": len(b.get("feature_names") or []),
        "fitted_on_rows": b.get("n_fit"),
        "tested_on_rows": b.get("n_test"),
        "holdout": b.get("holdout"),
        "comparison": b.get("comparison"),
        "inference_ms": b.get("inference_ms"),
        "limitations": b.get("limitations"),
    }
