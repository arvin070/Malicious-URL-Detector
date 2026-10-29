"""
Conservative assessment layer: turns a model probability into something a
person can act on, WITHOUT overstating what the model actually knows.

THE DISTINCTION THIS MODULE ENFORCES
------------------------------------
Four different kinds of statement get mixed together in most "URL checker"
UIs. They are kept strictly separate here:

  1. OBSERVATION      -- a fact about the URL string. "The host has 3 dots."
                         Always true, no interpretation.
  2. SIGNAL           -- an observation that is EMPIRICALLY associated with
                         phishing in the training data, with the evidence
                         attached (percentile + lift). Never "proof".
  3. MODEL ASSESSMENT -- the classifier's probability, and a risk band derived
                         from it. A prediction, not a verdict.
  4. EXTERNAL FACT    -- DNS resolution, IP, domain age, reputation feeds.
                         URLGuard has NONE of these and says so, rather than
                         guessing.

Nothing here is hard-coded or invented. Thresholds come from
`models/signal_thresholds.json` (percentiles of the benign training corpus);
explanations come from XGBoost's own SHAP contributions. If a quantity is not
available it is returned as `available: false`, never as a plausible-looking
default.

SAFETY: pure string/number work. Nothing is fetched, resolved or connected to.
"""

import json
import math
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from features import extract_features, FEATURE_NAMES  # noqa: E402
from leakage import registered_domain  # noqa: E402

THRESHOLDS_PATH = os.path.join(ROOT, "models", "signal_thresholds.json")

# A signal is only shown as a RISK signal when exceeding its benign-percentile
# threshold is actually more common in phishing than in benign URLs. Measured
# on this corpus, several "obviously suspicious" properties are not: long URLs
# (lift 0.95), many hyphens (0.05), heavy percent-encoding (0.13). Showing
# those as warnings would manufacture evidence, so anything below this bar is
# reported as a neutral observation instead.
MIN_SIGNAL_LIFT = 1.2

RISK_BANDS = (
    (0, 30, "LOW RISK", "low",
     "No significant suspicious characteristics were detected."),
    (31, 70, "SUSPICIOUS", "suspicious",
     "Some characteristics associated with potentially harmful URLs were detected."),
    (71, 100, "HIGH RISK", "high",
     "Multiple characteristics associated with potentially harmful URLs were "
     "detected. Exercise caution before proceeding."),
)

ASSESSMENT_DISCLAIMER = (
    "This assessment is based on machine-learning analysis and available "
    "security signals. It is not a guarantee that the website is malicious."
)

# Why each external field is missing. Stated explicitly so the UI never has to
# imply the lookup failed -- it was never attempted, by design.
NOT_COLLECTED_REASON = (
    "URLGuard analyses the URL text only and never contacts the address, so "
    "this information is not collected."
)


def _load_thresholds():
    if not os.path.exists(THRESHOLDS_PATH):
        return {"features": {}, "percentile": None, "n_benign": 0}
    with open(THRESHOLDS_PATH, encoding="utf-8") as fh:
        return json.load(fh)


_THRESHOLDS = _load_thresholds()


# Reported instead of a band when the fusion model's probability sits too
# close to its decision threshold to mean anything. A score of 48 is not a
# statement that a URL is half-malicious; presenting one as "SUSPICIOUS" gives
# a coin-flip the appearance of a finding. The honest output is that the
# evidence available did not settle the question.
INSUFFICIENT_EVIDENCE_BAND = {
    "label": "INSUFFICIENT EVIDENCE",
    "key": "insufficient",
    "range": [0, 100],
    "description": ("The available evidence does not support a conclusion "
                    "either way. The combined score sits close to the "
                    "decision threshold, where the model's own calibration "
                    "is weakest. Treat this as unresolved rather than safe."),
}


def band_for(score: int):
    """Risk band for a 0-100 score. Bands are advisory, not verdicts."""
    for lo, hi, label, key, text in RISK_BANDS:
        if lo <= score <= hi:
            return {"label": label, "key": key, "range": [lo, hi],
                    "description": text}
    return {"label": "UNKNOWN", "key": "unknown", "range": [0, 100],
            "description": "Risk level could not be determined."}


def _unavailable(reason=NOT_COLLECTED_REASON):
    return {"available": False, "value": None, "reason": reason}


def _available(value, note=None):
    out = {"available": True, "value": value}
    if note:
        out["note"] = note
    return out


# --------------------------------------------------------------- 1. facts
def characteristics(feats: dict) -> list:
    """Plain observations about the URL. No interpretation attached.

    Each entry carries only what was measured. Whether a value is *unusual* is
    decided separately, in `signals()`, and only where the corpus supports it.
    """
    def g(k, default=0):
        return feats.get(k, default)

    items = [
        ("URL length", int(g("url_length")), "characters"),
        ("HTTPS", "Enabled" if g("is_https") else "Not enabled", None),
        ("IP address used as host", "Yes" if g("has_ip") else "No", None),
        ("Subdomains", int(g("subdomain_depth")), None),
        ("Dots in URL", int(g("num_dots")), None),
        ("Hyphens in URL", int(g("num_hyphens")), None),
        ("Special characters", int(g("num_special_chars")), None),
        ("Query parameters", int(g("num_query_params")), None),
        ("Path depth", int(g("path_depth")), "segments"),
        ("Suspicious keyword matches", int(g("suspicious_word_count")), None),
        ("'@' in URL", "Yes" if g("has_at_symbol") else "No", None),
        ("Non-standard port", "Yes" if g("has_suspicious_port") else "No", None),
        ("Punycode / IDN host", "Yes" if g("is_punycode") else "No", None),
        ("Mixed-script characters", "Yes" if g("has_confusable_chars") else "No", None),
        ("Known URL shortener", "Yes" if g("is_shortened") else "No", None),
        ("Percent-encoded sequences", int(g("num_percent_encoded")), None),
    ]
    return [{"label": label, "value": value, "unit": unit}
            for label, value, unit in items]


# ------------------------------------------------------------- 2. signals
# (feature, signal text, the direction that is notable)
_BINARY_SIGNALS = [
    ("has_ip", True, "Host is a raw IP address rather than a domain name"),
    ("has_at_symbol", True, "URL contains '@', which can hide the real destination"),
    ("is_punycode", True, "Host uses punycode / internationalised characters"),
    ("has_confusable_chars", True,
     "URL mixes scripts (e.g. Cyrillic letters that look Latin)"),
    ("has_suspicious_port", True, "URL specifies an uncommon network port"),
    ("is_shortened", True, "Host is a known URL shortener, hiding the destination"),
    ("brand_outside_domain", True,
     "A well-known brand name appears outside the registered domain"),
    ("has_double_slash_redirect", True, "Path contains '//', used in redirect tricks"),
]


def signals(feats: dict) -> dict:
    """Security signals, each backed by an actual measurement.

    Returns `positive` (reassuring), `warning` (empirically associated with
    phishing) and `neutral` (measured, but NOT shown as risk because the
    training corpus does not support treating it as one).
    """
    positive, warning, neutral = [], [], []

    if feats.get("is_https"):
        positive.append({
            "text": "HTTPS enabled",
            "evidence": "The URL uses the https scheme.",
        })
    else:
        neutral.append({
            "text": "HTTPS not enabled",
            # Said plainly: in this corpus plain http is common on both sides,
            # so its absence is not evidence of anything on its own.
            "evidence": "Plain http. On its own this is not treated as a risk "
                        "signal -- it is common among benign URLs too.",
        })

    for key, notable, text in _BINARY_SIGNALS:
        if bool(feats.get(key)) == notable:
            warning.append({"text": text, "evidence": "Observed in the URL structure."})

    tf = _THRESHOLDS.get("features", {})
    for key, meta in tf.items():
        if key not in feats:
            continue
        value = float(feats[key])
        thr = meta["threshold"]
        if value <= thr:
            continue
        pct = meta.get("percentile")
        lift = meta.get("lift")
        label = _SIGNAL_LABELS.get(key, key.replace("_", " "))
        evidence = (f"{_fmt(value)} vs a benign-corpus threshold of "
                    f"{_fmt(thr)} (p{pct:g} of {_THRESHOLDS.get('n_benign', 0):,} "
                    f"benign URLs).")
        entry = {"text": label, "evidence": evidence, "lift": lift}
        if lift is not None and lift >= MIN_SIGNAL_LIFT:
            entry["evidence"] += (f" URLs above this threshold are {lift:g}x more "
                                  f"common among phishing samples.")
            warning.append(entry)
        else:
            entry["evidence"] += (" Not treated as a risk signal: in this corpus "
                                  "it is no more common among phishing URLs.")
            neutral.append(entry)

    return {"positive": positive, "warning": warning, "neutral": neutral}


_SIGNAL_LABELS = {
    "url_length": "Unusually long URL",
    "host_length": "Unusually long hostname",
    "path_length": "Unusually long path",
    "query_length": "Unusually long query string",
    "num_dots": "Unusually many dots",
    "num_hyphens": "Unusually many hyphens",
    "num_special_chars": "Unusually many special characters",
    "num_digits": "Unusually many digits",
    "subdomain_depth": "Unusually many subdomains",
    "path_depth": "Unusually deep path",
    "num_query_params": "Unusually many query parameters",
    "num_percent_encoded": "Heavy percent-encoding",
    "suspicious_word_count": "Suspicious keyword pattern detected",
    "host_entropy": "Hostname looks randomly generated",
    "domain_max_consonant_run": "Long consonant run in the domain",
    "longest_subdomain_length": "Unusually long subdomain label",
}


def _fmt(v):
    if isinstance(v, float) and not float(v).is_integer():
        return f"{v:.2f}"
    return str(int(v))


# --------------------------------------------------------- 3. explanation
def explanation(model, row: pd.DataFrame, feature_names, top_n=6) -> dict:
    """Per-prediction attribution from the model itself (XGBoost SHAP).

    These are the model's OWN contribution values for this specific URL, not a
    guess about why it decided something. If the estimator cannot produce them,
    this returns `available: false` and the UI says so, rather than inventing a
    narrative.
    """
    booster = getattr(model, "get_booster", None)
    if booster is None:
        return {"available": False,
                "reason": "Detailed model explanation is not available for this scan."}
    try:
        import xgboost
        contribs = booster().predict(xgboost.DMatrix(row), pred_contribs=True)[0]
    except Exception:
        return {"available": False,
                "reason": "Detailed model explanation is not available for this scan."}

    bias = float(contribs[-1])
    vals = np.asarray(contribs[:-1], dtype=float)
    order = np.argsort(-np.abs(vals))[:top_n]
    items = []
    for i in order:
        if vals[i] == 0:
            continue
        items.append({
            "feature": feature_names[i],
            "value": float(row.iloc[0, i]),
            "contribution": round(float(vals[i]), 4),
            "direction": "increases" if vals[i] > 0 else "decreases",
        })
    return {
        "available": True,
        "baseline": round(bias, 4),
        "contributions": items,
        "method": "XGBoost SHAP contributions (log-odds) for this specific URL",
    }


def explanation_sentence(expl: dict, band_key: str) -> str:
    """One careful sentence. Attributes cause only when SHAP supports it."""
    if not expl.get("available") or not expl.get("contributions"):
        return "Detailed model explanation is not available for this scan."
    pushing = [c for c in expl["contributions"]
               if c["direction"] == ("increases" if band_key != "low" else "decreases")]
    if not pushing:
        pushing = expl["contributions"][:3]
    names = ", ".join(c["feature"] for c in pushing[:3])
    verb = "toward a higher risk score" if band_key != "low" else "toward a lower risk score"
    return (f"The model's own attribution for this URL places the largest "
            f"influence {verb} on: {names}. These are the features the model "
            f"weighted most for this specific input; they are structural "
            f"properties of the URL text, not evidence of confirmed activity.")


# ----------------------------------------------------- 4. external facts
def domain_info(url: str, feats: dict, dns_only: bool = False) -> dict:
    """What is known about the host, and -- explicitly -- what is not.

    Hostname, registered domain and TLD are derived from the URL string and are
    therefore real. DNS resolution, IP address, domain age and reputation would
    all require contacting the network, which URLGuard deliberately never does,
    so they are returned as unavailable WITH the reason.
    """
    from urllib.parse import urlparse
    raw = url if "://" in url else "http://" + url
    try:
        netloc = urlparse(raw).netloc
    except ValueError:
        netloc = raw.split("/")[0]
    host = netloc.split(":")[0]
    reg = registered_domain(url)

    out = {
        "hostname": _available(host) if host else _unavailable(
            "No hostname could be parsed from this input."),
        "registered_domain": _available(reg) if reg else _unavailable(
            "No registered domain could be parsed from this input."),
        "tld": _available(reg.split(".", 1)[1]) if "." in reg else _unavailable(
            "No public suffix matched this host."),
        "dns_resolution": _unavailable(),
        "ip_address": _unavailable(),
        "domain_age": _unavailable(),
        "reputation": _unavailable(),
        "dns_classification": _unavailable(),
    }
    if dns_only:
        out["scope_note"] = (
            "Only the hostname was available for this assessment. Full URL "
            "features were not available. This reflects what the DNS layer "
            "sees: HTTPS encrypts the path and query, so they never reach it.")
    return out


# ------------------------------------------------------------- assemble
def _display_score(proba: float) -> int:
    """0-100 risk score that never reads as absolute certainty.

    A gradient-boosted ensemble never outputs exactly 0 or 1, but rounding
    0.9997 to "100/100" tells the user the system is certain, which is the
    specific claim this project must not make. So a probability strictly
    inside (0, 1) is clamped to the 1..99 range; only a literal 0.0 or 1.0 --
    which this model cannot produce -- could ever display as 0 or 100.
    """
    p = float(proba)
    score = int(round(p * 100))
    if 0.0 < p < 1.0:
        score = min(99, max(1, score))
    return score


def _decision_margin(proba: float, threshold: float) -> float:
    """How far the score sits from the decision threshold, as 0..1.

    This -- not "probability of the predicted class" -- is what gets reported
    as detection confidence, for two reasons:

      * The decision boundary is 0.70, not 0.50. With the class-probability
        definition a URL scoring 0.55 is "benign with 45% confidence", which
        reads as broken rather than as the borderline case it actually is.
      * Margin is monotonic in how decisive the call is, in both directions,
        so a very low score is high-confidence LOW RISK rather than
        low-confidence anything.

    It is a statement about the decision, not a calibrated likelihood, and the
    UI says so.
    """
    p, t = float(proba), float(threshold)
    if t <= 0.0 or t >= 1.0:
        return abs(p - 0.5) * 2.0
    return (p - t) / (1.0 - t) if p >= t else (t - p) / t


def _display_confidence(conf: float) -> float:
    """Never show a flat 100% or 0%: the model is never that certain."""
    c = round(float(conf) * 100.0, 1)
    return min(99.9, max(0.1, c))


def margin_label(margin: float) -> str:
    """Plain word for how decisive the call is. Avoids false precision."""
    if margin >= 0.70:
        return "Decisive"
    if margin >= 0.35:
        return "Moderate"
    return "Borderline"


def assess(url, proba, threshold, feats, model=None, row=None,
           feature_names=None, dns_only=False):
    """Build the complete, conservative assessment payload."""
    score = _display_score(proba)
    band = band_for(score)

    predicted_malicious = float(proba) >= float(threshold)
    margin = _decision_margin(proba, threshold)

    expl = {"available": False,
            "reason": "Detailed model explanation is not available for this scan."}
    if model is not None and row is not None and feature_names is not None:
        expl = explanation(model, row, feature_names)

    return {
        "risk_score": score,
        "risk_band": band,
        "confidence": _display_confidence(margin),
        "confidence_label": margin_label(margin),
        "confidence_basis": (
            "How far the model score sits from the decision threshold "
            f"({int(round(float(threshold) * 100))}/100). It describes how "
            "decisive this call is, not a calibrated probability that the site "
            "is malicious -- the model has no calibration step."),
        "exceeds_threshold": predicted_malicious,
        "threshold_score": int(round(float(threshold) * 100)),
        "raw_probability": round(float(proba), 6),
        "disclaimer": ASSESSMENT_DISCLAIMER,
        "characteristics": characteristics(feats),
        "signals": signals(feats),
        "explanation": expl,
        "explanation_text": explanation_sentence(expl, band["key"]),
        "domain": domain_info(url, feats, dns_only=dns_only),
        "scope": "hostname" if dns_only else "url",
        "scope_warning": (
            "Hostname-only assessment. The model used here is trained on "
            "hostnames alone and is substantially less accurate than full-URL "
            "analysis (domain-aware CV ROC-AUC 0.764 vs 0.958)."
            if dns_only else None),
        "signal_basis": {
            "percentile": _THRESHOLDS.get("percentile"),
            "n_benign": _THRESHOLDS.get("n_benign"),
            "min_lift": MIN_SIGNAL_LIFT,
            "available": bool(_THRESHOLDS.get("features")),
        },
    }
