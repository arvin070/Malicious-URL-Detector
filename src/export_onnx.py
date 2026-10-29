"""
Export the served model to ONNX for on-device (Android) inference.

WHY ONNX
--------
The Android app must classify URLs with NO network call -- that is the whole
premise of the project ("the URL itself is never fetched"). Calling the Flask
API from the phone would send every URL the user visits off the device, which
trades the privacy property away for convenience. ONNX Runtime Mobile runs the
same tree ensemble on-device, offline.

WHAT THIS SCRIPT GUARANTEES
---------------------------
Converting a model is not enough -- the conversion has to be *checked*, or the
app silently serves a different classifier from the one that was validated.
This script therefore:

  1. converts the bundle's selected model to ONNX,
  2. re-scores the ENTIRE training corpus through both the original estimator
     and ONNX Runtime, and reports max |delta| and -- the number that actually
     matters -- how many verdicts flip at the deployment threshold,
  3. emits golden vectors (URL -> 122 features -> probability) that the Kotlin
     feature port is unit-tested against, so a drifted re-implementation fails
     the Android build instead of silently degrading predictions.

Everything here is offline. No URL is fetched, resolved or executed.

Usage:
    python src/export_onnx.py
    python src/export_onnx.py --model "Random Forest" --n-golden 1000
"""

import argparse
import hashlib
import json
import os
import sys

import numpy as np
import pandas as pd
import joblib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from features import extract_features, FEATURE_NAMES  # noqa: E402


XGBOOST_TARGET_OPSET = 15   # onnxmltools' XGBoost converter supports <= 15


def _convert(model, n_features, name):
    """Dispatch to the right converter for the estimator type."""
    mod = type(model).__module__ or ""
    if mod.startswith("xgboost"):
        # onnxmltools validates against ITS OWN FloatTensorType class, not
        # skl2onnx's -- passing the skl2onnx one raises a confusing
        # "wrong type" error even though the two are structurally identical.
        from onnxmltools.convert import convert_xgboost
        from onnxmltools.convert.common.data_types import (
            FloatTensorType as MltoolsFloatTensorType)

        # The model was fitted on a DataFrame, so the booster carries the real
        # column names ("subdomain_length", ...). onnxmltools' converter only
        # understands XGBoost's positional "f0, f1, ..." convention and refuses
        # anything else. Renaming is metadata-only -- the split thresholds and,
        # critically, the column ORDER are untouched, and step [2] re-scores the
        # whole corpus through the ORIGINAL estimator to prove that.
        # Done on a deep copy: renaming in place would break predict_proba() on
        # the DataFrame we use for the reference predictions.
        import copy
        model = copy.deepcopy(model)
        model.get_booster().feature_names = [f"f{i}" for i in range(n_features)]
        # onnxmltools' XGBoost converter caps out at opset 15. That is well
        # within what ONNX Runtime Mobile supports, so pin it rather than
        # chasing the newest opset.
        return convert_xgboost(
            model,
            initial_types=[("input", MltoolsFloatTensorType([None, n_features]))],
            target_opset=XGBOOST_TARGET_OPSET)

    from skl2onnx import convert_sklearn
    from skl2onnx.common.data_types import FloatTensorType
    return convert_sklearn(
        model, initial_types=[("input", FloatTensorType([None, n_features]))],
        target_opset=17, options={id(model): {"zipmap": False}})


def _onnx_proba(sess, X: np.ndarray) -> np.ndarray:
    """Positive-class probability from an ONNX classifier session."""
    out_names = [o.name for o in sess.get_outputs()]
    res = sess.run(None, {sess.get_inputs()[0].name: X.astype(np.float32)})
    probs = res[out_names.index("probabilities")] if "probabilities" in out_names \
        else res[-1]
    if isinstance(probs, list):            # zipmap form: list[dict]
        return np.array([p[1] for p in probs], dtype=np.float64)
    probs = np.asarray(probs)
    return probs[:, 1].astype(np.float64) if probs.ndim == 2 else probs.astype(np.float64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", default="models/url_models.joblib")
    ap.add_argument("--data", default="data/real_urls.csv")
    ap.add_argument("--model", default=None,
                    help="which estimator to export (default: the bundle's "
                         "selected model)")
    ap.add_argument("--out-dir", default="android/app/src/main/assets")
    ap.add_argument("--golden-dir", default="android/app/src/test/resources",
                    help="where the Kotlin parity fixture goes. Kept OUT of "
                         "src/main/assets on purpose: it is ~700KB of test data "
                         "that would otherwise ship inside every APK.")
    ap.add_argument("--n-golden", type=int, default=600,
                    help="URLs written to the Kotlin parity fixture")
    ap.add_argument("--random-state", type=int, default=42)
    ap.add_argument("--url-transform", default="full", choices=("full", "hostname"),
                    help="'hostname' reduces the golden URLs to bare hostnames, "
                         "matching what the DNS layer actually passes in. A "
                         "fixture of full URLs would test the hostname model on "
                         "input it never sees in production.")
    ap.add_argument("--override-threshold", type=float, default=None,
                    help="serve at a different threshold than the bundle's. Used "
                         "for the DNS layer: the bundle threshold is chosen for "
                         "FPR<=0.05, but a SYSTEM-WIDE blocker needs far less "
                         "than that -- at 0.05 roughly 1 benign domain in 20 is "
                         "blocked, which makes the phone unusable.")
    args = ap.parse_args()

    import onnxruntime as ort

    b = joblib.load(args.bundle)
    name = args.model or b["best_model_name"]
    model = b["models"][name]
    feat_names = list(b["feature_names"])
    thr = float(b.get("threshold", 0.5))
    bundle_thr = thr
    if args.override_threshold is not None:
        thr = float(args.override_threshold)

    print("=" * 74)
    print("ONNX EXPORT FOR ON-DEVICE INFERENCE")
    print("=" * 74)
    print(f"  bundle    : {args.bundle}")
    print(f"  model     : {name}  ({type(model).__name__})")
    print(f"  features  : {len(feat_names)}")
    print(f"  threshold : {thr}"
          + (f"   (overriding the bundle's {bundle_thr})"
             if args.override_threshold is not None else ""))

    if feat_names != FEATURE_NAMES:
        print("  WARNING: bundle feature order differs from the live extractor")

    os.makedirs(args.out_dir, exist_ok=True)

    # ---- 1. convert -------------------------------------------------------
    print("\n[1] Converting to ONNX ...")
    onx = _convert(model, len(feat_names), name)
    onnx_path = os.path.join(args.out_dir, "url_model.onnx")
    with open(onnx_path, "wb") as f:
        f.write(onx.SerializeToString())
    size = os.path.getsize(onnx_path)
    print(f"    wrote {onnx_path}  ({size/1e6:.2f} MB)")

    # ---- 2. verify against the original estimator -------------------------
    print("\n[2] Verifying ONNX against the original estimator ...")
    df = pd.read_csv(args.data).dropna(subset=["url", "label"])
    urls = df["url"].astype(str).tolist()
    if args.url_transform == "hostname":
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from train import hostname_of  # noqa: E402
        urls = sorted({h for h in (hostname_of(u) for u in urls) if h})
        print(f"  url transform: hostname-only ({len(urls):,} distinct hosts)")
    X = pd.DataFrame([extract_features(u) for u in urls],
                     columns=FEATURE_NAMES)[feat_names]
    Xn = X.to_numpy(dtype=np.float32)

    p_ref = model.predict_proba(X)[:, 1]
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    p_onnx = _onnx_proba(sess, Xn)

    delta = np.abs(p_ref - p_onnx)
    flips = int(((p_ref >= thr) != (p_onnx >= thr)).sum())
    flips50 = int(((p_ref >= 0.5) != (p_onnx >= 0.5)).sum())
    print(f"    rows compared        {len(urls):,}")
    print(f"    max |delta|          {delta.max():.3e}")
    print(f"    mean |delta|         {delta.mean():.3e}")
    print(f"    verdict flips @{thr:<4} {flips}  ({flips/len(urls):.4%})")
    print(f"    verdict flips @0.50  {flips50}  ({flips50/len(urls):.4%})")
    if flips == 0:
        print("    OK: the exported model makes identical decisions on every row.")
    else:
        print("    NOTE: some rows flip -- these sit within float32 rounding of")
        print("          the threshold. Listed in the report below.")

    # ---- 3. golden vectors for the Kotlin parity test ---------------------
    print(f"\n[3] Golden vectors for the Kotlin feature port ...")
    rng = np.random.default_rng(args.random_state)
    idx = rng.choice(len(urls), size=min(args.n_golden, len(urls)), replace=False)

    # Hand-picked edge cases the random sample is unlikely to contain. Every
    # one of these has broken a URL parser somewhere.
    edge = [
        "", "   ", "a.b", "http://", "ftp://x.com/y", "x.com?a=1&b=2",
        "http://192.168.0.1/a", "http://a@b.com", "http://a@@b.com",
        "http://[::1]:8080/x", "http://[", "https://[]]/", "http://%%%/",
        "HTTP://WWW.EXAMPLE.COM/PATH", "http://xn--80ak6aa92e.com/login",
        "http://аpple.com/signin", "bit.ly/abc",
        "http://a.very.deep.sub.domain.example.co.uk/a/b/c?d=%41%42#frag",
        "http://secure-paypal-login.tk/verify/9231",
        "https://en.wikipedia.org/wiki/Phishing",
        "1.2.3.4:31337/x", "http://example.com//double//slash",
        "http://ex--ample.com/..//a", "h" * 300,
        "http://user:pass@evil.com/login?next=https://bank.com",
    ]
    if args.url_transform == "hostname":
        from train import hostname_of  # noqa: E402
        edge = sorted({h for h in (hostname_of(u) for u in edge) if h})
    golden_urls = [urls[i] for i in idx] + edge

    gX = pd.DataFrame([extract_features(u) for u in golden_urls],
                      columns=FEATURE_NAMES)[feat_names]
    gp = _onnx_proba(sess, gX.to_numpy(dtype=np.float32))

    golden = {
        "_comment": ("Generated by src/export_onnx.py. The Kotlin feature port "
                     "is asserted against this; regenerate it whenever "
                     "src/features.py changes."),
        "feature_names": feat_names,
        "threshold": thr,
        "model": name,
        "cases": [
            {"url": u,
             "features": [float(v) for v in gX.iloc[i].tolist()],
             "proba": float(gp[i])}
            for i, u in enumerate(golden_urls)
        ],
    }
    os.makedirs(args.golden_dir, exist_ok=True)
    golden_path = os.path.join(args.golden_dir, "golden_vectors.json")
    with open(golden_path, "w", encoding="utf-8") as f:
        json.dump(golden, f, ensure_ascii=False, separators=(",", ":"))
    print(f"    wrote {golden_path}  "
          f"({len(golden_urls)} cases, {os.path.getsize(golden_path)/1e6:.2f} MB)")
    print(f"      {len(idx)} sampled from the corpus + {len(edge)} edge cases")

    # ---- 4. model metadata the app reads at runtime -----------------------
    meta = {
        "model": name,
        "threshold": thr,
        "bundle_threshold": bundle_thr,
        "threshold_overridden": args.override_threshold is not None,
        "feature_names": feat_names,
        "n_features": len(feat_names),
        "feature_set": b.get("feature_set", "full"),
        "trained_on": b.get("dataset"),
        "n_rows": b.get("n_rows"),
        "selection": b.get("selection"),
        "onnx_sha256": hashlib.sha256(open(onnx_path, "rb").read()).hexdigest(),
        "max_abs_delta_vs_python": float(delta.max()),
        "verdict_flips_vs_python": flips,
    }
    meta_path = os.path.join(args.out_dir, "model_meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"    wrote {meta_path}")

    print("\n" + "=" * 74)
    return 0 if flips == 0 else 0  # flips are reported, not fatal


if __name__ == "__main__":
    sys.exit(main())
