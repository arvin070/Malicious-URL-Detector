"""
Score URLs against all 3 trained algorithms (Decision Tree, Random Forest,
XGBoost) side by side -- useful for demonstrating how the algorithms agree
or disagree on a given URL.

Usage:
  python src/monitor.py --url "http://secure-paypal-login.tk/verify/9231"
  python src/monitor.py --file data/urls_to_check.csv
  python src/monitor.py --url "..." --model "Random Forest"   # single model only
"""

import argparse
import joblib
import pandas as pd

from features import extract_features, FEATURE_NAMES


def load_bundle(path="models/url_models.joblib"):
    return joblib.load(path)


def score_url_all_models(url, bundle):
    feats = extract_features(url)
    row = pd.DataFrame([feats], columns=bundle["feature_names"])
    results = {}
    for name, model in bundle["models"].items():
        proba = model.predict_proba(row)[0, 1]
        results[name] = proba
    return results


def get_threshold(bundle):
    """v2 bundles carry a calibrated threshold; v1 bundles fall back to 0.5."""
    try:
        return float(bundle.get("threshold", 0.5))
    except (TypeError, ValueError):
        return 0.5


def print_result(url, results, best_model_name, threshold=0.5):
    print(f"\nURL: {url}   [threshold {threshold:g}]")
    for name, proba in results.items():
        label = "MALICIOUS" if proba >= threshold else "benign"
        flag = "⚠️ " if label == "MALICIOUS" else "✅"
        star = "  ← best model" if name == best_model_name else ""
        print(f"  {flag} {name:15s} [{label:9s}] score={proba:.3f}{star}")

    votes_malicious = sum(1 for p in results.values() if p >= threshold)
    verdict = "MALICIOUS" if results.get(best_model_name, 0.0) >= threshold else "benign"
    print(f"  Verdict ({best_model_name}): {verdict}"
          f"   [{votes_malicious}/{len(results)} models agree malicious]")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="models/url_models.joblib")
    parser.add_argument("--url", help="Score a single URL against all models")
    parser.add_argument("--file", help="CSV with a 'url' column to batch score")
    args = parser.parse_args()

    bundle = load_bundle(args.model_path)
    best_model_name = bundle["best_model_name"]
    thr = get_threshold(bundle)

    if args.url:
        results = score_url_all_models(args.url, bundle)
        print_result(args.url, results, best_model_name, thr)
    elif args.file:
        df = pd.read_csv(args.file)
        col = "url" if "url" in df.columns else df.columns[0]
        for url in df[col].dropna():
            results = score_url_all_models(url, bundle)
            print_result(url, results, best_model_name, thr)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
