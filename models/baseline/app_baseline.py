"""
URL Threat Scanner -- web app, now comparing 3 algorithms per scan.

Run with:
    pip install flask
    python webapp/app.py
Then open http://127.0.0.1:5000
"""

import os
import sys
import time
import joblib
import pandas as pd
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from features import extract_features  # noqa: E402

MODEL_PATH = os.path.join(ROOT, "models", "url_models.joblib")

app = Flask(__name__)
CORS(app)  # allow the browser extension (chrome-extension:// origin) to call this API
_bundle = None


def get_bundle():
    global _bundle
    if _bundle is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"No trained models found at {MODEL_PATH}. Run `python src/train.py` first."
            )
        _bundle = joblib.load(MODEL_PATH)
    return _bundle


@app.route("/")
def index():
    model_exists = os.path.exists(MODEL_PATH)
    model_mtime = None
    algo_names = []
    if model_exists:
        model_mtime = time.strftime("%b %d, %Y %H:%M", time.localtime(os.path.getmtime(MODEL_PATH)))
        try:
            bundle = get_bundle()
            algo_names = list(bundle["models"].keys())
        except Exception:
            pass
    return render_template("index.html", model_exists=model_exists,
                            model_mtime=model_mtime, algo_names=algo_names)


@app.route("/api/check", methods=["GET", "POST"])
def check():
    """Lightweight endpoint for the browser extension: called on every
    navigation, so it returns just the essentials for speed."""
    url = request.args.get("url") if request.method == "GET" else (request.get_json(force=True) or {}).get("url")
    url = (url or "").strip()
    if not url:
        return jsonify({"error": "No URL provided"}), 400

    try:
        bundle = get_bundle()
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 500

    feats = extract_features(url)
    row = pd.DataFrame([feats], columns=bundle["feature_names"])

    votes_malicious = 0
    best_score = 0.0
    for name, model in bundle["models"].items():
        proba = float(model.predict_proba(row)[0, 1])
        if proba >= 0.5:
            votes_malicious += 1
        if name == bundle["best_model_name"]:
            best_score = proba

    total = len(bundle["models"])
    is_malicious = votes_malicious >= (total / 2)

    return jsonify({
        "url": url,
        "malicious": is_malicious,
        "score": round(best_score, 4),
        "votes_malicious": votes_malicious,
        "total_models": total,
    })


@app.route("/api/scan", methods=["POST"])
def scan():
    data = request.get_json(force=True)
    url = (data.get("url") or "").strip()
    if not url:
        return jsonify({"error": "No URL provided"}), 400

    try:
        bundle = get_bundle()
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 500

    feats = extract_features(url)
    row = pd.DataFrame([feats], columns=bundle["feature_names"])

    per_model = []
    votes_malicious = 0
    for name, model in bundle["models"].items():
        proba = float(model.predict_proba(row)[0, 1])
        label = "malicious" if proba >= 0.5 else "benign"
        if label == "malicious":
            votes_malicious += 1
        per_model.append({
            "name": name, "score": round(proba, 4), "label": label,
            "is_best": name == bundle["best_model_name"],
        })

    consensus = "malicious" if votes_malicious >= (len(per_model) / 2) else "benign"

    best_model = bundle["models"][bundle["best_model_name"]]
    if hasattr(best_model, "feature_importances_"):
        importances = dict(zip(bundle["feature_names"], best_model.feature_importances_))
        ranked = sorted(feats.items(), key=lambda kv: importances.get(kv[0], 0), reverse=True)[:6]
    else:
        ranked = list(feats.items())[:6]

    return jsonify({
        "url": url,
        "consensus": consensus,
        "votes_malicious": votes_malicious,
        "total_models": len(per_model),
        "models": per_model,
        "best_model_name": bundle["best_model_name"],
        "top_features": [{"name": k, "value": v} for k, v in ranked],
    })


if __name__ == "__main__":
    print("Starting URL Threat Scanner at http://127.0.0.1:5000")
    app.run(debug=True, port=5000)
