"""
URL Threat Scanner -- web app, now comparing 3 algorithms per scan.

Run with:
    pip install flask
    python webapp/app.py
Then open http://127.0.0.1:5000
"""

import functools
import json
import os
import sys
import threading
import time
import uuid
import joblib
import pandas as pd
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from features import extract_features  # noqa: E402
import assessment
from assessment import INSUFFICIENT_EVIDENCE_BAND  # noqa: E402
import auth  # noqa: E402
import fusion
import risk_engine  # noqa: E402
import infra_graph  # noqa: E402
import threat_intel  # noqa: E402
from js_analysis import js_signals  # noqa: E402
from page_layers import analyse_page, empty_features  # noqa: E402
from web_fetch import fetch as web_fetch, FetchError  # noqa: E402
from web_features import extract_web_features, WEB_FEATURE_NAMES  # noqa: E402

# Prefer the v3 multi-source bundle; fall back to the original so an install
# that has not rebuilt the dataset still runs.
_V3 = os.path.join(ROOT, "models", "v3", "url_models.joblib")
_V2 = os.path.join(ROOT, "models", "url_models.joblib")
MODEL_PATH = _V3 if os.path.exists(_V3) else _V2

_V3H = os.path.join(ROOT, "models", "v3_hostname", "url_models.joblib")
_V2H = os.path.join(ROOT, "models", "hostname", "url_models.joblib")
# The hostname-only model the Android DNS layer serves. Loaded lazily and only
# when a hostname-scope assessment is requested, so the normal path is unaffected.
HOSTNAME_MODEL_PATH = _V3H if os.path.exists(_V3H) else _V2H
WEB_MODEL_PATH = os.path.join(ROOT, "models", "web", "web_model.joblib")
HISTORY_PATH = os.path.join(ROOT, "models", "scan_history.json")
HISTORY_MAX = 200
_history_lock = threading.Lock()

app = Flask(__name__)
CORS(app)  # allow the browser extension (chrome-extension:// origin) to call this API
auth.init_db()


# ------------------------------------------------------------ rate limiting
# Deliberately simple: an in-process sliding window. Enough to blunt credential
# stuffing and scan floods on a single-instance prototype, and honest about
# what it is -- a multi-process deployment would need a shared store.
_RATE = {}
_RATE_LOCK = threading.Lock()
RATE_RULES = {
    "login": (10, 300),        # 10 attempts per 5 minutes per ip+email
    "register": (5, 3600),
    "reset": (5, 3600),
    "scan": (120, 60),
}


def _client_ip():
    fwd = request.headers.get("X-Forwarded-For", "")
    return (fwd.split(",")[0].strip() if fwd else request.remote_addr) or "?"


def rate_limit(bucket, key_extra=""):
    limit, window = RATE_RULES[bucket]
    key = f"{bucket}:{_client_ip()}:{key_extra}"
    now = time.time()
    with _RATE_LOCK:
        hits = [t for t in _RATE.get(key, []) if now - t < window]
        if len(hits) >= limit:
            retry = int(window - (now - hits[0])) + 1
            return retry
        hits.append(now)
        _RATE[key] = hits
    return 0


def _rate_limited(retry):
    return jsonify({
        "error": "Too many requests. Please wait before trying again.",
        "code": "rate_limited", "retry_after_s": retry,
    }), 429


# ------------------------------------------------------------ auth helpers
def current_user(required=False):
    """Resolve the bearer token, if any.

    Scanning stays available anonymously -- the extension has no account -- but
    anything that touches stored history requires a verified token.
    """
    hdr = request.headers.get("Authorization", "")
    if hdr.startswith("Bearer "):
        payload = auth.decode_access(hdr[7:].strip())
        user = auth.get_user(int(payload["sub"]))
        if user is None:
            raise auth.AuthError("Account not found.", 401, "token_invalid")
        return user
    if required:
        raise auth.AuthError("Sign in to continue.", 401, "auth_required")
    return None


def require_auth(fn):
    @functools.wraps(fn)
    def wrapper(*a, **kw):
        try:
            user = current_user(required=True)
        except auth.AuthError as exc:
            return jsonify({"error": exc.message, "code": exc.code}), exc.status
        return fn(user, *a, **kw)
    return wrapper


@app.errorhandler(auth.AuthError)
def _auth_error(exc):
    return jsonify({"error": exc.message, "code": exc.code}), exc.status
_bundle = None


def get_threshold(bundle):
    """Deployment decision threshold.

    v2 bundles carry a threshold chosen on out-of-fold, domain-grouped CV.
    v1 bundles have no such key, so fall back to 0.5 and stay working."""
    try:
        return float(bundle.get("threshold", 0.5))
    except (TypeError, ValueError):
        return 0.5


_hostname_bundle = None
_web_bundle = None


_infra_graph = None


def _graph():
    """Load the prebuilt graph once. Built offline by src/infra_graph.py;
    never computed per request."""
    global _infra_graph
    if _infra_graph is None:
        path = os.path.join(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))), "models", "infra_graph.json")
        if os.path.exists(path):
            try:
                with open(path, encoding="utf-8") as fh:
                    _infra_graph = json.load(fh)
            except Exception:                       # noqa: BLE001
                _infra_graph = {"available": False,
                                "reason": "the graph file could not be read"}
        else:
            _infra_graph = {"available": False,
                            "reason": "no graph has been built; run "
                                      "python src/infra_graph.py"}
    return _infra_graph


def get_web_bundle():
    """Website-pattern model, or None if it was never trained."""
    global _web_bundle
    if _web_bundle is None:
        if not os.path.exists(WEB_MODEL_PATH):
            return None
        _web_bundle = joblib.load(WEB_MODEL_PATH)
    return _web_bundle


def analyse_website(url):
    """Fetch and score a page. Never raises -- returns an unavailable layer.

    Fail-open is the rule everywhere in URLGuard: a page that cannot be
    retrieved must leave the URL assessment intact rather than blocking the
    scan or being counted as either incriminating or exonerating.
    """
    wb = get_web_bundle()
    if wb is None:
        return risk_engine.layer_unavailable(
            "The website model is not installed on this server.",
            "model_missing"), None, None, empty_features()
    try:
        res = web_fetch(url)
    except FetchError as exc:
        return risk_engine.layer_unavailable(
            f"The page could not be retrieved: {exc.reason}", exc.code), None, None, empty_features()
    except Exception as exc:                        # noqa: BLE001
        return risk_engine.layer_unavailable(
            f"The page could not be retrieved ({type(exc).__name__}).",
            "fetch_failed"), None, None, empty_features()

    try:
        feats = extract_web_features(res["html"], res["final_url"], res)
        row = pd.DataFrame([feats], columns=wb["feature_names"])
        model = wb["models"][wb["best_model_name"]]
        proba = float(model.predict_proba(row)[0, 1])
    except Exception as exc:                        # noqa: BLE001
        return risk_engine.layer_unavailable(
            f"The page was retrieved but could not be analysed "
            f"({type(exc).__name__}).", "analysis_failed"), None, res, empty_features()

    # JavaScript, redirect-chain and structural-fingerprint layers. Computed
    # by the same helper the dataset builder uses, so the fusion model is fed
    # the numbers it was trained on. A failure here degrades the extra layers
    # without losing the page score that already succeeded.
    try:
        page_layers = analyse_page(res)
    except Exception as exc:                        # noqa: BLE001
        page_layers = {"features": empty_features(), "js": None,
                       "redirects": None, "fingerprint": None,
                       "error": f"{type(exc).__name__}"}

    score = min(99, max(1, int(round(proba * 100))))
    layer = {
        "available": True,
        "score": score,
        "level": risk_engine._level(score),
        "raw_probability": round(proba, 6),
        "threshold_score": int(round(float(wb.get("threshold", 0.5)) * 100)),
        "model_version": wb.get("model_version"),
        "features": feats,
        "basis": ("Static analysis of the retrieved HTML. No JavaScript was "
                  "executed, no form was submitted and no credential was sent."),
        "fetch": {
            "final_url": res["final_url"],
            "http_status": res["status"],
            "redirect_count": res["redirect_count"],
            "bytes": res["bytes"],
            "truncated": res["truncated"],
            "tls_verified": res["tls_verified"],
            "elapsed_ms": res["elapsed_ms"],
            "redirects": res["redirects"],
        },
        "javascript": page_layers.get("js"),
        "redirect_analysis": page_layers.get("redirects"),
        "fingerprint": page_layers.get("fingerprint"),
    }
    return layer, feats, res, page_layers.get("features") or empty_features()


def get_hostname_bundle():
    """Bundle for hostname-scope assessment, or None if it was never trained."""
    global _hostname_bundle
    if _hostname_bundle is None:
        if not os.path.exists(HOSTNAME_MODEL_PATH):
            return None
        _hostname_bundle = joblib.load(HOSTNAME_MODEL_PATH)
    return _hostname_bundle


_SCHEME_RE = __import__("re").compile(r"^([a-zA-Z][a-zA-Z0-9+.\-]*):")
_HOSTLIKE_RE = __import__("re").compile(r"^[^\s/?#@:]+\.[^\s/?#@:]{2,}$")


def validate_target(raw):
    """Reject inputs the model was never meant to judge.

    The classifier was trained on http/https web URLs. Scoring a
    `javascript:` or `mailto:` URI against it and reporting "LOW RISK" would
    present a non-web URI as a safe website -- a confident answer to a question
    the model cannot answer. Those are refused instead.

    Returns None when the input is acceptable, or an error message.
    """
    u = raw.strip()
    m = _SCHEME_RE.match(u)
    if m:
        scheme = m.group(1).lower()
        rest = u[m.end():]
        if rest.startswith("//"):
            # A real scheme: "https://host", "javascript://x".
            if scheme not in ("http", "https"):
                return (f"'{scheme}://' is not a web address scheme. URLGuard "
                        f"assesses http and https URLs only.")
        else:
            # Could be "mailto:a@b.com" (a scheme) or "example.com:8080/x"
            # (a host and port). Only the latter has digits up to the first
            # path/query separator.
            port = rest.split("/", 1)[0].split("?", 1)[0]
            is_host_port = port.isdigit() and 0 < int(port) <= 65535
            if not is_host_port and scheme not in ("http", "https"):
                return (f"'{scheme}:' is not a web address scheme. URLGuard "
                        f"assesses http and https URLs only.")

    host = hostname_of(u)
    if not host:
        return "No hostname could be parsed from that input."
    if host in ("localhost",) or _HOSTLIKE_RE.match(host):
        return None
    # bare IPv4 literal
    parts = host.split(".")
    if len(parts) == 4 and all(p.isdigit() and 0 <= int(p) <= 255 for p in parts):
        return None
    return ("That does not look like a web address. Enter a URL such as "
            "https://example.com/page.")


def hostname_of(u):
    """What a DNS query would carry: hostname only, no scheme/port/path."""
    import re
    s = str(u).strip()
    s = re.sub(r"^[a-zA-Z][a-zA-Z0-9+.\-]*://", "", s)
    s = s.split("/", 1)[0].split("?", 1)[0].split("#", 1)[0]
    s = s.rsplit("@", 1)[-1].split(":", 1)[0]
    return s.strip().lower()


def get_bundle():
    global _bundle
    if _bundle is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"No trained models found at {MODEL_PATH}. Run `python src/train.py` first."
            )
        _bundle = joblib.load(MODEL_PATH)
    return _bundle


# ----------------------------------------------------------------- history
# Real scans only. The dashboard reports "Insufficient data" on an empty store
# rather than showing sample numbers -- a security tool that invents its own
# statistics cannot be trusted about anything else either.
def _read_history():
    if not os.path.exists(HISTORY_PATH):
        return []
    try:
        with open(HISTORY_PATH, encoding="utf-8") as fh:
            data = json.load(fh)
        return data if isinstance(data, list) else []
    except (json.JSONDecodeError, OSError):
        return []


def _append_history(entry):
    with _history_lock:
        items = _read_history()
        items.insert(0, entry)
        del items[HISTORY_MAX:]
        tmp = HISTORY_PATH + ".tmp"
        try:
            os.makedirs(os.path.dirname(HISTORY_PATH), exist_ok=True)
            with open(tmp, "w", encoding="utf-8") as fh:
                json.dump(items, fh)
            os.replace(tmp, HISTORY_PATH)   # atomic: never a half-written file
        except OSError:
            pass
        return items


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

    thr = get_threshold(bundle)
    votes_malicious = 0
    best_score = 0.0
    for name, model in bundle["models"].items():
        proba = float(model.predict_proba(row)[0, 1])
        if proba >= thr:
            votes_malicious += 1
        if name == bundle["best_model_name"]:
            best_score = proba

    total = len(bundle["models"])
    # The verdict follows the SELECTED model at its calibrated threshold.
    # The consensus vote measured worse than the best model alone on an
    # independent corpus, so it is reported but no longer decides.
    is_malicious = best_score >= thr

    return jsonify({
        "url": url,
        "malicious": is_malicious,
        "score": round(best_score, 4),
        "votes_malicious": votes_malicious,
        "total_models": total,
        "threshold": round(thr, 4),
    })


@app.route("/api/scan", methods=["POST"])
def scan():
    data = request.get_json(force=True)
    url = (data.get("url") or "").strip()
    if not url:
        return jsonify({"error": "No URL provided"}), 400

    invalid = validate_target(url)
    if invalid:
        return jsonify({"error": invalid}), 400

    try:
        bundle = get_bundle()
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 500

    # scope="hostname" reproduces what the DNS layer can see: HTTPS hides the
    # path and query, so that layer only ever has a hostname. It is a different
    # model with materially lower accuracy, and the response says so.
    scope = (data.get("scope") or "url").strip().lower()
    dns_only = scope == "hostname"
    if dns_only:
        hb = get_hostname_bundle()
        if hb is None:
            return jsonify({"error": "Hostname-scope model is not available. Run "
                                     "`python src/train.py --url-transform hostname "
                                     "--model-dir models/hostname`."}), 503
        bundle = hb
        subject = hostname_of(url)
        if not subject:
            return jsonify({"error": "No hostname could be parsed from that input."}), 400
    else:
        subject = url

    started = time.perf_counter()
    feats = extract_features(subject)
    row = pd.DataFrame([feats], columns=bundle["feature_names"])

    thr = get_threshold(bundle)
    per_model = []
    votes_malicious = 0
    best_score = 0.0
    for name, model in bundle["models"].items():
        proba = float(model.predict_proba(row)[0, 1])
        label = "malicious" if proba >= thr else "benign"
        if label == "malicious":
            votes_malicious += 1
        if name == bundle["best_model_name"]:
            best_score = proba
        per_model.append({
            "name": name, "score": round(proba, 4), "label": label,
            "is_best": name == bundle["best_model_name"],
        })

    consensus = "malicious" if best_score >= thr else "benign"

    best_model = bundle["models"][bundle["best_model_name"]]
    if hasattr(best_model, "feature_importances_"):
        importances = dict(zip(bundle["feature_names"], best_model.feature_importances_))
        ranked = sorted(feats.items(), key=lambda kv: importances.get(kv[0], 0), reverse=True)[:6]
    else:
        ranked = list(feats.items())[:6]

    # ---- conservative assessment layer (additive; nothing above changed) ----
    detail = assessment.assess(
        url=subject, proba=best_score, threshold=thr, feats=feats,
        model=best_model, row=row, feature_names=bundle["feature_names"],
        dns_only=dns_only,
    )

    # ---- optional website layer -----------------------------------------
    # Opt-in per scan. Retrieving a page CONTACTS the site, which every other
    # path in URLGuard deliberately does not do, so it never happens unless
    # the caller asks for it.
    want_web = bool(data.get("analyze_website")) and not dns_only
    web_layer, web_feats, _web_res = (None, None, None)
    layer_feats = empty_features()
    if want_web:
        web_layer, web_feats, _web_res, layer_feats = analyse_website(subject)
    elif dns_only:
        web_layer = risk_engine.layer_unavailable(
            "Hostname-only assessment; webpage content was unavailable.",
            "hostname_scope")

    # Threat intel is offline (local snapshots + popularity list), so it runs
    # on every scan without contacting anything.
    try:
        intel = threat_intel.lookup_all(subject)
    except Exception:                               # noqa: BLE001
        intel = None
    fused = risk_engine.combine(detail["risk_score"], web=web_layer, intel=intel)

    # ---- calibrated fusion meta-model ------------------------------------
    # When a page was retrieved and a fusion model is installed, the learned
    # combination replaces the hand-tuned additive engine for the headline
    # score. The rule engine still runs: it supplies the per-layer breakdown
    # and the reputation adjustment, and it is the fallback whenever fusion
    # declines to answer (no page, no model, or an inference failure).
    model_fusion = fusion.score(best_score, thr, web_layer, web_feats,
                                layer_feats)
    detail["threat_intel"] = intel
    detail["layers"] = fused["layers"]
    detail["fusion"] = fused["fusion"]
    detail["fusion"]["model"] = model_fusion
    # `method` already holds the rule engine's human-readable account of what
    # it did; the headline source is recorded alongside it, not over it.
    detail["fusion"]["headline_source"] = ("calibrated_meta_model"
                                           if model_fusion else "rule_engine")
    detail["breakdown"] = risk_engine.breakdown_rows(fused, web_feats)
    detail["url_only_risk_score"] = detail["risk_score"]
    if fused["layers"]["website"].get("available"):
        # The headline becomes the fused score only when a second layer
        # actually contributed evidence.
        if model_fusion:
            headline = model_fusion["score"]
            # The reputation adjustment is applied on top of the model, not
            # learned by it: popularity is derived from the same Tranco
            # snapshot that supplied the benign labels, so training on it
            # would be circular. It stays an explicit, disclosed serving-time
            # correction -- the one that cut hard-benign false positives from
            # 37.5% to 6.2%.
            rep_points = (fused["fusion"] or {}).get("reputation_reduction") or 0
            if rep_points:
                headline = max(1, min(99, headline - int(round(rep_points))))
            detail["fusion"]["reputation_applied_to_model"] = round(rep_points, 1)
            detail["risk_score"] = headline
            detail["risk_band"] = assessment.band_for(headline)
            if model_fusion["undecided"]:
                detail["risk_band"] = INSUFFICIENT_EVIDENCE_BAND
        else:
            detail["risk_score"] = fused["risk_score"]
            detail["risk_band"] = assessment.band_for(fused["risk_score"])
        ps = risk_engine.page_signals(web_feats or {})
        detail["signals"]["warning"] = detail["signals"]["warning"] + [
            {"text": s["text"], "evidence": s["evidence"],
             "source": "website"} for s in ps["warning"]]
        detail["signals"]["positive"] = detail["signals"]["positive"] + [
            {"text": s["text"], "evidence": "Observed in the retrieved page.",
             "source": "website"} for s in ps["positive"]]
        detail["page_signals"] = ps
        detail["explanation_text"] = (
            detail["explanation_text"] + " The website layer contributed a "
            f"page-pattern score of {fused['layers']['website']['score']}/100 "
            "from static analysis of the retrieved HTML.")
        if model_fusion:
            detail["explanation_text"] += (
                f" The {model_fusion['model_name']} fusion model combined the "
                f"URL and page layers by learned weights into a calibrated "
                f"probability of {model_fusion['probability']:.3f}"
                + (", which sits too close to the decision threshold to call "
                   "either way." if model_fusion["undecided"] else "."))
        js = (web_layer or {}).get("javascript") or {}
        for sig in js_signals(js.get("features") or {}):
            detail["signals"]["warning"].append(
                {"text": sig["text"], "evidence": sig["evidence"],
                 "source": "javascript"})
        rdl = (web_layer or {}).get("redirect_analysis") or {}
        for sig in (rdl.get("signals") or []):
            detail["signals"]["warning"].append(
                {"text": sig["text"], "evidence": sig["evidence"],
                 "source": "redirect"})

    elapsed_ms = round((time.perf_counter() - started) * 1000, 1)
    scanned_at = time.time()

    try:
        user = current_user(required=False)
    except auth.AuthError:
        user = None       # an expired token must not block an anonymous scan

    entry = {
        "id": uuid.uuid4().hex[:12],
        "url": subject,
        "domain": detail["domain"]["registered_domain"]["value"],
        "risk_score": detail["risk_score"],
        "risk_band": detail["risk_band"]["label"],
        "risk_key": detail["risk_band"]["key"],
        "confidence": detail["confidence"],
        "model": bundle["best_model_name"],
        "source": "dns_scope" if dns_only else "web_scan",
        "scope": scope,
        "detection_ms": elapsed_ms,
        "scanned_at": scanned_at,
        "model_version": bundle.get("model_version"),
        "dataset_version": bundle.get("dataset_version"),
        "signals": [sig["text"] for sig in detail["signals"]["warning"]],
    }
    auth.record_scan(user["id"] if user else None, entry)
    _append_history(entry)

    payload = {
        # ---- existing contract, unchanged (web frontend + tests rely on it) --
        "url": url,
        "consensus": consensus,
        "votes_malicious": votes_malicious,
        "total_models": len(per_model),
        "models": per_model,
        "best_model_name": bundle["best_model_name"],
        "threshold": round(thr, 4),
        "top_features": [{"name": k, "value": v} for k, v in ranked],
        # ---- additions ------------------------------------------------------
        "scan_id": entry["id"],
        "scope": scope,
        "subject": subject,
        "model_version": bundle.get("model_version"),
        "dataset_version": bundle.get("dataset_version"),
        "scanned_at": scanned_at,
        "detection_ms": elapsed_ms,
        "model_agreement": {"flagged": votes_malicious, "total": len(per_model)},
        # Which layers actually contributed, so the client never has to infer
        # it from the presence or absence of a score.
        "website_analysed": bool(detail["layers"]["website"].get("available")),
        "web_model_version": (detail["layers"]["website"].get("model_version")
                              if detail["layers"]["website"].get("available")
                              else None),
        "assessment": detail,
    }
    return jsonify(payload)


# ---------------------------------------------------------- history / stats
def _uid():
    """User id for storage scoping, or None for anonymous.

    Anonymous rows are stored with a NULL user_id and are only ever returned
    to other anonymous callers -- a signed-in user never sees them mixed into
    their own history, and no user can read another's.
    """
    try:
        u = current_user(required=False)
    except auth.AuthError:
        return None
    return u["id"] if u else None


@app.route("/api/history", methods=["GET"])
def history():
    """Scans for the caller only, newest first. Real entries only."""
    try:
        limit = max(1, min(int(request.args.get("limit", 25)), HISTORY_MAX))
    except (TypeError, ValueError):
        limit = 25
    try:
        offset = max(0, int(request.args.get("offset", 0)))
    except (TypeError, ValueError):
        offset = 0
    items, total = auth.list_scans(
        _uid(), limit=limit, offset=offset,
        q=(request.args.get("q") or "").strip() or None,
        band=(request.args.get("band") or "").strip() or None,
        source=(request.args.get("source") or "").strip() or None,
        sort=request.args.get("sort", "scanned_at"),
        order=request.args.get("order", "desc"))
    return jsonify({"items": items, "total": total,
                    "limit": limit, "offset": offset})


@app.route("/api/history/<scan_id>", methods=["GET"])
def history_item(scan_id):
    row = auth.get_scan(_uid(), scan_id)
    if row is None:
        # Same response whether the id does not exist or belongs to someone
        # else: otherwise this endpoint becomes an existence oracle.
        return jsonify({"error": "Scan not found"}), 404
    return jsonify(row)


@app.route("/api/stats", methods=["GET"])
def stats():
    """Dashboard counters, computed from the stored scans.

    `sufficient_data` is false on an empty store so the UI can say
    "Insufficient data" instead of rendering zeroes that look like findings.
    """
    out = auth.stats(_uid())
    out["history_cap"] = HISTORY_MAX
    return jsonify(out)


@app.route("/api/history", methods=["DELETE"])
def clear_history():
    auth.clear_scans(_uid())
    with _history_lock:
        try:
            if os.path.exists(HISTORY_PATH):
                os.remove(HISTORY_PATH)
        except OSError:
            pass
    return jsonify({"ok": True, "total": 0})


@app.route("/api/graph", methods=["GET"])
def api_graph():
    """Infrastructure relationships observed in the recorded fetch corpus.

    With `?domain=` it returns only that domain's edges. A domain the corpus
    never saw returns an empty result with a note saying so -- which is not
    the same as the domain having no relationships.
    """
    g = _graph()
    if not g.get("available"):
        return jsonify({"available": False, "reason": g.get("reason")}), 200
    domain = (request.args.get("domain") or "").strip()
    if domain:
        return jsonify(infra_graph.neighbours(g, domain))
    return jsonify({
        "available": True,
        "node_count": g["node_count"],
        "edge_count": g["edge_count"],
        "counts": g["counts"],
        "clusters": g["kit_clusters"][:25],
        "cluster_count": g["kit_cluster_count"],
        "top_domains": g["nodes"][:25],
        "missing_edge_types": g["missing_edge_types"],
        "limitation": g["limitation"],
        "source": g["source"],
        "pages_examined": g["pages_examined"],
    })


@app.route("/api/meta", methods=["GET"])
def meta():
    """What the UI needs to describe the system accurately."""
    try:
        bundle = get_bundle()
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 500
    return jsonify({
        "models": list(bundle["models"].keys()),
        "best_model_name": bundle["best_model_name"],
        "model_version": bundle.get("model_version"),
        "dataset_version": bundle.get("dataset_version"),
        "threshold": round(get_threshold(bundle), 4),
        "n_features": len(bundle["feature_names"]),
        "bands": [
            {"label": lab, "key": key, "min": lo, "max": hi, "description": txt}
            for lo, hi, lab, key, txt in assessment.RISK_BANDS
        ],
        "disclaimer": assessment.ASSESSMENT_DISCLAIMER,
        "signal_basis": {
            "percentile": assessment._THRESHOLDS.get("percentile"),
            "n_benign": assessment._THRESHOLDS.get("n_benign"),
            "min_lift": assessment.MIN_SIGNAL_LIFT,
        },
        "insufficient_evidence_band": assessment.INSUFFICIENT_EVIDENCE_BAND,
        "website_analysis_available": get_web_bundle() is not None,
        "website_analysis_default": False,
        "fusion": fusion.model_info(),
        "fusion_available": fusion.available(),
        # Stated so the UI can be explicit about it rather than implying the
        # blocklist feeds contributed to the learned model.
        "threat_intel_is_serving_only": True,
        "threat_intel_note": (
            "Blocklist and popularity snapshots are applied as disclosed "
            "serving-time signals, not as learned model inputs: on this "
            "dataset they are the source of the labels, so training on them "
            "would be circular."),
        "page_layers": [
            "HTML structure and form destinations",
            "Inline JavaScript, matched statically and never executed",
            "Redirect chain, including shorteners and HTTPS downgrades",
            "Structural fingerprint for phishing-kit similarity",
        ],
        "url_analysis_never_contacts_url": True,
        "website_analysis_contacts_url": True,
        "website_analysis_safety": [
            "GET only; no form submitted, no credential sent.",
            "No JavaScript executed; HTML parsed statically.",
            "No cookies stored or returned.",
            "Private, loopback and link-local addresses refused (SSRF guard), "
            "re-checked on every redirect.",
            "Size, time and redirect limits enforced.",
        ],
    })


# ============================================================ auth endpoints
def _auth_payload(user, tokens):
    return {"user": {k: user[k] for k in
                     ("id", "name", "email", "created_at", "preferences")},
            **tokens}


@app.route("/api/auth/register", methods=["POST"])
def api_register():
    retry = rate_limit("register")
    if retry:
        return _rate_limited(retry)
    d = request.get_json(silent=True) or {}
    user = auth.create_user(d.get("name"), d.get("email"),
                            d.get("password"), d.get("confirm_password"))
    tokens = auth.issue_tokens(user, request.headers.get("User-Agent"))
    return jsonify(_auth_payload(user, tokens)), 201


@app.route("/api/auth/login", methods=["POST"])
def api_login():
    d = request.get_json(silent=True) or {}
    email = (d.get("email") or "").strip().lower()
    retry = rate_limit("login", email)
    if retry:
        return _rate_limited(retry)
    user = auth.verify_password(email, d.get("password"))
    tokens = auth.issue_tokens(user, request.headers.get("User-Agent"))
    return jsonify(_auth_payload(user, tokens))


@app.route("/api/auth/refresh", methods=["POST"])
def api_refresh():
    d = request.get_json(silent=True) or {}
    tokens, user = auth.refresh_tokens(d.get("refresh_token"),
                                       request.headers.get("User-Agent"))
    return jsonify(_auth_payload(user, tokens))


@app.route("/api/auth/logout", methods=["POST"])
def api_logout():
    d = request.get_json(silent=True) or {}
    if d.get("refresh_token"):
        auth.revoke_refresh(d["refresh_token"])
    if d.get("all_devices"):
        try:
            u = current_user(required=True)
            auth.revoke_all(u["id"])
        except auth.AuthError:
            pass
    return jsonify({"ok": True})


@app.route("/api/auth/me", methods=["GET"])
@require_auth
def api_me(user):
    return jsonify({"user": user})


@app.route("/api/auth/profile", methods=["PATCH"])
@require_auth
def api_profile(user):
    d = request.get_json(silent=True) or {}
    updated = auth.update_profile(user["id"], name=d.get("name"),
                                  preferences=d.get("preferences"))
    return jsonify({"user": updated})


@app.route("/api/auth/password", methods=["POST"])
@require_auth
def api_change_password(user):
    d = request.get_json(silent=True) or {}
    auth.change_password(user["id"], d.get("current_password"),
                         d.get("new_password"))
    return jsonify({"ok": True,
                    "message": "Password changed. Other sessions were signed out."})


@app.route("/api/auth/forgot", methods=["POST"])
def api_forgot():
    retry = rate_limit("reset")
    if retry:
        return _rate_limited(retry)
    d = request.get_json(silent=True) or {}
    token = auth.begin_password_reset((d.get("email") or "").strip().lower())
    # The response is identical whether or not the address exists, so this
    # endpoint cannot be used to discover who has an account.
    body = {
        "ok": True,
        "message": ("If that address has an account, a reset token has been "
                    "issued."),
        "delivery": ("This prototype does not send email. An operator must "
                     "relay the token out of band."),
    }
    if token and app.config.get("EXPOSE_RESET_TOKEN"):
        # Only in TESTING/dev, never in a normal run.
        body["reset_token"] = token
    return jsonify(body)


@app.route("/api/auth/reset", methods=["POST"])
def api_reset():
    d = request.get_json(silent=True) or {}
    auth.complete_password_reset(d.get("token"), d.get("new_password"))
    return jsonify({"ok": True, "message": "Password updated. Please sign in."})


@app.route("/api/analytics", methods=["GET"])
def api_analytics():
    """Aggregates over the caller's own scans only."""
    return jsonify(auth.stats(_uid()))


if __name__ == "__main__":
    print("Starting URL Threat Scanner at http://127.0.0.1:5000")
    app.run(debug=True, port=5000)
