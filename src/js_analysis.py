"""
Static analysis of JavaScript found in a fetched page.

NOTHING HERE RUNS. Not in a sandbox, not in a VM, not "carefully". The script
text is treated as a string and matched against patterns. That is the whole
design: a phishing kit's payload cannot do anything by being analysed, because
nothing ever evaluates it.

That restriction has a cost and the cost is stated rather than hidden. Static
matching sees `eval(atob("..."))`; it does not see what the decoded string
does. Heavily packed code therefore registers as "obfuscated" without its
behaviour being known, and the analysis says exactly that instead of guessing.

WHAT IS LOOKED FOR
------------------
  obfuscation   packing, encoding, character-code construction, dense escapes
  exfiltration  form submission to another origin, fetch/XHR/beacon posting,
                websocket use, keystroke capture
  evasion       devtools/debugger detection, right-click and copy blocking,
                bot/automation checks, console suppression
  redirection   location assignment, meta/timer-driven navigation
  credential    direct reads of password-field values

Every hit records the pattern name and a count. None of it is proof of intent:
real sites use `fetch` and legitimate SPAs set `location.href`. Weighting that
into a risk contribution is the fusion model's job, not this module's.
"""

import math
import re
from collections import Counter

# (compiled pattern, group, name). Patterns are deliberately shallow: a
# regex-based reader of a Turing-complete language is a heuristic, and treating
# it as anything more would be dishonest.
PATTERNS = [
    # ---- obfuscation -----------------------------------------------------
    (re.compile(r"\beval\s*\(", re.I), "obfuscation", "eval_call"),
    (re.compile(r"\bFunction\s*\(\s*['\"]", re.I), "obfuscation", "function_constructor"),
    (re.compile(r"\batob\s*\(", re.I), "obfuscation", "base64_decode"),
    (re.compile(r"\bunescape\s*\(", re.I), "obfuscation", "unescape"),
    (re.compile(r"decodeURIComponent\s*\(", re.I), "obfuscation", "decode_uri"),
    (re.compile(r"String\s*\.\s*fromCharCode", re.I), "obfuscation", "from_char_code"),
    (re.compile(r"document\s*\.\s*write\s*\(", re.I), "obfuscation", "document_write"),
    (re.compile(r"\\x[0-9a-fA-F]{2}(?:\\x[0-9a-fA-F]{2}){9,}"), "obfuscation",
     "hex_escape_run"),
    (re.compile(r"\\u[0-9a-fA-F]{4}(?:\\u[0-9a-fA-F]{4}){9,}"), "obfuscation",
     "unicode_escape_run"),
    (re.compile(r"[A-Za-z0-9+/]{200,}={0,2}"), "obfuscation", "long_base64_blob"),
    (re.compile(r"_0x[0-9a-f]{4,}", re.I), "obfuscation", "hex_identifier"),
    (re.compile(r"\[\s*['\"][^'\"]+['\"]\s*\]\s*\(", re.I), "obfuscation",
     "bracket_call"),

    # ---- exfiltration ----------------------------------------------------
    (re.compile(r"\bfetch\s*\(", re.I), "exfiltration", "fetch_call"),
    (re.compile(r"XMLHttpRequest", re.I), "exfiltration", "xhr"),
    (re.compile(r"navigator\s*\.\s*sendBeacon", re.I), "exfiltration", "send_beacon"),
    (re.compile(r"new\s+WebSocket", re.I), "exfiltration", "websocket"),
    (re.compile(r"\.\s*submit\s*\(\s*\)", re.I), "exfiltration", "form_submit_call"),
    (re.compile(r"new\s+FormData", re.I), "exfiltration", "formdata"),
    (re.compile(r"addEventListener\s*\(\s*['\"]key(down|press|up)", re.I),
     "exfiltration", "key_listener"),
    (re.compile(r"\.\s*(onkeydown|onkeypress|onkeyup)\s*=", re.I),
     "exfiltration", "key_handler"),

    # ---- evasion ---------------------------------------------------------
    (re.compile(r"\bdebugger\b"), "evasion", "debugger_statement"),
    (re.compile(r"devtools|firebug", re.I), "evasion", "devtools_check"),
    (re.compile(r"addEventListener\s*\(\s*['\"]contextmenu", re.I), "evasion",
     "contextmenu_block"),
    (re.compile(r"oncontextmenu\s*=", re.I), "evasion", "contextmenu_attr"),
    (re.compile(r"addEventListener\s*\(\s*['\"](copy|cut|selectstart)", re.I),
     "evasion", "copy_block"),
    (re.compile(r"navigator\s*\.\s*webdriver", re.I), "evasion", "webdriver_check"),
    (re.compile(r"console\s*\.\s*(log|clear)\s*=\s*function", re.I), "evasion",
     "console_override"),
    (re.compile(r"navigator\s*\.\s*userAgent", re.I), "evasion", "ua_inspection"),

    # ---- redirection -----------------------------------------------------
    (re.compile(r"(window|document|top|self)\s*\.\s*location\s*(\.\s*(href|replace|assign)\s*)?=",
                re.I), "redirection", "location_assign"),
    (re.compile(r"location\s*\.\s*(replace|assign)\s*\(", re.I), "redirection",
     "location_method"),
    (re.compile(r"setTimeout\s*\([^)]{0,80}location", re.I | re.S), "redirection",
     "timed_redirect"),
    (re.compile(r"history\s*\.\s*(pushState|replaceState)", re.I), "redirection",
     "history_rewrite"),

    # ---- credential access ------------------------------------------------
    (re.compile(r"getElementById\s*\(\s*['\"][^'\"]*pass", re.I), "credential",
     "password_field_read"),
    (re.compile(r"\[\s*['\"]password['\"]\s*\]", re.I), "credential",
     "password_key_access"),
    (re.compile(r"type\s*=\s*['\"]password['\"]", re.I), "credential",
     "password_selector"),
    (re.compile(r"localStorage\s*\.\s*setItem", re.I), "credential",
     "local_storage_write"),
]

GROUPS = ["obfuscation", "exfiltration", "evasion", "redirection", "credential"]

# Longest run of non-whitespace. Minified libraries have long lines too, which
# is exactly why this is reported as a measurement and not as a verdict.
_LONG_LINE = 400


def _entropy(s: str) -> float:
    if not s:
        return 0.0
    counts = Counter(s)
    n = len(s)
    return -sum((c / n) * math.log2(c / n) for c in counts.values())


def analyse_scripts(inline_sources, external_srcs=None) -> dict:
    """Analyse inline script text. External scripts are NOT downloaded.

    inline_sources : list of script bodies already present in the fetched HTML
    external_srcs  : list of src URLs, counted only -- never retrieved, because
                     fetching them would mean pulling further attacker-chosen
                     content, and executing them is never on the table.
    """
    sources = [s for s in (inline_sources or []) if s and s.strip()]
    blob = "\n".join(sources)
    external_srcs = external_srcs or []

    hits = Counter()
    by_group = Counter()
    for pat, group, name in PATTERNS:
        n = len(pat.findall(blob))
        if n:
            hits[name] = n
            by_group[group] += 1

    lines = blob.split("\n")
    longest = max((len(ln) for ln in lines), default=0)
    non_ws = re.sub(r"\s", "", blob)

    feats = {
        "js_inline_scripts": len(sources),
        "js_external_scripts": len(external_srcs),
        "js_total_length": len(blob),
        "js_longest_line": longest,
        "js_has_long_line": int(longest >= _LONG_LINE),
        "js_entropy": round(_entropy(blob[:40000]), 4),
        "js_nonws_ratio": round(len(non_ws) / len(blob), 4) if blob else 0.0,
        "js_distinct_patterns": len(hits),
        "js_total_hits": int(sum(hits.values())),
    }
    for g in GROUPS:
        feats[f"js_{g}_patterns"] = int(by_group.get(g, 0))
    # The combination that actually distinguishes a kit from a minified
    # library: encoding plus exfiltration plus credential access together.
    feats["js_obfuscated_and_exfil"] = int(
        by_group.get("obfuscation", 0) >= 2 and by_group.get("exfiltration", 0) >= 1)
    feats["js_credential_and_exfil"] = int(
        by_group.get("credential", 0) >= 1 and by_group.get("exfiltration", 0) >= 1)

    return {
        "features": feats,
        "hits": dict(hits.most_common()),
        "groups": {g: int(by_group.get(g, 0)) for g in GROUPS},
        "analysed_bytes": len(blob),
        "external_not_fetched": len(external_srcs),
        "method": ("Static pattern matching over inline script text. No script "
                   "was executed and no external script was downloaded."),
        "limitation": ("Packed or encoded code is recognised as obfuscated "
                       "without its behaviour being determined, because "
                       "determining it would require running it."),
    }


JS_FEATURE_NAMES = sorted(analyse_scripts([""])["features"].keys())


SIGNAL_RULES = [
    ("js_obfuscated_and_exfil", lambda v: v > 0,
     "Obfuscated script that also sends data elsewhere",
     "Inline script combines several encoding techniques with a network-send "
     "call. Matched statically; nothing was executed."),
    ("js_credential_and_exfil", lambda v: v > 0,
     "Script reads credential fields and can send data",
     "Inline script references password fields and contains a network-send "
     "call."),
    ("js_evasion_patterns", lambda v: v >= 2,
     "Script contains analysis-evasion patterns",
     "Devtools detection, right-click or copy blocking, or automation checks."),
    ("js_obfuscation_patterns", lambda v: v >= 3,
     "Heavily obfuscated inline script",
     "Several independent encoding or packing techniques in one page."),
    ("js_redirection_patterns", lambda v: v >= 2,
     "Script-driven redirection",
     "Script assigns the page location or schedules navigation."),
    ("js_has_long_line", lambda v: v > 0,
     "Very long single script line",
     "A line over 400 characters, typical of minified or packed code. Common "
     "on legitimate sites too, so weak on its own."),
]


def js_signals(feats: dict):
    out = []
    for key, test, text, evidence in SIGNAL_RULES:
        v = feats.get(key)
        if v is None:
            continue
        try:
            if test(v):
                out.append({"text": text, "evidence": evidence,
                            "feature": key, "value": v})
        except TypeError:
            continue
    return out
