"""
Risk engine: combine URL evidence, website evidence and DNS/reputation into
one assessment, without pretending to know more than the inputs support.

    URL model        -> URL-pattern risk          (always available)
    Website model    -> page-pattern risk         (only if the page was fetched)
    DNS / reputation -> additional evidence       (not held; always unavailable)
                              |
                              v
                         Risk engine -> final score + per-layer breakdown

THE RULE THIS FILE EXISTS TO ENFORCE
------------------------------------
A layer that produced no evidence must not quietly act as evidence. If the page
could not be fetched, the website layer reports `available: false` with the
reason, contributes nothing, and the final score is the URL score alone -- it is
NOT treated as a clean bill of health. "We could not look" and "we looked and
found nothing" are different findings and are rendered differently.

Fusion is a transparent weighted blend, not a third learned model. A learned
stacker would need its own leakage-safe training on pairs where both layers
exist, and the website dataset is too small and too biased by takedowns for
that to be honest. The weights are declared here and shown in the response, so
a reader can see exactly how the number was produced.

Nothing here fetches anything; it consumes results computed elsewhere.
"""

# How much the website layer may move the URL layer's score.
#
# WHY THIS IS AN ADJUSTMENT AND NOT AN AVERAGE
# --------------------------------------------
# The first version averaged the two scores. That is wrong, and measurably so:
# the two models have different decision thresholds (URL 63/100, website
# 95/100), so the same number means different things to each. example.com
# scored 94 on the website model -- BELOW that model's own threshold, i.e. the
# website model's answer was "not malicious" -- and averaging still dragged the
# combined score from 7 up to 42.
#
# So the website layer now moves the URL score by how far it sits from ITS OWN
# threshold. At threshold it contributes nothing; the further above, the more
# it pushes up; below, it pulls down gently.
#
# The push is asymmetric on purpose. Evidence of harm is stronger than absence
# of evidence: a page can look clean because it is clean, or because it is
# JavaScript-rendered, cloaked against scanners, or already taken down. A clean
# page therefore earns a small reduction, never a clean bill of health.
WEB_MAX_PUSH_UP = 35      # points added when the page model is maximally sure
WEB_MAX_PUSH_DOWN = 12    # points removed when it is maximally unconcerned

# Page patterns strong enough to raise the floor on their own.
#
# Each requires a COMBINATION, not a single feature. An earlier version floored
# on brand mismatch alone, which flagged Wikipedia's article about phishing at
# 60/100 -- the article mentions brands, so "brand claimed, not that brand's
# domain" was technically true and completely irrelevant. Impersonation is a
# brand claim PLUS a credential form; either alone is not the pattern.
DECISIVE_PAGE_PATTERNS = [
    {
        "requires": ["password_form_posts_offsite"],
        "text": ("A form collecting a password submits to a different "
                 "registered domain"),
        "floor": 70,
    },
    {
        "requires": ["brand_domain_mismatch", "has_credential_pair"],
        "text": ("The page asks for credentials while presenting itself as a "
                 "brand it is not served from"),
        "floor": 65,
    },
    {
        "requires": ["brand_lookalike_domain", "is_login_page"],
        "text": ("A login page served from a domain containing a brand name "
                 "that brand does not use"),
        "floor": 60,
    },
]


# How much a well-known domain may reduce the URL score.
#
# WHY A REPUTATION LAYER EXISTS AT ALL
# ------------------------------------
# Measured on the hard-benign suite, the URL model alone flagged 43.8% of
# well-known sites (banking 75%, major tech 62%, legitimate logins 50%) while
# posting 2.6% FPR on its own holdout. The gap is structural, not a tuning
# problem: nothing about the STRING "google.com" distinguishes it from
# "g00gle-secure.com". Only knowing which domain is real does, and that is
# reputation.
#
# Bounded, and overridable, because popularity is not safety. This project's
# own index contains 7,609 phishing URLs on google.com domains -- user content,
# shared hosting and compromised pages are real. So:
#   * the reduction is capped,
#   * it shrinks with rank,
#   * and it is CANCELLED outright when page-level evidence fires, because a
#     credential form posting offsite is not excused by the domain being
#     popular.
REPUTATION_MAX_REDUCTION = 45     # top-ranked domain, no contrary evidence
REPUTATION_MIN_RANK_FOR_ANY = 1_000_000


def reputation_reduction(rank):
    """Points removed for domain popularity. 0 when unranked."""
    if not rank or rank > REPUTATION_MIN_RANK_FOR_ANY:
        return 0.0
    import math
    # log-scaled: rank 1 gets the full reduction, rank 1e6 gets none.
    frac = 1.0 - (math.log10(max(rank, 1)) / math.log10(REPUTATION_MIN_RANK_FOR_ANY))
    return REPUTATION_MAX_REDUCTION * max(0.0, min(1.0, frac))


def _band(score):
    if score <= 30:
        return "LOW RISK", "low"
    if score <= 70:
        return "SUSPICIOUS", "suspicious"
    return "HIGH RISK", "high"


def _level(score):
    """Per-layer wording. Deliberately coarse: a layer's contribution does not
    deserve two decimal places."""
    if score is None:
        return "Not available"
    if score <= 30:
        return "Low"
    if score <= 70:
        return "Medium"
    return "High"


def layer_unavailable(reason, code=None):
    return {"available": False, "score": None, "level": "Not available",
            "reason": reason, "code": code}


def combine(url_score, web=None, dns=None, intel=None):
    """Fuse the layers.

    url_score : 0-100 from the URL model (bands are defined on this scale)
    web       : {"available", "score", "threshold_score", "features", ...}
    dns       : layer dict; unavailable by design
    intel     : threat_intel.lookup_all() result, or None
    """
    url_layer = {"available": True, "score": int(url_score),
                 "level": _level(url_score),
                 "basis": "Lexical and structural features of the URL text"}

    web = web or layer_unavailable(
        "The page was not retrieved, so no website-pattern evidence is "
        "available.", "not_requested")
    dns = dns or layer_unavailable(
        "URLGuard holds no DNS resolution, IP reputation or domain-age data.",
        "not_collected")

    final = float(url_score)
    adjustment = 0.0
    contributions = [{"layer": "url", "score": int(url_score),
                      "role": "base score"}]
    method_parts = []

    # ---- website layer ---------------------------------------------------
    feats = (web or {}).get("features") or {}
    if web.get("available") and web.get("score") is not None:
        w_score = float(web["score"])
        w_thr = float(web.get("threshold_score") or 50)
        if w_score >= w_thr:
            margin = (w_score - w_thr) / max(100.0 - w_thr, 1.0)
            adjustment = WEB_MAX_PUSH_UP * margin
        else:
            margin = (w_score - w_thr) / max(w_thr, 1.0)
            adjustment = WEB_MAX_PUSH_DOWN * margin
        final += adjustment
        method_parts.append(
            "website model adjusted by its distance from its own threshold "
            f"({int(w_thr)}/100)")
        contributions.append({
            "layer": "website", "score": int(w_score), "threshold": int(w_thr),
            "role": f"{adjustment:+.1f} points",
            "adjustment": round(adjustment, 1)})
    else:
        method_parts.append(
            "URL model only -- the website layer produced no evidence and is "
            "counted as neither supporting nor clearing")

    # ---- pattern floors --------------------------------------------------
    floors = []
    if web.get("available"):
        for pat in DECISIVE_PAGE_PATTERNS:
            if all(feats.get(k) for k in pat["requires"]):
                floors.append({"features": pat["requires"],
                               "description": pat["text"],
                               "floor": pat["floor"]})

    # ---- reputation layer ------------------------------------------------
    rep_layer = layer_unavailable(
        "No reputation source is configured.", "not_configured")
    rep_reduction = 0.0
    if intel:
        pop = (intel.get("providers") or {}).get("popularity", {})
        listed = bool(intel.get("any_listed"))
        if pop.get("available"):
            rank = pop.get("rank")
            rep_layer = {
                "available": True,
                "ranked": pop.get("verdict") == "ranked",
                "rank": rank, "tier": pop.get("tier"),
                "domain": pop.get("domain"),
                "listed_in_blocklist": listed,
                "source": pop.get("source"),
                "note": pop.get("note"),
            }
            # A domain that is itself listed gets no benefit from being popular.
            if rank and not listed and not floors:
                rep_reduction = reputation_reduction(rank)
                final -= rep_reduction
                method_parts.append(
                    f"reduced {rep_reduction:.0f} points because the domain is "
                    f"ranked #{rank:,} by popularity")
                contributions.append({
                    "layer": "reputation", "rank": rank,
                    "role": f"-{rep_reduction:.1f} points",
                    "adjustment": round(-rep_reduction, 1)})
            elif rank and floors:
                method_parts.append(
                    "popularity reduction withheld: page-level evidence "
                    "overrides domain reputation")
            elif rank and listed:
                method_parts.append(
                    "popularity reduction withheld: the URL is itself listed "
                    "in a blocklist snapshot")
            rep_layer["reduction_applied"] = round(rep_reduction, 1)

    # floors are applied AFTER reputation so reputation cannot undercut them
    if floors:
        highest = max(f["floor"] for f in floors)
        if final < highest:
            final = highest
            method_parts.append(
                f"raised to {highest} by a measured page pattern")

    score = int(round(final))
    score = min(99, max(1, score))
    label, key = _band(score)

    return {
        "risk_score": score,
        "risk_band": {"label": label, "key": key},
        "layers": {"url": url_layer, "website": web, "dns": dns,
                   "reputation": rep_layer},
        "fusion": {
            "method": "; ".join(method_parts),
            "contributions": contributions,
            "pattern_floors": floors,
            "website_adjustment": round(adjustment, 1),
            "reputation_reduction": round(rep_reduction, 1),
            "max_push": {"up": WEB_MAX_PUSH_UP, "down": WEB_MAX_PUSH_DOWN,
                         "reputation": REPUTATION_MAX_REDUCTION},
        },
    }


# ------------------------------------------------------------- page signals
# Each entry maps a measured feature to plain wording. The number shown is
# always the measurement; the wording never asserts intent.
PAGE_SIGNAL_RULES = [
    ("password_form_posts_offsite", lambda v: v > 0,
     "Password form submits to a different domain",
     "A form containing a password field posts to a registered domain other "
     "than the page's own."),
    ("brand_domain_mismatch", lambda v: v > 0,
     "Brand mentioned that the domain does not belong to",
     "The page names a known brand and is not served from a domain that brand "
     "is known to use. On its own this is weak -- any page discussing a brand "
     "matches it -- and it only contributes materially when the page also "
     "collects credentials."),
    ("brand_lookalike_domain", lambda v: v > 0,
     "Brand name in an unrelated domain",
     "A brand name appears inside the domain, which is not one the brand "
     "uses."),
    ("has_credential_pair", lambda v: v > 0,
     "Credential form detected",
     "The page contains both a password field and a username or email field."),
    ("is_login_page", lambda v: v > 0,
     "Login page",
     "The page contains a password field inside a form."),
    ("num_sensitive_fields", lambda v: v > 0,
     "Requests sensitive personal or payment details",
     "Fields referring to card, bank, identity or date-of-birth data."),
    ("any_form_offsite", lambda v: v > 0,
     "Form submits to an external domain",
     "At least one form posts outside the page's own registered domain."),
    ("form_action_insecure", lambda v: v > 0,
     "Form submits over plain HTTP",
     "Form data would be sent unencrypted."),
    ("num_zero_size_iframes", lambda v: v > 0,
     "Zero-size iframe present",
     "An iframe sized 0 or 1 pixel, which a visitor cannot see."),
    ("num_hidden_iframes", lambda v: v > 0,
     "Hidden iframe present",
     "An iframe hidden by inline style attributes."),
    ("obfuscation_signal_count", lambda v: v >= 2,
     "Obfuscated script patterns",
     "Inline script uses several of eval, document.write, decode calls, "
     "character-code construction or long encoded blobs. Counted statically; "
     "nothing was executed."),
    ("urgency_terms", lambda v: v >= 2,
     "Urgency or account-threat wording",
     "Phrases such as account suspension or time-limited verification."),
    ("tls_failed", lambda v: v > 0,
     "TLS certificate did not validate",
     "The page was served over HTTPS but its certificate failed verification."),
    ("mixed_content_resources", lambda v: v > 0,
     "Mixed content",
     "An HTTPS page loads sub-resources over plain HTTP."),
    ("redirect_changed_domain", lambda v: v > 0,
     "Redirected to a different domain",
     "The final page is served from a different registered domain than the "
     "URL requested."),
    ("has_meta_refresh", lambda v: v > 0,
     "Automatic meta-refresh redirect",
     "The page redirects itself without user action."),
    ("hidden_input_ratio", lambda v: v >= 0.5,
     "Mostly hidden form inputs",
     "Half or more of the form inputs are hidden fields."),
]

POSITIVE_PAGE_RULES = [
    ("page_is_https", lambda v: v > 0, "Served over HTTPS"),
    ("tls_verified", lambda v: v > 0, "TLS certificate validated"),
    ("brand_on_official_domain", lambda v: v > 0,
     "Brand claim matches the domain it is served from"),
    ("has_favicon", lambda v: v > 0, "Has a favicon"),
]


def page_signals(features: dict):
    """Plain-language page signals, each backed by a measurement."""
    warn, good = [], []
    for key, test, text, evidence in PAGE_SIGNAL_RULES:
        v = features.get(key)
        if v is None:
            continue
        try:
            hit = bool(test(v))
        except TypeError:
            continue
        if hit:
            warn.append({"text": text, "evidence": evidence,
                         "feature": key, "value": v})
    for key, test, text in POSITIVE_PAGE_RULES:
        v = features.get(key)
        if v is not None and test(v):
            good.append({"text": text, "feature": key, "value": v})
    return {"warning": warn, "positive": good}


def breakdown_rows(result, page_feats=None):
    """The compact 'Detection breakdown' the UI renders."""
    L = result["layers"]
    feats = page_feats or {}
    rows = [
        {"label": "URL pattern", "value": L["url"]["level"],
         "available": True, "score": L["url"]["score"]},
        {"label": "Website structure",
         "value": L["website"]["level"],
         "available": bool(L["website"].get("available")),
         "score": L["website"].get("score"),
         "reason": L["website"].get("reason")},
    ]
    if L["website"].get("available"):
        rows += [
            {"label": "Credential form",
             "value": "Detected" if feats.get("has_credential_pair")
                      else "Not detected", "available": True},
            {"label": "Form destination",
             "value": ("Suspicious" if feats.get("password_form_posts_offsite")
                       else "External" if feats.get("any_form_offsite")
                       else "Same domain" if feats.get("num_forms")
                       else "No forms"),
             "available": True},
            {"label": "Brand/domain match",
             "value": ("Mismatch" if feats.get("brand_domain_mismatch")
                       else "Match" if feats.get("brand_on_official_domain")
                       else "No brand claimed"),
             "available": True},
        ]
    else:
        for lab in ("Credential form", "Form destination", "Brand/domain match"):
            rows.append({"label": lab, "value": "Not available",
                         "available": False,
                         "reason": L["website"].get("reason")})
    rep = L.get("reputation", {})
    if rep.get("available"):
        rows.append({
            "label": "Domain reputation",
            "value": (f"Ranked #{rep['rank']:,}" if rep.get("rank")
                      else "Not in the popular-domain list"),
            "available": True})
    else:
        rows.append({"label": "Domain reputation", "value": "Not available",
                     "available": False, "reason": rep.get("reason")})
    rows.append({"label": "DNS/WHOIS/certificates", "value": "Not available",
                 "available": False, "reason": L["dns"].get("reason")})
    return rows
