"""
Website / DOM feature extraction for the page-pattern model.

WHAT THIS DOES AND DOES NOT DO
------------------------------
It parses already-downloaded HTML as a static document. It does not run
JavaScript, does not resolve or request any sub-resource, does not submit
anything and does not follow any link. A `<script>` is counted and its `src`
host is compared with the page host -- the script is never fetched or executed.

Features are grouped so the frontend can explain itself:

  structure   forms, inputs, scripts, iframes, links, DOM size and depth
  credential  password fields, login wording, autocomplete hints
  destination where forms post to, relative to the page's own domain
  resources   how much of the page comes from somewhere else
  concealment hidden inputs, zero-size iframes, off-screen elements
  identity    title, brand mentions, brand/domain agreement
  transport   scheme, mixed content, redirect count, TLS verification

EVERY feature is a measurement, not a verdict. "This form posts to a different
registered domain" is a fact; whether that is malicious is the model's problem,
and even the model's answer is an assessment rather than proof.

Determinism: the same HTML always yields the same vector, so Python and any
future port can be checked against each other the way the URL features already
are.
"""

import math
import re
from collections import Counter
from urllib.parse import urljoin, urlsplit

from bs4 import BeautifulSoup

from features import TARGET_BRANDS, SUSPICIOUS_WORDS, shannon_entropy
from leakage import registered_domain

# Wording that indicates a page is asking for credentials. Matched against
# visible text, titles, button labels and field names.
LOGIN_TERMS = [
    "log in", "login", "sign in", "signin", "log on", "logon",
    "username", "user name", "password", "passcode", "credentials",
    "verify your account", "confirm your identity", "account verification",
    "two-factor", "one-time", "otp", "security code", "pin",
]

URGENCY_TERMS = [
    "suspended", "unusual activity", "verify now", "immediately",
    "within 24 hours", "account locked", "limited access", "confirm now",
    "failure to", "will be closed", "unauthorized", "restricted",
]

SENSITIVE_FIELD_HINTS = [
    "card", "cvv", "cvc", "expiry", "ssn", "social security", "sort code",
    "account number", "routing", "iban", "pin", "dob", "date of birth",
    "mother", "passport",
]

# Official registered domains for the brands the URL model already tracks.
#
# Needed because "is the brand name in the domain?" is the WRONG test. A page
# titled "PayPal - Log In" served from paypal-secure-verify.tk contains the
# brand in its domain, so a substring check calls it a match -- when it is the
# textbook impersonation case. Comparing against the brand's real eTLD+1
# answers the question that was actually being asked.
#
# Static, curated domain knowledge. It is intentionally small and intentionally
# incomplete: a brand absent from this map simply produces no mismatch signal,
# which under-reports rather than inventing one.
BRAND_DOMAINS = {
    "paypal": {"paypal.com", "paypal.me"},
    "apple": {"apple.com", "icloud.com"},
    "icloud": {"icloud.com", "apple.com"},
    "microsoft": {"microsoft.com", "live.com", "office.com", "office365.com",
                  "microsoftonline.com"},
    "office365": {"office.com", "office365.com", "microsoft.com"},
    "outlook": {"outlook.com", "live.com", "microsoft.com"},
    "google": {"google.com", "google.co.uk", "gmail.com", "youtube.com"},
    "amazon": {"amazon.com", "amazon.co.uk", "amazon.in", "amazon.de"},
    "facebook": {"facebook.com", "fb.com", "meta.com"},
    "instagram": {"instagram.com", "facebook.com"},
    "whatsapp": {"whatsapp.com", "facebook.com"},
    "netflix": {"netflix.com"},
    "chase": {"chase.com", "jpmorganchase.com"},
    "wellsfargo": {"wellsfargo.com"},
    "bankofamerica": {"bankofamerica.com", "bofa.com"},
    "ebay": {"ebay.com", "ebay.co.uk"},
    "dropbox": {"dropbox.com"},
    "linkedin": {"linkedin.com"},
    "steam": {"steampowered.com", "steamcommunity.com", "valvesoftware.com"},
    "irs": {"irs.gov"},
}

# Patterns that often accompany obfuscated or generated pages. Counted, never
# evaluated.
OBFUSCATION_PATTERNS = [
    (re.compile(r"\beval\s*\(", re.I), "eval"),
    (re.compile(r"document\s*\.\s*write\s*\(", re.I), "document_write"),
    (re.compile(r"\b(unescape|atob)\s*\(", re.I), "decode_call"),
    (re.compile(r"fromCharCode", re.I), "fromcharcode"),
    (re.compile(r"[A-Za-z0-9+/]{200,}={0,2}"), "long_base64_blob"),
    (re.compile(r"\\x[0-9a-f]{2}(?:\\x[0-9a-f]{2}){10,}", re.I), "hex_escape_run"),
]

_WS = re.compile(r"\s+")


def _host_of(url: str) -> str:
    try:
        return (urlsplit(url).hostname or "").lower()
    except ValueError:
        return ""


def _safe_join(base: str, ref: str) -> str:
    """urljoin also raises on malformed bracketed hosts."""
    try:
        return urljoin(base, ref)
    except ValueError:
        return ref


def _ratio(n, d):
    return (n / d) if d else 0.0


def _dom_depth(node, depth=0, cap=60):
    """Maximum nesting depth, bounded so a pathological page cannot hang this."""
    if depth >= cap:
        return depth
    best = depth
    for child in getattr(node, "children", []):
        if getattr(child, "name", None):
            d = _dom_depth(child, depth + 1, cap)
            if d > best:
                best = d
    return best


def _is_hidden(tag) -> bool:
    """Static check for elements the page is trying not to show.

    Only inline style/attributes are considered: stylesheets are never fetched,
    so a class-based rule is invisible here. That under-counts rather than
    guesses, which is the right direction.
    """
    if tag.has_attr("hidden"):
        return True
    style = (tag.get("style") or "").lower().replace(" ", "")
    if any(p in style for p in ("display:none", "visibility:hidden",
                                "opacity:0", "height:0", "width:0")):
        return True
    for dim in ("width", "height"):
        v = (tag.get(dim) or "").strip()
        if v in ("0", "0px", "1"):
            return True
    return False


def extract_web_features(html: str, page_url: str, fetch_meta=None) -> dict:
    """Return the website feature vector for one page.

    `fetch_meta` is the dict from web_fetch.fetch(); when absent the transport
    features are reported as 0 rather than guessed.
    """
    meta = fetch_meta or {}
    soup = BeautifulSoup(html or "", "lxml")
    page_host = _host_of(page_url)
    page_domain = registered_domain(page_url)
    # urlsplit raises on malformed bracketed hosts ("http://["). Feature
    # extraction runs on attacker-controlled input and must never raise -- the
    # URL extractor has the same rule for the same reason.
    try:
        scheme = (urlsplit(page_url).scheme or "").lower() if "://" in page_url else ""
    except ValueError:
        scheme = page_url.split("://", 1)[0].lower() if "://" in page_url else ""

    text = _WS.sub(" ", soup.get_text(" ", strip=True)).lower()
    html_l = (html or "").lower()
    title_tag = soup.title.string if soup.title and soup.title.string else ""
    title = _WS.sub(" ", str(title_tag)).strip()
    title_l = title.lower()

    f = {}

    # ---------------------------------------------------------- structure
    forms = soup.find_all("form")
    inputs = soup.find_all("input")
    scripts = soup.find_all("script")
    iframes = soup.find_all("iframe")
    links = soup.find_all("a", href=True)
    images = soup.find_all("img")
    all_tags = soup.find_all(True)

    f["html_length"] = len(html or "")
    f["dom_node_count"] = len(all_tags)
    f["dom_max_depth"] = _dom_depth(soup)
    f["distinct_tag_types"] = len({t.name for t in all_tags})
    f["text_length"] = len(text)
    f["text_to_html_ratio"] = _ratio(len(text), len(html or ""))
    f["num_forms"] = len(forms)
    f["num_inputs"] = len(inputs)
    f["num_scripts"] = len(scripts)
    f["num_iframes"] = len(iframes)
    f["num_links"] = len(links)
    f["num_images"] = len(images)
    f["num_buttons"] = len(soup.find_all(["button"])) + \
        len([i for i in inputs if (i.get("type") or "").lower() in ("submit", "button")])
    f["num_meta_tags"] = len(soup.find_all("meta"))
    f["has_doctype"] = int(bool(re.match(r"\s*<!doctype", html_l)))
    f["has_head"] = int(soup.head is not None)
    f["has_body"] = int(soup.body is not None)

    # --------------------------------------------------------- credential
    pw_inputs = [i for i in inputs if (i.get("type") or "").lower() == "password"]
    email_inputs = [i for i in inputs
                    if (i.get("type") or "").lower() == "email"]
    hidden_inputs = [i for i in inputs if (i.get("type") or "").lower() == "hidden"]
    text_inputs = [i for i in inputs
                   if (i.get("type") or "text").lower() in ("text", "")]

    def _field_blob(i):
        return " ".join(str(i.get(k) or "") for k in
                        ("name", "id", "placeholder", "aria-label", "autocomplete")).lower()

    user_like = [i for i in inputs if any(
        w in _field_blob(i) for w in ("user", "email", "login", "account", "signin"))]
    sensitive = [i for i in inputs if any(
        w in _field_blob(i) for w in SENSITIVE_FIELD_HINTS)]

    f["num_password_fields"] = len(pw_inputs)
    f["num_email_fields"] = len(email_inputs)
    f["num_hidden_inputs"] = len(hidden_inputs)
    f["num_text_inputs"] = len(text_inputs)
    f["num_username_like_fields"] = len(user_like)
    f["num_sensitive_fields"] = len(sensitive)
    f["hidden_input_ratio"] = _ratio(len(hidden_inputs), len(inputs))
    f["has_credential_pair"] = int(bool(pw_inputs) and bool(user_like or email_inputs))
    f["login_terms_in_text"] = sum(1 for t in LOGIN_TERMS if t in text)
    f["login_terms_in_title"] = sum(1 for t in LOGIN_TERMS if t in title_l)
    f["urgency_terms"] = sum(1 for t in URGENCY_TERMS if t in text)
    f["suspicious_words_in_text"] = sum(1 for w in SUSPICIOUS_WORDS if w in text)
    f["autocomplete_off"] = int("autocomplete=\"off\"" in html_l
                                or "autocomplete='off'" in html_l)
    # A login page with no password field at all is unusual; so is a page with
    # several independent password fields.
    f["is_login_page"] = int(bool(pw_inputs) and bool(forms))
    f["multiple_password_fields"] = int(len(pw_inputs) > 1)

    # -------------------------------------------------------- destination
    ext_actions = 0
    rel_actions = 0
    empty_actions = 0
    js_actions = 0
    insecure_actions = 0
    action_domains = set()
    pw_form_external = 0
    for form in forms:
        action = (form.get("action") or "").strip()
        has_pw = bool(form.find_all("input", attrs={"type": re.compile("^password$", re.I)}))
        if not action or action in ("#", "."):
            empty_actions += 1
            continue
        if action.lower().startswith("javascript:"):
            js_actions += 1
            continue
        absolute = _safe_join(page_url, action)
        if absolute.lower().startswith("http://"):
            insecure_actions += 1
        a_dom = registered_domain(absolute)
        if a_dom:
            action_domains.add(a_dom)
        if "://" not in action and not action.startswith("//"):
            rel_actions += 1
        elif a_dom and page_domain and a_dom != page_domain:
            ext_actions += 1
            if has_pw:
                pw_form_external += 1

    f["form_action_external"] = ext_actions
    f["form_action_relative"] = rel_actions
    f["form_action_empty"] = empty_actions
    f["form_action_javascript"] = js_actions
    f["form_action_insecure"] = insecure_actions
    f["form_action_domains"] = len(action_domains)
    # The strongest single page-level pattern in credential phishing: a form
    # that collects a password and posts it somewhere else entirely.
    f["password_form_posts_offsite"] = pw_form_external
    f["any_form_offsite"] = int(ext_actions > 0)

    # ---------------------------------------------------------- resources
    def _res_hosts(tags, attr):
        hosts = []
        for t in tags:
            v = (t.get(attr) or "").strip()
            if not v or v.startswith(("data:", "javascript:", "#", "about:")):
                continue
            hosts.append(_host_of(_safe_join(page_url, v)))
        return [h for h in hosts if h]

    script_hosts = _res_hosts(scripts, "src")
    img_hosts = _res_hosts(images, "src")
    link_hosts = _res_hosts(links, "href")
    css_hosts = _res_hosts(soup.find_all("link", href=True), "href")

    def _external(hosts):
        return [h for h in hosts
                if page_domain and registered_domain("http://" + h) != page_domain]

    ext_scripts = _external(script_hosts)
    ext_links = _external(link_hosts)
    ext_imgs = _external(img_hosts)
    ext_css = _external(css_hosts)

    f["num_external_scripts"] = len(ext_scripts)
    f["num_inline_scripts"] = len([s for s in scripts if not s.get("src")])
    f["external_script_ratio"] = _ratio(len(ext_scripts), max(len(script_hosts), 1))
    f["num_external_links"] = len(ext_links)
    f["external_link_ratio"] = _ratio(len(ext_links), max(len(link_hosts), 1))
    f["num_external_images"] = len(ext_imgs)
    f["num_external_css"] = len(ext_css)
    f["distinct_external_domains"] = len({
        registered_domain("http://" + h)
        for h in ext_scripts + ext_links + ext_imgs + ext_css})
    f["num_internal_links"] = len(link_hosts) - len(ext_links)
    f["links_to_self_ratio"] = _ratio(len(link_hosts) - len(ext_links),
                                      max(len(link_hosts), 1))
    f["num_anchor_empty"] = len([a for a in links
                                 if (a.get("href") or "").strip() in ("#", "", "javascript:void(0)")])
    f["empty_anchor_ratio"] = _ratio(f["num_anchor_empty"], max(len(links), 1))

    inline_js = " ".join(s.get_text() or "" for s in scripts if not s.get("src"))
    f["inline_script_length"] = len(inline_js)
    f["inline_script_entropy"] = shannon_entropy(inline_js[:20000])
    obf = Counter()
    for pat, name in OBFUSCATION_PATTERNS:
        n = len(pat.findall(inline_js)) + len(pat.findall(html or ""))
        if n:
            obf[name] = n
    f["obfuscation_signal_count"] = len(obf)
    f["obfuscation_hit_total"] = int(sum(obf.values()))

    # -------------------------------------------------------- concealment
    hidden_tags = [t for t in all_tags if _is_hidden(t)]
    f["num_hidden_elements"] = len(hidden_tags)
    f["hidden_element_ratio"] = _ratio(len(hidden_tags), max(len(all_tags), 1))
    f["num_hidden_iframes"] = len([t for t in iframes if _is_hidden(t)])
    f["num_zero_size_iframes"] = len([
        t for t in iframes
        if (t.get("width") or "").strip() in ("0", "1")
        or (t.get("height") or "").strip() in ("0", "1")])
    f["has_meta_refresh"] = int(bool(soup.find(
        "meta", attrs={"http-equiv": re.compile("^refresh$", re.I)})))
    f["num_noscript"] = len(soup.find_all("noscript"))

    # ------------------------------------------------------------ identity
    f["has_title"] = int(bool(title))
    f["title_length"] = len(title)
    f["title_word_count"] = len(title.split())
    f["has_favicon"] = int(bool(soup.find(
        "link", rel=lambda v: v and "icon" in " ".join(v).lower())))

    brands_in_title = [b for b in TARGET_BRANDS if b in title_l]
    brands_in_text = [b for b in TARGET_BRANDS if b in text]
    dom_l = (page_domain or "").lower()
    brand_substring_in_domain = any(b in dom_l for b in TARGET_BRANDS)

    def _is_official(brands):
        """Is the page served from a domain this brand actually owns?

        Substring matching is explicitly NOT used here: paypal-secure-verify.tk
        contains "paypal" and is not PayPal.
        """
        for b in brands:
            if dom_l in BRAND_DOMAINS.get(b, set()):
                return True
        return False

    claimed = brands_in_title or brands_in_text
    on_official = _is_official(claimed)
    known = [b for b in claimed if b in BRAND_DOMAINS]

    f["brand_in_title"] = len(brands_in_title)
    f["brand_in_text"] = len(brands_in_text)
    f["brand_in_page_domain"] = int(brand_substring_in_domain)
    f["brand_on_official_domain"] = int(on_official)
    # Impersonation: the page claims a brand whose real domains are known, and
    # it is not being served from any of them.
    f["brand_domain_mismatch"] = int(bool(known) and not on_official)
    f["brand_mismatch_in_title"] = int(
        bool([b for b in brands_in_title if b in BRAND_DOMAINS]) and not on_official)
    # Brand name embedded in a domain that is not the brand's: the lookalike
    # pattern, distinct from simply mentioning a brand in the text.
    f["brand_lookalike_domain"] = int(brand_substring_in_domain and not on_official)
    f["title_mentions_page_domain"] = int(
        bool(page_domain) and page_domain.split(".")[0] in title_l)
    f["num_copyright_mentions"] = text.count("©") + text.count("copyright")

    # ----------------------------------------------------------- transport
    f["page_is_https"] = int(scheme == "https")
    f["redirect_count"] = int(meta.get("redirect_count", 0) or 0)
    f["redirect_changed_domain"] = int(bool(
        meta.get("final_url") and meta.get("requested_url")
        and registered_domain(meta["final_url"])
        != registered_domain(meta["requested_url"])))
    f["tls_verified"] = int(bool(meta.get("tls_verified")))
    f["tls_failed"] = int(meta.get("tls_verified") is False)
    f["http_status"] = int(meta.get("status", 0) or 0)
    f["response_bytes"] = int(meta.get("bytes", 0) or 0)
    f["response_truncated"] = int(bool(meta.get("truncated")))
    # Mixed content: an https page pulling http sub-resources.
    http_resources = len([u for u in
                          [(t.get("src") or t.get("href") or "") for t in
                           scripts + images + soup.find_all("link", href=True)]
                          if u.lower().startswith("http://")])
    f["mixed_content_resources"] = http_resources if scheme == "https" else 0

    # Guard: every value must be a finite number, because this vector feeds a
    # model and a NaN would propagate silently.
    for k, v in list(f.items()):
        if isinstance(v, bool):
            f[k] = int(v)
        elif not isinstance(v, (int, float)) or not math.isfinite(v):
            f[k] = 0.0
    return f


WEB_FEATURE_NAMES = list(extract_web_features(
    "<html><head><title>x</title></head><body></body></html>",
    "http://example.com/").keys())

FEATURE_GROUPS = {
    "structure": ["html_length", "dom_node_count", "dom_max_depth",
                  "distinct_tag_types", "text_length", "text_to_html_ratio",
                  "num_forms", "num_inputs", "num_scripts", "num_iframes",
                  "num_links", "num_images", "num_buttons", "num_meta_tags",
                  "has_doctype", "has_head", "has_body"],
    "credential": ["num_password_fields", "num_email_fields",
                   "num_hidden_inputs", "num_text_inputs",
                   "num_username_like_fields", "num_sensitive_fields",
                   "hidden_input_ratio", "has_credential_pair",
                   "login_terms_in_text", "login_terms_in_title",
                   "urgency_terms", "suspicious_words_in_text",
                   "autocomplete_off", "is_login_page",
                   "multiple_password_fields"],
    "destination": ["form_action_external", "form_action_relative",
                    "form_action_empty", "form_action_javascript",
                    "form_action_insecure", "form_action_domains",
                    "password_form_posts_offsite", "any_form_offsite"],
    "resources": ["num_external_scripts", "num_inline_scripts",
                  "external_script_ratio", "num_external_links",
                  "external_link_ratio", "num_external_images",
                  "num_external_css", "distinct_external_domains",
                  "num_internal_links", "links_to_self_ratio",
                  "num_anchor_empty", "empty_anchor_ratio",
                  "inline_script_length", "inline_script_entropy",
                  "obfuscation_signal_count", "obfuscation_hit_total"],
    "concealment": ["num_hidden_elements", "hidden_element_ratio",
                    "num_hidden_iframes", "num_zero_size_iframes",
                    "has_meta_refresh", "num_noscript"],
    "identity": ["has_title", "title_length", "title_word_count",
                 "has_favicon", "brand_in_title", "brand_in_text",
                 "brand_in_page_domain", "brand_on_official_domain",
                 "brand_domain_mismatch", "brand_mismatch_in_title",
                 "brand_lookalike_domain", "title_mentions_page_domain",
                 "num_copyright_mentions"],
    "transport": ["page_is_https", "redirect_count", "redirect_changed_domain",
                  "tls_verified", "tls_failed", "http_status",
                  "response_bytes", "response_truncated",
                  "mixed_content_resources"],
}
