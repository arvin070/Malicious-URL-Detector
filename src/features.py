"""
Feature extraction for phishing / malicious URL detection.
Purely lexical/structural features computed from the URL string alone --
no page fetch, no network call, safe to run on unverified URLs.

INTERFACE (unchanged, relied on by train.py / monitor.py / webapp / tests):
    extract_features(url: str) -> dict
    FEATURE_NAMES: list[str]
    shannon_entropy(s), has_ip_address(host), count_suspicious_words(url)

The ORIGINAL 21 features keep their exact names, values and ordering so that
old model bundles stay loadable (webapp/app.py selects columns by the
`feature_names` stored in the bundle, so a bundle trained on 21 features still
works against this richer extractor). New features are appended after them.
"""

import re
import math
import unicodedata
import zlib
from collections import Counter
from urllib.parse import urlparse
import tldextract

_extract = tldextract.TLDExtract(suffix_list_urls=())  # fully offline

SUSPICIOUS_WORDS = [
    "login", "verify", "update", "secure", "account", "banking",
    "confirm", "signin", "webscr", "ebayisapi", "paypal", "free",
    "bonus", "gift", "urgent", "suspend", "password", "wallet",
]

SHORTENER_DOMAINS = {
    "bit.ly", "tinyurl.com", "goo.gl", "t.co", "ow.ly", "is.gd",
    "buff.ly", "adf.ly", "shorte.st", "rebrand.ly",
}

# Brands most often impersonated -- used only for "brand appears somewhere it
# shouldn't" signals, never to whitelist anything.
TARGET_BRANDS = [
    "paypal", "apple", "microsoft", "google", "amazon", "facebook", "netflix",
    "chase", "wellsfargo", "bankofamerica", "ebay", "dropbox", "linkedin",
    "instagram", "whatsapp", "outlook", "office365", "icloud", "steam", "irs",
]

# Ports that are legitimate on the open web. Anything else is unusual.
COMMON_PORTS = {"", "80", "443", "8080", "8443"}

# TLDs that are free or near-free to register, or that registry operators and
# abuse reports consistently rank among the most abused. This is a STATIC list
# of documented domain knowledge, deliberately NOT fitted from our own labels:
# a target-encoded "phishing rate per TLD" would leak the training labels into
# a feature and inflate every metric.
ABUSE_PRONE_TLDS = {
    # Freenom's free TLDs, historically the single largest phishing source
    "tk", "ml", "ga", "cf", "gq",
    # cheap gTLDs repeatedly topping registry abuse reports
    "xyz", "top", "work", "click", "link", "buzz", "icu", "cyou", "rest",
    "quest", "monster", "bar", "best", "cam", "surf", "fit", "beauty",
    "sbs", "cfd", "lol", "makeup", "hair", "skin", "mom", "store",
    # TLDs that collide with file extensions, usable to disguise a link
    "zip", "mov",
}

# Extensions associated with executable payload delivery rather than pages.
# Relevant because URLhaus (malware distribution) is now a training source.
_EXECUTABLE_EXT = re.compile(
    r"\.(exe|apk|scr|bat|cmd|com|pif|msi|jar|vbs|js|ps1|hta|dll|"
    r"zip|rar|7z|gz|iso|img|bin|sh|elf)$", re.IGNORECASE)

_VOWELS = set("aeiou")
_HEX = re.compile(r"%[0-9a-fA-F]{2}")
_TOKEN_SPLIT = re.compile(r"[^a-zA-Z0-9]+")
_SCHEME_RE = re.compile(r"^[a-zA-Z][a-zA-Z0-9+.\-]*://")

# ---- hashed n-gram configuration -------------------------------------------
# A full char-n-gram vocabulary would be ~10^5 columns. Hashing into a small
# fixed number of buckets keeps the block O(len(url)) per URL and keeps the
# matrix dense and tree-friendly, which is what makes n-grams practical here.
#
# zlib.crc32 is used instead of Python's hash(): hash() is randomised per
# interpreter run (PYTHONHASHSEED), which would make features -- and therefore
# predictions -- differ between processes. crc32 is stable forever, so a saved
# model keeps meaning the same thing.
CHAR_NGRAM_N = 3
CHAR_NGRAM_BUCKETS = 24
TOKEN_NGRAM_BUCKETS = 12


def _bucket(s: str, mod: int) -> int:
    return zlib.crc32(s.encode("utf-8", "replace")) % mod


def _hashed_char_ngrams(s: str, n: int = CHAR_NGRAM_N,
                        buckets: int = CHAR_NGRAM_BUCKETS):
    """Relative frequency of hashed character n-grams.

    Normalised by the n-gram count so the block encodes *composition* rather
    than length -- url_length already carries length, and leaving these as raw
    counts would just add 24 more copies of it.
    """
    counts = [0] * buckets
    total = max(len(s) - n + 1, 0)
    for i in range(total):
        counts[_bucket(s[i:i + n], buckets)] += 1
    if not total:
        return counts  # all zeros
    return [c / total for c in counts]


def _hashed_token_bigrams(tokens, buckets: int = TOKEN_NGRAM_BUCKETS):
    """Relative frequency of hashed adjacent-token bigrams ("secure|login")."""
    counts = [0] * buckets
    total = max(len(tokens) - 1, 0)
    for a, b in zip(tokens, tokens[1:]):
        counts[_bucket(a + "|" + b, buckets)] += 1
    if not total:
        return counts
    return [c / total for c in counts]


def strip_scheme(u: str) -> str:
    """Remove a leading scheme. Shared with the n-gram block so that adding
    'https://' to a URL cannot change its n-gram profile."""
    return _SCHEME_RE.sub("", str(u))


def shannon_entropy(s: str) -> float:
    if not s:
        return 0.0
    probs = [s.count(c) / len(s) for c in set(s)]
    return -sum(p * math.log2(p) for p in probs)


def has_ip_address(host: str) -> int:
    ipv4 = r"^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$"
    return int(bool(re.match(ipv4, host or "")))


def count_suspicious_words(url: str) -> int:
    url_l = url.lower()
    return sum(1 for w in SUSPICIOUS_WORDS if w in url_l)


# ---------------------------------------------------------------- helpers
def _ratio(n, d):
    return (n / d) if d else 0.0


def _max_run(s: str, charset) -> int:
    best = cur = 0
    for ch in s:
        cur = cur + 1 if ch in charset else 0
        best = max(best, cur)
    return best


def _tokens(s: str):
    return [t for t in _TOKEN_SPLIT.split(s) if t]


def _is_confusable(s: str) -> int:
    """Non-ASCII characters that render like Latin letters (homograph risk).

    Pure string inspection via unicodedata -- no network, no IDNA resolution.
    """
    for ch in s:
        if ord(ch) < 128:
            continue
        try:
            name = unicodedata.name(ch)
        except ValueError:
            return 1
        # e.g. CYRILLIC SMALL LETTER A, GREEK SMALL LETTER OMICRON
        if any(scr in name for scr in ("CYRILLIC", "GREEK", "ARMENIAN")) and \
                ("LETTER" in name):
            return 1
    return 0


class _NullParse:
    """Fallback when urlparse refuses a malformed URL."""
    __slots__ = ("scheme", "netloc", "path", "query", "fragment")

    def __init__(self, raw):
        self.scheme = raw.split("://", 1)[0].lower() if "://" in raw else ""
        rest = raw.split("://", 1)[1] if "://" in raw else raw
        rest, _, self.fragment = rest.partition("#")
        rest, _, self.query = rest.partition("?")
        self.netloc, _, path = rest.partition("/")
        self.path = "/" + path if path else ""


def _safe_urlparse(raw: str):
    """urlparse() raises on some malformed inputs (e.g. bad bracketed IPv6
    netlocs). Feature extraction must never raise: it runs on untrusted,
    attacker-controlled strings coming straight from the API and the browser
    extension, so a crash would be a denial-of-service vector."""
    try:
        return urlparse(raw)
    except ValueError:
        return _NullParse(raw)


def extract_features(url: str) -> dict:
    url = url.strip()
    parsed = _safe_urlparse(url if "://" in url else "http://" + url)
    netloc = parsed.netloc
    host = netloc.split(":")[0]
    port = netloc.split(":")[1] if ":" in netloc else ""
    ext = _extract(url)
    registered_domain = ".".join(p for p in [ext.domain, ext.suffix] if p)

    path = parsed.path or ""
    query = parsed.query or ""
    fragment = parsed.fragment or ""
    url_l = url.lower()
    host_l = host.lower()

    # ---- ORIGINAL 21 FEATURES (names, order and semantics unchanged) ----
    feats = {
        "url_length": len(url),
        "host_length": len(host),
        "path_length": len(path),
        "num_dots": url.count("."),
        "num_hyphens": url.count("-"),
        "num_underscores": url.count("_"),
        "num_slashes": url.count("/"),
        "num_digits": sum(c.isdigit() for c in url),
        "num_special_chars": len(re.findall(r"[^a-zA-Z0-9./:-]", url)),
        "num_subdomains": max(len(ext.subdomain.split(".")), 0) if ext.subdomain else 0,
        "has_ip": has_ip_address(host),
        "has_at_symbol": int("@" in url),
        "has_double_slash_redirect": int("//" in path),
        "is_https": int(parsed.scheme == "https"),
        "is_shortened": int(registered_domain in SHORTENER_DOMAINS),
        "suspicious_word_count": count_suspicious_words(url),
        "entropy": shannon_entropy(host),
        "query_length": len(query),
        "num_params": query.count("=") if query else 0,
        "tld_length": len(ext.suffix) if ext.suffix else 0,
        "digit_ratio_in_host": (sum(c.isdigit() for c in host) / len(host)) if host else 0,
    }

    # ================= NEW FEATURES (appended) =================

    # ---- component lengths -------------------------------------------
    feats["fragment_length"] = len(fragment)
    feats["domain_length"] = len(ext.domain or "")
    feats["subdomain_length"] = len(ext.subdomain or "")
    feats["netloc_length"] = len(netloc)

    # ---- per-component separator / special-char counts ----------------
    feats["host_num_dots"] = host.count(".")
    feats["host_num_hyphens"] = host.count("-")
    feats["host_num_digits"] = sum(c.isdigit() for c in host)
    feats["path_num_dots"] = path.count(".")
    feats["path_num_hyphens"] = path.count("-")
    feats["path_num_digits"] = sum(c.isdigit() for c in path)
    feats["domain_num_hyphens"] = (ext.domain or "").count("-")
    feats["query_num_special"] = len(re.findall(r"[^a-zA-Z0-9=&_.\-]", query))

    # ---- query / path structure ---------------------------------------
    feats["path_depth"] = len([s for s in path.split("/") if s])
    feats["num_query_params"] = len([p for p in query.split("&") if p]) if query else 0
    feats["max_param_value_length"] = max(
        (len(p.split("=", 1)[1]) for p in query.split("&") if "=" in p), default=0)
    feats["has_fragment"] = int(bool(fragment))
    feats["path_has_file_ext"] = int(bool(re.search(r"\.[a-zA-Z0-9]{2,5}$", path)))

    # ---- encoding / obfuscation ---------------------------------------
    feats["num_percent_encoded"] = len(_HEX.findall(url))
    feats["num_percent_signs"] = url.count("%")
    # NB: computed on the URL with the scheme removed. Counting the "//" of
    # "http://" would make this feature a proxy for "was a scheme supplied",
    # which is a property of the data source, not of the URL.
    _no_scheme = strip_scheme(url)
    feats["repeated_separators"] = len(
        re.findall(r"(//|\.\.|--|__|&&|\?\?)", _no_scheme))
    feats["num_equals"] = url.count("=")
    feats["num_ampersands"] = url.count("&")
    feats["num_question_marks"] = url.count("?")
    feats["has_hex_like_segment"] = int(bool(re.search(r"\b[0-9a-f]{16,}\b", url_l)))
    feats["has_base64_like_segment"] = int(
        bool(re.search(r"[A-Za-z0-9+/]{24,}={0,2}", url)))

    # ---- port ----------------------------------------------------------
    feats["has_port"] = int(bool(port))
    feats["has_suspicious_port"] = int(bool(port) and port not in COMMON_PORTS)

    # ---- internationalisation / homograph -------------------------------
    feats["is_punycode"] = int("xn--" in host_l)
    feats["has_non_ascii"] = int(any(ord(c) > 127 for c in url))
    feats["has_confusable_chars"] = _is_confusable(url)

    # ---- entropy, per component -----------------------------------------
    feats["host_entropy"] = shannon_entropy(host)
    feats["path_entropy"] = shannon_entropy(path)
    feats["query_entropy"] = shannon_entropy(query)
    feats["domain_entropy"] = shannon_entropy(ext.domain or "")
    feats["url_entropy"] = shannon_entropy(url)

    # ---- character-class ratios ------------------------------------------
    letters_url = sum(c.isalpha() for c in url)
    feats["digit_ratio_in_url"] = _ratio(feats["num_digits"], len(url))
    feats["letter_ratio_in_url"] = _ratio(letters_url, len(url))
    feats["letter_ratio_in_host"] = _ratio(sum(c.isalpha() for c in host), len(host))
    feats["digit_ratio_in_path"] = _ratio(feats["path_num_digits"], len(path))
    feats["special_ratio_in_url"] = _ratio(feats["num_special_chars"], len(url))
    feats["path_ratio_of_url"] = _ratio(len(path), len(url))
    feats["query_ratio_of_url"] = _ratio(len(query), len(url))
    feats["host_ratio_of_url"] = _ratio(len(host), len(url))

    # ---- domain / subdomain structure -------------------------------------
    sub_parts = [p for p in (ext.subdomain or "").split(".") if p]
    feats["subdomain_depth"] = len(sub_parts)
    feats["longest_subdomain_length"] = max((len(p) for p in sub_parts), default=0)
    feats["has_www"] = int(bool(sub_parts) and sub_parts[0].lower() == "www")
    feats["tld_in_subdomain"] = int(bool(ext.suffix) and ext.suffix in (ext.subdomain or ""))
    feats["tld_in_path"] = int(bool(ext.suffix) and ("." + ext.suffix) in path.lower())
    feats["domain_is_digits_only"] = int(bool(ext.domain) and (ext.domain or "").isdigit())

    # ---- brand-abuse signals (string matching only) -------------------------
    brand_in_url = [b for b in TARGET_BRANDS if b in url_l]
    feats["brand_in_url"] = len(brand_in_url)
    feats["brand_in_subdomain"] = int(any(b in (ext.subdomain or "").lower()
                                          for b in TARGET_BRANDS))
    feats["brand_in_path"] = int(any(b in path.lower() for b in TARGET_BRANDS))
    # brand mentioned somewhere, but NOT the registered domain -> impersonation
    feats["brand_outside_domain"] = int(
        bool(brand_in_url) and not any(b in (ext.domain or "").lower()
                                       for b in TARGET_BRANDS))
    feats["suspicious_words_in_host"] = sum(1 for w in SUSPICIOUS_WORDS if w in host_l)
    feats["suspicious_words_in_path"] = sum(1 for w in SUSPICIOUS_WORDS
                                            if w in path.lower())

    # ---- token statistics (what char/token n-grams capture, aggregated) -----
    url_tokens = _tokens(url_l)
    host_tokens = _tokens(host_l)
    feats["num_tokens"] = len(url_tokens)
    feats["mean_token_length"] = _ratio(sum(len(t) for t in url_tokens), len(url_tokens))
    feats["max_token_length"] = max((len(t) for t in url_tokens), default=0)
    feats["num_host_tokens"] = len(host_tokens)
    feats["max_host_token_length"] = max((len(t) for t in host_tokens), default=0)

    # randomness-of-name indicators (cheap stand-ins for char n-gram models)
    dom = (ext.domain or "").lower()
    feats["domain_vowel_ratio"] = _ratio(sum(c in _VOWELS for c in dom), len(dom))
    feats["domain_max_consonant_run"] = _max_run(
        dom, set("bcdfghjklmnpqrstvwxyz"))
    feats["domain_digit_letter_switches"] = sum(
        1 for a, b in zip(dom, dom[1:]) if a.isdigit() != b.isdigit())
    feats["url_char_repeat_max"] = max(Counter(url_l).values()) if url_l else 0
    feats["host_hyphen_groups"] = len([p for p in host_l.split("-") if p]) - 1 \
        if "-" in host_l else 0

    # ---- TLD and payload indicators ----------------------------------------
    feats["tld_abuse_prone"] = int((ext.suffix or "").lower().rsplit(".", 1)[-1]
                                   in ABUSE_PRONE_TLDS)
    feats["path_has_executable_ext"] = int(
        bool(_EXECUTABLE_EXT.search(path)))

    # ---- hashed character / token n-grams ----------------------------------
    # Computed on the lowercased, scheme-stripped URL: the training corpora
    # differ in whether they store a scheme, so including it would let the
    # n-gram block encode "which corpus did this row come from".
    _ng_src = strip_scheme(url_l)
    for i, v in enumerate(_hashed_char_ngrams(_ng_src)):
        feats[f"cng3_{i:02d}"] = v
    for i, v in enumerate(_hashed_token_bigrams(_tokens(_ng_src))):
        feats[f"tng2_{i:02d}"] = v

    return feats


FEATURE_NAMES = list(extract_features("http://example.com/test").keys())

# Named subsets, so the n-gram block can be ablated against the purely
# structural features under one identical validation protocol.
NGRAM_FEATURE_NAMES = [f for f in FEATURE_NAMES
                       if f.startswith(("cng3_", "tng2_"))]
STRUCTURAL_FEATURE_NAMES = [f for f in FEATURE_NAMES
                            if f not in set(NGRAM_FEATURE_NAMES)]

# The original 21, preserved for explicit old-vs-new comparisons.
LEGACY_FEATURE_NAMES = [
    "url_length", "host_length", "path_length", "num_dots", "num_hyphens",
    "num_underscores", "num_slashes", "num_digits", "num_special_chars",
    "num_subdomains", "has_ip", "has_at_symbol", "has_double_slash_redirect",
    "is_https", "is_shortened", "suspicious_word_count", "entropy",
    "query_length", "num_params", "tld_length", "digit_ratio_in_host",
]

FEATURE_SETS = {
    "legacy": LEGACY_FEATURE_NAMES,          # the original 21
    "structural": STRUCTURAL_FEATURE_NAMES,  # 21 + new lexical/structural
    "full": FEATURE_NAMES,                   # structural + hashed n-grams
}
