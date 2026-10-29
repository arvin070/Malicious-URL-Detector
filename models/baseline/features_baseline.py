"""
Feature extraction for phishing / malicious URL detection.
Purely lexical/structural features computed from the URL string alone --
no page fetch, no network call, safe to run on unverified URLs.
"""

import re
import math
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


def extract_features(url: str) -> dict:
    url = url.strip()
    parsed = urlparse(url if "://" in url else "http://" + url)
    host = parsed.netloc.split(":")[0]
    ext = _extract(url)
    registered_domain = ".".join(p for p in [ext.domain, ext.suffix] if p)

    path = parsed.path or ""
    query = parsed.query or ""

    return {
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


FEATURE_NAMES = list(extract_features("http://example.com/test").keys())
