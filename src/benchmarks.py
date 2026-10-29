"""
Hard-benign and adversarial benchmark suites.

These are not training data and never become training data. They are fixed,
hand-curated probes that answer two questions the headline metrics hide:

  HARD BENIGN   Does the system flag the sites people actually use? Holdout
                FPR is measured on the benign distribution the dataset
                happens to contain, which under-represents major sites,
                banking, universities and legitimate login pages. A model can
                post 2.6% holdout FPR and still flag github.com.

  ADVERSARIAL   Does a trivial transformation of a known-bad URL evade it, and
                does a trivial transformation of a known-good URL trip it?
                Homoglyphs, punycode, percent-encoding, added subdomains,
                case changes, port notation, shorteners.

Expected outcomes are stated per case. An adversarial *variant* of a benign URL
is still benign -- encoding a URL does not make it malicious -- and a variant of
a phishing URL is still phishing. Where the honest expectation is "unknown",
the case is marked `expect=None` and only reported, never scored.

Nothing here contacts the network: every case is a URL string.
"""

import unicodedata

# ---------------------------------------------------------------- hard benign
# Sites whose misclassification would be immediately visible to any user.
# Grouped so the report says WHICH category fails, not just a count.
HARD_BENIGN = {
    "major_tech": [
        "https://www.google.com", "https://github.com",
        "https://github.com/explore", "https://stackoverflow.com",
        "https://stackoverflow.com/questions", "https://www.microsoft.com",
        "https://www.apple.com", "https://www.amazon.com",
        "https://news.ycombinator.com", "https://www.reddit.com/r/programming",
        "https://developer.mozilla.org/en-US/docs/Web/JavaScript",
        "https://www.cloudflare.com", "https://openai.com",
    ],
    "reference": [
        "https://en.wikipedia.org/wiki/Phishing", "https://www.wikipedia.org",
        "https://arxiv.org/abs/1708.07747", "https://doi.org/10.1038/nature14539",
        "https://www.bbc.co.uk/news/world", "https://www.nasa.gov/missions",
        "https://docs.python.org/3/library/os.html",
    ],
    "banking_finance": [
        "https://www.chase.com", "https://www.wellsfargo.com",
        "https://www.bankofamerica.com", "https://www.hsbc.co.uk",
        "https://www.paypal.com", "https://www.barclays.co.uk",
        "https://www.icicibank.com", "https://www.sbi.co.in",
    ],
    "university": [
        "https://www.ox.ac.uk", "https://www.mit.edu", "https://www.stanford.edu",
        "https://www.iitm.ac.in", "https://www.cam.ac.uk",
        "https://www.ed.ac.uk/studying",
    ],
    # The category most at risk of false positives: real login pages look like
    # phishing pages by construction.
    "legitimate_login": [
        "https://accounts.google.com/signin",
        "https://github.com/login",
        "https://login.microsoftonline.com",
        "https://www.paypal.com/signin",
        "https://appleid.apple.com/sign-in",
        "https://www.linkedin.com/login",
        "https://id.atlassian.com/login",
        "https://auth.openai.com/log-in",
        "https://signin.aws.amazon.com/signin",
        "https://www.dropbox.com/login",
    ],
    "government": [
        "https://www.gov.uk/browse/visas-immigration",
        "https://www.irs.gov/payments", "https://www.india.gov.in",
        "https://www.usa.gov",
    ],
}

ALL_HARD_BENIGN = [(u, cat) for cat, urls in HARD_BENIGN.items() for u in urls]


# --------------------------------------------------------------- adversarial
CYRILLIC_LOOKALIKE = {
    "a": "а", "e": "е", "o": "о", "p": "р",
    "c": "с", "x": "х", "y": "у", "i": "і",
}


def homoglyph(text: str, limit=2) -> str:
    """Replace up to `limit` Latin letters with Cyrillic lookalikes."""
    out, n = [], 0
    for ch in text:
        low = ch.lower()
        if n < limit and low in CYRILLIC_LOOKALIKE:
            out.append(CYRILLIC_LOOKALIKE[low])
            n += 1
        else:
            out.append(ch)
    return "".join(out)


def to_punycode(host: str) -> str:
    try:
        return host.encode("idna").decode("ascii")
    except (UnicodeError, UnicodeDecodeError):
        return host


def percent_encode_path(url: str) -> str:
    """Encode a few path characters without changing what the URL resolves to."""
    if "://" not in url:
        return url
    scheme, rest = url.split("://", 1)
    if "/" not in rest:
        return url
    host, path = rest.split("/", 1)
    enc = "".join(f"%{ord(c):02X}" if c.isalpha() and i % 3 == 0 else c
                  for i, c in enumerate(path))
    return f"{scheme}://{host}/{enc}"


def build_adversarial(benign_seed, phishing_seed):
    """Generate variant cases with the honest expected label for each.

    Transforming a URL does not change what it is: a percent-encoded benign URL
    is benign, and a phishing URL with an extra subdomain is still phishing.
    Where a transformation genuinely changes the destination (a different host)
    the expectation is left unknown.
    """
    cases = []

    def add(url, expect, family, note):
        cases.append({"url": url, "expect": expect, "family": family,
                      "note": note})

    for u in benign_seed:
        host = u.split("://", 1)[-1].split("/")[0]
        rest = u[len(u.split("://")[0]) + 3 + len(host):]
        scheme = u.split("://")[0]

        add(percent_encode_path(u), "benign", "encoding",
            "Percent-encoding path characters does not change the destination.")
        add(f"{scheme}://{host.upper()}{rest}", "benign", "case",
            "Hostnames are case-insensitive.")
        add(f"{scheme}://{host}:443{rest}" if scheme == "https"
            else f"{scheme}://{host}:80{rest}", "benign", "explicit_port",
            "Explicit default port is equivalent.")
        add(f"{scheme}://{host}{rest or '/'}?utm_source=newsletter&utm_campaign=x",
            "benign", "tracking_params",
            "Marketing parameters are ubiquitous on legitimate links.")
        add(f"{scheme}://www.{host}{rest}" if not host.startswith("www.")
            else f"{scheme}://{host[4:]}{rest}", "benign", "www_variant",
            "Adding or removing www is equivalent for most sites.")

    for u in phishing_seed:
        host = u.split("://", 1)[-1].split("/")[0]
        rest = u[len(u.split("://")[0]) + 3 + len(host):]
        scheme = u.split("://")[0]

        add(percent_encode_path(u), "phishing", "encoding",
            "Encoding the path must not hide a phishing URL.")
        add(f"{scheme}://{host}{rest}#fragment", "phishing", "fragment",
            "A fragment is not sent to the server and changes nothing.")
        add(f"{scheme}://login.secure.{host}{rest}", "phishing", "subdomain",
            "Extra subdomains on the same registered domain.")
        add(f"{scheme}://{host.upper()}{rest}", "phishing", "case",
            "Case change on the host.")
        add(f"{scheme}://{host}{rest}?redirect=https://www.paypal.com",
            "phishing", "benign_param",
            "A legitimate-looking parameter appended to a phishing URL.")

    # Homoglyph and punycode variants of MAJOR BRANDS: these are new domains an
    # attacker could register, so the honest expectation is that they look
    # suspicious -- they are not the real site.
    for brand in ["google.com", "paypal.com", "apple.com", "microsoft.com",
                  "github.com", "amazon.com"]:
        hg = homoglyph(brand)
        add(f"https://{hg}/login", "phishing", "homoglyph",
            f"Cyrillic lookalike of {brand}; a different domain entirely.")
        add(f"https://{to_punycode(hg)}/login", "phishing", "punycode",
            f"Punycode encoding of the homoglyph variant of {brand}.")
        add(f"https://{brand.split('.')[0]}-secure-login.com/", "phishing",
            "brand_in_other_domain",
            f"Brand name inside an unrelated registered domain.")

    return cases


def default_adversarial():
    benign_seed = [
        "https://www.google.com/search",
        "https://github.com/explore",
        "https://en.wikipedia.org/wiki/Phishing",
        "https://www.bbc.co.uk/news/world",
        "https://accounts.google.com/signin",
    ]
    phishing_seed = [
        "http://secure-paypal-login.tk/verify/9231",
        "http://update-account-signin.ml/confirm",
        "http://verify-apple-id.cfd/secure/login",
    ]
    return build_adversarial(benign_seed, phishing_seed)


def summarise(results):
    """results: list of {case..., score, flagged}. Returns per-family stats."""
    from collections import defaultdict
    by = defaultdict(lambda: {"n": 0, "wrong": 0, "scores": []})
    for r in results:
        fam = r.get("family") or r.get("category") or "all"
        b = by[fam]
        b["n"] += 1
        b["scores"].append(r["score"])
        if r.get("expect") == "benign" and r["flagged"]:
            b["wrong"] += 1
        elif r.get("expect") == "phishing" and not r["flagged"]:
            b["wrong"] += 1
    out = {}
    for fam, b in by.items():
        out[fam] = {
            "cases": b["n"],
            "incorrect": b["wrong"],
            "error_rate": round(b["wrong"] / b["n"], 4) if b["n"] else 0.0,
            "mean_score": round(sum(b["scores"]) / len(b["scores"]), 1)
                          if b["scores"] else None,
            "max_score": max(b["scores"]) if b["scores"] else None,
        }
    return out
