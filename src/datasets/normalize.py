"""
Per-source parsers that all emit ONE schema.

    url                  the URL exactly as published (never canonicalised here)
    label                1 malicious / 0 benign
    source               source key
    label_confidence     verified | reported | presumed
    first_seen           ISO-8601 UTC, or None when the feed carries no date
    collected_at         when this project downloaded the feed

Canonicalisation deliberately does NOT happen at this stage: the model must see
URLs as a browser would, and folding case or scheme here would destroy real
signal. Canonical forms are computed separately, only for comparison.
"""

import csv
import io
import re

_VALID = re.compile(r"^(?:[a-zA-Z][a-zA-Z0-9+.\-]*://)?[^\s/?#@:]+\.[^\s/?#]{2,}",
                    re.IGNORECASE)


def looks_like_url(u: str) -> bool:
    """Cheap structural sanity check. Rejects obvious junk, not bad sites."""
    if not u or len(u) > 2000 or len(u) < 4:
        return False
    if any(ch.isspace() for ch in u):
        return False
    return bool(_VALID.match(u))


def _rows_from_csv(text, comment_prefix="#"):
    lines = [ln for ln in text.splitlines()
             if ln.strip() and not ln.lstrip().startswith(comment_prefix)]
    if not lines:
        return []
    return list(csv.DictReader(io.StringIO("\n".join(lines))))


def parse_phishtank(text, source, collected_at):
    out = []
    for r in _rows_from_csv(text):
        # Only entries a PhishTank reviewer actually confirmed. Unverified
        # submissions are exactly the "appears on a list" case the brief warns
        # against treating as fact.
        if (r.get("verified") or "").strip().lower() != "yes":
            continue
        url = (r.get("url") or "").strip()
        if not looks_like_url(url):
            continue
        out.append({
            "url": url,
            "label": 1,
            "source": source.key,
            "label_confidence": "verified",
            "first_seen": (r.get("submission_time") or "").strip() or None,
            "collected_at": collected_at,
        })
    return out


def _rows_from_commented_header_csv(text):
    """URLhaus puts its CSV header INSIDE the comment block.

    The dump starts with a banner of `#` lines, the last of which is
    `# id,dateadded,url,...` -- the actual header. Stripping comments naively
    therefore removes the header and yields zero rows, which is exactly what
    happened the first time this ran.
    """
    header = None
    data = []
    for ln in text.splitlines():
        stripped = ln.strip().lstrip("\ufeff")
        if not stripped:
            continue
        if stripped.startswith("#"):
            candidate = stripped.lstrip("#").strip()
            if header is None and candidate.lower().startswith("id,"):
                header = candidate
            continue
        data.append(stripped)
    if header is None:
        return _rows_from_csv(text)
    return list(csv.DictReader(io.StringIO("\n".join([header] + data))))


def parse_urlhaus(text, source, collected_at):
    out = []
    for r in _rows_from_commented_header_csv(text):
        if (r.get("url_status") or "").strip().lower() not in ("online", ""):
            continue
        url = (r.get("url") or "").strip().strip('"')
        if not looks_like_url(url):
            continue
        out.append({
            "url": url,
            "label": 1,
            "source": source.key,
            "label_confidence": "verified",
            "first_seen": (r.get("dateadded") or "").strip() or None,
            "collected_at": collected_at,
        })
    return out


def parse_url_list(text, source, collected_at):
    """Plain one-URL-per-line feeds (OpenPhish, Phishing.Database)."""
    out = []
    for line in text.splitlines():
        url = line.strip()
        if not url or url.startswith("#") or not looks_like_url(url):
            continue
        out.append({
            "url": url,
            "label": source.label,
            "source": source.key,
            "label_confidence": source.label_confidence,
            "first_seen": None,
            "collected_at": collected_at,
        })
    return out


def parse_tranco(text, source, collected_at):
    """Tranco is `rank,domain`. Emitted as the BARE DOMAIN, nothing added.

    An earlier version wrote `http://{domain}/` and that single line of
    synthesis poisoned the dataset three separate ways, each of which the
    model found before a human did:

      * the scheme: every Tranco row became http, so "https" turned into a 5x
        malicious indicator;
      * the trailing slash: it gives the URL a path of "/", while real
        phishing roots from the feeds have none -- so "path_ratio_of_url == 0"
        became a near-perfect malicious signal (SHAP +4.55 on github.com);
      * both together made benign roots structurally unlike malicious roots in
        ways that have nothing to do with phishing.

    Tranco publishes a domain and a rank. It says nothing about scheme or
    path, so nothing about scheme or path is invented here. The scheme is
    assigned later, at the same rate in both classes, by balance_scheme().
    """
    out = []
    for line in text.splitlines():
        line = line.strip()
        if not line or "," not in line:
            continue
        _rank, _, domain = line.partition(",")
        domain = domain.strip()
        if not looks_like_url(domain):
            continue
        out.append({
            "url": domain,
            "label": 0,
            "source": source.key,
            "label_confidence": "presumed",
            "first_seen": None,
            "collected_at": collected_at,
        })
    return out


def parse_local_csv(text, source, collected_at):
    out = []
    for r in _rows_from_csv(text):
        url = (r.get("url") or "").strip()
        if not looks_like_url(url):
            continue
        try:
            label = int(r.get("label")) if source.label < 0 else source.label
        except (TypeError, ValueError):
            continue
        if label not in (0, 1):
            continue
        out.append({
            "url": url,
            "label": label,
            "source": source.key,
            "label_confidence": source.label_confidence,
            "first_seen": None,
            "collected_at": collected_at,
        })
    return out


PARSERS = {
    "phishtank": parse_phishtank,
    "urlhaus": parse_urlhaus,
    "openphish": parse_url_list,
    "phishing_database": parse_url_list,
    "tranco": parse_tranco,
    "local_corpus": parse_local_csv,
    "local_hard_negatives": parse_local_csv,
}


def parse(source, text, collected_at):
    return PARSERS[source.key](text, source, collected_at)
