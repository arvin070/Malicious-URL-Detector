"""
The per-page analysis layers that sit beside the HTML feature extractor.

WHY THIS MODULE EXISTS
----------------------
JavaScript analysis, redirect-chain analysis and structural fingerprinting all
need the same thing: the fetched HTML, while it is still in hand. Both the
dataset builder and the live server need all three, computed identically. When
that logic lived in the dataset builder, serving had to reimplement it, and a
model trained on one implementation would be fed the other's numbers -- the
same class of parity bug the Python/Kotlin golden vectors exist to prevent.

So it lives here once, and both callers import it.

Nothing in this module executes a script, fetches an external resource, or
follows a link. External script `src` URLs are counted, never retrieved.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from js_analysis import analyse_scripts, JS_FEATURE_NAMES          # noqa: E402,F401
from kit_similarity import fingerprint                             # noqa: E402
from redirect_analysis import (analyse as analyse_redirects,        # noqa: E402,F401
                               REDIRECT_FEATURE_NAMES)

# Feature columns contributed by this module, in a stable order.
LAYER_FEATURE_NAMES = list(JS_FEATURE_NAMES) + list(REDIRECT_FEATURE_NAMES)


def split_scripts(html):
    """Separate inline script bodies from external script sources.

    External sources are returned as URLs so they can be COUNTED. They are not
    downloaded: doing so would pull further attacker-chosen content for no
    analytical gain that static matching could use.
    """
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(html or "", "lxml")
    tags = soup.find_all("script")
    inline = [t.get_text() or "" for t in tags if not t.get("src")]
    external = [t.get("src") for t in tags if t.get("src")]
    return inline, external


def analyse_page(res):
    """Run every page layer over one fetch result.

    `res` is a src.web_fetch result: it must carry `html`, `requested_url`,
    `final_url` and `redirects`.

    Returns a dict with:
        features     flat, model-ready floats/ints (JS + redirect)
        js           the full JS report, including which patterns matched
        redirects    the full redirect report
        fingerprint  tag_simhash / form_sha, for phishing-kit similarity
    """
    html = res.get("html") or ""
    inline, external = split_scripts(html)
    js = analyse_scripts(inline, external)
    rd = analyse_redirects(res.get("requested_url") or res.get("final_url") or "",
                           res.get("final_url") or "",
                           res.get("redirects") or [])
    fp = fingerprint(html)

    feats = {}
    feats.update(js["features"])
    feats.update(rd["features"])
    return {"features": feats, "js": js, "redirects": rd, "fingerprint": fp}


def empty_features():
    """Zeros for every layer column, for use when no page was retrieved.

    Returned explicitly rather than left as missing values so that a caller
    pairing this with an availability flag is stating "no page" rather than
    "a page with no JavaScript" -- two different things the model must not
    confuse.
    """
    return {c: 0 for c in LAYER_FEATURE_NAMES}
