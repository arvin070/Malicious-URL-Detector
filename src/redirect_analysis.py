"""
Redirect-chain analysis.

Consumes the chain web_fetch.fetch() already recorded while following hops
manually. It performs no requests of its own.

What a chain can show, and what it cannot: a hop from https to http is a
measurable downgrade; a hop to a different registered domain is a measurable
change of destination. Neither is misconduct on its own -- link shorteners,
CDNs, SSO flows and marketing trackers all redirect across domains, and some
still land on http. The measurements are reported; weighting them is the
fusion model's job.
"""

from urllib.parse import urlsplit

from leakage import registered_domain

SHORTENERS = {
    "bit.ly", "tinyurl.com", "goo.gl", "t.co", "ow.ly", "is.gd", "buff.ly",
    "adf.ly", "shorte.st", "rebrand.ly", "cutt.ly", "lnkd.in", "t.ly",
    "rb.gy", "shorturl.at", "tiny.cc", "s.id",
}


def _scheme(u):
    try:
        return (urlsplit(u).scheme or "").lower()
    except ValueError:
        return ""


def analyse(requested_url, final_url, chain):
    """Return redirect features plus a human-readable layer dict."""
    chain = chain or []
    hops = len(chain)
    start_dom = registered_domain(requested_url)
    end_dom = registered_domain(final_url or requested_url)

    domains = [start_dom]
    downgrades = 0
    cross_domain = 0
    for hop in chain:
        frm, to = hop.get("from", ""), hop.get("to", "")
        d_to = registered_domain(to)
        if domains and d_to != domains[-1]:
            cross_domain += 1
        domains.append(d_to)
        if _scheme(frm) == "https" and _scheme(to) == "http":
            downgrades += 1

    distinct = len({d for d in domains if d})
    feats = {
        "redirect_hops": hops,
        "redirect_distinct_domains": distinct,
        "redirect_cross_domain_hops": cross_domain,
        "redirect_https_downgrades": downgrades,
        "redirect_final_is_https": int(_scheme(final_url or "") == "https"),
        "redirect_start_is_shortener": int(start_dom in SHORTENERS),
        "redirect_domain_changed": int(bool(start_dom and end_dom
                                            and start_dom != end_dom)),
        "redirect_chain_through_shortener": int(
            any(registered_domain(h.get("to", "")) in SHORTENERS for h in chain)),
    }

    signals = []
    if downgrades:
        signals.append({
            "text": "Redirect downgraded from HTTPS to HTTP",
            "evidence": f"{downgrades} hop(s) moved from https to http, so the "
                        f"final request was unencrypted."})
    if feats["redirect_domain_changed"]:
        signals.append({
            "text": "Final destination is a different domain",
            "evidence": f"Started at {start_dom or 'unknown'} and ended at "
                        f"{end_dom or 'unknown'}. Common for shorteners and "
                        f"SSO, so weak on its own."})
    if hops >= 3:
        signals.append({
            "text": f"Long redirect chain ({hops} hops)",
            "evidence": "Multiple consecutive redirects before the final page."})
    if feats["redirect_start_is_shortener"]:
        signals.append({
            "text": "Link begins at a URL shortener",
            "evidence": f"{start_dom} hides the destination until it is "
                        f"followed."})

    return {
        "features": feats,
        "signals": signals,
        "chain": [{"from": h.get("from"), "to": h.get("to"),
                   "status": h.get("status")} for h in chain],
        "start_domain": start_dom,
        "final_domain": end_dom,
    }


REDIRECT_FEATURE_NAMES = sorted(
    analyse("http://a.com/", "http://a.com/", [])["features"].keys())
