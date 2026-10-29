"""
Threat-intelligence adapters.

THE RULE
--------
An adapter that is not configured returns `available: false` with a reason. It
never returns a neutral score, never returns "clean", and never guesses. The
difference between "no provider is configured", "the provider was asked and
found nothing" and "the provider has an entry for this" is preserved all the
way to the user interface, because collapsing them is how a scanner starts
claiming knowledge it does not have.

WHAT IS ACTUALLY AVAILABLE HERE
-------------------------------
One adapter is real and offline: `LocalFeedAdapter` looks a URL up in the
blocklists this project already downloaded to build its dataset (PhishTank,
URLhaus, OpenPhish, Phishing.Database). That is a genuine independent
observation -- "this exact URL appears in PhishTank's verified feed collected
on <date>" is a fact about a published list, not a model output -- and it is
the one source that can legitimately be described as *confirmed by a
third party* rather than *assessed by a model*.

It comes with a caveat that is enforced, not just documented: a URL from the
training feeds will of course be found in the training feeds. The adapter
records whether the hit came from a feed snapshot that the serving model was
also trained on, so the UI can say "corroborated by the same source that
labelled it" instead of presenting it as independent confirmation.

DNS, WHOIS, certificate transparency and commercial reputation adapters are
defined by interface only. They need network lookups or API keys this project
does not have, so they report unavailable. Stub implementations that returned
plausible values would be fabrication.
"""

import datetime as dt
import json
import os
import sys
import threading

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from leakage import canonicalize, registered_domain  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(ROOT, "data", "raw")
INDEX_PATH = os.path.join(ROOT, "models", "threat_intel_index.json")


def unavailable(name, reason, code="not_configured"):
    return {"provider": name, "available": False, "verdict": None,
            "reason": reason, "code": code}


class Adapter:
    """Interface every provider implements."""

    name = "adapter"
    kind = "generic"

    def configured(self) -> bool:
        return False

    def lookup(self, url: str) -> dict:
        return unavailable(self.name, "Adapter not implemented.")


# --------------------------------------------------------------- local feeds
class LocalFeedAdapter(Adapter):
    """Exact-URL and domain lookup against downloaded blocklist snapshots."""

    name = "local_feeds"
    kind = "blocklist"

    def __init__(self, index_path=INDEX_PATH):
        self.index_path = index_path
        self._lock = threading.Lock()
        self._idx = None

    def configured(self) -> bool:
        return os.path.exists(self.index_path)

    def _load(self):
        if self._idx is not None:
            return self._idx
        with self._lock:
            if self._idx is None:
                if not os.path.exists(self.index_path):
                    self._idx = {"urls": {}, "domains": {}, "sources": {}}
                else:
                    with open(self.index_path, encoding="utf-8") as fh:
                        self._idx = json.load(fh)
        return self._idx

    def lookup(self, url: str) -> dict:
        if not self.configured():
            return unavailable(
                self.name,
                "No local feed index is built. Run "
                "`python src/threat_intel.py --build` after building the "
                "dataset.", "not_built")
        idx = self._load()
        key = canonicalize(url)
        dom = registered_domain(url)

        url_hit = idx["urls"].get(key)
        dom_rec = idx["domains"].get(dom)

        # DOMAIN HITS ARE CONTEXT, NOT A LISTING.
        #
        # An earlier version treated "some URL on this domain is in a feed" as
        # "this domain is listed", and google.com came back listed -- because
        # Phishing.Database contains phishing pages hosted on Google
        # subdomains. The same is true of blogspot, weebly, 000webhostapp,
        # amazonaws and every other shared host.
        #
        # A blocklist of URLs licenses a claim about those URLs. Inferring the
        # domain is malicious from one bad page on it is exactly the kind of
        # overclaiming this project refuses elsewhere, so a domain hit is now
        # reported as a count for context and never as a verdict.
        if not url_hit:
            searched = {
                "provider": self.name, "available": True, "verdict": "no_match",
                "matched": False,
                "sources_searched": sorted(idx["sources"].keys()),
                "snapshot_dates": idx["sources"],
                "note": ("This exact URL is not present in the blocklist "
                         "snapshots held locally. These feeds cover a small "
                         "fraction of all phishing, so absence is not evidence "
                         "of safety."),
            }
            if dom_rec:
                searched["domain_context"] = {
                    "domain": dom,
                    "listed_urls_on_domain": dom_rec["n"],
                    "sources": dom_rec["sources"],
                    "shared_hosting_candidate": bool(dom_rec.get("popular")),
                    "note": ("Other URLs on this domain appear in the feeds. "
                             "On shared hosting that says nothing about this "
                             "page, so it is reported as context only and is "
                             "not a listing."),
                }
            return searched

        return {
            "provider": self.name,
            "available": True,
            "verdict": "listed",
            "matched": True,
            "match_level": "url",
            "sources": url_hit["sources"],
            "first_seen": url_hit.get("first_seen"),
            "label_confidence": url_hit.get("confidence"),
            # The honesty gate: these are the same snapshots the model trained
            # on, so a hit is corroboration from the labelling source, not an
            # independent second opinion.
            "independent_of_training_data": False,
            "note": ("This exact URL is listed in a blocklist snapshot this "
                     "project also used to label training data. Treat as "
                     "corroboration from the labelling source, not as "
                     "independent confirmation."),
            "snapshot_dates": idx["sources"],
        }


# ------------------------------------------------------ domain popularity
POPULARITY_PATH = os.path.join(ROOT, "models", "popular_domains.json")


class PopularityAdapter(Adapter):
    """Is this registered domain among the most-visited sites on the web?

    WHY THIS LAYER EXISTS
    ---------------------
    Measured on the hard-benign suite, the URL model alone flagged 43.8% of
    well-known sites -- 75% of banking domains, 62% of major tech, 50% of
    legitimate login pages -- while posting 2.6% FPR on its own holdout. That
    gap is not a tuning problem. A purely lexical model cannot distinguish
    google.com from g00gle-secure.com, because the thing that separates them is
    not how the string is shaped: it is knowing which domain is real. That is
    reputation, and no amount of character n-grams supplies it.

    Tranco (the ranked list already used as a benign source) supplies it
    directly and citably.

    WHAT POPULARITY IS NOT
    ----------------------
    It is not safety, and this adapter never says it is. Measured in this
    project's own index, 7,609 URLs on google.com domains appear in phishing
    blocklists -- shared hosting, user content and compromised pages are real.
    So the adapter reports a rank and a shared-hosting flag; deciding how much
    that is worth, and refusing to let it override page-level evidence, is the
    fusion engine's job.
    """

    name, kind = "popularity", "reputation"

    def __init__(self, path=POPULARITY_PATH):
        self.path = path
        self._lock = threading.Lock()
        self._idx = None

    def configured(self):
        return os.path.exists(self.path)

    def _load(self):
        if self._idx is None:
            with self._lock:
                if self._idx is None:
                    if not os.path.exists(self.path):
                        self._idx = {"ranks": {}, "source": None}
                    else:
                        with open(self.path, encoding="utf-8") as fh:
                            self._idx = json.load(fh)
        return self._idx

    def lookup(self, url):
        if not self.configured():
            return unavailable(
                self.name,
                "No popular-domain list is installed. Run "
                "`python src/threat_intel.py --build` after building the "
                "dataset.", "not_built")
        idx = self._load()
        dom = registered_domain(url)
        rank = idx["ranks"].get(dom)
        if rank is None:
            return {
                "provider": self.name, "available": True, "verdict": "not_ranked",
                "domain": dom, "rank": None,
                "note": ("Not in the top-ranked domain list. The list covers "
                         "popular sites only, so absence is expected for most "
                         "legitimate small sites and means very little."),
                "source": idx.get("source"),
            }
        return {
            "provider": self.name, "available": True, "verdict": "ranked",
            "domain": dom, "rank": int(rank),
            "tier": ("top_1k" if rank <= 1000 else
                     "top_10k" if rank <= 10000 else
                     "top_100k" if rank <= 100000 else "top_1m"),
            "note": ("Popularity is not safety: popular domains host user "
                     "content and get compromised. This reduces, but never "
                     "eliminates, risk, and page-level evidence overrides it."),
            "source": idx.get("source"),
        }


# ------------------------------------------- interface-only (not configured)
class DnsAdapter(Adapter):
    name, kind = "dns", "dns"

    def lookup(self, url):
        return unavailable(
            self.name,
            "DNS resolution is not performed. URL analysis deliberately does "
            "not contact the network, and no passive-DNS provider is "
            "configured.")


class WhoisAdapter(Adapter):
    name, kind = "whois", "registration"

    def lookup(self, url):
        return unavailable(
            self.name,
            "No WHOIS/RDAP provider is configured, so domain age and "
            "registrar information are not collected.")


class CertificateAdapter(Adapter):
    name, kind = "certificates", "tls"

    def lookup(self, url):
        return unavailable(
            self.name,
            "No certificate-transparency source is configured. The page "
            "fetcher records whether TLS validated, but issuer and CT-log "
            "history are not collected.")


class ReputationAdapter(Adapter):
    name, kind = "reputation", "reputation"

    def lookup(self, url):
        return unavailable(
            self.name,
            "No commercial reputation provider is configured (none is bundled; "
            "all require an API key).")


ADAPTERS = [LocalFeedAdapter(), PopularityAdapter(), DnsAdapter(),
            WhoisAdapter(), CertificateAdapter(), ReputationAdapter()]


def lookup_all(url: str) -> dict:
    """Query every adapter. Returns per-provider results plus a summary that
    never conflates 'unavailable' with 'clean'."""
    results = {}
    for ad in ADAPTERS:
        try:
            results[ad.name] = ad.lookup(url)
        except Exception as exc:                    # noqa: BLE001
            results[ad.name] = unavailable(
                ad.name, f"Adapter failed: {type(exc).__name__}", "error")

    listed = [r for r in results.values()
              if r.get("available") and r.get("verdict") == "listed"]
    searched = [r for r in results.values() if r.get("available")]
    return {
        "providers": results,
        "any_listed": bool(listed),
        "listed_by": [r["provider"] for r in listed],
        "providers_available": len(searched),
        "providers_total": len(results),
        "summary": (
            f"Listed by {', '.join(r['provider'] for r in listed)}."
            if listed else
            "Searched the configured sources and found no entry."
            if searched else
            "No threat-intelligence provider is configured."),
    }


def intel_features(res: dict) -> dict:
    """Numeric view for the fusion model. Unavailable stays 0 AND is flagged,
    so the model can tell 'absent' from 'searched and clean'."""
    providers = res.get("providers", {})
    local = providers.get("local_feeds", {})
    pop = providers.get("popularity", {})
    ctx = local.get("domain_context") or {}
    rank = pop.get("rank")
    return {
        "intel_domain_ranked": int(pop.get("verdict") == "ranked"),
        "intel_domain_rank_log": (
            round(__import__("math").log10(max(rank, 1)), 3) if rank else 0.0),
        "intel_domain_top1k": int(bool(rank) and rank <= 1000),
        "intel_domain_top10k": int(bool(rank) and rank <= 10000),
        "intel_available": int(bool(res.get("providers_available"))),
        "intel_listed": int(bool(res.get("any_listed"))),
        "intel_url_match": int(local.get("match_level") == "url"),
        # Context only. Kept as a separate, weaker feature so the model can
        # learn how little it is worth rather than having it folded into
        # "listed", where it caused google.com to be reported as listed.
        "intel_domain_other_urls": min(int(ctx.get("listed_urls_on_domain", 0)), 500),
        "intel_searched_clean": int(
            bool(res.get("providers_available")) and not res.get("any_listed")),
    }


INTEL_FEATURE_NAMES = sorted(intel_features({}).keys())


# --------------------------------------------------------------- index build
def build_index(out_path=INDEX_PATH, raw_dir=RAW_DIR):
    """Build the local blocklist index from the feed snapshots already
    downloaded by src/build_dataset.py. No new network access."""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from datasets.sources import SOURCES, BY_KEY
    from datasets import fetch as fetchmod, normalize

    urls, domains, sources_meta = {}, {}, {}
    for src in SOURCES:
        if src.label != 1 or src.fmt == "local":
            continue
        blob, meta = fetchmod.cache_paths(raw_dir, src.key)
        if not (os.path.exists(blob) and os.path.exists(meta)):
            print(f"  {src.key:<22} no snapshot cached, skipped")
            continue
        with open(meta, encoding="utf-8") as fh:
            m = json.load(fh)
        with open(blob, "rb") as fh:
            payload = fh.read()
        try:
            text = fetchmod.decode(src, payload)
            rows = normalize.parse(src, text, m["collected_at"])
        except Exception as exc:                    # noqa: BLE001
            print(f"  {src.key:<22} parse failed: {type(exc).__name__}")
            continue
        sources_meta[src.key] = m["collected_at"]
        for r in rows:
            k = canonicalize(r["url"])
            e = urls.setdefault(k, {"sources": [], "first_seen": r["first_seen"],
                                    "confidence": r["label_confidence"]})
            if src.key not in e["sources"]:
                e["sources"].append(src.key)
            d = registered_domain(r["url"])
            if d:
                de = domains.setdefault(d, {"sources": [], "n": 0})
                de["n"] += 1
                if src.key not in de["sources"]:
                    de["sources"].append(src.key)
        print(f"  {src.key:<22} {len(rows):>7,} entries  "
              f"(snapshot {m['collected_at']})")

    # Mark domains that also appear in the popular-domain list: those are
    # almost always shared hosting, where one bad page says nothing about the
    # next one.
    popular = set()
    try:
        from datasets import fetch as _f, normalize as _n
        tr = BY_KEY["tranco"]
        blob, meta = _f.cache_paths(raw_dir, tr.key)
        if os.path.exists(blob):
            with open(blob, "rb") as fh:
                rows = _n.parse(tr, _f.decode(tr, fh.read()), "")
            popular = {registered_domain(r["url"]) for r in rows}
    except Exception:                               # noqa: BLE001
        pass
    flagged = 0
    for d, rec in domains.items():
        if d in popular:
            rec["popular"] = 1
            flagged += 1
    print(f"  {flagged:,} listed domains also appear in the popular-domain "
          f"list (shared hosting)")

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump({"urls": urls, "domains": domains, "sources": sources_meta,
                   "built_at": dt.datetime.now(dt.timezone.utc).isoformat(
                       timespec="seconds")}, fh)
    print(f"\n  wrote {out_path}")
    print(f"  {len(urls):,} URLs / {len(domains):,} domains from "
          f"{len(sources_meta)} snapshots")

    # Popularity index, from the same cached Tranco snapshot.
    try:
        from datasets import fetch as _f2, normalize as _n2
        tr = BY_KEY["tranco"]
        blob, meta = _f2.cache_paths(raw_dir, tr.key)
        if os.path.exists(blob):
            with open(meta, encoding="utf-8") as fh:
                tm = json.load(fh)
            with open(blob, "rb") as fh:
                text = _f2.decode(tr, fh.read())
            ranks = {}
            for line in text.splitlines():
                line = line.strip()
                if "," not in line:
                    continue
                r, _, d = line.partition(",")
                d = registered_domain(d.strip())
                if d and d not in ranks:
                    try:
                        ranks[d] = int(r)
                    except ValueError:
                        continue
            with open(POPULARITY_PATH, "w", encoding="utf-8") as fh:
                json.dump({"ranks": ranks,
                           "source": f"{tr.attribution} (snapshot "
                                     f"{tm['collected_at']})"}, fh)
            print(f"  wrote {POPULARITY_PATH}  ({len(ranks):,} ranked domains)")
    except Exception as exc:                        # noqa: BLE001
        print(f"  popularity index skipped: {type(exc).__name__}: {exc}")
    return out_path


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--lookup")
    a = ap.parse_args()
    if a.build:
        print("Building local threat-intel index from cached feed snapshots")
        build_index()
    if a.lookup:
        print(json.dumps(lookup_all(a.lookup), indent=2))
