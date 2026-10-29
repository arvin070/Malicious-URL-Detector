"""
Phishing-kit / template reuse detection via DOM structural fingerprinting.

THE IDEA
--------
Phishing kits are redistributed. The same page markup turns up across hundreds
of hostnames with the text, images and branding swapped. Text changes; DOM
*shape* does not. Fingerprinting the shape therefore links pages that a
content-based comparison would call unrelated.

HOW
---
Two fingerprints per page, both computed from structure only:

  tag_simhash   a 64-bit SimHash over tag-path shingles ("html>body>form>input")
                so near-identical layouts land near each other in Hamming space
  form_sha      an exact hash of the form skeleton (input types and order),
                which is the part a kit almost never edits

Lookup is against a corpus built from pages already in the training set, with
their labels. A match is reported as "this page's structure matches N known
pages, M of them labelled phishing" -- a provenance statement, not a verdict.
A benign template (Bootstrap, a CMS theme) will also match many pages, which is
why the label breakdown is always shown rather than just the match count.

Nothing here fetches or executes anything.
"""

import hashlib
import json
import os
import re
from collections import Counter

from bs4 import BeautifulSoup

SIMHASH_BITS = 64
SHINGLE_DEPTH = 4          # tag-path length used as a shingle
# Hamming distance at or below which two pages are called structurally similar.
# 8/64 bits allows a kit to have been lightly edited while still matching.
NEAR_DUPLICATE_DISTANCE = 8


def _tag_paths(html: str, max_nodes=4000):
    """Tag-only paths through the DOM. No text, no attributes, no classes --
    a kit that renames every class still produces the same paths."""
    soup = BeautifulSoup(html or "", "lxml")
    paths = []
    for i, tag in enumerate(soup.find_all(True)):
        if i >= max_nodes:
            break
        chain = []
        node = tag
        for _ in range(SHINGLE_DEPTH):
            if node is None or not getattr(node, "name", None):
                break
            chain.append(node.name)
            node = node.parent
        if chain:
            paths.append(">".join(reversed(chain)))
    return paths


def simhash(tokens, bits=SIMHASH_BITS) -> int:
    if not tokens:
        return 0
    v = [0] * bits
    for tok, weight in Counter(tokens).items():
        h = int(hashlib.blake2b(tok.encode("utf-8"),
                                digest_size=8).hexdigest(), 16)
        for b in range(bits):
            v[b] += weight if (h >> b) & 1 else -weight
    out = 0
    for b in range(bits):
        if v[b] > 0:
            out |= (1 << b)
    return out


def hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def form_skeleton(html: str) -> str:
    """Input types and order per form, with names and labels stripped."""
    soup = BeautifulSoup(html or "", "lxml")
    parts = []
    for form in soup.find_all("form"):
        types = [(i.get("type") or "text").lower() for i in form.find_all("input")]
        parts.append(",".join(types))
    return "|".join(sorted(parts))


def fingerprint(html: str) -> dict:
    paths = _tag_paths(html)
    skel = form_skeleton(html)
    h = simhash(paths)
    return {
        "tag_simhash": h,
        # The same value as a hex string. A 64-bit integer does not survive
        # JSON into a JavaScript Number -- it is silently rounded to the
        # nearest double -- so anything that displays or compares this value
        # outside Python must use the hex form. Hamming distance is still
        # computed on the integer, server-side.
        "tag_simhash_hex": f"{h:016x}",
        "form_sha": hashlib.sha256(skel.encode("utf-8")).hexdigest()[:16]
                    if skel else "",
        "form_skeleton": skel,
        "node_sample": len(paths),
    }


class KitIndex:
    """Corpus of fingerprints with labels. Built offline, loaded read-only."""

    def __init__(self, entries=None):
        self.entries = entries or []

    @classmethod
    def load(cls, path):
        if not os.path.exists(path):
            return cls([])
        with open(path, encoding="utf-8") as fh:
            return cls(json.load(fh).get("entries", []))

    def save(self, path, meta=None):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump({"meta": meta or {}, "entries": self.entries}, fh)

    def add(self, fp, label, domain):
        self.entries.append({"h": fp["tag_simhash"], "f": fp["form_sha"],
                             "y": int(label), "d": domain})

    def query(self, fp, exclude_domain=None, max_distance=NEAR_DUPLICATE_DISTANCE):
        """Structural neighbours of a page, with their label breakdown."""
        h = fp["tag_simhash"]
        if not h or not self.entries:
            return {"available": False,
                    "reason": ("No phishing-kit index is installed, or the "
                               "page produced no usable structure."),
                    "matches": 0}
        near, near_phish, domains = 0, 0, set()
        exact_form, exact_form_phish = 0, 0
        for e in self.entries:
            if exclude_domain and e["d"] == exclude_domain:
                continue          # a page matching itself proves nothing
            if hamming(h, e["h"]) <= max_distance:
                near += 1
                near_phish += e["y"]
                domains.add(e["d"])
            if fp["form_sha"] and e["f"] == fp["form_sha"]:
                exact_form += 1
                exact_form_phish += e["y"]
        return {
            "available": True,
            "matches": near,
            "matching_domains": len(domains),
            "matches_labelled_phishing": near_phish,
            "phishing_share": round(near_phish / near, 3) if near else 0.0,
            "form_skeleton_matches": exact_form,
            "form_skeleton_phishing": exact_form_phish,
            "max_distance": max_distance,
            "index_size": len(self.entries),
            "method": ("64-bit SimHash over tag-path shingles, compared by "
                       "Hamming distance. Structure only -- no text, no "
                       "attributes, no images."),
            "caveat": ("Shared structure also results from common frameworks "
                       "and CMS themes, so the label breakdown matters more "
                       "than the match count."),
        }


def kit_features(match: dict) -> dict:
    if not match or not match.get("available"):
        return {"kit_matches": 0, "kit_phishing_share": 0.0,
                "kit_form_matches": 0, "kit_available": 0}
    return {
        "kit_matches": min(match["matches"], 500),
        "kit_phishing_share": match["phishing_share"],
        "kit_form_matches": min(match["form_skeleton_matches"], 500),
        "kit_available": 1,
    }
