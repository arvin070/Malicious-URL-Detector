"""
Leakage control primitives -- the single source of truth for "are these two
URLs the same thing?" and "do these two sets share a domain?".

WHY THIS MODULE EXISTS
----------------------
`canon()` and the registered-domain lookup were previously copy-pasted into
train.py, validate_unseen.py, build_hard_negatives.py and audit_dataset.py.
Four copies of a leakage check is four chances for them to drift, and a
leakage check that silently disagrees with the one used to build the data is
worse than no check at all.  Everything now imports from here.

WHAT CANONICALIZATION IS FOR
----------------------------
`canonicalize()` exists ONLY to decide whether two rows are duplicates of each
other.  It is never used to build features and never written back to the
dataset: the model must see the URL exactly as a browser would.  Folding case,
scheme, "www." and the trailing slash would destroy real signal (`is_https`,
host casing) if it ever reached the feature extractor.

SAFETY: every function here is pure string manipulation.  Nothing is resolved,
fetched, crawled or executed.
"""

import re

from urllib.parse import urlsplit

import pandas as pd
import tldextract

# suffix_list_urls=() -> never downloads the public-suffix list; uses the
# snapshot bundled with tldextract. Keeps this module import-safe offline.
_extract = tldextract.TLDExtract(suffix_list_urls=())

_SCHEME_RE = re.compile(r"^[a-zA-Z][a-zA-Z0-9+.\-]*://")
_WWW_RE = re.compile(r"^www\.")


def canonicalize(url: str) -> str:
    """Comparison key for duplicate detection. NOT a feature, NOT stored.

    Folds the differences that make two records the same resource:
    case, scheme, a leading "www.", and the trailing slash.
    """
    u = str(url).strip().lower()
    u = _SCHEME_RE.sub("", u)
    u = _WWW_RE.sub("", u)
    return u.rstrip("/")


# Back-compat alias: validate_unseen.py and build_hard_negatives.py already
# call this name.
canon = canonicalize


def registered_domain(url: str) -> str:
    """eTLD+1 ("login.secure.example.co.uk" -> "example.co.uk").

    This is the grouping key for all domain-aware validation, so what it does
    when extraction FAILS matters as much as when it succeeds.

    Fallback order: public suffix + 1, then the hostname, then "<unparsed>".

    The hostname step is load-bearing. It was previously the raw URL, which
    looks harmless and is not: an IP-literal host has no public suffix, so
    every path on it became its own group. `http://1.2.3.4/a` and
    `http://1.2.3.4/b` were treated as unrelated domains and could therefore
    land on opposite sides of a "domain-disjoint" split -- the precise leak
    this module exists to prevent. Measured on data/urlguard_dataset.csv when
    the bug was found: 1,554 URLs collapsing to 834 real hosts, with 97 hosts
    appearing in more than one split.

    "<unparsed>" remains for input with no hostname at all. It is an explicit
    bucket rather than an empty string so blank and malformed rows share one
    visible group in leakage reports instead of silently vanishing.
    """
    s = str(url)
    suffix = _extract(s).top_domain_under_public_suffix
    if suffix:
        return suffix
    try:
        host = (urlsplit(s if "//" in s else "//" + s.strip()).hostname or "")
    except ValueError:
        host = ""
    return host.lower() or "<unparsed>"


dom = registered_domain  # back-compat alias


def registered_domains(urls) -> pd.Series:
    return pd.Series(list(urls), dtype=object).map(registered_domain)


# --------------------------------------------------------------- duplicates
def drop_exact_duplicates(df: pd.DataFrame, url_col: str = "url"):
    """Drop rows that are the same URL after canonicalization.

    Returns (deduped_df, report).  Conflicting labels on the same canonical
    URL are reported separately -- those are not merely redundant rows, they
    are contradictory ground truth, and averaging them away would hide the
    problem.
    """
    out = df.copy()
    out["_key"] = out[url_col].map(canonicalize)

    n_before = len(out)
    exact_dupes = int(out[url_col].astype(str).duplicated().sum())
    canon_dupes = int(out["_key"].duplicated().sum())

    conflicts = 0
    if "label" in out.columns:
        per_key = out.groupby("_key")["label"].nunique()
        conflicting_keys = set(per_key[per_key > 1].index)
        conflicts = len(conflicting_keys)

    out = out.drop_duplicates(subset="_key").drop(columns="_key")
    out = out.reset_index(drop=True)

    return out, {
        "rows_before": n_before,
        "rows_after": len(out),
        "exact_string_duplicates": exact_dupes,
        "canonical_duplicates_removed": canon_dupes,
        "label_conflicting_canonical_urls": conflicts,
    }


# ------------------------------------------------------------ set overlap
def url_overlap(a_urls, b_urls) -> dict:
    """Canonical-URL overlap between two corpora."""
    a = set(pd.Series(list(a_urls), dtype=object).map(canonicalize))
    b = set(pd.Series(list(b_urls), dtype=object).map(canonicalize))
    shared = a & b
    return {
        "n_a": len(a), "n_b": len(b), "shared": len(shared),
        "frac_of_b": (len(shared) / len(b)) if b else 0.0,
        "examples": sorted(shared)[:5],
    }


def domain_overlap(a_urls, b_urls) -> dict:
    """Registered-domain overlap between two corpora."""
    a = set(registered_domains(a_urls))
    b = set(registered_domains(b_urls))
    shared = a & b
    return {
        "n_a": len(a), "n_b": len(b), "shared": len(shared),
        "frac_of_b": (len(shared) / len(b)) if b else 0.0,
        "examples": sorted(shared)[:5],
    }


# ---------------------------------------------------- split verification
def split_domain_leakage(groups, train_idx, val_idx) -> dict:
    """Domains present on BOTH sides of one split. Empty == clean split."""
    g = pd.Series(list(groups), dtype=object)
    tr = set(g.iloc[list(train_idx)])
    va = set(g.iloc[list(val_idx)])
    shared = tr & va
    n_leaked_rows = int(g.iloc[list(val_idx)].isin(shared).sum())
    return {
        "train_domains": len(tr),
        "val_domains": len(va),
        "shared_domains": len(shared),
        "leaked_val_rows": n_leaked_rows,
        "leaked_val_frac": n_leaked_rows / max(len(val_idx), 1),
        "examples": sorted(shared)[:5],
    }


def assert_no_domain_leakage(groups, train_idx, val_idx, label: str = "split"):
    """Raise if a split shares any registered domain across its two sides.

    Used as a hard gate in train.py: a domain-aware protocol that silently
    stopped being domain-aware would produce optimistic numbers that look
    entirely plausible, so this fails loudly instead of warning.
    """
    rep = split_domain_leakage(groups, train_idx, val_idx)
    if rep["shared_domains"]:
        raise AssertionError(
            f"{label}: {rep['shared_domains']} registered domain(s) appear in "
            f"both train and validation "
            f"({rep['leaked_val_rows']} validation rows affected, "
            f"e.g. {rep['examples']})")
    return rep


def audit_folds(groups, folds, label: str = "fold") -> pd.DataFrame:
    """Verify every fold of a CV splitter and return a per-fold report."""
    rows = []
    for i, (tr, va) in enumerate(folds, 1):
        rep = assert_no_domain_leakage(groups, tr, va, f"{label} {i}")
        rows.append({"fold": i, "train_rows": len(tr), "val_rows": len(va),
                     **{k: rep[k] for k in ("train_domains", "val_domains",
                                            "shared_domains")}})
    return pd.DataFrame(rows)
