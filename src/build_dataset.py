"""
Build the versioned multi-source URLGuard dataset.

    sources -> normalise -> dedupe -> label validation -> leakage-safe splits
            -> manifest

WHAT THIS GUARDS AGAINST
------------------------
Three failure modes, each of which produces a model that looks excellent and
is not:

1. DUPLICATE LEAKAGE. The same page reachable as http/https, with or without
   "www.", with or without a trailing slash, is one page. Splitting those
   copies across train and test lets the model memorise rather than
   generalise. Dedupe is canonical, not exact-string.

2. DOMAIN LEAKAGE. 50 phishing URLs on one host are one campaign, not 50
   independent samples. Splits are grouped by registered domain so no domain
   appears on both sides.

3. STRUCTURAL CONFOUNDING. This one is specific to multi-source data and is
   the most dangerous. Tranco supplies bare domains; the phishing feeds supply
   deep URLs. Train on that naively and "has a path" becomes a near-perfect
   label predictor -- the model learns which FEED a row came from, scores ~0.99
   in validation, and collapses in the field. The build measures this
   explicitly and refuses to pretend it is not happening.

Three splits are produced:
  * grouped   domain-disjoint random split (the default for training)
  * temporal  older URLs train / newer URLs test, using real feed timestamps
  * holdout   a domain-disjoint slice kept aside entirely

SAFETY: URLs are handled as strings. Feeds are downloaded; no URL from the
dataset is ever resolved, fetched or executed.

Usage:
    python src/build_dataset.py
    python src/build_dataset.py --refresh          # re-download feeds
    python src/build_dataset.py --max-per-domain 3
"""

import argparse
import datetime as dt
import hashlib
import json
import os
import sys
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from datasets.sources import SOURCES, trust_rank  # noqa: E402
from datasets import fetch as fetchmod, normalize  # noqa: E402
from leakage import (canonicalize, registered_domain, registered_domains,  # noqa: E402
                     drop_exact_duplicates)

SCHEMA = ["url", "label", "source", "label_confidence", "first_seen",
          "collected_at"]

LOCAL_FILES = {
    "local_corpus": os.path.join("data", "real_urls.csv"),
    "local_hard_negatives": os.path.join("data", "hard_negatives.csv"),
}


def _utcnow():
    return dt.datetime.now(dt.timezone.utc)


def _parse_ts(value):
    """Lenient ISO-8601 parse. Returns tz-aware UTC datetime or NaT."""
    if not value:
        return pd.NaT
    try:
        ts = pd.to_datetime(value, utc=True, errors="coerce")
        return ts
    except (ValueError, TypeError):
        return pd.NaT


# ----------------------------------------------------------------- collect
def collect(args):  # noqa: C901
    frames, provenance = [], []
    for src in SOURCES:
        if not src.enabled:
            continue
        if src.scope not in ("both", args.scope):
            print(f"  {src.key:<22} skipped (scoped to '{src.scope}')")
            continue
        try:
            if src.fmt == "local":
                path = LOCAL_FILES[src.key]
                if not os.path.exists(path):
                    print(f"  {src.key:<22} SKIPPED (missing {path})")
                    continue
                with open(path, encoding="utf-8", errors="replace") as fh:
                    text = fh.read()
                meta = {
                    "source_key": src.key, "source_name": src.name,
                    "url": path, "bytes": os.path.getsize(path),
                    "sha256": hashlib.sha256(
                        open(path, "rb").read()).hexdigest(),
                    "collected_at": dt.datetime.fromtimestamp(
                        os.path.getmtime(path), dt.timezone.utc
                    ).isoformat(timespec="seconds"),
                    "licence": src.licence, "attribution": src.attribution,
                }
            else:
                payload, meta = fetchmod.fetch(src, cache_dir=args.cache,
                                               refresh=args.refresh)
                text = fetchmod.decode(src, payload)

            rows = normalize.parse(src, text, meta["collected_at"])
            if src.max_rows and len(rows) > src.max_rows:
                # Deterministic cap: keep the most recent where dates exist,
                # otherwise the head of the feed. Prevents one large feed from
                # dominating and skewing the class balance.
                dated = [r for r in rows if r["first_seen"]]
                if len(dated) >= src.max_rows:
                    dated.sort(key=lambda r: r["first_seen"], reverse=True)
                    rows = dated[:src.max_rows]
                else:
                    rows = rows[:src.max_rows]
            df = pd.DataFrame(rows, columns=SCHEMA)
            frames.append(df)
            meta["rows_parsed"] = len(rows)
            meta["label"] = src.label
            meta["label_confidence"] = src.label_confidence
            meta["has_timestamps"] = src.has_timestamps
            meta["notes"] = src.notes
            provenance.append(meta)
            print(f"  {src.key:<22} {len(rows):>7,} rows   "
                  f"[{src.label_confidence}]  collected {meta['collected_at']}")
        except Exception as exc:                      # noqa: BLE001
            print(f"  {src.key:<22} FAILED: {type(exc).__name__}: {exc}")
            provenance.append({
                "source_key": src.key, "error": f"{type(exc).__name__}: {exc}",
                "collected_at": _utcnow().isoformat(timespec="seconds"),
            })
    if not frames:
        raise SystemExit("No sources produced data; aborting.")
    return pd.concat(frames, ignore_index=True), provenance


# ------------------------------------------------------- dedupe + validate
def dedupe_and_validate(df, report):
    n0 = len(df)
    df = df[df["url"].map(normalize.looks_like_url)].copy()
    report["dropped_malformed"] = n0 - len(df)

    df["canonical"] = df["url"].map(canonicalize)
    df["trust"] = df["label_confidence"].map(trust_rank)

    # --- label conflicts across sources -----------------------------------
    # The same URL appearing as both benign and malicious is contradictory
    # ground truth. It is resolved by source trust, and the count is reported
    # rather than quietly averaged away.
    per = df.groupby("canonical")["label"].nunique()
    conflicting = set(per[per > 1].index)
    report["label_conflicts"] = len(conflicting)
    if conflicting:
        sub = df[df["canonical"].isin(conflicting)]
        pairs = Counter()
        for canon_url, grp in sub.groupby("canonical"):
            srcs = tuple(sorted(set(grp["source"])))
            pairs[srcs] += 1
        report["label_conflict_sources"] = {
            " + ".join(k): v for k, v in pairs.most_common(10)}

    # Keep the most trusted row per canonical URL; ties break on the earliest
    # first_seen so temporal ordering is not distorted by dedupe.
    df["_ts"] = df["first_seen"].map(_parse_ts)
    df = df.sort_values(["trust", "_ts"], ascending=[True, True], na_position="last")
    before = len(df)
    df = df.drop_duplicates(subset="canonical", keep="first").reset_index(drop=True)
    report["dropped_duplicates"] = before - len(df)

    df["registered_domain"] = registered_domains(df["url"]).to_numpy()
    return df, report


def cap_per_domain(df, max_per_domain, seed):
    """Limit rows per registered domain.

    One phishing campaign can contribute thousands of URLs on a single host.
    Left uncapped, the dataset measures how well the model memorises that one
    campaign. Capping also keeps the domain-grouped splits balanced.
    """
    if max_per_domain <= 0:
        return df, 0
    rng = np.random.default_rng(seed)
    keep = []
    for _dom, idx in df.groupby("registered_domain").groups.items():
        idx = np.asarray(list(idx))
        if len(idx) > max_per_domain:
            idx = rng.choice(idx, size=max_per_domain, replace=False)
        keep.extend(idx.tolist())
    out = df.loc[sorted(keep)].reset_index(drop=True)
    return out, len(df) - len(out)


# --------------------------------------------------------- confound checks
def structural_confound(df):
    """Can a single structural property separate the classes?

    Reports the strongest single-feature AUC over a handful of cheap
    structural properties. A value near 1.0 means the dataset is separable
    without any real phishing signal -- usually because the benign and
    malicious rows came from feeds with different shapes.
    """
    from sklearn.metrics import roc_auc_score
    from urllib.parse import urlparse

    def _parts(u):
        raw = u if "://" in u else "http://" + u
        try:
            p = urlparse(raw)
        except ValueError:
            return 0, 0, 0
        return len(p.path or ""), len(p.query or ""), int(bool(p.scheme == "https"))

    parts = [_parts(u) for u in df["url"]]
    # Concatenating frames can leave `label` as object dtype, which roc_auc_score
    # rejects with an opaque "unknown format" error.
    y = df["label"].astype(int).to_numpy()
    probes = {
        "path_length": np.array([p[0] for p in parts], dtype=float),
        "query_length": np.array([p[1] for p in parts], dtype=float),
        "is_https": np.array([p[2] for p in parts], dtype=float),
        "url_length": df["url"].str.len().to_numpy(dtype=float),
        "has_path": np.array([1.0 if p[0] > 1 else 0.0 for p in parts]),
    }
    out = {}
    for name, x in probes.items():
        if len(np.unique(x)) < 2:
            continue
        auc = roc_auc_score(y, x)
        out[name] = round(float(max(auc, 1 - auc)), 4)
    return dict(sorted(out.items(), key=lambda kv: -kv[1]))


def structure_stratum(url: str) -> tuple:
    """Coarse shape of a URL, used to neutralise feed-shape confounding.

    Only the gross structural properties that differ between FEEDS -- whether
    there is a path at all, whether there is a query, and a coarse length
    bucket. Anything finer would start matching on the very signal the model
    is supposed to learn.
    """
    from urllib.parse import urlparse
    raw = url if "://" in url else "http://" + url
    try:
        p = urlparse(raw)
        path, query = p.path or "", p.query or ""
    except ValueError:
        path, query = "", ""
    has_path = len(path) > 1
    has_query = len(query) > 0
    n = len(url)
    bucket = 0 if n < 40 else 1 if n < 80 else 2 if n < 160 else 3
    return (has_path, has_query, bucket)


def balance_structure(df, seed, report, max_auc, max_ratio=1.0):
    """Downsample within structural strata until shape stops predicting label.

    Reporting a 0.92 structural AUC and training anyway would mean knowingly
    publishing a model that learned its data sources. Within each stratum the
    larger class is downsampled to the smaller, so by construction the strata
    carry no label information; what survives is signal the structure cannot
    explain.

    This is expensive in rows, and that cost is reported rather than hidden.
    """
    rng = np.random.default_rng(seed)
    df = df.copy()
    df["_stratum"] = df["url"].map(structure_stratum)

    keep = []
    strata_report = []
    for stratum, grp in df.groupby("_stratum"):
        n0 = int((grp.label == 0).sum())
        n1 = int((grp.label == 1).sum())
        floor = min(n0, n1)
        # Strict 1:1 parity per stratum removes the confound completely but is
        # brutally wasteful: with only ~600 malicious bare roots available it
        # discarded ~19,000 benign popular domains, and the model then had
        # almost no evidence of what a legitimate short root looks like.
        # Allowing a bounded imbalance keeps far more of that evidence; the
        # residual confound it admits is measured and printed, not assumed
        # away.
        cap = int(floor * max_ratio) if floor else 0
        strata_report.append({"stratum": str(stratum), "benign": n0,
                              "malicious": n1, "floor": floor, "cap": cap})
        if floor == 0:
            continue
        for lab, n in ((0, n0), (1, n1)):
            idx = grp.index[grp.label == lab].to_numpy()
            limit = max(floor, min(n, cap))
            if len(idx) > limit:
                idx = rng.choice(idx, size=limit, replace=False)
            keep.extend(idx.tolist())

    before = len(df)
    out = df.loc[sorted(keep)].drop(columns="_stratum").reset_index(drop=True)
    report["structure_balancing"] = {
        "rows_before": before,
        "rows_after": len(out),
        "dropped": before - len(out),
        "target_max_auc": max_auc,
        "max_stratum_ratio": max_ratio,
        "strata": strata_report,
    }
    return out, report


def balance_scheme(df, seed, report, target=0.5):
    """Make http/https carry zero label information.

    Measured on the collected feeds: 9.0% of benign rows and 47.1% of malicious
    rows were https. That 5x gap is not a property of phishing -- it is a
    property of how each FEED stores URLs. Tranco publishes bare domains (this
    pipeline synthesised "http://" for them), Phishing.Database normalises
    everything to http, the project's hard negatives carry no scheme at all,
    and PhishTank preserves the original, which today is mostly https.

    Left alone, "https" became a 5x malicious indicator and the model scored
    https://github.com at 0.9995 while rating the same site's http form benign.

    So the scheme is reassigned at the SAME rate in both classes. This is not
    discarding evidence: for most of these rows the scheme was never evidence
    in the first place, it was a storage convention. `is_https` stays in the
    feature set and stays meaningful at inference time; it simply cannot be
    used as a shortcut during training.

    This mirrors the per-class scheme augmentation already in src/train.py,
    which exists for the same reason.
    """
    rng = np.random.default_rng(seed)
    out = df.copy()
    before = {}
    for lab in (0, 1):
        m = out["label"] == lab
        before[lab] = round(float(out.loc[m, "url"].str.startswith("https://").mean()), 4)

    stripped = out["url"].str.replace(r"^[a-zA-Z][a-zA-Z0-9+.\-]*://", "",
                                      regex=True)
    new_urls = stripped.copy()
    for lab in (0, 1):
        idx = out.index[out["label"] == lab].to_numpy()
        n_https = int(round(target * len(idx)))
        pick = rng.choice(idx, size=n_https, replace=False) if n_https else []
        is_https = np.zeros(len(out), dtype=bool)
        is_https[np.searchsorted(out.index.to_numpy(), pick)] = True
        new_urls.loc[pick] = "https://" + stripped.loc[pick]
        rest = np.setdiff1d(idx, pick)
        new_urls.loc[rest] = "http://" + stripped.loc[rest]
    out["url"] = new_urls

    after = {}
    for lab in (0, 1):
        m = out["label"] == lab
        after[lab] = round(float(out.loc[m, "url"].str.startswith("https://").mean()), 4)
    report["scheme_balancing"] = {
        "https_rate_before": {"benign": before[0], "malicious": before[1]},
        "https_rate_after": {"benign": after[0], "malicious": after[1]},
        "target": target,
        "why": ("Scheme distribution differed by source, not by threat; see "
                "balance_scheme() in build_dataset.py."),
    }
    return out, report


# ------------------------------------------------------------- splitting
def make_splits(df, args, report):
    from sklearn.model_selection import GroupShuffleSplit

    groups = df["registered_domain"].to_numpy()
    y = df["label"].to_numpy()
    idx = np.arange(len(df))

    # --- holdout: domain-disjoint, never touched by training or tuning ----
    gss = GroupShuffleSplit(n_splits=1, test_size=args.holdout_size,
                            random_state=args.seed)
    rest_i, hold_i = next(gss.split(idx, y, groups))
    assert not (set(groups[rest_i]) & set(groups[hold_i])), "holdout leaked"

    # --- grouped train/val from what remains ------------------------------
    gss2 = GroupShuffleSplit(n_splits=1, test_size=args.val_size,
                             random_state=args.seed)
    tr_rel, va_rel = next(gss2.split(rest_i, y[rest_i], groups[rest_i]))
    train_i, val_i = rest_i[tr_rel], rest_i[va_rel]
    assert not (set(groups[train_i]) & set(groups[val_i])), "val leaked"

    split = np.array(["train"] * len(df), dtype=object)
    split[val_i] = "val"
    split[hold_i] = "holdout"
    df["split"] = split

    report["split_grouped"] = {
        "train": int((split == "train").sum()),
        "val": int((split == "val").sum()),
        "holdout": int((split == "holdout").sum()),
        "shared_domains_train_val": 0,
        "shared_domains_train_holdout": 0,
    }

    # --- temporal split ---------------------------------------------------
    # Real deployment sees URLs that did not exist at training time. Where the
    # feeds carry dates, a cut-off split measures that directly.
    ts = df["first_seen"].map(_parse_ts)
    df["first_seen_ts"] = ts
    dated = ts.notna()
    report["rows_with_timestamps"] = int(dated.sum())

    df["temporal_split"] = "unusable"
    dated_mal = dated & (df["label"] == 1)
    report["dated_malicious"] = int(dated_mal.sum())
    report["dated_benign"] = int((dated & (df["label"] == 0)).sum())

    if dated_mal.sum() >= args.min_temporal_rows:
        # IMPORTANT, AND A REAL LIMITATION:
        # Only the malicious feeds publish dates. PhishTank gives
        # submission_time and URLhaus gives dateadded; the benign sources are
        # undated lists. A strict both-sides time split is therefore
        # impossible, and faking benign dates (e.g. using the collection
        # timestamp) would make every benign row share one date and turn the
        # experiment into nonsense.
        #
        # So the MALICIOUS side is time-ordered and the BENIGN side is
        # domain-disjoint but not time-ordered. That still answers the
        # question that matters operationally -- does the model still catch
        # phishing that appeared after it was trained? -- because drift here
        # is driven by attacker campaigns rotating, not by benign sites
        # changing shape. The asymmetry is recorded in the manifest so no
        # reader mistakes this for a full temporal split.
        cut = ts[dated_mal].quantile(args.temporal_quantile)
        temporal = np.where(dated_mal & (ts <= cut), "train",
                            np.where(dated_mal, "test", "unusable"))
        df["temporal_split"] = temporal

        # Benign rows: split by DOMAIN so the two sides stay disjoint.
        from sklearn.model_selection import GroupShuffleSplit as _GSS
        ben = np.where(df["label"].to_numpy() == 0)[0]
        if len(ben) > 50:
            frac_test = 1.0 - args.temporal_quantile
            g2 = _GSS(n_splits=1, test_size=frac_test, random_state=args.seed)
            b_tr, b_te = next(g2.split(ben, df["label"].to_numpy()[ben],
                                       groups[ben]))
            df.iloc[ben[b_tr], df.columns.get_loc("temporal_split")] = "train"
            df.iloc[ben[b_te], df.columns.get_loc("temporal_split")] = "test"

        # Domains straddling the cut-off would leak across it.
        tr_dom = set(df.loc[df.temporal_split == "train", "registered_domain"])
        straddle = df.temporal_split.eq("test") & \
            df["registered_domain"].isin(tr_dom)
        df.loc[straddle, "temporal_split"] = "unusable"
        report["temporal"] = {
            "cutoff": cut.isoformat(),
            "train": int((df.temporal_split == "train").sum()),
            "test": int((df.temporal_split == "test").sum()),
            "train_malicious": int(((df.temporal_split == "train") &
                                    (df.label == 1)).sum()),
            "test_malicious": int(((df.temporal_split == "test") &
                                   (df.label == 1)).sum()),
            "dropped_straddling_domains": int(straddle.sum()),
            "quantile": args.temporal_quantile,
            "time_ordered_side": "malicious only",
            "caveat": ("Only the malicious feeds carry timestamps. The benign "
                       "side is domain-disjoint but NOT time-ordered, so this "
                       "measures drift in phishing campaigns rather than a "
                       "full both-sides temporal shift."),
        }
    else:
        report["temporal"] = {
            "available": False,
            "reason": (f"only {int(dated_mal.sum())} dated malicious rows "
                       f"(need {args.min_temporal_rows})"),
        }
    return df, report


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scope", default="url", choices=("url", "hostname"),
                    help="which dataset to build. Sources declare the scope "
                         "their rows are shaped for; see datasets/sources.py.")
    ap.add_argument("--out", default=None)
    ap.add_argument("--manifest", default=None)
    ap.add_argument("--cache", default="data/raw")
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-per-domain", type=int, default=5)
    ap.add_argument("--val-size", type=float, default=0.15)
    ap.add_argument("--holdout-size", type=float, default=0.15)
    ap.add_argument("--temporal-quantile", type=float, default=0.70)
    ap.add_argument("--min-temporal-rows", type=int, default=2000)
    ap.add_argument("--balance", action="store_true", default=True,
                    help="downsample the larger class toward parity")
    ap.add_argument("--no-balance", dest="balance", action="store_false")
    ap.add_argument("--max-stratum-ratio", type=float, default=1.0,
                    help="largest benign:malicious imbalance tolerated inside "
                         "one structural stratum.\n"
                         "Measured on this corpus (ratio -> strongest "
                         "structural AUC / rows / holdout FPR / false "
                         "positives on a fixed set of 17 well-known sites):\n"
                         "  1.0 -> 0.61 / 21.8k / 0.0262 / 5\n"
                         "  2.0 -> 0.76 / 32.6k / 0.0402 / 7\n"
                         "  3.0 -> 0.82 / 42.5k\n"
                         "  6.0 -> 0.89 / 67.2k\n"
                         "Strict parity keeps the fewest rows and still wins "
                         "on real sites: removing the shape confound matters "
                         "more here than dataset size.")
    ap.add_argument("--scheme-rate", type=float, default=0.5,
                    help="https rate forced onto BOTH classes, so scheme cannot "
                         "stand in for the label (it is a feed artifact here)")
    ap.add_argument("--max-structure-auc", type=float, default=0.75,
                    help="if a single structural property separates the classes "
                         "above this AUC, stratify until it does not")
    args = ap.parse_args()
    if args.out is None:
        args.out = (f"data/urlguard_dataset.csv" if args.scope == "url"
                    else f"data/urlguard_dataset_hostname.csv")
    if args.manifest is None:
        args.manifest = (f"data/dataset_manifest.json" if args.scope == "url"
                         else f"data/dataset_manifest_hostname.json")

    report = {"scope": args.scope}
    print("=" * 78)
    print("URLGUARD MULTI-SOURCE DATASET BUILD")
    print("=" * 78)
    print("\n[1] Collecting sources")
    df, provenance = collect(args)
    df["label"] = pd.to_numeric(df["label"], errors="coerce")
    df = df[df["label"].isin([0, 1])].copy()
    df["label"] = df["label"].astype(int)
    report["rows_collected"] = len(df)

    print(f"\n[2] Normalised schema: {len(df):,} rows")
    print(f"    columns: {', '.join(SCHEMA)}")

    if args.scope == "hostname":
        # A DNS query carries a hostname and nothing else: no scheme, no path,
        # no query. Reducing here -- before dedupe -- is deliberate, because
        # many URLs collapse onto one host and counting them separately would
        # inflate every metric for the hostname model.
        import re as _re
        def _host(u):
            v = _re.sub(r"^[a-zA-Z][a-zA-Z0-9+.\-]*://", "", str(u).strip())
            v = v.split("/", 1)[0].split("?", 1)[0].split("#", 1)[0]
            return v.rsplit("@", 1)[-1].split(":", 1)[0].strip().lower()
        before_h = len(df)
        df["url"] = df["url"].map(_host)
        df = df[df["url"].str.len() > 0].reset_index(drop=True)
        print(f"\n[2b] Reduced {before_h:,} URLs to hostnames "
              f"({df['url'].nunique():,} distinct)")

    print("\n[3] De-duplication and label validation")
    df, report = dedupe_and_validate(df, report)
    print(f"    malformed dropped      {report['dropped_malformed']:>8,}")
    print(f"    duplicates dropped     {report['dropped_duplicates']:>8,}")
    print(f"    label conflicts        {report['label_conflicts']:>8,}")
    if report.get("label_conflict_sources"):
        for k, v in report["label_conflict_sources"].items():
            print(f"      {k}: {v}")
    print(f"    rows remaining         {len(df):>8,}")

    print(f"\n[4] Per-domain cap ({args.max_per_domain} rows/domain)")
    df, capped = cap_per_domain(df, args.max_per_domain, args.seed)
    report["dropped_domain_cap"] = capped
    print(f"    dropped                {capped:>8,}")
    print(f"    rows remaining         {len(df):>8,}")
    print(f"    distinct domains       {df['registered_domain'].nunique():>8,}")

    if args.balance:
        counts = df["label"].value_counts()
        if len(counts) == 2:
            target = int(counts.min())
            rng = np.random.default_rng(args.seed)
            keep = []
            for lab, grp in df.groupby("label"):
                gi = grp.index.to_numpy()
                if len(gi) > target:
                    gi = rng.choice(gi, size=target, replace=False)
                keep.extend(gi.tolist())
            before = len(df)
            df = df.loc[sorted(keep)].reset_index(drop=True)
            report["dropped_balancing"] = before - len(df)
            print(f"\n[5] Class balancing -> {len(df):,} rows "
                  f"({before - len(df):,} dropped)")

    print("\n[6] Structural confound check "
          "(single-property AUC; ~1.0 means the feeds, not phishing, "
          "are separable)")
    conf = structural_confound(df)
    report["structural_confound_auc"] = conf
    for k, v in conf.items():
        flag = "  <-- DOMINATES" if v > 0.90 else ""
        print(f"    {k:<16} {v:.4f}{flag}")
    worst = max(conf.values()) if conf else 0.0
    if worst > args.max_structure_auc:
        print(f"    Strongest structural AUC {worst:.4f} exceeds the "
              f"{args.max_structure_auc} ceiling.")
        print("    Stratifying by URL shape so the feeds cannot stand in for "
              "the label ...")
        df, report = balance_structure(df, args.seed, report,
                                       args.max_structure_auc,
                                       args.max_stratum_ratio)
        sb = report["structure_balancing"]
        print(f"    {sb['rows_before']:,} -> {sb['rows_after']:,} rows "
              f"({sb['dropped']:,} dropped)")
        conf_after = structural_confound(df)
        report["structural_confound_auc_after"] = conf_after
        print("    after stratification:")
        for k, v in conf_after.items():
            print(f"      {k:<16} {v:.4f}")
        worst_after = max(conf_after.values()) if conf_after else 0.0
        report["structural_confound_resolved"] = \
            bool(worst_after <= args.max_structure_auc)
        if worst_after > args.max_structure_auc:
            print(f"      NOTE: {worst_after:.4f} still exceeds the ceiling; "
                  f"residual shape difference is reported, not hidden.")
    else:
        report["structural_confound_resolved"] = True

    print("\n[6b] Scheme balancing")
    df, report = balance_scheme(df, args.seed, report, args.scheme_rate)
    sb = report["scheme_balancing"]
    print(f"    https rate  benign {sb['https_rate_before']['benign']:.1%} -> "
          f"{sb['https_rate_after']['benign']:.1%}")
    print(f"                malicious {sb['https_rate_before']['malicious']:.1%} -> "
          f"{sb['https_rate_after']['malicious']:.1%}")
    df, _ = drop_exact_duplicates(df)
    conf2 = structural_confound(df)
    report["structural_confound_auc_final"] = conf2
    print("    structural AUC after scheme balancing:")
    for k, v in conf2.items():
        print(f"      {k:<16} {v:.4f}")

    n_pos = int(df["label"].sum())
    report["rows_final"] = len(df)
    report["n_malicious"] = n_pos
    report["n_benign"] = len(df) - n_pos
    report["by_source"] = df["source"].value_counts().to_dict()
    report["by_confidence"] = df["label_confidence"].value_counts().to_dict()

    print("\n[7] Leakage-safe splits")
    df, report = make_splits(df, args, report)
    g = report["split_grouped"]
    print(f"    grouped   train {g['train']:,} / val {g['val']:,} "
          f"/ holdout {g['holdout']:,}   (0 shared domains)")
    t = report["temporal"]
    if t.get("available") is False:
        print(f"    temporal  unavailable: {t['reason']}")
    else:
        print(f"    temporal  cutoff {t['cutoff']}")
        print(f"              train {t['train']:,} "
              f"({t['train_malicious']:,} malicious) / "
              f"test {t['test']:,} ({t['test_malicious']:,} malicious)")
        print(f"              {t['dropped_straddling_domains']:,} straddling "
              f"rows dropped; time-ordered side: {t['time_ordered_side']}")

    # ---- write ----------------------------------------------------------
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    cols = SCHEMA + ["registered_domain", "split", "temporal_split"]
    df[cols].to_csv(args.out, index=False)

    version = f"URLGuard-MultiSource-{_utcnow():%Y-%m-%d}"
    manifest = {
        "dataset_version": version,
        "built_at": _utcnow().isoformat(timespec="seconds"),
        "output": args.out,
        "sha256": hashlib.sha256(open(args.out, "rb").read()).hexdigest(),
        "schema": cols,
        "parameters": {
            "seed": args.seed, "max_per_domain": args.max_per_domain,
            "val_size": args.val_size, "holdout_size": args.holdout_size,
            "temporal_quantile": args.temporal_quantile,
            "balanced": args.balance,
        },
        "counts": report,
        "sources": provenance,
        "leakage_controls": [
            "Canonical de-duplication (scheme, www., case, trailing slash).",
            "Splits grouped by registered domain; 0 domains shared across "
            "train/val/holdout.",
            f"At most {args.max_per_domain} rows per registered domain, so a "
            "single campaign cannot dominate.",
            "Temporal split drops domains that straddle the cut-off.",
            "Holdout slice is domain-disjoint and excluded from all tuning.",
            "Structural confound AUC reported so feed shape cannot silently "
            "stand in for the label.",
        ],
    }
    with open(args.manifest, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, default=str)

    print(f"\n[8] Wrote {args.out}  ({len(df):,} rows)")
    print(f"    {n_pos:,} malicious / {len(df) - n_pos:,} benign")
    print(f"    manifest -> {args.manifest}")
    print(f"    version  -> {version}")
    print("=" * 78)


if __name__ == "__main__":
    main()
