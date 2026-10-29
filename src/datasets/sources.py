"""
Declarative registry of the URL feeds URLGuard trains on.

WHY A REGISTRY RATHER THAN INLINE URLS
--------------------------------------
Every row in the final dataset has to be traceable to a source, a licence and a
collection date. Keeping that metadata next to the download URL -- instead of
scattered through fetch code -- makes it impossible to add a feed without also
declaring where it came from and what may legally be done with it.

ON SCOPE
--------
`scope` records which task a source's ROWS ARE SHAPED FOR, and it exists
because of a measured failure. Tranco publishes bare domains; the phishing
feeds publish deep URLs. Mixed naively into one full-URL dataset, "does this
URL have a path" separated the classes at AUC 0.92 -- benign 11% vs malicious
96%. A model trained on that learns which FEED a row came from, not what
phishing looks like.

So Tranco is scoped to `hostname`, where a bare domain is exactly the right
shape (a DNS query carries nothing else), and the full-URL dataset draws its
benign rows from sources that contain real deep links.

ON LABEL TRUST
--------------
`label_confidence` is NOT a guess about how dangerous a URL is. It records how
much the SOURCE's labelling process can be relied on:

  verified   a human or automated pipeline confirmed the classification
             (PhishTank entries marked verified=yes; URLhaus malware URLs)
  reported   submitted to a feed but not independently confirmed
  presumed   inferred from a proxy, never confirmed -- Tranco ranks domains by
             POPULARITY, not safety. A popular domain is not a safe domain, and
             nothing downstream may describe Tranco rows as "verified benign".

This distinction exists because the brief is explicit: do not claim a URL is
malicious merely because it appears in a third-party list.
"""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Source:
    key: str
    name: str
    url: str
    label: int                 # 1 = malicious/phishing, 0 = benign
    label_confidence: str      # verified | reported | presumed
    licence: str
    licence_url: str
    attribution: str
    fmt: str                   # csv | csv.gz | txt | zip
    has_timestamps: bool
    notes: str = ""
    max_rows: int = 0          # 0 = no cap
    scope: str = "both"        # url | hostname | both -- see note below
    enabled: bool = True
    extra: dict = field(default_factory=dict)


SOURCES = [
    Source(
        key="phishtank",
        name="PhishTank (verified online)",
        url="https://data.phishtank.com/data/online-valid.csv.gz",
        label=1,
        label_confidence="verified",
        licence="Free for research/non-commercial use with attribution",
        licence_url="https://phishtank.org/api_info.php",
        attribution="Data from PhishTank (OpenDNS/Cisco), phishtank.org",
        fmt="csv.gz",
        has_timestamps=True,
        notes=("Only rows with verified=yes are kept. The feed also carries "
               "submission_time, which is what makes temporal validation "
               "possible."),
        max_rows=40000,
    ),
    Source(
        key="urlhaus",
        name="abuse.ch URLhaus (full dump)",
        url="https://urlhaus.abuse.ch/downloads/csv/",
        label=1,
        label_confidence="verified",
        licence="CC0 1.0 (public domain dedication)",
        licence_url="https://urlhaus.abuse.ch/api/#tos",
        attribution="Data from abuse.ch URLhaus",
        fmt="zip",
        has_timestamps=True,
        notes=("Malware-distribution URLs rather than credential phishing. "
               "The FULL dump is used rather than csv_recent: it carries far "
               "more dated rows, which is what makes the temporal experiment "
               "possible. Only entries with status=online are kept."),
        max_rows=30000,
    ),
    Source(
        key="openphish",
        name="OpenPhish community feed",
        url="https://openphish.com/feed.txt",
        label=1,
        label_confidence="reported",
        licence="Free community feed; research use with attribution, "
                "redistribution restricted",
        licence_url="https://openphish.com/terms.html",
        attribution="Data from the OpenPhish community feed",
        fmt="txt",
        has_timestamps=False,
        notes=("Plain URL list with no timestamps and no per-entry "
               "verification metadata, so rows are labelled 'reported'."),
        max_rows=20000,
    ),
    Source(
        key="phishing_database",
        name="Phishing.Database (active links)",
        url=("https://raw.githubusercontent.com/mitchellkrogza/"
             "Phishing.Database/master/phishing-links-ACTIVE.txt"),
        label=1,
        label_confidence="reported",
        licence="MIT",
        licence_url=("https://github.com/mitchellkrogza/Phishing.Database/"
                     "blob/master/LICENSE"),
        attribution="Phishing.Database by Mitchell Krog (MIT)",
        fmt="txt",
        has_timestamps=False,
        notes=("Aggregated from several upstream feeds. Treated as 'reported' "
               "because entries are not individually verified here."),
        max_rows=25000,
    ),
    Source(
        key="tranco",
        name="Tranco top sites",
        url="https://tranco-list.eu/top-1m.csv.zip",
        label=0,
        label_confidence="presumed",
        licence="Free for research; cite the Tranco paper",
        licence_url="https://tranco-list.eu/",
        attribution=("Tranco list (Le Pochat et al., NDSS 2019), "
                     "https://tranco-list.eu/"),
        fmt="zip",
        has_timestamps=False,
        notes=("POPULARITY ranking, not a safety judgement. Popular domains "
               "do host compromised pages. Rows are 'presumed' benign and "
               "must never be described as verified-safe. Also: Tranco gives "
               "bare DOMAINS, so using it alone would make 'has a path' a "
               "perfect benign/malicious separator -- see the structural "
               "confound check in build_dataset.py."),
        max_rows=40000,
        # Scoped to BOTH after a measured regression. Excluding Tranco from the
        # full-URL set removed every popular short domain from the benign
        # class, leaving only long-tail blog and forum URLs -- and the model
        # then scored github.com at 0.92 and stackoverflow.com/questions at
        # 0.81. The shape confound Tranco introduces is handled properly by
        # the structural stratification in build_dataset.py, which is the
        # right tool; dropping the source was a blunt one that traded a
        # measurable confound for a worse, less visible failure.
        scope="both",
    ),
    Source(
        key="local_corpus",
        name="Existing URLGuard corpus",
        url="",                       # local file, not downloaded
        label=-1,                     # labels come from the file itself
        label_confidence="reported",
        licence="As per the original upstream datasets (see README section 3)",
        licence_url="",
        attribution="Existing project corpus (data/real_urls.csv)",
        fmt="local",
        has_timestamps=False,
        notes=("Kept so the new dataset is a superset of what the previous "
               "model was validated on, and so benign rows include real deep "
               "links rather than only Tranco roots."),
    ),
    Source(
        key="local_hard_negatives",
        name="Existing hard negatives",
        url="",
        label=0,
        label_confidence="reported",
        licence="As per the original upstream dataset",
        licence_url="",
        attribution="Existing project hard negatives (data/hard_negatives.csv)",
        fmt="local",
        has_timestamps=False,
        notes=("Benign URLs that look phishing-like. Essential: without them "
               "the benign class is dominated by short clean roots."),
    ),
]

BY_KEY = {s.key: s for s in SOURCES}

# Ordered most- to least-trusted. Used to resolve a URL appearing in several
# feeds with conflicting labels.
TRUST_ORDER = ["verified", "reported", "presumed"]


def trust_rank(confidence: str) -> int:
    try:
        return TRUST_ORDER.index(confidence)
    except ValueError:
        return len(TRUST_ORDER)
