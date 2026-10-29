"""
Cached downloads that record their own provenance.

Every fetch writes a sidecar `.meta.json` next to the payload holding the URL,
the HTTP status, the byte count, a SHA-256 and the UTC timestamp of collection.
That sidecar is what lets the dataset manifest state *when* each feed was
collected rather than when the build happened to run.

Downloads are cached by default: rebuilding the dataset should not re-hammer
volunteer-run feeds, and a reproducible build needs the same bytes.
"""

import datetime as _dt
import hashlib
import io
import json
import os
import urllib.request
import zipfile

USER_AGENT = "URLGuard-research/2.0 (+dataset pipeline; contact: local)"
DEFAULT_CACHE = os.path.join("data", "raw")
TIMEOUT = 180


def _utcnow():
    return _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")


def cache_paths(cache_dir, key):
    return (os.path.join(cache_dir, f"{key}.bin"),
            os.path.join(cache_dir, f"{key}.meta.json"))


def fetch(source, cache_dir=DEFAULT_CACHE, refresh=False):
    """Download a source, or reuse the cached copy.

    Returns (payload_bytes, meta_dict). Raises on network failure so a build
    cannot silently proceed with a missing feed.
    """
    os.makedirs(cache_dir, exist_ok=True)
    blob_path, meta_path = cache_paths(cache_dir, source.key)

    if not refresh and os.path.exists(blob_path) and os.path.exists(meta_path):
        with open(meta_path, encoding="utf-8") as fh:
            meta = json.load(fh)
        with open(blob_path, "rb") as fh:
            return fh.read(), meta

    # Volunteer-run feeds rate-limit. Back off rather than hammering them, and
    # let the caller fall back to the cache if one stays unavailable.
    import time
    import urllib.error
    last_err = None
    payload = status = None
    for attempt, delay in enumerate((0, 15, 45), start=1):
        if delay:
            time.sleep(delay)
        req = urllib.request.Request(source.url,
                                     headers={"User-Agent": USER_AGENT})
        try:
            with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
                payload = resp.read()
                status = resp.status
            break
        except urllib.error.HTTPError as exc:
            last_err = exc
            if exc.code not in (429, 503) or attempt == 3:
                raise
        except (urllib.error.URLError, TimeoutError) as exc:
            last_err = exc
            if attempt == 3:
                raise
    if payload is None:
        raise last_err or RuntimeError(f"could not fetch {source.url}")

    meta = {
        "source_key": source.key,
        "source_name": source.name,
        "url": source.url,
        "http_status": status,
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "collected_at": _utcnow(),
        "licence": source.licence,
        "attribution": source.attribution,
    }
    with open(blob_path, "wb") as fh:
        fh.write(payload)
    with open(meta_path, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)
    return payload, meta


def decode(source, payload):
    """Turn raw bytes into text, handling gzip and zip containers."""
    if source.fmt == "csv.gz":
        import gzip
        try:
            return gzip.decompress(payload).decode("utf-8", "replace")
        except (OSError, EOFError):
            # Truncated stream: salvage what decompressed before the cut.
            import zlib
            dec = zlib.decompressobj(16 + zlib.MAX_WBITS)
            return dec.decompress(payload).decode("utf-8", "replace")
    if source.fmt == "zip":
        with zipfile.ZipFile(io.BytesIO(payload)) as zf:
            name = zf.namelist()[0]
            return zf.read(name).decode("utf-8", "replace")
    return payload.decode("utf-8", "replace")
