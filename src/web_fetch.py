"""
Controlled, non-interactive retrieval of page HTML.

READ THIS BEFORE CHANGING ANYTHING HERE
---------------------------------------
Every other part of URLGuard analyses URLs as text and never contacts the
address. This module is the ONE place that deliberately breaks that, so it is
also the one place where getting the sandbox wrong turns the scanner into an
attack tool. The guarantees are:

1. NO CODE EXECUTION. The response is parsed as text. No browser, no
   JavaScript engine, no DOM execution, no plugin. A phishing page cannot run
   anything by being analysed.

2. NO INTERACTION. GET only. No form is ever submitted, no credential is ever
   sent, no cookie is stored or returned, no authentication header is attached.
   Nothing on the page is "clicked".

3. NO SSRF. The scanner takes a URL from an untrusted user, so without this it
   would happily fetch http://169.254.169.254/ (cloud metadata),
   http://127.0.0.1:5000/api/... (its own admin surface) or anything inside a
   private network. EVERY hostname is resolved and EVERY resolved address is
   checked against loopback, private, link-local, multicast and reserved
   ranges -- before connecting, and again after each redirect.

4. BOUNDED. Hard caps on bytes, time, and redirect hops, so a malicious server
   cannot exhaust the scanner with an endless or enormous response.

5. ANNOUNCED. The User-Agent identifies the request as an automated security
   scan. Fetching is opt-in per scan and is never performed by the Android
   interception path or the DNS layer.

What is deliberately NOT mitigated, and is documented rather than hidden:
DNS rebinding. Addresses are validated at resolution time; a server that
returns a public address to the validating lookup and a private one to the
connecting lookup could still slip through. Closing that needs connect-time
address pinning with TLS SNI handling, which is noted as a limitation.
"""

import gzip
import ipaddress
import socket
import ssl
import time
import urllib.error
import urllib.parse
import urllib.request
import zlib

USER_AGENT = ("URLGuard-Scanner/3.1 (+automated security analysis; "
              "no javascript executed; GET only)")

MAX_BYTES = 1_500_000        # 1.5 MB of HTML is already far past typical
MAX_REDIRECTS = 5
CONNECT_TIMEOUT = 8
READ_TIMEOUT = 12
ALLOWED_SCHEMES = ("http", "https")

# Content we are willing to parse. Anything else is recorded and discarded --
# there is no reason for a page analyser to pull down a 2 GB video.
TEXTUAL_TYPES = ("text/html", "application/xhtml+xml", "text/plain")


class FetchError(Exception):
    """Fetching failed. Always non-fatal: callers fall back to URL-only."""

    def __init__(self, reason, code="fetch_failed"):
        super().__init__(reason)
        self.reason = reason
        self.code = code


def _is_ip_literal(host: str) -> bool:
    """True only when the host is already an IP address, not a name."""
    try:
        ipaddress.ip_address(host.strip("[]"))
        return True
    except ValueError:
        return False


def _is_blocked_ip(ip_str: str) -> bool:
    try:
        ip = ipaddress.ip_address(ip_str.strip("[]"))
    except ValueError:
        # Unparseable: fail closed. Only ever called on addresses that came
        # back from the resolver, so this means something is badly wrong.
        return True
    return (ip.is_private or ip.is_loopback or ip.is_link_local
            or ip.is_multicast or ip.is_reserved or ip.is_unspecified)


def resolve_and_check(host: str):
    """Resolve a hostname and reject it if ANY address is internal.

    All addresses are checked, not just the first: a host that resolves to one
    public and one loopback address must not be fetchable, because which one
    gets connected to is not under our control.
    """
    try:
        infos = socket.getaddrinfo(host, None, proto=socket.IPPROTO_TCP)
    except socket.gaierror as exc:
        raise FetchError(f"DNS resolution failed for {host}", "dns_failed") from exc
    addrs = sorted({i[4][0] for i in infos})
    if not addrs:
        raise FetchError(f"No addresses for {host}", "dns_failed")
    blocked = [a for a in addrs if _is_blocked_ip(a)]
    if blocked:
        raise FetchError(
            f"{host} resolves to a non-public address ({blocked[0]}); refused "
            f"to prevent server-side request forgery", "blocked_address")
    return addrs


def _check_url(url: str):
    parts = urllib.parse.urlsplit(url)
    if parts.scheme.lower() not in ALLOWED_SCHEMES:
        raise FetchError(f"Scheme '{parts.scheme}' is not fetchable "
                         f"(http/https only)", "bad_scheme")
    host = parts.hostname
    if not host:
        raise FetchError("No hostname to fetch", "bad_url")
    # A literal internal IP never even reaches DNS. Note this must NOT use
    # _is_blocked_ip, which answers "block it" for anything unparseable --
    # correct for a resolved address, but it would reject every hostname.
    if _is_ip_literal(host) and _is_blocked_ip(host):
        raise FetchError(f"{host} is a non-public address; refused",
                         "blocked_address")
    addrs = resolve_and_check(host)
    return parts, addrs


def _decode_body(raw: bytes, encoding: str, charset: str) -> str:
    if encoding in ("gzip", "x-gzip"):
        try:
            raw = gzip.decompress(raw)
        except (OSError, EOFError):
            raw = zlib.decompressobj(16 + zlib.MAX_WBITS).decompress(raw)
    elif encoding == "deflate":
        try:
            raw = zlib.decompress(raw)
        except zlib.error:
            raw = zlib.decompressobj(-zlib.MAX_WBITS).decompress(raw)
    for enc in (charset, "utf-8", "latin-1"):
        if not enc:
            continue
        try:
            return raw.decode(enc, "replace")
        except (LookupError, UnicodeDecodeError):
            continue
    return raw.decode("utf-8", "replace")


def fetch(url: str, max_bytes: int = MAX_BYTES,
          max_redirects: int = MAX_REDIRECTS, verify_tls: bool = True) -> dict:
    """Retrieve page HTML safely. Never raises for ordinary failure modes
    other than FetchError, which callers are expected to treat as 'no page
    data available' rather than as a verdict.

    Returns a dict with html, status, final_url, redirect chain, timing and
    TLS details. Redirects are followed MANUALLY so every hop is re-validated;
    urllib's automatic handler would happily follow a public URL to
    http://127.0.0.1/.
    """
    started = time.perf_counter()
    chain = []
    current = url if "://" in url else "http://" + url
    tls_ok = None
    cert_error = None

    ctx = ssl.create_default_context()
    if not verify_tls:
        # Phishing sites frequently have broken or self-signed certificates.
        # Refusing them would mean refusing to analyse exactly the pages most
        # worth analysing, so the failure is RECORDED as a signal instead.
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE

    for hop in range(max_redirects + 1):
        parts, _addrs = _check_url(current)
        req = urllib.request.Request(current, method="GET", headers={
            "User-Agent": USER_AGENT,
            "Accept": "text/html,application/xhtml+xml;q=0.9,*/*;q=0.1",
            "Accept-Language": "en",
            "Accept-Encoding": "gzip, deflate",
            # No cookies are sent and none are kept: a redirect handler that
            # carried them could leak one site's session to another.
            "Cache-Control": "no-cache",
        })
        opener = urllib.request.build_opener(
            urllib.request.HTTPSHandler(context=ctx),
            _NoRedirect(),              # we follow them ourselves
            _NoCookies(),
        )
        try:
            resp = opener.open(req, timeout=READ_TIMEOUT)
            status = resp.status
        except urllib.error.HTTPError as exc:
            resp, status = exc, exc.code
        except ssl.SSLCertVerificationError as exc:
            if verify_tls:
                # Retry once without verification, and remember why.
                return fetch(url, max_bytes, max_redirects, verify_tls=False) | {
                    "tls_verified": False,
                    "tls_error": str(exc.verify_message or exc),
                }
            raise FetchError(f"TLS error: {exc}", "tls_error") from exc
        except (urllib.error.URLError, socket.timeout, TimeoutError,
                ConnectionError, OSError) as exc:
            raise FetchError(f"Could not reach {parts.hostname}: "
                             f"{type(exc).__name__}", "unreachable") from exc

        if verify_tls and parts.scheme == "https":
            tls_ok = True

        location = resp.headers.get("Location")
        if status in (301, 302, 303, 307, 308) and location:
            nxt = urllib.parse.urljoin(current, location)
            chain.append({"from": current, "to": nxt, "status": status})
            resp.close()
            current = nxt
            if hop == max_redirects:
                raise FetchError(f"More than {max_redirects} redirects",
                                 "too_many_redirects")
            continue

        ctype = (resp.headers.get("Content-Type") or "").lower()
        mime = ctype.split(";")[0].strip()
        charset = ""
        if "charset=" in ctype:
            charset = ctype.split("charset=", 1)[1].split(";")[0].strip()

        raw = resp.read(max_bytes + 1)
        truncated = len(raw) > max_bytes
        raw = raw[:max_bytes]
        enc = (resp.headers.get("Content-Encoding") or "").lower()
        resp.close()

        if mime and not any(mime.startswith(t) for t in TEXTUAL_TYPES):
            raise FetchError(f"Response is '{mime}', not an HTML page",
                             "not_html")

        html = _decode_body(raw, enc, charset)
        return {
            "ok": True,
            "html": html,
            "status": status,
            "final_url": current,
            "requested_url": url,
            "redirects": chain,
            "redirect_count": len(chain),
            "bytes": len(raw),
            "truncated": truncated,
            "content_type": mime,
            "elapsed_ms": round((time.perf_counter() - started) * 1000, 1),
            "tls_verified": tls_ok,
            "tls_error": cert_error,
            "fetched_at": time.time(),
        }

    raise FetchError("Redirect limit reached", "too_many_redirects")


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    """Disables urllib's automatic redirect following.

    Automatic redirects are an SSRF hole: a public URL can 302 to
    http://169.254.169.254/ and the handler would follow it without any of the
    address checks above ever running again.
    """

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


class _NoCookies(urllib.request.HTTPCookieProcessor):
    def __init__(self):
        super().__init__(cookiejar=_NullJar())


class _NullJar:
    def add_cookie_header(self, request):
        pass

    def extract_cookies(self, response, request):
        pass
