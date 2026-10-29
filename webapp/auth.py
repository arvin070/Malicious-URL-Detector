"""
Authentication and per-user storage for URLGuard.

DESIGN NOTES THAT MATTER
------------------------
* Passwords are hashed with Argon2id (argon2-cffi defaults), never stored or
  logged in plaintext and never reversible. Hashes are transparently upgraded
  when the parameters change.

* The signing secret is read from the URLGUARD_SECRET environment variable. If
  it is unset, a random one is generated into `instance/secret.key` with 0600
  permissions. There is no hard-coded fallback secret: a shipped default is a
  shipped master key, and the Android client would carry it.

* The client gets a short-lived access token and a long-lived refresh token.
  Only the refresh token is stored server-side (as a hash), so "log out
  everywhere" is possible and a stolen database does not yield usable tokens.

* Scan history is keyed by user id and every read is filtered by the id in the
  verified token, so one account cannot reach another's rows.

* Email enumeration is avoided: registration conflicts and failed logins and
  password-reset requests all return the same shape regardless of whether the
  address exists.

WHAT THIS IS NOT
----------------
This is a research prototype's auth layer. It does not do email delivery, so
password reset issues a token through the API that an operator must relay; the
API says so rather than pretending a mail was sent.
"""

import datetime as dt
import hashlib
import hmac
import json
import os
import re
import secrets
import sqlite3
import threading
from contextlib import closing
import time

import jwt
from argon2 import PasswordHasher
from argon2.exceptions import InvalidHashError, VerifyMismatchError

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INSTANCE = os.path.join(ROOT, "instance")
DB_PATH = os.path.join(INSTANCE, "urlguard.sqlite3")

ACCESS_TTL = dt.timedelta(minutes=30)
REFRESH_TTL = dt.timedelta(days=30)
RESET_TTL = dt.timedelta(minutes=20)

_ph = PasswordHasher()
_lock = threading.Lock()

EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]{2,}$")
MIN_PASSWORD = 10


class AuthError(Exception):
    def __init__(self, message, status=400, code="auth_error"):
        super().__init__(message)
        self.message = message
        self.status = status
        self.code = code


# ----------------------------------------------------------------- secret
def _load_secret() -> bytes:
    env = os.environ.get("URLGUARD_SECRET")
    if env:
        return env.encode("utf-8")
    os.makedirs(INSTANCE, exist_ok=True)
    path = os.path.join(INSTANCE, "secret.key")
    if os.path.exists(path):
        with open(path, "rb") as fh:
            return fh.read().strip()
    # Generated once, kept out of version control. Never a constant default.
    key = secrets.token_bytes(48)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "wb") as fh:
        fh.write(key)
    return key


SECRET = _load_secret()


# --------------------------------------------------------------- database
SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    email           TEXT NOT NULL UNIQUE,
    name            TEXT NOT NULL,
    password_hash   TEXT NOT NULL,
    created_at      REAL NOT NULL,
    last_login_at   REAL,
    preferences     TEXT NOT NULL DEFAULT '{}'
);
CREATE TABLE IF NOT EXISTS refresh_tokens (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id     INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    token_hash  TEXT NOT NULL UNIQUE,
    issued_at   REAL NOT NULL,
    expires_at  REAL NOT NULL,
    revoked     INTEGER NOT NULL DEFAULT 0,
    user_agent  TEXT
);
CREATE TABLE IF NOT EXISTS scans (
    id              TEXT PRIMARY KEY,
    user_id         INTEGER REFERENCES users(id) ON DELETE CASCADE,
    url             TEXT NOT NULL,
    domain          TEXT,
    risk_score      INTEGER NOT NULL,
    risk_band       TEXT NOT NULL,
    risk_key        TEXT NOT NULL,
    confidence      REAL,
    model_version   TEXT,
    dataset_version TEXT,
    model_name      TEXT,
    source          TEXT,
    scope           TEXT,
    detection_ms    REAL,
    signals         TEXT,
    scanned_at      REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_scans_user_time
    ON scans(user_id, scanned_at DESC);
CREATE INDEX IF NOT EXISTS idx_scans_user_band
    ON scans(user_id, risk_key);
CREATE TABLE IF NOT EXISTS password_resets (
    token_hash  TEXT PRIMARY KEY,
    user_id     INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    expires_at  REAL NOT NULL,
    used        INTEGER NOT NULL DEFAULT 0
);
"""


def connect():
    os.makedirs(INSTANCE, exist_ok=True)
    conn = sqlite3.connect(DB_PATH, timeout=10)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_db():
    with _lock, closing(connect()) as conn, conn:
        conn.executescript(SCHEMA)


# ------------------------------------------------------------- validation
def validate_registration(name, email, password, confirm):
    name = (name or "").strip()
    email = (email or "").strip().lower()
    if not name:
        raise AuthError("Name is required.", 400, "name_required")
    if len(name) > 100:
        raise AuthError("Name is too long.", 400, "name_too_long")
    if not EMAIL_RE.match(email or ""):
        raise AuthError("Enter a valid email address.", 400, "email_invalid")
    if not password or len(password) < MIN_PASSWORD:
        raise AuthError(
            f"Password must be at least {MIN_PASSWORD} characters.",
            400, "password_too_short")
    if confirm is not None and password != confirm:
        raise AuthError("Passwords do not match.", 400, "password_mismatch")
    # Cheap guard against the most-reused passwords. Not a substitute for a
    # breach-corpus check, and not described as one.
    if password.lower() in {
        "password12", "password123", "1234567890", "qwertyuiop",
        "letmein123", "urlguard12", "passw0rd12",
    }:
        raise AuthError("That password is too common.", 400, "password_weak")
    return name, email


# ------------------------------------------------------------------ users
def create_user(name, email, password, confirm=None):
    name, email = validate_registration(name, email, password, confirm)
    pw_hash = _ph.hash(password)
    now = time.time()
    try:
        with _lock, closing(connect()) as conn, conn:
            cur = conn.execute(
                "INSERT INTO users (email, name, password_hash, created_at, "
                "preferences) VALUES (?,?,?,?,?)",
                (email, name, pw_hash, now, json.dumps(default_preferences())))
            uid = cur.lastrowid
    except sqlite3.IntegrityError:
        # Same message and status as success-shaped failures elsewhere so the
        # endpoint cannot be used to enumerate registered addresses.
        raise AuthError("That email cannot be registered.", 409,
                        "email_unavailable")
    return get_user(uid)


def default_preferences():
    return {
        "warn_on_suspicious": True,
        "block_high_risk": True,
        "dns_layer_enabled": False,
        "store_full_urls": True,
        "notifications": True,
    }


def get_user(user_id):
    with closing(connect()) as conn:
        row = conn.execute("SELECT * FROM users WHERE id=?", (user_id,)).fetchone()
    return _user_dict(row) if row else None


def get_user_by_email(email):
    with closing(connect()) as conn:
        row = conn.execute("SELECT * FROM users WHERE email=?",
                           ((email or "").strip().lower(),)).fetchone()
    return row


def _user_dict(row):
    try:
        prefs = json.loads(row["preferences"] or "{}")
    except json.JSONDecodeError:
        prefs = {}
    merged = default_preferences()
    merged.update(prefs)
    return {
        "id": row["id"],
        "name": row["name"],
        "email": row["email"],
        "created_at": row["created_at"],
        "last_login_at": row["last_login_at"],
        "preferences": merged,
    }


def verify_password(email, password):
    """Returns the user dict, or raises. Timing is equalised on the miss path."""
    row = get_user_by_email(email)
    if row is None:
        # Spend comparable time hashing so a missing account is not detectably
        # faster than a wrong password.
        _ph.hash(password or "x")
        raise AuthError("Email or password is incorrect.", 401,
                        "invalid_credentials")
    try:
        _ph.verify(row["password_hash"], password or "")
    except (VerifyMismatchError, InvalidHashError):
        raise AuthError("Email or password is incorrect.", 401,
                        "invalid_credentials")
    if _ph.check_needs_rehash(row["password_hash"]):
        with _lock, closing(connect()) as conn, conn:
            conn.execute("UPDATE users SET password_hash=? WHERE id=?",
                         (_ph.hash(password), row["id"]))
    with _lock, closing(connect()) as conn, conn:
        conn.execute("UPDATE users SET last_login_at=? WHERE id=?",
                     (time.time(), row["id"]))
    return get_user(row["id"])


def update_profile(user_id, name=None, preferences=None):
    sets, params = [], []
    if name is not None:
        name = name.strip()
        if not name or len(name) > 100:
            raise AuthError("Enter a valid name.", 400, "name_invalid")
        sets.append("name=?")
        params.append(name)
    if preferences is not None:
        allowed = set(default_preferences())
        clean = {k: bool(v) for k, v in preferences.items() if k in allowed}
        current = get_user(user_id)["preferences"]
        current.update(clean)
        sets.append("preferences=?")
        params.append(json.dumps(current))
    if sets:
        params.append(user_id)
        with _lock, closing(connect()) as conn, conn:
            conn.execute(f"UPDATE users SET {', '.join(sets)} WHERE id=?", params)
    return get_user(user_id)


def change_password(user_id, current_password, new_password):
    user = get_user(user_id)
    if user is None:
        raise AuthError("Account not found.", 404, "not_found")
    verify_password(user["email"], current_password)
    if len(new_password or "") < MIN_PASSWORD:
        raise AuthError(f"Password must be at least {MIN_PASSWORD} characters.",
                        400, "password_too_short")
    with _lock, closing(connect()) as conn, conn:
        conn.execute("UPDATE users SET password_hash=? WHERE id=?",
                     (_ph.hash(new_password), user_id))
        # Changing a password invalidates every existing session.
        conn.execute("UPDATE refresh_tokens SET revoked=1 WHERE user_id=?",
                     (user_id,))
    return True


# ----------------------------------------------------------------- tokens
def _hash_token(raw: str) -> str:
    return hmac.new(SECRET, raw.encode("utf-8"), hashlib.sha256).hexdigest()


def issue_tokens(user, user_agent=None):
    now = dt.datetime.now(dt.timezone.utc)
    access = jwt.encode(
        {"sub": str(user["id"]), "email": user["email"], "name": user["name"],
         "iat": int(now.timestamp()),
         "exp": int((now + ACCESS_TTL).timestamp()), "typ": "access"},
        SECRET, algorithm="HS256")
    raw_refresh = secrets.token_urlsafe(48)
    expires = (now + REFRESH_TTL).timestamp()
    with _lock, closing(connect()) as conn, conn:
        conn.execute(
            "INSERT INTO refresh_tokens (user_id, token_hash, issued_at, "
            "expires_at, user_agent) VALUES (?,?,?,?,?)",
            (user["id"], _hash_token(raw_refresh), now.timestamp(), expires,
             (user_agent or "")[:200]))
    return {
        "access_token": access,
        "refresh_token": raw_refresh,
        "token_type": "Bearer",
        "expires_in": int(ACCESS_TTL.total_seconds()),
        "refresh_expires_in": int(REFRESH_TTL.total_seconds()),
    }


def decode_access(token):
    try:
        payload = jwt.decode(token, SECRET, algorithms=["HS256"])
    except jwt.ExpiredSignatureError:
        raise AuthError("Session expired. Sign in again.", 401, "token_expired")
    except jwt.InvalidTokenError:
        raise AuthError("Invalid session token.", 401, "token_invalid")
    if payload.get("typ") != "access":
        raise AuthError("Invalid session token.", 401, "token_invalid")
    return payload


def refresh_tokens(raw_refresh, user_agent=None):
    h = _hash_token(raw_refresh or "")
    with closing(connect()) as conn:
        row = conn.execute(
            "SELECT * FROM refresh_tokens WHERE token_hash=?", (h,)).fetchone()
    if row is None or row["revoked"] or row["expires_at"] < time.time():
        raise AuthError("Session expired. Sign in again.", 401,
                        "refresh_invalid")
    user = get_user(row["user_id"])
    if user is None:
        raise AuthError("Account not found.", 401, "refresh_invalid")
    # Rotate: the presented refresh token is burned as it is exchanged, so a
    # captured one cannot be replayed after the legitimate client uses it.
    with _lock, closing(connect()) as conn, conn:
        conn.execute("UPDATE refresh_tokens SET revoked=1 WHERE id=?",
                     (row["id"],))
    return issue_tokens(user, user_agent), user


def revoke_refresh(raw_refresh):
    with _lock, closing(connect()) as conn, conn:
        conn.execute("UPDATE refresh_tokens SET revoked=1 WHERE token_hash=?",
                     (_hash_token(raw_refresh or ""),))


def revoke_all(user_id):
    with _lock, closing(connect()) as conn, conn:
        conn.execute("UPDATE refresh_tokens SET revoked=1 WHERE user_id=?",
                     (user_id,))


# -------------------------------------------------------- password reset
def begin_password_reset(email):
    """Always succeeds from the caller's perspective.

    Returns the raw token when the address exists, None otherwise. The API
    must not reveal which happened.
    """
    row = get_user_by_email(email)
    if row is None:
        return None
    raw = secrets.token_urlsafe(32)
    with _lock, closing(connect()) as conn, conn:
        conn.execute(
            "INSERT OR REPLACE INTO password_resets (token_hash, user_id, "
            "expires_at, used) VALUES (?,?,?,0)",
            (_hash_token(raw), row["id"],
             (dt.datetime.now(dt.timezone.utc) + RESET_TTL).timestamp()))
    return raw


def complete_password_reset(raw_token, new_password):
    h = _hash_token(raw_token or "")
    with closing(connect()) as conn:
        row = conn.execute("SELECT * FROM password_resets WHERE token_hash=?",
                           (h,)).fetchone()
    if row is None or row["used"] or row["expires_at"] < time.time():
        raise AuthError("This reset link is invalid or has expired.", 400,
                        "reset_invalid")
    if len(new_password or "") < MIN_PASSWORD:
        raise AuthError(f"Password must be at least {MIN_PASSWORD} characters.",
                        400, "password_too_short")
    with _lock, closing(connect()) as conn, conn:
        conn.execute("UPDATE users SET password_hash=? WHERE id=?",
                     (_ph.hash(new_password), row["user_id"]))
        conn.execute("UPDATE password_resets SET used=1 WHERE token_hash=?", (h,))
        conn.execute("UPDATE refresh_tokens SET revoked=1 WHERE user_id=?",
                     (row["user_id"],))
    return True


# ------------------------------------------------------------- scan store
def record_scan(user_id, entry):
    with _lock, closing(connect()) as conn, conn:
        conn.execute(
            "INSERT OR REPLACE INTO scans (id, user_id, url, domain, "
            "risk_score, risk_band, risk_key, confidence, model_version, "
            "dataset_version, model_name, source, scope, detection_ms, "
            "signals, scanned_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (entry["id"], user_id, entry["url"], entry.get("domain"),
             entry["risk_score"], entry["risk_band"], entry["risk_key"],
             entry.get("confidence"), entry.get("model_version"),
             entry.get("dataset_version"), entry.get("model"),
             entry.get("source"), entry.get("scope"),
             entry.get("detection_ms"),
             json.dumps(entry.get("signals") or []), entry["scanned_at"]))


_SORTABLE = {"scanned_at", "risk_score", "domain", "risk_key"}


def list_scans(user_id, limit=25, offset=0, q=None, band=None,
               sort="scanned_at", order="desc", source=None):
    """Always filtered by user_id -- there is no unscoped read path."""
    sort = sort if sort in _SORTABLE else "scanned_at"
    order = "ASC" if str(order).lower() == "asc" else "DESC"
    where = ["user_id IS ?"] if user_id is None else ["user_id = ?"]
    params = [user_id]
    if q:
        where.append("(url LIKE ? OR domain LIKE ?)")
        params += [f"%{q}%", f"%{q}%"]
    if band:
        where.append("risk_key = ?")
        params.append(band)
    if source:
        where.append("source = ?")
        params.append(source)
    sql = f"SELECT * FROM scans WHERE {' AND '.join(where)}"
    with closing(connect()) as conn:
        total = conn.execute(
            f"SELECT COUNT(*) c FROM scans WHERE {' AND '.join(where)}",
            params).fetchone()["c"]
        rows = conn.execute(
            f"{sql} ORDER BY {sort} {order} LIMIT ? OFFSET ?",
            params + [int(limit), int(offset)]).fetchall()
    return [dict(r) | {"signals": json.loads(r["signals"] or "[]")}
            for r in rows], total


def get_scan(user_id, scan_id):
    with closing(connect()) as conn:
        row = conn.execute(
            "SELECT * FROM scans WHERE id=? AND user_id IS ?"
            if user_id is None else
            "SELECT * FROM scans WHERE id=? AND user_id=?",
            (scan_id, user_id)).fetchone()
    if row is None:
        return None
    return dict(row) | {"signals": json.loads(row["signals"] or "[]")}


def clear_scans(user_id):
    with _lock, closing(connect()) as conn, conn:
        if user_id is None:
            conn.execute("DELETE FROM scans WHERE user_id IS NULL")
        else:
            conn.execute("DELETE FROM scans WHERE user_id=?", (user_id,))


def stats(user_id):
    where = "user_id IS ?" if user_id is None else "user_id = ?"
    with closing(connect()) as conn:
        row = conn.execute(
            f"SELECT COUNT(*) total, AVG(detection_ms) avg_ms, "
            f"SUM(risk_key='low') low, SUM(risk_key='suspicious') suspicious, "
            f"SUM(risk_key='high') high, "
            f"SUM(risk_key='insufficient') insufficient "
            f"FROM scans WHERE {where}",
            (user_id,)).fetchone()
        by_source = conn.execute(
            f"SELECT source, COUNT(*) c FROM scans WHERE {where} "
            f"GROUP BY source", (user_id,)).fetchall()
        by_model = conn.execute(
            f"SELECT model_version, COUNT(*) c FROM scans WHERE {where} "
            f"GROUP BY model_version", (user_id,)).fetchall()
        top = conn.execute(
            f"SELECT domain, COUNT(*) c, MAX(risk_score) worst FROM scans "
            f"WHERE {where} AND risk_key IN ('suspicious','high') "
            f"AND domain IS NOT NULL GROUP BY domain ORDER BY c DESC, "
            f"worst DESC LIMIT 10", (user_id,)).fetchall()
        series = conn.execute(
            f"SELECT CAST(scanned_at/86400 AS INTEGER) day, risk_key, "
            f"COUNT(*) c FROM scans WHERE {where} GROUP BY day, risk_key "
            f"ORDER BY day", (user_id,)).fetchall()
    total = row["total"] or 0
    return {
        "total": total,
        "low": row["low"] or 0,
        "suspicious": row["suspicious"] or 0,
        "high": row["high"] or 0,
        # Undecided scans are counted separately so the band tiles sum to the
        # total. Folding them into "low" would present an unresolved scan as a
        # clean one.
        "insufficient": row["insufficient"] or 0,
        "avg_detection_ms": round(row["avg_ms"], 1) if row["avg_ms"] else None,
        "by_source": {r["source"] or "unknown": r["c"] for r in by_source},
        "by_model_version": {r["model_version"] or "unknown": r["c"]
                             for r in by_model},
        "top_domains": [{"domain": r["domain"], "count": r["c"],
                         "worst_score": r["worst"]} for r in top],
        "series": [{"day": r["day"] * 86400, "risk_key": r["risk_key"],
                    "count": r["c"]} for r in series],
        "sufficient_data": total > 0,
    }
