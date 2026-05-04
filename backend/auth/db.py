"""
Auth database — SQLite-backed user accounts and session tokens.
Uses pbkdf2_hmac for password hashing (no extra dependencies).
"""
import sqlite3
import hashlib
import hmac
import os
import secrets
import uuid
from datetime import datetime

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "users.db")


def _conn():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    con = sqlite3.connect(DB_PATH, check_same_thread=False)
    con.row_factory = sqlite3.Row
    return con


def init_db():
    with _conn() as con:
        con.executescript("""
            CREATE TABLE IF NOT EXISTS users (
                user_id   TEXT PRIMARY KEY,
                email     TEXT UNIQUE NOT NULL COLLATE NOCASE,
                name      TEXT NOT NULL,
                pw_hash   TEXT NOT NULL,
                pw_salt   TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS sessions (
                token      TEXT PRIMARY KEY,
                user_id    TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY(user_id) REFERENCES users(user_id)
            );
        """)


def _hash_password(password: str, salt: str) -> str:
    dk = hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), 260_000)
    return dk.hex()


def create_user(email: str, password: str, name: str) -> dict:
    init_db()
    salt = secrets.token_hex(32)
    pw_hash = _hash_password(password, salt)
    user_id = str(uuid.uuid4())
    now = datetime.utcnow().isoformat()
    with _conn() as con:
        con.execute(
            "INSERT INTO users (user_id, email, name, pw_hash, pw_salt, created_at) VALUES (?,?,?,?,?,?)",
            (user_id, email.lower().strip(), name.strip(), pw_hash, salt, now),
        )
    return {"user_id": user_id, "email": email.lower().strip(), "name": name.strip()}


def verify_user(email: str, password: str) -> dict | None:
    init_db()
    with _conn() as con:
        row = con.execute(
            "SELECT * FROM users WHERE email = ?", (email.lower().strip(),)
        ).fetchone()
    if not row:
        return None
    expected = _hash_password(password, row["pw_salt"])
    if not hmac.compare_digest(expected, row["pw_hash"]):
        return None
    return {"user_id": row["user_id"], "email": row["email"], "name": row["name"]}


def create_session(user_id: str) -> str:
    init_db()
    token = secrets.token_urlsafe(48)
    now = datetime.utcnow().isoformat()
    with _conn() as con:
        con.execute(
            "INSERT INTO sessions (token, user_id, created_at) VALUES (?,?,?)",
            (token, user_id, now),
        )
    return token


def get_user_by_token(token: str) -> dict | None:
    if not token:
        return None
    init_db()
    with _conn() as con:
        row = con.execute(
            """SELECT u.user_id, u.email, u.name
               FROM sessions s JOIN users u ON s.user_id = u.user_id
               WHERE s.token = ?""",
            (token,),
        ).fetchone()
    if not row:
        return None
    return {"user_id": row["user_id"], "email": row["email"], "name": row["name"]}


def delete_session(token: str):
    init_db()
    with _conn() as con:
        con.execute("DELETE FROM sessions WHERE token = ?", (token,))


def email_exists(email: str) -> bool:
    init_db()
    with _conn() as con:
        row = con.execute(
            "SELECT 1 FROM users WHERE email = ?", (email.lower().strip(),)
        ).fetchone()
    return row is not None
