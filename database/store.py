from __future__ import annotations

import hashlib
import hmac
import os
import secrets
import sqlite3
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
DB_PATH = Path(os.environ.get("AIVQ_DB_PATH", ROOT / "app.sqlite3"))
SCHEMA_PATH = ROOT / "schema.sql"
SESSION_SECONDS = 7 * 24 * 60 * 60
PASSWORD_ITERATIONS = 240_000


class DuplicateUserError(ValueError):
    pass


def connect() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(DB_PATH)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


def init_db() -> Path:
    with connect() as connection:
        connection.executescript(SCHEMA_PATH.read_text(encoding="utf-8"))
    return DB_PATH


def hash_password(password: str) -> str:
    salt = secrets.token_hex(16)
    digest = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        bytes.fromhex(salt),
        PASSWORD_ITERATIONS,
    ).hex()
    return f"pbkdf2_sha256${PASSWORD_ITERATIONS}${salt}${digest}"


def verify_password(password: str, stored_hash: str) -> bool:
    try:
        algorithm, iterations, salt, digest = stored_hash.split("$", 3)
    except ValueError:
        return False

    if algorithm != "pbkdf2_sha256":
        return False

    candidate = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        bytes.fromhex(salt),
        int(iterations),
    ).hex()
    return hmac.compare_digest(candidate, digest)


def create_user(username: str, password: str) -> dict[str, Any]:
    username = username.strip()
    if not username or not password:
        raise ValueError("Username and password are required.")

    try:
        with connect() as connection:
            cursor = connection.execute(
                "INSERT INTO users (username, password_hash) VALUES (?, ?)",
                (username, hash_password(password)),
            )
            user_id = cursor.lastrowid
            row = connection.execute(
                "SELECT id, username, created_at FROM users WHERE id = ?",
                (user_id,),
            ).fetchone()
    except sqlite3.IntegrityError as exc:
        raise DuplicateUserError("Username already exists.") from exc

    return dict(row)


def authenticate_user(username: str, password: str) -> dict[str, Any] | None:
    with connect() as connection:
        row = connection.execute(
            "SELECT id, username, password_hash, created_at FROM users WHERE username = ?",
            (username.strip(),),
        ).fetchone()

    if not row or not verify_password(password, row["password_hash"]):
        return None

    return {
        "id": row["id"],
        "username": row["username"],
        "created_at": row["created_at"],
    }


def create_session(user_id: int) -> str:
    token = secrets.token_urlsafe(32)
    expires_at = int(time.time()) + SESSION_SECONDS

    with connect() as connection:
        connection.execute(
            "INSERT INTO sessions (token, user_id, expires_at) VALUES (?, ?, ?)",
            (token, user_id, expires_at),
        )

    return token


def delete_session(token: str | None) -> None:
    if not token:
        return

    with connect() as connection:
        connection.execute("DELETE FROM sessions WHERE token = ?", (token,))


def get_user_by_session(token: str | None) -> dict[str, Any] | None:
    if not token:
        return None

    now = int(time.time())
    with connect() as connection:
        connection.execute("DELETE FROM sessions WHERE expires_at <= ?", (now,))
        row = connection.execute(
            """
            SELECT users.id, users.username, users.created_at
            FROM sessions
            JOIN users ON users.id = sessions.user_id
            WHERE sessions.token = ? AND sessions.expires_at > ?
            """,
            (token, now),
        ).fetchone()

    return dict(row) if row else None


def create_bookmark(
    user_id: int,
    video_id: str | None,
    video_title: str | None,
    log_text: str,
) -> dict[str, Any]:
    if not log_text.strip():
        raise ValueError("Log text is required.")

    with connect() as connection:
        cursor = connection.execute(
            """
            INSERT INTO bookmarks (user_id, video_id, video_title, log_text)
            VALUES (?, ?, ?, ?)
            """,
            (user_id, video_id, video_title, log_text),
        )
        row = connection.execute(
            """
            SELECT id, video_id, video_title, log_text, created_at
            FROM bookmarks
            WHERE id = ?
            """,
            (cursor.lastrowid,),
        ).fetchone()

    return dict(row)


def list_bookmarks(user_id: int) -> list[dict[str, Any]]:
    with connect() as connection:
        rows = connection.execute(
            """
            SELECT id, video_id, video_title, log_text, created_at
            FROM bookmarks
            WHERE user_id = ?
            ORDER BY created_at DESC, id DESC
            """,
            (user_id,),
        ).fetchall()

    return [dict(row) for row in rows]


def delete_bookmark(user_id: int, bookmark_id: int) -> bool:
    with connect() as connection:
        cursor = connection.execute(
            "DELETE FROM bookmarks WHERE id = ? AND user_id = ?",
            (bookmark_id, user_id),
        )
    return cursor.rowcount > 0
