from __future__ import annotations

import sqlite3
from pathlib import Path

from config import settings


def get_db_path() -> Path:
    path = Path(settings.db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def get_connection() -> sqlite3.Connection:
    """
    Создаёт подключение к SQLite.

    На Amvera база должна лежать в /data/bot.db,
    чтобы она не пропадала после пересборки.
    """
    db_path = get_db_path()

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA synchronous = NORMAL")

    return conn


def init_db() -> None:
    """
    Создаёт все таблицы, если их ещё нет.
    """
    with get_connection() as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS users (
                user_id INTEGER PRIMARY KEY,
                username TEXT,
                full_name TEXT,
                status TEXT NOT NULL DEFAULT 'active',
                source TEXT,
                approved_by INTEGER,
                approved_at TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_users_status
            ON users(status);


            CREATE TABLE IF NOT EXISTS access_requests (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                username TEXT,
                full_name TEXT,
                status TEXT NOT NULL DEFAULT 'pending',
                requested_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                decided_by INTEGER,
                decided_at TEXT
            );

            CREATE INDEX IF NOT EXISTS idx_access_requests_user_id
            ON access_requests(user_id);

            CREATE INDEX IF NOT EXISTS idx_access_requests_status
            ON access_requests(status);


            CREATE TABLE IF NOT EXISTS query_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                answer_id TEXT NOT NULL UNIQUE,
                created_at TEXT NOT NULL,
                user_id INTEGER,
                username TEXT,
                full_name TEXT,
                question TEXT NOT NULL,
                normalized_question TEXT,
                answer TEXT,
                sources_json TEXT
            );

            CREATE INDEX IF NOT EXISTS idx_query_logs_answer_id
            ON query_logs(answer_id);

            CREATE INDEX IF NOT EXISTS idx_query_logs_user_id
            ON query_logs(user_id);

            CREATE INDEX IF NOT EXISTS idx_query_logs_normalized_question
            ON query_logs(normalized_question);

            CREATE INDEX IF NOT EXISTS idx_query_logs_created_at
            ON query_logs(created_at);


            CREATE TABLE IF NOT EXISTS feedback_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                answer_id TEXT NOT NULL,
                feedback TEXT NOT NULL,
                feedback_user_id INTEGER,

                question TEXT,
                answer TEXT,
                sources_json TEXT,

                original_user_id INTEGER,
                original_username TEXT,
                original_full_name TEXT,
                original_created_at TEXT
            );

            CREATE INDEX IF NOT EXISTS idx_feedback_logs_answer_id
            ON feedback_logs(answer_id);

            CREATE INDEX IF NOT EXISTS idx_feedback_logs_feedback
            ON feedback_logs(feedback);

            CREATE INDEX IF NOT EXISTS idx_feedback_logs_created_at
            ON feedback_logs(created_at);


            CREATE TABLE IF NOT EXISTS admin_actions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                admin_id INTEGER,
                action TEXT NOT NULL,
                target_user_id INTEGER,
                details TEXT
            );

            CREATE INDEX IF NOT EXISTS idx_admin_actions_admin_id
            ON admin_actions(admin_id);

            CREATE INDEX IF NOT EXISTS idx_admin_actions_target_user_id
            ON admin_actions(target_user_id);
            """
        )