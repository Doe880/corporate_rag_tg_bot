from __future__ import annotations

import json
import re
import uuid
from collections import Counter
from datetime import datetime
from typing import Any

from config import settings
from db import get_connection, init_db


def now_str() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def normalize_query(text: str) -> str:
    text = text.lower()
    text = text.replace("ё", "е")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False)


def json_loads_list(value: str | None) -> list[str]:
    if not value:
        return []

    try:
        data = json.loads(value)
    except json.JSONDecodeError:
        return []

    if isinstance(data, list):
        return [str(item) for item in data]

    return []


class AnalyticsStore:
    """
    Хранилище логов запросов и оценок в SQLite.

    На Amvera:
    DB_PATH=/data/bot.db
    """

    def __init__(self) -> None:
        init_db()

    def _make_answer_id(self) -> str:
        """
        Генерирует короткий answer_id и проверяет уникальность.
        """
        for _ in range(5):
            answer_id = uuid.uuid4().hex[:12]

            with get_connection() as conn:
                row = conn.execute(
                    """
                    SELECT answer_id
                    FROM query_logs
                    WHERE answer_id = ?
                    """,
                    (answer_id,),
                ).fetchone()

            if row is None:
                return answer_id

        return uuid.uuid4().hex

    def log_query(
        self,
        user_id: int | None,
        username: str | None,
        full_name: str | None,
        question: str,
        answer: str,
        sources: list[str],
    ) -> str:
        """
        Логирует каждый запрос пользователя.
        Возвращает answer_id, который потом используется в кнопках feedback.
        """
        answer_id = self._make_answer_id()

        with get_connection() as conn:
            conn.execute(
                """
                INSERT INTO query_logs (
                    answer_id,
                    created_at,
                    user_id,
                    username,
                    full_name,
                    question,
                    normalized_question,
                    answer,
                    sources_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    answer_id,
                    now_str(),
                    user_id,
                    username,
                    full_name,
                    question,
                    normalize_query(question),
                    answer,
                    json_dumps(sources),
                ),
            )

        return answer_id

    def get_query_by_answer_id(self, answer_id: str) -> dict[str, Any] | None:
        with get_connection() as conn:
            row = conn.execute(
                """
                SELECT
                    answer_id,
                    created_at,
                    user_id,
                    username,
                    full_name,
                    question,
                    normalized_question,
                    answer,
                    sources_json
                FROM query_logs
                WHERE answer_id = ?
                """,
                (answer_id,),
            ).fetchone()

        if not row:
            return None

        record = dict(row)
        record["sources"] = json_loads_list(record.pop("sources_json", None))

        return record

    def log_feedback(
        self,
        answer_id: str,
        feedback: str,
        feedback_user_id: int | None,
    ) -> tuple[bool, dict[str, Any] | None]:
        """
        feedback:
        - good
        - bad
        """
        query_record = self.get_query_by_answer_id(answer_id)

        if not query_record:
            return False, None

        feedback_record = {
            "created_at": now_str(),
            "answer_id": answer_id,
            "feedback": feedback,
            "feedback_user_id": feedback_user_id,

            "question": query_record.get("question"),
            "answer": query_record.get("answer"),
            "sources": query_record.get("sources") or [],

            "original_user_id": query_record.get("user_id"),
            "original_username": query_record.get("username"),
            "original_full_name": query_record.get("full_name"),
            "original_created_at": query_record.get("created_at"),
        }

        with get_connection() as conn:
            conn.execute(
                """
                INSERT INTO feedback_logs (
                    created_at,
                    answer_id,
                    feedback,
                    feedback_user_id,
                    question,
                    answer,
                    sources_json,
                    original_user_id,
                    original_username,
                    original_full_name,
                    original_created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    feedback_record["created_at"],
                    feedback_record["answer_id"],
                    feedback_record["feedback"],
                    feedback_record["feedback_user_id"],
                    feedback_record["question"],
                    feedback_record["answer"],
                    json_dumps(feedback_record["sources"]),
                    feedback_record["original_user_id"],
                    feedback_record["original_username"],
                    feedback_record["original_full_name"],
                    feedback_record["original_created_at"],
                ),
            )

        return True, feedback_record

    def get_stats(self) -> dict[str, Any]:
        with get_connection() as conn:
            total_queries = conn.execute(
                """
                SELECT COUNT(*) AS count
                FROM query_logs
                """
            ).fetchone()["count"]

            unique_users = conn.execute(
                """
                SELECT COUNT(DISTINCT user_id) AS count
                FROM query_logs
                WHERE user_id IS NOT NULL
                """
            ).fetchone()["count"]

            total_feedback = conn.execute(
                """
                SELECT COUNT(*) AS count
                FROM feedback_logs
                """
            ).fetchone()["count"]

            feedback_rows = conn.execute(
                """
                SELECT feedback, COUNT(*) AS count
                FROM feedback_logs
                GROUP BY feedback
                """
            ).fetchall()

        feedback_counter = Counter()

        for row in feedback_rows:
            feedback_counter[row["feedback"]] = row["count"]

        db_label = f"sqlite:{settings.db_path}"

        return {
            "total_queries": total_queries,
            "unique_users": unique_users,
            "total_feedback": total_feedback,
            "good_feedback": feedback_counter.get("good", 0),
            "bad_feedback": feedback_counter.get("bad", 0),

            # Оставляем эти ключи для совместимости с текущим bot.py
            "query_log_path": f"{db_label}#query_logs",
            "feedback_log_path": f"{db_label}#feedback_logs",
        }

    def get_popular_queries(self, limit: int = 10) -> list[dict[str, Any]]:
        with get_connection() as conn:
            rows = conn.execute(
                """
                SELECT
                    normalized_question,
                    MIN(question) AS question,
                    COUNT(*) AS count
                FROM query_logs
                WHERE normalized_question IS NOT NULL
                  AND normalized_question != ''
                GROUP BY normalized_question
                ORDER BY count DESC, normalized_question ASC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()

        result: list[dict[str, Any]] = []

        for row in rows:
            result.append(
                {
                    "question": row["question"],
                    "normalized_question": row["normalized_question"],
                    "count": row["count"],
                }
            )

        return result

    def get_recent_bad_feedback(self, limit: int = 10) -> list[dict[str, Any]]:
        with get_connection() as conn:
            rows = conn.execute(
                """
                SELECT
                    created_at,
                    answer_id,
                    feedback,
                    feedback_user_id,
                    question,
                    answer,
                    sources_json,
                    original_user_id,
                    original_username,
                    original_full_name,
                    original_created_at
                FROM feedback_logs
                WHERE feedback = 'bad'
                ORDER BY created_at DESC, id DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()

        result: list[dict[str, Any]] = []

        for row in rows:
            item = dict(row)
            item["sources"] = json_loads_list(item.pop("sources_json", None))
            result.append(item)

        return result