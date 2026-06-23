from __future__ import annotations

import json
import re
import uuid
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

from config import settings


def now_str() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def normalize_query(text: str) -> str:
    text = text.lower()
    text = text.replace("ё", "е")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


class AnalyticsStore:
    """
    Хранилище логов запросов и оценок.

    На Amvera рекомендуется:
    LOGS_DIR=/data/logs

    Файлы:
    /data/logs/query_log.jsonl
    /data/logs/feedback_log.jsonl
    """

    def __init__(self) -> None:
        self.logs_dir = Path(settings.logs_dir)
        self.logs_dir.mkdir(parents=True, exist_ok=True)

        self.query_log_path = self.logs_dir / "query_log.jsonl"
        self.feedback_log_path = self.logs_dir / "feedback_log.jsonl"

    def _append_jsonl(self, path: Path, record: dict[str, Any]) -> None:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _read_jsonl(self, path: Path) -> list[dict[str, Any]]:
        if not path.exists():
            return []

        records: list[dict[str, Any]] = []

        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()

                if not line:
                    continue

                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

        return records

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
        answer_id = uuid.uuid4().hex[:12]

        record = {
            "answer_id": answer_id,
            "created_at": now_str(),
            "user_id": user_id,
            "username": username,
            "full_name": full_name,
            "question": question,
            "normalized_question": normalize_query(question),
            "answer": answer,
            "sources": sources,
        }

        self._append_jsonl(self.query_log_path, record)

        return answer_id

    def get_query_by_answer_id(self, answer_id: str) -> dict[str, Any] | None:
        records = self._read_jsonl(self.query_log_path)

        for record in reversed(records):
            if record.get("answer_id") == answer_id:
                return record

        return None

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

            # сохраняем данные, чтобы потом видеть, где ошибка
            "question": query_record.get("question"),
            "answer": query_record.get("answer"),
            "sources": query_record.get("sources"),
            "original_user_id": query_record.get("user_id"),
            "original_username": query_record.get("username"),
            "original_full_name": query_record.get("full_name"),
            "original_created_at": query_record.get("created_at"),
        }

        self._append_jsonl(self.feedback_log_path, feedback_record)

        return True, feedback_record

    def get_stats(self) -> dict[str, Any]:
        queries = self._read_jsonl(self.query_log_path)
        feedback = self._read_jsonl(self.feedback_log_path)

        unique_users = {
            item.get("user_id")
            for item in queries
            if item.get("user_id") is not None
        }

        feedback_counter = Counter(
            item.get("feedback")
            for item in feedback
        )

        return {
            "total_queries": len(queries),
            "unique_users": len(unique_users),
            "total_feedback": len(feedback),
            "good_feedback": feedback_counter.get("good", 0),
            "bad_feedback": feedback_counter.get("bad", 0),
            "query_log_path": str(self.query_log_path),
            "feedback_log_path": str(self.feedback_log_path),
        }

    def get_popular_queries(self, limit: int = 10) -> list[dict[str, Any]]:
        queries = self._read_jsonl(self.query_log_path)

        counter: Counter[str] = Counter()
        examples: dict[str, str] = {}

        for item in queries:
            normalized = item.get("normalized_question")
            original = item.get("question")

            if not normalized:
                continue

            counter[normalized] += 1

            if normalized not in examples and original:
                examples[normalized] = original

        result: list[dict[str, Any]] = []

        for normalized, count in counter.most_common(limit):
            result.append(
                {
                    "question": examples.get(normalized, normalized),
                    "normalized_question": normalized,
                    "count": count,
                }
            )

        return result

    def get_recent_bad_feedback(self, limit: int = 10) -> list[dict[str, Any]]:
        feedback = self._read_jsonl(self.feedback_log_path)

        bad_items = [
            item
            for item in feedback
            if item.get("feedback") == "bad"
        ]

        return list(reversed(bad_items))[:limit]