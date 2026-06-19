from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np


NEGATIVE_ANSWER_PREFIXES = (
    "В базе знаний нет информации",
    "Не удалось сформировать ответ",
)


def str_to_bool(value: str | None, default: bool = True) -> bool:
    if value is None:
        return default

    return value.strip().lower() in {"1", "true", "yes", "y", "да", "on"}


def normalize_question(question: str) -> str:
    return " ".join(
        question.lower()
        .replace("ё", "е")
        .replace("—", " ")
        .replace("–", " ")
        .replace("-", " ")
        .split()
    )


def vector_to_list(vector: np.ndarray) -> list[float]:
    return [float(x) for x in vector.tolist()]


def list_to_vector(values: list[float]) -> np.ndarray:
    vector = np.array(values, dtype=np.float32)
    norm = np.linalg.norm(vector)
    if norm == 0:
        return vector
    return vector / norm


@dataclass
class CacheHit:
    answer: str
    question: str
    similarity: float
    hit_type: str


class AnswerCache:
    """
    Кэш ответов.

    Хранит:
    - исходный вопрос
    - нормализованный вопрос
    - embedding вопроса
    - готовый ответ
    - hash базы знаний

    Если база знаний изменилась, старый кэш автоматически не используется.
    """

    def __init__(
        self,
        storage_dir: str | Path,
        knowledge_base_hash: str,
    ) -> None:
        self.enabled = str_to_bool(os.getenv("CACHE_ENABLED"), default=True)
        self.similarity_threshold = float(
            os.getenv("CACHE_SIMILARITY_THRESHOLD", "0.92")
        )
        self.cache_file_name = os.getenv("CACHE_FILE", "answer_cache.json")

        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)

        self.cache_path = self.storage_dir / self.cache_file_name
        self.knowledge_base_hash = knowledge_base_hash

        self.items: list[dict] = []

        if self.enabled:
            self._load()

    def _load(self) -> None:
        if not self.cache_path.exists():
            self.items = []
            return

        try:
            with open(self.cache_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            self.items = []
            return

        if data.get("knowledge_base_hash") != self.knowledge_base_hash:
            # База знаний изменилась — старый кэш использовать нельзя.
            self.items = []
            return

        self.items = data.get("items", [])

    def _save(self) -> None:
        if not self.enabled:
            return

        data = {
            "knowledge_base_hash": self.knowledge_base_hash,
            "updated_at": int(time.time()),
            "items": self.items,
        }

        temp_path = self.cache_path.with_suffix(".tmp")

        with open(temp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        temp_path.replace(self.cache_path)

    def find_exact(self, question: str) -> CacheHit | None:
        """
        Точный поиск по нормализованному вопросу.
        Не требует embedding, поэтому самый дешёвый.
        """
        if not self.enabled:
            return None

        normalized = normalize_question(question)

        for item in self.items:
            if item.get("normalized_question") == normalized:
                return CacheHit(
                    answer=item["answer"],
                    question=item["question"],
                    similarity=1.0,
                    hit_type="exact",
                )

        return None

    def find_semantic(
        self,
        question: str,
        question_vector: np.ndarray,
    ) -> CacheHit | None:
        """
        Семантический поиск по похожим вопросам.
        Требует embedding вопроса, но экономит chat completion.
        """
        if not self.enabled or not self.items:
            return None

        best_item: dict | None = None
        best_similarity = -1.0

        for item in self.items:
            cached_vector_raw = item.get("question_embedding")

            if not cached_vector_raw:
                continue

            cached_vector = list_to_vector(cached_vector_raw)
            similarity = float(np.dot(question_vector, cached_vector))

            if similarity > best_similarity:
                best_similarity = similarity
                best_item = item

        if best_item and best_similarity >= self.similarity_threshold:
            return CacheHit(
                answer=best_item["answer"],
                question=best_item["question"],
                similarity=best_similarity,
                hit_type="semantic",
            )

        return None

    def add(
        self,
        question: str,
        question_vector: np.ndarray,
        answer: str,
    ) -> None:
        """
        Сохраняет ответ в кэш.
        Негативные ответы не кэшируем, чтобы случайная ошибка поиска
        не закрепилась в кэше.
        """
        if not self.enabled:
            return

        clean_answer = answer.strip()

        if not clean_answer:
            return

        if clean_answer.startswith(NEGATIVE_ANSWER_PREFIXES):
            return

        normalized = normalize_question(question)

        # Если такой вопрос уже есть — обновим ответ.
        for item in self.items:
            if item.get("normalized_question") == normalized:
                item["answer"] = clean_answer
                item["question_embedding"] = vector_to_list(question_vector)
                item["updated_at"] = int(time.time())
                self._save()
                return

        self.items.append(
            {
                "question": question,
                "normalized_question": normalized,
                "question_embedding": vector_to_list(question_vector),
                "answer": clean_answer,
                "created_at": int(time.time()),
                "updated_at": int(time.time()),
            }
        )

        self._save()


def file_sha256(path: Path) -> str:
    hasher = hashlib.sha256()

    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            hasher.update(chunk)

    return hasher.hexdigest()