from __future__ import annotations

import json
import logging
import math
import re
from collections import Counter
from pathlib import Path

import numpy as np
from openai import AsyncOpenAI

from cache import AnswerCache, file_sha256
from config import settings
from ingest import INDEX_FILE, CHUNKS_FILE, normalize_vectors

logger = logging.getLogger(__name__)


SYSTEM_PROMPT = """
Ты корпоративный AI-ассистент, который отвечает строго по базе знаний.

Правила:
1. Используй только информацию из блока CONTEXT.
2. Не используй внешние знания и не додумывай факты.
3. Если в CONTEXT нет ответа, напиши: "В базе знаний нет информации для ответа на этот вопрос."
4. Если данных недостаточно, честно скажи, каких данных не хватает.
5. Не ссылайся на несуществующие документы, страницы или пункты.
6. Отвечай кратко, понятно и по делу.
""".strip()


STOPWORDS = {
    "что",
    "это",
    "как",
    "какой",
    "какая",
    "какие",
    "какое",
    "где",
    "когда",
    "зачем",
    "почему",
    "для",
    "про",
    "при",
    "или",
    "и",
    "а",
    "в",
    "во",
    "на",
    "с",
    "со",
    "по",
    "из",
    "у",
    "от",
    "до",
    "за",
    "под",
    "над",
    "об",
    "обо",
    "без",
    "есть",
    "ли",
    "же",
    "бы",
    "мне",
    "найди",
    "расскажи",
    "напиши",
    "покажи",
    "информация",
    "информацию",
    "база",
    "базе",
    "знаний",
}


IMPORTANT_FIELDS = {
    "слоган",
    "девиз",
    "состав",
    "компоненты",
    "вещество",
    "вещества",
    "противопоказания",
    "показания",
    "дозировка",
    "применение",
    "прием",
    "принимать",
    "рекомендации",
    "сообщение",
    "форма",
    "выпуска",
    "описание",
    "имидж",
    "ключевое",
    "эффективность",
    "аналоги",
    "преимущества",
    "выгода",
}


# Query expansion:
# Если пользователь задаёт вопрос другими словами,
# бот расширяет запрос синонимами и близкими формулировками.
QUERY_EXPANSIONS = {
    "слоган": [
        "девиз",
        "ключевая фраза",
        "ключевое сообщение",
        "коммуникация",
        "позиционирование",
    ],
    "девиз": [
        "слоган",
        "ключевая фраза",
        "ключевое сообщение",
        "коммуникация",
        "позиционирование",
    ],
    "состав": [
        "компоненты",
        "активные вещества",
        "ингредиенты",
        "вещество",
        "вещества",
        "формула",
    ],
    "компоненты": [
        "состав",
        "активные вещества",
        "ингредиенты",
        "вещество",
        "вещества",
        "формула",
    ],
    "вещество": [
        "состав",
        "компоненты",
        "активные вещества",
        "ингредиенты",
    ],
    "вещества": [
        "состав",
        "компоненты",
        "активные вещества",
        "ингредиенты",
    ],
    "применение": [
        "как принимать",
        "прием",
        "способ применения",
        "рекомендации",
        "дозировка",
        "курс",
    ],
    "принимать": [
        "применение",
        "прием",
        "способ применения",
        "рекомендации",
        "дозировка",
        "курс",
    ],
    "прием": [
        "применение",
        "как принимать",
        "способ применения",
        "рекомендации",
        "дозировка",
        "курс",
    ],
    "дозировка": [
        "доза",
        "прием",
        "как принимать",
        "способ применения",
        "применение",
    ],
    "показания": [
        "для чего",
        "кому подходит",
        "назначение",
        "применение",
        "рекомендации",
    ],
    "противопоказания": [
        "нельзя",
        "ограничения",
        "кому нельзя",
        "не рекомендуется",
        "предостережения",
    ],
    "форма": [
        "форма выпуска",
        "выпуск",
        "упаковка",
        "таблетки",
        "капсулы",
        "саше",
    ],
    "выпуска": [
        "форма выпуска",
        "выпуск",
        "упаковка",
        "таблетки",
        "капсулы",
        "саше",
    ],
    "упаковка": [
        "форма выпуска",
        "выпуск",
        "количество",
        "таблетки",
        "капсулы",
        "саше",
    ],
    "преимущества": [
        "выгоды",
        "плюсы",
        "отличия",
        "сильные стороны",
        "польза",
    ],
    "выгоды": [
        "преимущества",
        "плюсы",
        "отличия",
        "сильные стороны",
        "польза",
    ],
    "отличия": [
        "преимущества",
        "аналоги",
        "конкуренты",
        "чем отличается",
        "сравнение",
    ],
    "аналоги": [
        "конкуренты",
        "сравнение",
        "отличия",
        "похожие продукты",
    ],
    "эффективность": [
        "результат",
        "доказательства",
        "исследования",
        "данные",
        "эффект",
    ],
    "исследования": [
        "данные",
        "доказательства",
        "эффективность",
        "результаты",
    ],
    "сообщение": [
        "ключевое сообщение",
        "слоган",
        "девиз",
        "позиционирование",
        "коммуникация",
    ],
    "позиционирование": [
        "слоган",
        "девиз",
        "ключевое сообщение",
        "коммуникация",
    ],
}


BM25_K1 = 1.5
BM25_B = 0.75


def normalize_text_for_search(text: str) -> str:
    text = text.lower()
    text = text.replace("ё", "е")
    text = text.replace("—", " ")
    text = text.replace("–", " ")
    text = text.replace("-", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def tokenize(text: str) -> list[str]:
    text = normalize_text_for_search(text)
    tokens = re.findall(r"[a-zа-я0-9]+", text, flags=re.IGNORECASE)

    return [
        token
        for token in tokens
        if token not in STOPWORDS and len(token) >= 2
    ]


def unique_preserve_order(items: list[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()

    for item in items:
        if item in seen:
            continue

        seen.add(item)
        result.append(item)

    return result


def expand_query_tokens(question: str) -> tuple[list[str], list[str], str]:
    """
    Возвращает:
    1. original_tokens — токены исходного вопроса
    2. expanded_tokens — исходные токены + расширенные токены
    3. expanded_text — текст вопроса + расширенные слова

    Пример:
    вопрос: "девиз продукта"
    expanded_tokens: ["девиз", "продукта", "слоган", "ключевая", "фраза", ...]
    """
    original_tokens = tokenize(question)
    expanded_tokens: list[str] = []

    for token in original_tokens:
        expanded_tokens.append(token)

        expansions = QUERY_EXPANSIONS.get(token, [])

        for phrase in expansions:
            expanded_tokens.extend(tokenize(phrase))

    expanded_tokens = unique_preserve_order(expanded_tokens)

    additional_tokens = [
        token
        for token in expanded_tokens
        if token not in original_tokens
    ]

    if additional_tokens:
        expanded_text = question + " " + " ".join(additional_tokens)
    else:
        expanded_text = question

    return original_tokens, expanded_tokens, expanded_text


def build_query_token_weights(
    original_tokens: list[str],
    expanded_tokens: list[str],
) -> dict[str, float]:
    """
    Вес исходных токенов выше, вес расширенных токенов ниже.
    Это нужно, чтобы query expansion помогал, но не перебивал исходный смысл вопроса.
    """
    weights: dict[str, float] = {}

    for token in expanded_tokens:
        weights[token] = 0.6

    for token in original_tokens:
        weights[token] = 1.0

    return weights


def safe_float(value: float | int | None, default: float = 0.0) -> float:
    try:
        if value is None:
            return default

        return float(value)
    except Exception:
        return default


class RAGEngine:
    def __init__(self) -> None:
        self.client = AsyncOpenAI(api_key=settings.openai_api_key)
        self.storage_dir = Path(settings.storage_dir)

        self.index_path = self.storage_dir / INDEX_FILE
        self.chunks_path = self.storage_dir / CHUNKS_FILE

        self.embeddings: np.ndarray
        self.chunks: list[dict]
        self.knowledge_base_hash: str

        self.bm25_index: list[dict]
        self.bm25_idf: dict[str, float]
        self.bm25_avg_doc_len: float

        self.cache: AnswerCache

        self.reload()

    def reload(self) -> None:
        """
        Полностью перезагружает базу знаний из index.npz и chunks.json.
        Используется командой /reload.
        """
        self.embeddings = self._load_embeddings()
        self.chunks = self._load_chunks()
        self.knowledge_base_hash = file_sha256(self.chunks_path)

        self.bm25_index, self.bm25_idf, self.bm25_avg_doc_len = self._build_bm25_index()

        self.cache = AnswerCache(
            storage_dir=self.storage_dir,
            knowledge_base_hash=self.knowledge_base_hash,
        )

        logger.info(
            "RAG reloaded. chunks=%s, embeddings_shape=%s, kb_hash=%s, bm25_docs=%s",
            len(self.chunks),
            self.embeddings.shape,
            self.knowledge_base_hash[:12],
            len(self.bm25_index),
        )

    def _load_embeddings(self) -> np.ndarray:
        if not self.index_path.exists():
            raise RuntimeError(
                "Не найден векторный индекс. "
                "Сначала запустите: python ingest.py"
            )

        data = np.load(self.index_path)
        return data["embeddings"]

    def _load_chunks(self) -> list[dict]:
        if not self.chunks_path.exists():
            raise RuntimeError(
                "Не найден файл chunks.json. "
                "Сначала запустите: python ingest.py"
            )

        with open(self.chunks_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _build_bm25_index(self) -> tuple[list[dict], dict[str, float], float]:
        """
        Строит BM25-индекс по chunks.

        BM25 лучше простого keyword search, потому что учитывает:
        - частоту слова в chunk
        - редкость слова во всей базе
        - длину документа/chunk
        """
        index: list[dict] = []
        doc_freq: Counter[str] = Counter()
        total_doc_len = 0

        for chunk in self.chunks:
            text = chunk.get("text", "")
            tokens = tokenize(text)
            token_counts = Counter(tokens)
            token_set = set(tokens)
            doc_len = len(tokens)

            total_doc_len += doc_len
            doc_freq.update(token_set)

            index.append(
                {
                    "id": chunk["id"],
                    "file_name": chunk["file_name"],
                    "page": chunk["page"],
                    "chunk_no": chunk["chunk_no"],
                    "text": text,
                    "normalized_text": normalize_text_for_search(text),
                    "tokens": tokens,
                    "token_counts": token_counts,
                    "token_set": token_set,
                    "doc_len": doc_len,
                }
            )

        docs_count = len(index)
        avg_doc_len = total_doc_len / docs_count if docs_count else 0.0

        idf: dict[str, float] = {}

        for token, df in doc_freq.items():
            # Классическая сглаженная формула IDF для BM25.
            idf[token] = math.log(1 + ((docs_count - df + 0.5) / (df + 0.5)))

        return index, idf, avg_doc_len

    async def _embed_query(self, question: str) -> np.ndarray:
        response = await self.client.embeddings.create(
            model=settings.embedding_model,
            input=question,
        )

        vector = np.array([response.data[0].embedding], dtype=np.float32)
        return normalize_vectors(vector)[0]

    async def vector_search(
        self,
        question_vector: np.ndarray,
        limit: int,
    ) -> list[dict]:
        """
        Semantic/vector search через embeddings.
        """
        scores = self.embeddings @ question_vector
        top_indices = np.argsort(scores)[::-1][:limit]

        results: list[dict] = []

        for idx in top_indices:
            score = float(scores[idx])

            if score < settings.min_relevance_score:
                continue

            chunk = self.chunks[int(idx)]

            results.append(
                {
                    "id": chunk["id"],
                    "vector_score": score,
                    "bm25_score": 0.0,
                    "score": score,
                    "rank_score": score,
                    "search_type": "vector",
                    "search_types": {"vector"},
                    "file_name": chunk["file_name"],
                    "page": chunk["page"],
                    "chunk_no": chunk["chunk_no"],
                    "text": chunk["text"],
                }
            )

        return results

    def bm25_search(
        self,
        question: str,
        limit: int,
    ) -> list[dict]:
        """
        BM25-поиск по chunks с query expansion.

        Исходный вопрос:
        "девиз продукта"

        Расширенный поиск:
        "девиз продукта слоган ключевая фраза ключевое сообщение ..."
        """
        original_tokens, expanded_tokens, expanded_text = expand_query_tokens(question)
        normalized_question = normalize_text_for_search(question)

        if not expanded_tokens:
            return []

        token_weights = build_query_token_weights(
            original_tokens=original_tokens,
            expanded_tokens=expanded_tokens,
        )

        original_token_set = set(original_tokens)
        expanded_token_set = set(expanded_tokens)

        results: list[dict] = []

        avg_doc_len = self.bm25_avg_doc_len or 1.0

        for item in self.bm25_index:
            token_counts: Counter = item["token_counts"]
            token_set: set[str] = item["token_set"]
            doc_len = item["doc_len"] or 1
            text = item["normalized_text"]

            # Быстрый пропуск: если нет ни одного совпадающего термина.
            if not expanded_token_set.intersection(token_set):
                continue

            bm25_score = 0.0

            for token in expanded_tokens:
                tf = token_counts.get(token, 0)

                if tf <= 0:
                    continue

                idf = self.bm25_idf.get(token, 0.0)
                weight = token_weights.get(token, 0.6)

                denominator = tf + BM25_K1 * (
                    1 - BM25_B + BM25_B * (doc_len / avg_doc_len)
                )

                bm25_score += weight * idf * ((tf * (BM25_K1 + 1)) / denominator)

            # Дополнительные бонусы поверх BM25.
            exact_phrase_bonus = 0.0
            field_bonus = 0.0
            all_original_terms_bonus = 0.0
            important_field_bonus = 0.0
            expansion_match_bonus = 0.0

            if normalized_question and normalized_question in text:
                exact_phrase_bonus = 5.0

            # Если совпадает поле из исходного вопроса, даём сильный бонус.
            for token in original_tokens:
                if f"{token}:" in text:
                    field_bonus += 3.0

            # Если совпадает поле из расширенного запроса, даём более слабый бонус.
            for token in expanded_tokens:
                if token in original_token_set:
                    continue

                if f"{token}:" in text:
                    field_bonus += 1.5

            if original_tokens and all(token in token_set for token in original_tokens):
                all_original_terms_bonus = 2.0

            matched_important_fields = IMPORTANT_FIELDS.intersection(expanded_token_set)

            for field in matched_important_fields:
                if field in token_set:
                    important_field_bonus += 2.0

            # Если расширение помогло найти термин, которого не было в исходном запросе.
            expanded_only_tokens = expanded_token_set - original_token_set

            if expanded_only_tokens.intersection(token_set):
                expansion_match_bonus = 1.0

            final_bm25_score = (
                bm25_score
                + exact_phrase_bonus
                + field_bonus
                + all_original_terms_bonus
                + important_field_bonus
                + expansion_match_bonus
            )

            if final_bm25_score <= 0:
                continue

            results.append(
                {
                    "id": item["id"],
                    "vector_score": 0.0,
                    "bm25_score": final_bm25_score,
                    "score": final_bm25_score,
                    "rank_score": final_bm25_score,
                    "search_type": "bm25",
                    "search_types": {"bm25"},
                    "file_name": item["file_name"],
                    "page": item["page"],
                    "chunk_no": item["chunk_no"],
                    "text": item["text"],
                    "expanded_query": expanded_text,
                }
            )

        results.sort(key=lambda x: x["bm25_score"], reverse=True)

        return results[:limit]

    def merge_candidates(
        self,
        vector_results: list[dict],
        bm25_results: list[dict],
    ) -> list[dict]:
        """
        Объединяет результаты vector search и BM25.
        Если chunk найден двумя способами, сохраняем оба score.
        """
        combined: dict[int, dict] = {}

        def add_result(result: dict) -> None:
            chunk_id = int(result["id"])

            if chunk_id not in combined:
                combined[chunk_id] = result.copy()
                combined[chunk_id]["search_types"] = set(result.get("search_types", set()))
                return

            existing = combined[chunk_id]

            existing["vector_score"] = max(
                safe_float(existing.get("vector_score")),
                safe_float(result.get("vector_score")),
            )
            existing["bm25_score"] = max(
                safe_float(existing.get("bm25_score")),
                safe_float(result.get("bm25_score")),
            )

            existing["score"] = max(
                safe_float(existing.get("score")),
                safe_float(result.get("score")),
            )

            existing["search_types"].update(result.get("search_types", set()))

            if result.get("expanded_query"):
                existing["expanded_query"] = result["expanded_query"]

        for result in vector_results:
            add_result(result)

        for result in bm25_results:
            add_result(result)

        return list(combined.values())

    def rerank_candidates(
        self,
        question: str,
        candidates: list[dict],
    ) -> list[dict]:
        """
        Локальный reranking найденных chunks.

        Учитывает:
        - vector_score
        - bm25_score
        - query expansion
        - покрытие слов исходного запроса
        - покрытие слов расширенного запроса
        - точное совпадение фразы
        - поля вида "Слоган:"
        - найден ли chunk сразу двумя способами
        """
        if not candidates:
            return []

        original_tokens, expanded_tokens, _expanded_text = expand_query_tokens(question)

        original_token_set = set(original_tokens)
        expanded_token_set = set(expanded_tokens)
        expanded_only_token_set = expanded_token_set - original_token_set

        normalized_question = normalize_text_for_search(question)

        max_bm25 = max(safe_float(item.get("bm25_score")) for item in candidates) or 1.0

        reranked: list[dict] = []

        for item in candidates:
            text = item.get("text", "")
            normalized_text = normalize_text_for_search(text)
            token_set = set(tokenize(text))

            vector_score = safe_float(item.get("vector_score"))
            bm25_score = safe_float(item.get("bm25_score"))

            # Нормализация vector score.
            if vector_score <= 0:
                vector_norm = 0.0
            elif vector_score >= settings.min_relevance_score:
                vector_norm = min(
                    1.0,
                    (vector_score - settings.min_relevance_score)
                    / max(1e-6, 1.0 - settings.min_relevance_score),
                )
            else:
                vector_norm = 0.0

            # Нормализация BM25.
            bm25_norm = bm25_score / max_bm25 if max_bm25 else 0.0

            # Покрытие исходного запроса.
            if original_token_set:
                original_coverage = len(original_token_set.intersection(token_set)) / len(original_token_set)
            else:
                original_coverage = 0.0

            # Покрытие расширенного запроса.
            if expanded_token_set:
                expanded_coverage = len(expanded_token_set.intersection(token_set)) / len(expanded_token_set)
            else:
                expanded_coverage = 0.0

            # Query expansion не должен перебивать исходный вопрос,
            # поэтому исходное покрытие весит сильнее.
            coverage = 0.75 * original_coverage + 0.25 * expanded_coverage

            # Бонус за точную фразу.
            exact_phrase_bonus = 0.0

            if normalized_question and normalized_question in normalized_text:
                exact_phrase_bonus = 0.15

            # Бонус за поля вида "слоган:", "состав:".
            field_bonus = 0.0

            for token in original_tokens:
                if f"{token}:" in normalized_text:
                    field_bonus += 0.10

            for token in expanded_only_token_set:
                if f"{token}:" in normalized_text:
                    field_bonus += 0.05

            field_bonus = min(field_bonus, 0.25)

            # Бонус за важные поля.
            important_bonus = 0.0
            matched_important = IMPORTANT_FIELDS.intersection(expanded_token_set)

            for field in matched_important:
                if field in token_set:
                    important_bonus += 0.05

            important_bonus = min(important_bonus, 0.15)

            # Бонус, если query expansion реально помог найти совпадение.
            expansion_bonus = 0.0

            if expanded_only_token_set.intersection(token_set):
                expansion_bonus = 0.08

            # Бонус, если chunk найден и в vector, и в BM25.
            search_types = item.get("search_types", set())
            both_search_bonus = 0.10 if len(search_types) > 1 else 0.0

            # Финальный rerank score.
            rerank_score = (
                0.48 * vector_norm
                + 0.34 * bm25_norm
                + 0.18 * coverage
                + exact_phrase_bonus
                + field_bonus
                + important_bonus
                + expansion_bonus
                + both_search_bonus
            )

            item = item.copy()
            item["rank_score"] = rerank_score
            item["score"] = rerank_score
            item["coverage"] = coverage
            item["original_coverage"] = original_coverage
            item["expanded_coverage"] = expanded_coverage
            item["vector_norm"] = vector_norm
            item["bm25_norm"] = bm25_norm

            reranked.append(item)

        reranked.sort(key=lambda x: x["rank_score"], reverse=True)

        return reranked

    async def search(
        self,
        question: str,
        question_vector: np.ndarray,
    ) -> list[dict]:
        """
        Новый поиск:
        1. Vector search
        2. BM25 search с query expansion
        3. Merge candidates
        4. Reranking
        5. TOP_K лучших chunks
        """
        candidate_limit = max(settings.top_k * 5, 25)

        vector_results = await self.vector_search(
            question_vector=question_vector,
            limit=candidate_limit,
        )

        bm25_results = self.bm25_search(
            question=question,
            limit=candidate_limit,
        )

        candidates = self.merge_candidates(
            vector_results=vector_results,
            bm25_results=bm25_results,
        )

        reranked = self.rerank_candidates(
            question=question,
            candidates=candidates,
        )

        return reranked[: settings.top_k]

    async def debug_search(self, question: str) -> list[dict]:
        """
        Поиск chunks без генерации ответа.
        Используется командой /debug_search.
        """
        question_vector = await self._embed_query(question)
        return await self.search(question=question, question_vector=question_vector)

    async def answer(self, question: str) -> str:
        exact_cache_hit = self.cache.find_exact(question)

        if exact_cache_hit:
            logger.info("Ответ найден в exact cache")
            return exact_cache_hit.answer

        question_vector = await self._embed_query(question)

        semantic_cache_hit = self.cache.find_semantic(
            question=question,
            question_vector=question_vector,
        )

        if semantic_cache_hit:
            logger.info(
                "Ответ найден в semantic cache. Similarity: %.3f",
                semantic_cache_hit.similarity,
            )
            return semantic_cache_hit.answer

        hits = await self.search(
            question=question,
            question_vector=question_vector,
        )

        if not hits:
            return "В базе знаний нет информации для ответа на этот вопрос."

        context = "\n\n".join(
            f"[Источник {i}]\n"
            f"Файл: {hit['file_name']}\n"
            f"Страница: {hit['page']}\n"
            f"Текст:\n{hit['text']}"
            for i, hit in enumerate(hits, start=1)
        )

        user_prompt = f"""
CONTEXT:
{context}

QUESTION:
{question}
""".strip()

        response = await self.client.chat.completions.create(
            model=settings.chat_model,
            temperature=0,
            max_tokens=settings.max_answer_tokens,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
        )

        answer_text = (response.choices[0].message.content or "").strip()

        if not answer_text:
            answer_text = "Не удалось сформировать ответ по найденным данным."

        sources = []
        used = set()

        for hit in hits:
            key = (hit["file_name"], hit["page"])

            if key in used:
                continue

            used.add(key)
            sources.append(f"• {hit['file_name']}, стр. {hit['page']}")

        # Источники добавляются в технический ответ,
        # но bot.py скрывает их от пользователя и сохраняет в логи.
        final_answer = f"{answer_text}\n\n📎 Источники:\n" + "\n".join(sources)

        self.cache.add(
            question=question,
            question_vector=question_vector,
            answer=final_answer,
        )

        return final_answer

    def clear_cache(self) -> int:
        return self.cache.clear()

    def get_status(self) -> dict:
        return {
            "storage_dir": str(self.storage_dir),
            "index_path": str(self.index_path),
            "chunks_path": str(self.chunks_path),
            "index_exists": self.index_path.exists(),
            "chunks_exists": self.chunks_path.exists(),
            "chunks_count": len(self.chunks),
            "embeddings_shape": tuple(self.embeddings.shape),
            "knowledge_base_hash": self.knowledge_base_hash,
            "cache_enabled": self.cache.enabled,
            "cache_path": str(self.cache.cache_path),
            "cache_items": self.cache.size(),
            "chat_model": settings.chat_model,
            "embedding_model": settings.embedding_model,
            "top_k": settings.top_k,
            "min_relevance_score": settings.min_relevance_score,
            "bm25_docs": len(self.bm25_index),
            "bm25_avg_doc_len": round(self.bm25_avg_doc_len, 2),
            "query_expansion_terms": len(QUERY_EXPANSIONS),
        }

    def get_version_text(self) -> str:
        status = self.get_status()

        return (
            "📦 Версия базы знаний\n\n"
            f"Hash: <code>{status['knowledge_base_hash']}</code>\n"
            f"Short hash: <code>{status['knowledge_base_hash'][:12]}</code>\n"
            f"Chunks: <code>{status['chunks_count']}</code>\n"
            f"BM25 docs: <code>{status['bm25_docs']}</code>\n"
            f"BM25 avg doc len: <code>{status['bm25_avg_doc_len']}</code>\n"
            f"Query expansion terms: <code>{status['query_expansion_terms']}</code>\n"
            f"Embeddings shape: <code>{status['embeddings_shape']}</code>\n"
            f"Embedding model: <code>{status['embedding_model']}</code>"
        )