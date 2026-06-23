from __future__ import annotations

import json
import logging
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


class RAGEngine:
    def __init__(self) -> None:
        self.client = AsyncOpenAI(api_key=settings.openai_api_key)
        self.storage_dir = Path(settings.storage_dir)

        self.index_path = self.storage_dir / INDEX_FILE
        self.chunks_path = self.storage_dir / CHUNKS_FILE

        self.embeddings: np.ndarray
        self.chunks: list[dict]
        self.knowledge_base_hash: str
        self.keyword_index: list[dict]
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
        self.keyword_index = self._build_keyword_index()

        self.cache = AnswerCache(
            storage_dir=self.storage_dir,
            knowledge_base_hash=self.knowledge_base_hash,
        )

        logger.info(
            "RAG reloaded. chunks=%s, embeddings_shape=%s, kb_hash=%s",
            len(self.chunks),
            self.embeddings.shape,
            self.knowledge_base_hash[:12],
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

    def _build_keyword_index(self) -> list[dict]:
        index: list[dict] = []

        for chunk in self.chunks:
            text = chunk.get("text", "")
            tokens = tokenize(text)

            index.append(
                {
                    "id": chunk["id"],
                    "file_name": chunk["file_name"],
                    "page": chunk["page"],
                    "chunk_no": chunk["chunk_no"],
                    "text": text,
                    "normalized_text": normalize_text_for_search(text),
                    "token_counts": Counter(tokens),
                    "token_set": set(tokens),
                }
            )

        return index

    async def _embed_query(self, question: str) -> np.ndarray:
        response = await self.client.embeddings.create(
            model=settings.embedding_model,
            input=question,
        )

        vector = np.array([response.data[0].embedding], dtype=np.float32)
        return normalize_vectors(vector)[0]

    async def vector_search(self, question_vector: np.ndarray) -> list[dict]:
        scores = self.embeddings @ question_vector
        top_indices = np.argsort(scores)[::-1][: settings.top_k]

        results: list[dict] = []

        for idx in top_indices:
            score = float(scores[idx])

            if score < settings.min_relevance_score:
                continue

            chunk = self.chunks[int(idx)]

            results.append(
                {
                    "id": chunk["id"],
                    "score": score,
                    "rank_score": score,
                    "search_type": "vector",
                    "file_name": chunk["file_name"],
                    "page": chunk["page"],
                    "chunk_no": chunk["chunk_no"],
                    "text": chunk["text"],
                }
            )

        return results

    def keyword_search(self, question: str) -> list[dict]:
        normalized_question = normalize_text_for_search(question)
        query_tokens = tokenize(question)

        if not query_tokens and not normalized_question:
            return []

        results: list[dict] = []

        important_fields = {
            "слоган",
            "состав",
            "противопоказания",
            "показания",
            "дозировка",
            "применение",
            "рекомендации",
            "сообщение",
            "форма",
            "выпуска",
            "описание",
            "имидж",
        }

        for item in self.keyword_index:
            score = 0.0

            text = item["normalized_text"]
            token_counts: Counter = item["token_counts"]
            token_set: set[str] = item["token_set"]

            if normalized_question and normalized_question in text:
                score += 20.0

            for token in query_tokens:
                count = token_counts.get(token, 0)

                if count:
                    score += count * 3.0

            if query_tokens and all(token in token_set for token in query_tokens):
                score += 8.0

            for token in query_tokens:
                if f"{token}:" in text:
                    score += 15.0

            matched_important_fields = important_fields.intersection(query_tokens)

            for field in matched_important_fields:
                if field in token_set:
                    score += 10.0

            if score > 0:
                results.append(
                    {
                        "id": item["id"],
                        "score": score,
                        "rank_score": min(score / 10.0, 2.0),
                        "search_type": "keyword",
                        "file_name": item["file_name"],
                        "page": item["page"],
                        "chunk_no": item["chunk_no"],
                        "text": item["text"],
                    }
                )

        results.sort(key=lambda x: x["rank_score"], reverse=True)

        return results[: max(settings.top_k, 10)]

    async def search(
        self,
        question: str,
        question_vector: np.ndarray,
    ) -> list[dict]:
        vector_results = await self.vector_search(question_vector)
        keyword_results = self.keyword_search(question)

        combined: dict[int, dict] = {}

        def add_result(result: dict) -> None:
            chunk_id = int(result["id"])

            if chunk_id not in combined:
                combined[chunk_id] = result.copy()
                combined[chunk_id]["search_types"] = {result["search_type"]}
            else:
                combined[chunk_id]["rank_score"] += result["rank_score"]
                combined[chunk_id]["search_types"].add(result["search_type"])
                combined[chunk_id]["score"] = max(
                    float(combined[chunk_id].get("score", 0)),
                    float(result.get("score", 0)),
                )

        for result in vector_results:
            add_result(result)

        for result in keyword_results:
            add_result(result)

        final_results = list(combined.values())

        for result in final_results:
            if len(result.get("search_types", set())) > 1:
                result["rank_score"] += 0.5

        final_results.sort(key=lambda x: x["rank_score"], reverse=True)

        return final_results[: settings.top_k]

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

        final_answer = f"{answer_text}\n\n📎 Источники:\n" + "\n".join(sources)

        self.cache.add(
            question=question,
            question_vector=question_vector,
            answer=final_answer,
        )

        return final_answer

    def clear_cache(self) -> int:
        """
        Очищает кэш ответов.
        """
        return self.cache.clear()

    def get_status(self) -> dict:
        """
        Возвращает технический статус RAG.
        """
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
        }

    def get_version_text(self) -> str:
        """
        Короткаяz информация о версии базы знаний.
        """
        status = self.get_status()

        return (
            "📦 Версия базы знаний\n\n"
            f"Hash: <code>{status['knowledge_base_hash']}</code>\n"
            f"Short hash: <code>{status['knowledge_base_hash'][:12]}</code>\n"
            f"Chunks: <code>{status['chunks_count']}</code>\n"
            f"Embeddings shape: <code>{status['embeddings_shape']}</code>\n"
            f"Embedding model: <code>{status['embedding_model']}</code>"
        )