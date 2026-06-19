from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np
import tiktoken
from openai import OpenAI

from config import settings
from pdf_loader import iter_pdf_pages


INDEX_FILE = "index.npz"
CHUNKS_FILE = "chunks.json"


def get_tokenizer():
    # Для разбиения текста достаточно универсальной кодировки.
    return tiktoken.get_encoding("cl100k_base")


def split_text_by_tokens(text: str, max_tokens: int, overlap_tokens: int) -> list[str]:
    if overlap_tokens >= max_tokens:
        raise ValueError("CHUNK_OVERLAP_TOKENS должен быть меньше CHUNK_SIZE_TOKENS")

    enc = get_tokenizer()
    tokens = enc.encode(text)
    chunks: list[str] = []

    start = 0
    while start < len(tokens):
        end = min(start + max_tokens, len(tokens))
        chunk_tokens = tokens[start:end]
        chunk_text = enc.decode(chunk_tokens).strip()
        if chunk_text:
            chunks.append(chunk_text)
        if end == len(tokens):
            break
        start = end - overlap_tokens

    return chunks


def batched(items: list[str], batch_size: int) -> Iterable[list[str]]:
    for i in range(0, len(items), batch_size):
        yield items[i:i + batch_size]


def normalize_vectors(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1
    return vectors / norms


def main() -> None:
    docs_dir = Path(settings.private_docs_dir)
    storage_dir = Path(settings.storage_dir)
    storage_dir.mkdir(parents=True, exist_ok=True)

    pdf_files = sorted(docs_dir.glob("*.pdf"))
    if not pdf_files:
        raise RuntimeError(
            f"В папке {docs_dir.resolve()} нет PDF-файлов. "
            "Положите туда корпоративные PDF и повторите индексацию."
        )

    chunks: list[dict] = []
    for pdf_path in pdf_files:
        print(f"Читаю PDF: {pdf_path.name}")
        for page in iter_pdf_pages(pdf_path):
            for chunk_no, chunk_text in enumerate(
                split_text_by_tokens(
                    page["text"],
                    max_tokens=settings.chunk_size_tokens,
                    overlap_tokens=settings.chunk_overlap_tokens,
                ),
                start=1,
            ):
                chunks.append({
                    "id": len(chunks),
                    "file_name": page["file_name"],
                    "page": page["page"],
                    "chunk_no": chunk_no,
                    "text": chunk_text,
                })

    if not chunks:
        raise RuntimeError("Не удалось извлечь текст из PDF. Возможно, PDF состоит из сканов без OCR.")

    print(f"Всего chunks: {len(chunks)}")
    texts = [chunk["text"] for chunk in chunks]

    client = OpenAI(api_key=settings.openai_api_key)
    embeddings: list[list[float]] = []

    for batch in batched(texts, batch_size=80):
        response = client.embeddings.create(
            model=settings.embedding_model,
            input=batch,
        )
        embeddings.extend([item.embedding for item in response.data])
        print(f"Создано embeddings: {len(embeddings)}/{len(texts)}")

    matrix = np.array(embeddings, dtype=np.float32)
    matrix = normalize_vectors(matrix)

    np.savez_compressed(storage_dir / INDEX_FILE, embeddings=matrix)
    with open(storage_dir / CHUNKS_FILE, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)

    print("Индекс готов:")
    print(f"- {storage_dir / INDEX_FILE}")
    print(f"- {storage_dir / CHUNKS_FILE}")
    print("Важно: папку storage нельзя коммитить в публичный GitHub.")


if __name__ == "__main__":
    main()
