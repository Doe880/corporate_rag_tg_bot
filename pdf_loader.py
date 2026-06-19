from __future__ import annotations

import re
from pathlib import Path
from typing import Iterator

import fitz  # PyMuPDF


def clean_text(text: str) -> str:
    """Убирает лишние пробелы и переносы, чтобы chunks были чище."""
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def iter_pdf_pages(pdf_path: Path) -> Iterator[dict]:
    """
    Возвращает текст по страницам PDF.
    Важно: сканы без OCR не прочитаются. Для сканов нужен отдельный OCR-этап.
    """
    with fitz.open(pdf_path) as doc:
        for page_index, page in enumerate(doc, start=1):
            text = clean_text(page.get_text("text"))
            if text:
                yield {
                    "file_name": pdf_path.name,
                    "page": page_index,
                    "text": text,
                }
