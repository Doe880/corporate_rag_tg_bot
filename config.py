import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()


def _get_required(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Не задана переменная окружения {name}. Проверьте файл .env")
    return value


def _get_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    return int(raw) if raw else default


def _get_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    return float(raw) if raw else default


@dataclass(frozen=True)
class Settings:
    telegram_bot_token: str
    openai_api_key: str
    chat_model: str
    embedding_model: str
    private_docs_dir: str
    storage_dir: str
    chunk_size_tokens: int
    chunk_overlap_tokens: int
    top_k: int
    min_relevance_score: float
    max_answer_tokens: int


settings = Settings(
    telegram_bot_token=_get_required("TELEGRAM_BOT_TOKEN"),
    openai_api_key=_get_required("OPENAI_API_KEY"),
    chat_model=os.getenv("OPENAI_CHAT_MODEL", "gpt-4.1-mini"),
    embedding_model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
    private_docs_dir=os.getenv("PRIVATE_DOCS_DIR", "private_docs"),
    storage_dir=os.getenv("STORAGE_DIR", "storage"),
    chunk_size_tokens=_get_int("CHUNK_SIZE_TOKENS", 350),
    chunk_overlap_tokens=_get_int("CHUNK_OVERLAP_TOKENS", 70),
    top_k=_get_int("TOP_K", 10),
    min_relevance_score=_get_float("MIN_RELEVANCE_SCORE", 0.20),
    max_answer_tokens=_get_int("MAX_ANSWER_TOKENS", 900),
)