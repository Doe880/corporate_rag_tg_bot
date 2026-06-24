import os
from dataclasses import dataclass
from pathlib import Path

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


def _get_int_set(name: str) -> set[int]:
    """
    Читает список Telegram user_id из переменной окружения.

    Пример:
    ADMIN_USER_IDS=123456789,987654321
    """
    raw = os.getenv(name, "").strip()

    if not raw:
        return set()

    result: set[int] = set()

    for item in raw.replace(";", ",").split(","):
        item = item.strip()

        if not item:
            continue

        try:
            result.add(int(item))
        except ValueError:
            raise RuntimeError(
                f"Некорректное значение в {name}: {item}. "
                "Ожидаются только числовые Telegram user_id через запятую."
            )

    return result


@dataclass(frozen=True)
class Settings:
    telegram_bot_token: str

    admin_user_ids: set[int]
    allowed_user_ids: set[int]

    db_path: str

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


storage_dir = os.getenv("STORAGE_DIR", "storage")

settings = Settings(
    telegram_bot_token=_get_required("TELEGRAM_BOT_TOKEN"),

    admin_user_ids=_get_int_set("ADMIN_USER_IDS"),
    allowed_user_ids=_get_int_set("ALLOWED_USER_IDS"),

    db_path=os.getenv("DB_PATH", str(Path(storage_dir) / "bot.db")),

    openai_api_key=_get_required("OPENAI_API_KEY"),
    chat_model=os.getenv("OPENAI_CHAT_MODEL", "gpt-4.1-mini"),
    embedding_model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),

    private_docs_dir=os.getenv("PRIVATE_DOCS_DIR", "private_docs"),
    storage_dir=storage_dir,

    chunk_size_tokens=_get_int("CHUNK_SIZE_TOKENS", 350),
    chunk_overlap_tokens=_get_int("CHUNK_OVERLAP_TOKENS", 70),
    top_k=_get_int("TOP_K", 10),
    min_relevance_score=_get_float("MIN_RELEVANCE_SCORE", 0.20),
    max_answer_tokens=_get_int("MAX_ANSWER_TOKENS", 900),
)