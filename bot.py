from __future__ import annotations

import asyncio
import logging
import os
from html import escape
from pathlib import Path

from aiogram import Bot, Dispatcher, F
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ChatAction, ParseMode
from aiogram.filters import Command, CommandStart
from aiogram.types import CallbackQuery, InlineKeyboardButton, InlineKeyboardMarkup, Message

from analytics import AnalyticsStore
from auth import AuthStore
from config import settings
from ingest import CHUNKS_FILE, INDEX_FILE
from rag import RAGEngine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

rag_engine: RAGEngine | None = None
auth_store: AuthStore | None = None
analytics_store: AnalyticsStore | None = None


def split_for_telegram(text: str, limit: int = 3900) -> list[str]:
    """
    Telegram ограничивает длину одного сообщения.
    Делим длинный ответ на части.
    """
    parts: list[str] = []

    while len(text) > limit:
        cut = text.rfind("\n", 0, limit)

        if cut == -1:
            cut = limit

        parts.append(text[:cut].strip())
        text = text[cut:].strip()

    if text:
        parts.append(text)

    return parts


def get_user_id(message: Message) -> int | None:
    if message.from_user is None:
        return None

    return message.from_user.id


def get_callback_user_id(callback: CallbackQuery) -> int | None:
    if callback.from_user is None:
        return None

    return callback.from_user.id


def get_command_arg(message: Message) -> str:
    text = message.text or ""
    parts = text.split(maxsplit=1)

    if len(parts) < 2:
        return ""

    return parts[1].strip()


def parse_user_id(value: str) -> int | None:
    value = value.strip()

    if not value:
        return None

    try:
        return int(value)
    except ValueError:
        return None


def format_user_record(record: dict) -> str:
    user_id = record.get("user_id")
    username = record.get("username")
    full_name = record.get("full_name")

    username_text = f"@{escape(username)}" if username else "без username"
    full_name_text = escape(full_name) if full_name else "без имени"

    return f"<code>{user_id}</code> — {username_text} — {full_name_text}"


def auth_keyboard(user_id: int) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(
                    text="✅ Разрешить",
                    callback_data=f"auth:approve:{user_id}",
                ),
                InlineKeyboardButton(
                    text="❌ Отклонить",
                    callback_data=f"auth:deny:{user_id}",
                ),
            ]
        ]
    )


def feedback_keyboard(answer_id: str) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(
                    text="👍 Полезно",
                    callback_data=f"fb:good:{answer_id}",
                ),
                InlineKeyboardButton(
                    text="👎 Неверно",
                    callback_data=f"fb:bad:{answer_id}",
                ),
            ]
        ]
    )


def extract_sources_from_answer(answer: str) -> list[str]:
    """
    Достаёт источники из текста ответа.

    Ожидаемый формат:
    📎 Источники:
    • file.pdf, стр. 3
    """
    marker = "📎 Источники:"

    if marker not in answer:
        return []

    sources_block = answer.split(marker, maxsplit=1)[1]

    sources: list[str] = []

    for line in sources_block.splitlines():
        line = line.strip()

        if not line:
            continue

        line = line.lstrip("•").strip()

        if line:
            sources.append(line)

    return sources


def expected_paths_text() -> str:
    storage_dir = Path(settings.storage_dir)
    index_path = storage_dir / INDEX_FILE
    chunks_path = storage_dir / CHUNKS_FILE
    cache_file = os.getenv("CACHE_FILE", "answer_cache.json")
    cache_path = storage_dir / cache_file

    return (
        f"STORAGE_DIR: <code>{escape(str(storage_dir))}</code>\n"
        f"index.npz: <code>{escape(str(index_path))}</code>\n"
        f"chunks.json: <code>{escape(str(chunks_path))}</code>\n"
        f"cache: <code>{escape(str(cache_path))}</code>"
    )


async def notify_admins_about_request(bot: Bot, user_record: dict) -> None:
    if not settings.admin_user_ids:
        logger.error("ADMIN_USER_IDS пустой. Некому отправить заявку на доступ.")
        return

    user_id = int(user_record["user_id"])
    username = user_record.get("username")
    full_name = user_record.get("full_name")
    requested_at = user_record.get("requested_at")

    username_text = f"@{escape(username)}" if username else "не указан"
    full_name_text = escape(full_name) if full_name else "не указано"

    text = (
        "🔐 Новая заявка на доступ к RAG-боту\n\n"
        f"User ID: <code>{user_id}</code>\n"
        f"Username: {username_text}\n"
        f"Имя: {full_name_text}\n"
        f"Дата заявки: {escape(str(requested_at))}\n\n"
        "Разрешить доступ?"
    )

    for admin_id in settings.admin_user_ids:
        try:
            await bot.send_message(
                chat_id=admin_id,
                text=text,
                reply_markup=auth_keyboard(user_id),
            )
        except Exception:
            logger.exception("Не удалось отправить заявку админу %s", admin_id)


async def request_access(message: Message) -> None:
    global auth_store

    if auth_store is None:
        await message.answer("Система авторизации не инициализирована.")
        return

    if message.from_user is None:
        await message.answer("Не удалось определить ваш Telegram user_id.")
        return

    user_id = message.from_user.id
    username = message.from_user.username
    full_name = message.from_user.full_name

    is_new = auth_store.add_pending(
        user_id=user_id,
        username=username,
        full_name=full_name,
    )

    if is_new:
        pending = auth_store.get_pending(user_id)

        if pending:
            await notify_admins_about_request(message.bot, pending)

        await message.answer(
            "⏳ У вас пока нет доступа к этому боту.\n\n"
            "Заявка отправлена администратору. "
            "Когда администратор подтвердит доступ, вы сможете пользоваться ботом."
        )
    else:
        await message.answer(
            "⏳ У вас пока нет доступа к этому боту.\n\n"
            "Ваша заявка уже отправлена администратору. "
            "Дождитесь подтверждения."
        )


async def ensure_access(message: Message) -> bool:
    global auth_store

    if auth_store is None:
        await message.answer("Система авторизации не инициализирована.")
        return False

    user_id = get_user_id(message)

    if not auth_store.is_allowed(user_id):
        logger.warning("Попытка доступа без разрешения. user_id=%s", user_id)
        await request_access(message)
        return False

    return True


async def ensure_admin(message: Message) -> bool:
    global auth_store

    if auth_store is None:
        await message.answer("Система авторизации не инициализирована.")
        return False

    user_id = get_user_id(message)

    if not auth_store.is_admin(user_id):
        await message.answer("⛔ Эта команда доступна только администратору.")
        return False

    return True


async def cmd_id(message: Message) -> None:
    if message.from_user is None:
        await message.answer("Не удалось определить ваш Telegram user_id.")
        return

    user_id = message.from_user.id
    username = message.from_user.username
    full_name = escape(message.from_user.full_name or "")

    username_text = f"@{escape(username)}" if username else "не указан"

    await message.answer(
        "Ваши данные Telegram:\n\n"
        f"User ID: <code>{user_id}</code>\n"
        f"Username: {username_text}\n"
        f"Имя: {full_name}"
    )


async def cmd_start(message: Message) -> None:
    if not await ensure_access(message):
        return

    await message.answer(
        "Здравствуйте! Я корпоративный AI-ассистент.\n\n"
        "Задайте вопрос, а я найду ответ в базе знаний. "
        "Если информации в базе нет, я так и отвечу."
    )


async def cmd_help(message: Message) -> None:
    if not await ensure_access(message):
        return

    await message.answer(
        "Как пользоваться:\n"
        "1. Напишите вопрос обычным сообщением.\n"
        "2. Я выполню поиск по базе.\n"
        "3. Ответ будет сформирован только по найденным фрагментам.\n\n"
        "Основные команды:\n"
        "/id — узнать свой Telegram user_id\n"
        "/help — помощь\n\n"
        "Админ-команды RAG:\n"
        "/status — статус базы знаний\n"
        "/reload — перезагрузить базу без перезапуска контейнера\n"
        "/clear_cache — очистить кэш ответов\n"
        "/version — версия базы знаний\n"
        "/debug_search запрос — показать найденные chunks\n\n"
        "Админ-команды авторизации:\n"
        "/users — список пользователей\n"
        "/pending — список заявок\n"
        "/allow user_id — добавить пользователя\n"
        "/revoke user_id — удалить пользователя\n\n"
        "Админ-команды аналитики:\n"
        "/stats — статистика\n"
        "/popular — популярные запросы\n"
        "/feedback — последние негативные оценки"
    )


async def cmd_status(message: Message) -> None:
    global rag_engine

    if not await ensure_admin(message):
        return

    if rag_engine is None:
        await message.answer(
            "⚠️ RAG-индекс не загружен.\n\n"
            "Ожидаемые пути:\n"
            f"{expected_paths_text()}\n\n"
            "Проверьте, что index.npz и chunks.json загружены в STORAGE_DIR."
        )
        return

    status = rag_engine.get_status()

    text = (
        "✅ RAG-индекс загружен\n\n"
        f"Storage: <code>{escape(status['storage_dir'])}</code>\n"
        f"Index exists: <code>{status['index_exists']}</code>\n"
        f"Chunks exists: <code>{status['chunks_exists']}</code>\n"
        f"Chunks count: <code>{status['chunks_count']}</code>\n"
        f"Embeddings shape: <code>{status['embeddings_shape']}</code>\n"
        f"KB hash: <code>{status['knowledge_base_hash'][:12]}</code>\n"
        f"Cache enabled: <code>{status['cache_enabled']}</code>\n"
        f"Cache items: <code>{status['cache_items']}</code>\n"
        f"Chat model: <code>{escape(status['chat_model'])}</code>\n"
        f"Embedding model: <code>{escape(status['embedding_model'])}</code>\n"
        f"TOP_K: <code>{status['top_k']}</code>\n"
        f"MIN_RELEVANCE_SCORE: <code>{status['min_relevance_score']}</code>"
    )

    if "bm25_docs" in status:
        text += (
            f"\nBM25 docs: <code>{status['bm25_docs']}</code>"
            f"\nBM25 avg doc len: <code>{status['bm25_avg_doc_len']}</code>"
        )

    await message.answer(text)


async def cmd_reload(message: Message) -> None:
    global rag_engine

    if not await ensure_admin(message):
        return

    await message.answer("🔄 Перезагружаю RAG-индекс...")

    try:
        if rag_engine is None:
            rag_engine = RAGEngine()
        else:
            rag_engine.reload()

        status = rag_engine.get_status()

        await message.answer(
            "✅ RAG-индекс успешно перезагружен.\n\n"
            f"Chunks: <code>{status['chunks_count']}</code>\n"
            f"Embeddings shape: <code>{status['embeddings_shape']}</code>\n"
            f"KB hash: <code>{status['knowledge_base_hash'][:12]}</code>"
        )
    except Exception as exc:
        logger.exception("Ошибка при reload RAG")
        rag_engine = None

        await message.answer(
            "❌ Не удалось перезагрузить RAG-индекс.\n\n"
            f"Ошибка: <code>{escape(str(exc))}</code>\n\n"
            "Проверьте файлы:\n"
            f"{expected_paths_text()}"
        )


async def cmd_clear_cache(message: Message) -> None:
    global rag_engine

    if not await ensure_admin(message):
        return

    if rag_engine is not None:
        deleted_count = rag_engine.clear_cache()

        await message.answer(
            "✅ Кэш очищен.\n\n"
            f"Удалено записей: <code>{deleted_count}</code>"
        )
        return

    storage_dir = Path(settings.storage_dir)
    cache_file = os.getenv("CACHE_FILE", "answer_cache.json")
    cache_path = storage_dir / cache_file

    if cache_path.exists():
        cache_path.unlink()
        await message.answer(
            "✅ Файл кэша удалён.\n\n"
            f"Путь: <code>{escape(str(cache_path))}</code>"
        )
    else:
        await message.answer(
            "Кэш не найден.\n\n"
            f"Путь: <code>{escape(str(cache_path))}</code>"
        )


async def cmd_version(message: Message) -> None:
    global rag_engine

    if not await ensure_admin(message):
        return

    if rag_engine is None:
        await message.answer(
            "⚠️ RAG-индекс не загружен. Версия базы недоступна.\n\n"
            f"{expected_paths_text()}"
        )
        return

    await message.answer(rag_engine.get_version_text())


async def cmd_debug_search(message: Message) -> None:
    global rag_engine

    if not await ensure_admin(message):
        return

    query = get_command_arg(message)

    if not query:
        await message.answer(
            "Укажите запрос.\n\n"
            "Пример:\n"
            "<code>/debug_search слоган</code>"
        )
        return

    if rag_engine is None:
        await message.answer("⚠️ RAG-индекс не загружен.")
        return

    await message.bot.send_chat_action(message.chat.id, ChatAction.TYPING)

    try:
        hits = await rag_engine.debug_search(query)
    except Exception as exc:
        logger.exception("Ошибка debug_search")
        await message.answer(f"❌ Ошибка поиска: <code>{escape(str(exc))}</code>")
        return

    if not hits:
        await message.answer(
            "Ничего не найдено.\n\n"
            f"Запрос: <code>{escape(query)}</code>"
        )
        return

    lines = [
        "🔎 Debug search",
        "",
        f"Запрос: <code>{escape(query)}</code>",
        f"Найдено chunks: <code>{len(hits)}</code>",
        "",
    ]

    for index, hit in enumerate(hits, start=1):
        search_types = hit.get("search_types", {hit.get("search_type", "unknown")})

        if isinstance(search_types, set):
            search_types_text = ", ".join(sorted(search_types))
        elif isinstance(search_types, list):
            search_types_text = ", ".join(search_types)
        else:
            search_types_text = str(search_types)

        preview = hit.get("text", "")
        preview = preview.replace("\n", " ")
        preview = preview[:700]

        lines.append(
            f"{index}. <b>{escape(hit['file_name'])}</b>, стр. <code>{hit['page']}</code>\n"
            f"chunk_no: <code>{hit['chunk_no']}</code>\n"
            f"score: <code>{round(float(hit.get('score', 0)), 4)}</code>\n"
            f"rank_score: <code>{round(float(hit.get('rank_score', 0)), 4)}</code>\n"
            f"search: <code>{escape(search_types_text)}</code>\n"
            f"preview: {escape(preview)}\n"
        )

    full_text = "\n".join(lines)

    for part in split_for_telegram(full_text):
        await message.answer(part)


async def cmd_users(message: Message) -> None:
    global auth_store

    if not await ensure_admin(message):
        return

    assert auth_store is not None

    users = auth_store.list_users()

    if not users:
        await message.answer("Список разрешённых пользователей пуст.")
        return

    lines = ["👥 Разрешённые пользователи:\n"]

    for index, user in enumerate(users, start=1):
        lines.append(f"{index}. {format_user_record(user)}")

    await message.answer("\n".join(lines))


async def cmd_pending(message: Message) -> None:
    global auth_store

    if not await ensure_admin(message):
        return

    assert auth_store is not None

    pending = auth_store.list_pending()

    if not pending:
        await message.answer("Новых заявок нет.")
        return

    lines = ["⏳ Заявки на доступ:\n"]

    for index, user in enumerate(pending, start=1):
        lines.append(f"{index}. {format_user_record(user)}")

    await message.answer("\n".join(lines))


async def cmd_allow(message: Message) -> None:
    global auth_store

    if not await ensure_admin(message):
        return

    assert auth_store is not None

    user_id = parse_user_id(get_command_arg(message))

    if user_id is None:
        await message.answer(
            "Укажите user_id.\n\n"
            "Пример:\n"
            "<code>/allow 123456789</code>"
        )
        return

    admin_id = get_user_id(message)

    if admin_id is None:
        await message.answer("Не удалось определить ID администратора.")
        return

    auth_store.allow_user(
        user_id=user_id,
        admin_id=admin_id,
    )

    await message.answer(f"✅ Пользователь <code>{user_id}</code> добавлен.")


async def cmd_revoke(message: Message) -> None:
    global auth_store

    if not await ensure_admin(message):
        return

    assert auth_store is not None

    user_id = parse_user_id(get_command_arg(message))

    if user_id is None:
        await message.answer(
            "Укажите user_id.\n\n"
            "Пример:\n"
            "<code>/revoke 123456789</code>"
        )
        return

    success, text = auth_store.revoke_user(user_id)

    if success:
        await message.answer(f"✅ {text}\nUser ID: <code>{user_id}</code>")
    else:
        await message.answer(f"⚠️ {text}\nUser ID: <code>{user_id}</code>")


async def cmd_stats(message: Message) -> None:
    global analytics_store

    if not await ensure_admin(message):
        return

    if analytics_store is None:
        await message.answer("Система аналитики не инициализирована.")
        return

    stats = analytics_store.get_stats()

    text = (
        "📊 Статистика бота\n\n"
        f"Всего запросов: <code>{stats['total_queries']}</code>\n"
        f"Уникальных пользователей: <code>{stats['unique_users']}</code>\n"
        f"Всего оценок: <code>{stats['total_feedback']}</code>\n"
        f"👍 Полезно: <code>{stats['good_feedback']}</code>\n"
        f"👎 Неверно: <code>{stats['bad_feedback']}</code>\n\n"
        f"Query log: <code>{escape(stats['query_log_path'])}</code>\n"
        f"Feedback log: <code>{escape(stats['feedback_log_path'])}</code>"
    )

    await message.answer(text)


async def cmd_popular(message: Message) -> None:
    global analytics_store

    if not await ensure_admin(message):
        return

    if analytics_store is None:
        await message.answer("Система аналитики не инициализирована.")
        return

    items = analytics_store.get_popular_queries(limit=10)

    if not items:
        await message.answer("Пока нет статистики по запросам.")
        return

    lines = ["🔥 Самые популярные запросы:\n"]

    for index, item in enumerate(items, start=1):
        question = escape(item["question"])
        count = item["count"]

        lines.append(f"{index}. <code>{count}</code> — {question}")

    await message.answer("\n".join(lines))


async def cmd_feedback(message: Message) -> None:
    global analytics_store

    if not await ensure_admin(message):
        return

    if analytics_store is None:
        await message.answer("Система аналитики не инициализирована.")
        return

    items = analytics_store.get_recent_bad_feedback(limit=5)

    if not items:
        await message.answer("Негативных оценок пока нет.")
        return

    lines = ["👎 Последние ответы с оценкой «Неверно»:\n"]

    for index, item in enumerate(items, start=1):
        question = escape(str(item.get("question", "")))
        sources = item.get("sources") or []
        created_at = escape(str(item.get("created_at", "")))
        user_id = item.get("original_user_id")

        sources_text = "; ".join(sources) if sources else "источники не указаны"

        lines.append(
            f"{index}. Дата: <code>{created_at}</code>\n"
            f"User ID: <code>{user_id}</code>\n"
            f"Вопрос: {question}\n"
            f"Источники: {escape(sources_text)}\n"
        )

    text = "\n".join(lines)

    for part in split_for_telegram(text):
        await message.answer(part)


async def handle_auth_callback(callback: CallbackQuery) -> None:
    global auth_store

    if auth_store is None:
        await callback.answer("Система авторизации не инициализирована.", show_alert=True)
        return

    admin_id = get_callback_user_id(callback)

    if not auth_store.is_admin(admin_id):
        await callback.answer("Нет прав администратора.", show_alert=True)
        return

    data = callback.data or ""
    parts = data.split(":")

    if len(parts) != 3:
        await callback.answer("Некорректная команда.", show_alert=True)
        return

    _, action, user_id_raw = parts
    user_id = parse_user_id(user_id_raw)

    if user_id is None:
        await callback.answer("Некорректный user_id.", show_alert=True)
        return

    if admin_id is None:
        await callback.answer("Не удалось определить администратора.", show_alert=True)
        return

    if action == "approve":
        record = auth_store.approve_user(
            user_id=user_id,
            admin_id=admin_id,
        )

        await callback.answer("Доступ разрешён.")

        if callback.message:
            await callback.message.edit_text(
                "✅ Доступ разрешён\n\n"
                f"{format_user_record(record)}"
            )

        try:
            await callback.bot.send_message(
                chat_id=user_id,
                text=(
                    "✅ Ваша заявка одобрена.\n\n"
                    "Теперь вы можете пользоваться ботом. "
                    "Напишите /start или задайте вопрос."
                ),
            )
        except Exception:
            logger.exception("Не удалось уведомить пользователя %s", user_id)

        return

    if action == "deny":
        record = auth_store.deny_user(
            user_id=user_id,
            admin_id=admin_id,
        )

        await callback.answer("Заявка отклонена.")

        if callback.message:
            if record:
                await callback.message.edit_text(
                    "❌ Заявка отклонена\n\n"
                    f"{format_user_record(record)}"
                )
            else:
                await callback.message.edit_text(
                    "❌ Заявка отклонена.\n\n"
                    f"User ID: <code>{user_id}</code>"
                )

        try:
            await callback.bot.send_message(
                chat_id=user_id,
                text="❌ Ваша заявка на доступ к боту отклонена.",
            )
        except Exception:
            logger.exception("Не удалось уведомить пользователя %s", user_id)

        return

    await callback.answer("Неизвестное действие.", show_alert=True)


async def handle_feedback_callback(callback: CallbackQuery) -> None:
    global analytics_store
    global auth_store

    if analytics_store is None:
        await callback.answer("Система аналитики не инициализирована.", show_alert=True)
        return

    if auth_store is None:
        await callback.answer("Система авторизации не инициализирована.", show_alert=True)
        return

    user_id = get_callback_user_id(callback)

    if not auth_store.is_allowed(user_id):
        await callback.answer("У вас нет доступа.", show_alert=True)
        return

    data = callback.data or ""
    parts = data.split(":")

    if len(parts) != 3:
        await callback.answer("Некорректная оценка.", show_alert=True)
        return

    _, feedback, answer_id = parts

    if feedback not in {"good", "bad"}:
        await callback.answer("Некорректная оценка.", show_alert=True)
        return

    success, _record = analytics_store.log_feedback(
        answer_id=answer_id,
        feedback=feedback,
        feedback_user_id=user_id,
    )

    if not success:
        await callback.answer("Не удалось найти ответ для оценки.", show_alert=True)
        return

    if feedback == "good":
        await callback.answer("Спасибо за оценку 👍")
    else:
        await callback.answer("Спасибо. Я сохранил этот ответ для проверки 👎")

    # Убираем кнопки, чтобы пользователь не нажимал несколько раз
    if callback.message:
        try:
            await callback.message.edit_reply_markup(reply_markup=None)
        except Exception:
            logger.exception("Не удалось убрать feedback-кнопки")


async def handle_question(message: Message) -> None:
    global rag_engine
    global analytics_store

    if not await ensure_access(message):
        return

    question = (message.text or "").strip()

    if not question:
        await message.answer("Пришлите вопрос текстом.")
        return

    if rag_engine is None:
        await message.answer(
            "База знаний ещё не загружена.\n\n"
            "Проверьте, что файлы index.npz и chunks.json загружены в storage "
            "или в /data/storage на Amvera."
        )
        return

    await message.bot.send_chat_action(message.chat.id, ChatAction.TYPING)

    try:
        answer = await rag_engine.answer(question)
    except Exception as exc:
        logger.exception("Ошибка при обработке вопроса")
        await message.answer(f"Произошла ошибка: {exc}")
        return

    sources = extract_sources_from_answer(answer)

    answer_id: str | None = None

    if analytics_store is not None:
        username = message.from_user.username if message.from_user else None
        full_name = message.from_user.full_name if message.from_user else None
        user_id = get_user_id(message)

        answer_id = analytics_store.log_query(
            user_id=user_id,
            username=username,
            full_name=full_name,
            question=question,
            answer=answer,
            sources=sources,
        )

    parts = split_for_telegram(answer)

    for index, part in enumerate(parts):
        is_last_part = index == len(parts) - 1

        if is_last_part and answer_id:
            await message.answer(
                part,
                reply_markup=feedback_keyboard(answer_id),
            )
        else:
            await message.answer(part)


async def main() -> None:
    global rag_engine
    global auth_store
    global analytics_store

    auth_store = AuthStore()
    analytics_store = AnalyticsStore()

    if not settings.admin_user_ids:
        logger.warning(
            "ADMIN_USER_IDS пустой. "
            "Пользователи смогут отправлять заявки, но некому будет их подтверждать."
        )

    try:
        logger.info("Загрузка RAG-индекса...")
        rag_engine = RAGEngine()
        logger.info("RAG-индекс загружен")
    except Exception as exc:
        logger.exception(
            "RAG-индекс не загружен. "
            "Бот стартует без базы знаний: %s",
            exc,
        )
        rag_engine = None

    bot = Bot(
        token=settings.telegram_bot_token,
        default=DefaultBotProperties(parse_mode=ParseMode.HTML),
    )

    dp = Dispatcher()

    # Команды, доступные всем
    dp.message.register(cmd_id, Command("id"))

    # Callback-кнопки авторизации
    dp.callback_query.register(handle_auth_callback, F.data.startswith("auth:"))

    # Callback-кнопки оценки ответов
    dp.callback_query.register(handle_feedback_callback, F.data.startswith("fb:"))

    # Обычные команды
    dp.message.register(cmd_start, CommandStart())
    dp.message.register(cmd_help, Command("help"))

    # Админ-команды эксплуатации RAG
    dp.message.register(cmd_status, Command("status"))
    dp.message.register(cmd_reload, Command("reload"))
    dp.message.register(cmd_clear_cache, Command("clear_cache"))
    dp.message.register(cmd_version, Command("version"))
    dp.message.register(cmd_debug_search, Command("debug_search"))

    # Админ-команды авторизации
    dp.message.register(cmd_users, Command("users"))
    dp.message.register(cmd_pending, Command("pending"))
    dp.message.register(cmd_allow, Command("allow"))
    dp.message.register(cmd_revoke, Command("revoke"))

    # Админ-команды аналитики
    dp.message.register(cmd_stats, Command("stats"))
    dp.message.register(cmd_popular, Command("popular"))
    dp.message.register(cmd_feedback, Command("feedback"))

    # Все остальные текстовые сообщения — вопросы к RAG
    dp.message.register(handle_question, F.text)

    try:
        logger.info("Бот запущен")
        await dp.start_polling(bot, polling_timeout=30)
    finally:
        logger.info("Закрытие сессии бота")
        await bot.session.close()


if __name__ == "__main__":
    asyncio.run(main())