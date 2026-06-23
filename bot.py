from __future__ import annotations

import asyncio
import logging
from html import escape

from aiogram import Bot, Dispatcher, F
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ChatAction, ParseMode
from aiogram.filters import Command, CommandStart
from aiogram.types import CallbackQuery, InlineKeyboardButton, InlineKeyboardMarkup, Message

from auth import AuthStore
from config import settings
from rag import RAGEngine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

rag_engine: RAGEngine | None = None
auth_store: AuthStore | None = None


def split_for_telegram(text: str, limit: int = 3900) -> list[str]:
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
    username = message.from_user.username or "не указан"
    full_name = escape(message.from_user.full_name or "")

    await message.answer(
        "Ваши данные Telegram:\n\n"
        f"User ID: <code>{user_id}</code>\n"
        f"Username: @{escape(username)}\n"
        f"Имя: {full_name}"
    )


async def cmd_start(message: Message) -> None:
    if not await ensure_access(message):
        return

    await message.answer(
        "Здравствуйте! Я корпоративный AI-ассистент.\n\n"
        "Задайте вопрос, а я найду ответ в базе знаний PDF. "
        "Если информации в базе нет, я так и отвечу."
    )


async def cmd_help(message: Message) -> None:
    if not await ensure_access(message):
        return

    await message.answer(
        "Как пользоваться:\n"
        "1. Напишите вопрос обычным сообщением.\n"
        "2. Я выполню поиск по PDF-базе.\n"
        "3. Ответ будет сформирован только по найденным фрагментам.\n\n"
        "Команды:\n"
        "/id — узнать свой Telegram user_id\n"
        "/help — помощь\n\n"
        "Админ-команды:\n"
        "/users — список пользователей\n"
        "/pending — список заявок\n"
        "/allow user_id — добавить пользователя\n"
        "/revoke user_id — удалить пользователя"
    )


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


async def handle_question(message: Message) -> None:
    global rag_engine

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

    for part in split_for_telegram(answer):
        await message.answer(part)


async def main() -> None:
    global rag_engine
    global auth_store

    auth_store = AuthStore()

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

    # Обычные команды
    dp.message.register(cmd_start, CommandStart())
    dp.message.register(cmd_help, Command("help"))

    # Админ-команды
    dp.message.register(cmd_users, Command("users"))
    dp.message.register(cmd_pending, Command("pending"))
    dp.message.register(cmd_allow, Command("allow"))
    dp.message.register(cmd_revoke, Command("revoke"))

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