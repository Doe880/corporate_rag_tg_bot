from __future__ import annotations

import asyncio
import logging

from aiogram import Bot, Dispatcher, F
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ChatAction, ParseMode
from aiogram.filters import Command, CommandStart
from aiogram.types import Message

from config import settings
from rag import RAGEngine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

rag_engine: RAGEngine | None = None


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


async def cmd_start(message: Message) -> None:
    await message.answer(
        "Здравствуйте! Я корпоративный AI-ассистент.\n\n"
        "Задайте вопрос, а я найду ответ в базе знаний. "
        "Если информации в базе нет, я так и отвечу."
    )


async def cmd_help(message: Message) -> None:
    await message.answer(
        "Как пользоваться:\n"
        "1. Напишите вопрос обычным сообщением.\n"
        "2. Я выполню поиск по базе.\n"
        "3. Ответ будет сформирован по полученным данным.\n\n"
        "Примеры:\n"
        "• Какой слоган у продукта?\n"
        "• Какой состав у продукта?\n"
        "• Какие рекомендации по применению?"
    )


async def handle_question(message: Message) -> None:
    global rag_engine

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

    dp.message.register(cmd_start, CommandStart())
    dp.message.register(cmd_help, Command("help"))
    dp.message.register(handle_question, F.text)

    try:
        logger.info("Бот запущен")
        await dp.start_polling(bot, polling_timeout=30)
    finally:
        logger.info("Закрытие сессии бота")
        await bot.session.close()


if __name__ == "__main__":
    asyncio.run(main())