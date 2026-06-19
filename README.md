# Corporate RAG Telegram Bot

Telegram-бот на Python, который отвечает на вопросы пользователей строго по корпоративной PDF-базе знаний.

## Что внутри

- Telegram bot: `aiogram`
- PDF parsing: `PyMuPDF`
- Embeddings: OpenAI `text-embedding-3-small`
- Chat model: OpenAI `gpt-4.1-mini` по умолчанию
- Vector search: локальный numpy cosine-search
- Корпоративные PDF и векторный индекс не коммитятся в GitHub

## Структура

```text
corporate_rag_tg_bot/
├── bot.py
├── config.py
├── ingest.py
├── pdf_loader.py
├── rag.py
├── requirements.txt
├── .env.example
├── .gitignore
├── private_docs/   # сюда кладутся PDF, не коммитить
└── storage/        # тут индекс и chunks, не коммитить
```

## Локальный запуск

### 1. Создать виртуальное окружение

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

macOS/Linux:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Создать `.env`

Скопируйте `.env.example` в `.env` и заполните:

```text
TELEGRAM_BOT_TOKEN=...
OPENAI_API_KEY=...
OPENAI_CHAT_MODEL=gpt-4.1-mini
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
```

### 3. Добавить PDF

Положите PDF-файлы в папку:

```text
private_docs/
```

PDF и индекс не должны попадать в публичный GitHub.

### 4. Создать векторный индекс

```bash
python ingest.py
```

После этого появятся файлы:

```text
storage/index.npz
storage/chunks.json
```

### 5. Запустить Telegram-бота

```bash
python bot.py
```

## Важное про GitHub

Публичный GitHub должен содержать только код. Нельзя коммитить:

- `.env`
- `private_docs/`
- `storage/`

Telegram-бот должен работать на сервере с постоянным процессом: VPS, Render, Railway, Fly.io и т.п. GitHub Pages не подходит для long-running Telegram bot.

## Если PDF — это сканы

Если PDF содержит только картинки, `PyMuPDF` не извлечёт текст. Нужно сначала сделать OCR, например Tesseract или другой OCR-сервис, и уже потом индексировать распознанный текст.
