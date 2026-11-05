# VPg07 — Search your data

Telegram-бот для поиска по своим данным: память диалога и разбор загруженных
файлов (PDF, DOCX и др.) через DocLing, с хранением эмбеддингов в Pinecone.

## Что в репозитории

| Путь | Назначение |
|------|------------|
| `hay_v2_bot/` | Основной бот v2: Haystack-пайплайны + DocLing + Pinecone |
| `telegram_bot.py` | Первая версия бота на OpenAI + `PineconeManager` |
| `pinecone_manager.py` | Обёртка над Pinecone для v1 |
| `hay/hay-telegram-bot.py` | Отдельный Haystack-бот с агентом и инструментами |
| `.env.example` | Шаблон переменных окружения |

## Быстрый старт (hay_v2_bot)

1. Скопируйте `.env.example` в `.env` и заполните ключи.
2. Установите зависимости:

```bash
pip install -r hay_v2_bot/requirements.txt
```

3. Запустите бота:

```bash
python hay_v2_bot/main.py
```

## Возможности hay_v2_bot

- текстовый диалог с долговременной памятью пользователя;
- команды `/start` и `/forget`;
- загрузка PDF, DOCX, PPTX, HTML, Markdown, TXT, CSV и изображений документов;
- `ingestion_pipeline`: DocLing → чанки → **OpenAI embeddings** → Pinecone;
- `generation_pipeline`: OpenAI embed запроса → retriever → ответ LLM;
- после обработки файла — подтверждение и **одно предложение** резюме.

## Переменные окружения

Настройки берутся из корневого `.env`:

| Переменная | Пример / назначение |
|---|---|
| `OPENAI_BASE_URL` | `https://api.openai.com/v1` |
| `OPENAI_API_KEY` | ключ OpenAI |
| `OPENAI_CHAT_MODEL` | `gpt-4o-mini` |
| `OPENAI_EMBEDDING_MODEL` | `text-embedding-3-small` |
| `OPENAI_EMBEDDING_DIMENSIONS` | `1536` |
| `TELEGRAM_BOT_TOKEN` | токен BotFather |
| `PINECONE_API_KEY` | ключ Pinecone |
| `HAYSTACK_PINECONE_INDEX` | dense-индекс для v2, например `vpg07-search-your-data` |
| `PINECONE_INDEX_NAME` | индекс для v1 (`telegram_bot.py` / `PineconeManager`) |

Локальные Sentence Transformers в v2 **не используются**: chat и embeddings
идут через OpenAI API.

## Структура hay_v2_bot

```text
hay_v2_bot/
  components/   # конфиг, OpenAI-фабрика, кастомные Haystack-компоненты
  pipelines/    # store, ingestion, memory_index, generation
  bot/          # Telegram-оркестрация и скачивание файлов
  main.py
```

## Пайплайны

1. **ingestion** — `DoclingConverter` → `DocumentMetaEnricher` → `OpenAIDocumentEmbedder` → writer  
2. **memory_index** — `OpenAIDocumentEmbedder` → writer  
3. **generation** — `OpenAITextEmbedder` → Pinecone retriever → `ContextChatResponder`  
4. **retrieval** — короткий поиск для дедупликации памяти

В метаданные файловых чанков пишутся `filename`, `chunk_index`, `page` (если есть),
`section`, `user_id`, `doc_type=file_chunk`.
