# Haystack Telegram Bot v2

Модульный Telegram-бот на Haystack-пайплайнах с разбором файлов через DocLing
и хранением чанков в Pinecone.

## Возможности

- текстовый диалог с долговременной памятью пользователя;
- команды `/start` и `/forget`;
- загрузка PDF, DOCX, PPTX, HTML, Markdown, TXT, CSV и изображений документов;
- `ingestion_pipeline`: DocLing → чанки → **OpenAI embeddings** → Pinecone;
- `generation_pipeline`: OpenAI embed запроса → retriever → ответ LLM;
- после обработки файла бот присылает подтверждение и **одно предложение** резюме.

## OpenAI API

Локальные эмбеддинги (Sentence Transformers) **не используются**.

Настройки берутся из корневого `.env`:

| Переменная | Значение в проекте |
|---|---|
| `OPENAI_BASE_URL` | `https://api.openai.com/v1` |
| `OPENAI_CHAT_MODEL` | `gpt-4o-mini` |
| `OPENAI_EMBEDDING_MODEL` | `text-embedding-3-small` |
| `OPENAI_EMBEDDING_DIMENSIONS` | `1536` |
| `HAYSTACK_PINECONE_INDEX` | `vpg07-search-your-data` |

Также обязательны: `OPENAI_API_KEY`, `TELEGRAM_BOT_TOKEN`, `PINECONE_API_KEY`.

`PINECONE_INDEX_NAME` используется старым `telegram_bot.py` / `PineconeManager`, не Haystack v2.

## Структура

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

## Запуск

Из корня проекта:

```bash
pip install -r hay_v2_bot/requirements.txt
python hay_v2_bot/main.py
```
