"""Константы бота v2: лимиты, типы документов и поддерживаемые файлы."""

from __future__ import annotations

# Поле схемы / тип записи в метаданных Pinecone.
DOC_TYPE_CHAT = 'chat_memory'
DOC_TYPE_FILE = 'file_chunk'
DOC_TYPE_PROFILE = 'profile'

# Пороги и лимиты диалога.
MAX_HISTORY_MESSAGES = 10
MEMORY_TOP_K = 5
MIN_COSINE_SCORE = 0.35
COSINE_DEDUP_THRESHOLD = 0.85
FORGET_ROUNDS = 20
TELEGRAM_MESSAGE_LIMIT = 4000
MAX_MEMORY_CHARS = 8000
MAX_SUMMARY_SOURCE_CHARS = 6000
MAX_FILE_BYTES = 20 * 1024 * 1024

# Telegram polling.
POLLING_HTTP_TIMEOUT = 30
LONG_POLLING_TIMEOUT = 10
POLLING_SESSION_SECONDS = 120
TYPING_INTERVAL_SECONDS = 4.0
TYPING_JOIN_SECONDS = 1.0

# Генерация.
CHAT_TEMPERATURE = 0.7
SUMMARY_TEMPERATURE = 0.3
MAX_REPLY_TOKENS = 800
MAX_SUMMARY_TOKENS = 120
OPENAI_TIMEOUT_SECONDS = 90.0
OPENAI_MAX_RETRIES = 2

# Значения по умолчанию — как в корневом .env проекта.
DEFAULT_OPENAI_BASE_URL = 'https://api.openai.com/v1'
DEFAULT_CHAT_MODEL = 'gpt-4o-mini'
DEFAULT_INDEX_NAME = 'vpg07-search-your-data'
DEFAULT_NAMESPACE = 'default'
DEFAULT_EMBEDDING_MODEL = 'text-embedding-3-small'
# Полная размерность text-embedding-3-small; можно укоротить через env.
DEFAULT_EMBEDDING_DIMENSION = 1536
DEFAULT_CHUNK_MAX_TOKENS = 8191
DEFAULT_CLOUD = 'aws'
DEFAULT_REGION = 'us-east-1'
MODEL_DIMENSIONS = {
    'text-embedding-3-small': 1536,
    'text-embedding-3-large': 3072,
    'text-embedding-ada-002': 1536,
}
ALLOWED_CLOUDS = frozenset({'aws', 'gcp', 'azure'})

# Расширения, которые DocLing обычно уверенно разбирает.
SUPPORTED_FILE_EXTENSIONS = frozenset({
    '.pdf',
    '.docx',
    '.pptx',
    '.html',
    '.htm',
    '.md',
    '.txt',
    '.csv',
    '.png',
    '.jpg',
    '.jpeg',
    '.tif',
    '.tiff',
    '.webp',
})
SUPPORTED_MIME_TYPES = frozenset({
    'application/pdf',
    'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
    'application/vnd.openxmlformats-officedocument.presentationml.presentation',
    'text/html',
    'text/markdown',
    'text/plain',
    'text/csv',
    'image/png',
    'image/jpeg',
    'image/tiff',
    'image/webp',
})

USER_ROLE = 'user'
ASSISTANT_ROLE = 'assistant'
