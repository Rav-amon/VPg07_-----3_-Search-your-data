"""Пайплайн генерации: OpenAI embed query → retrieve → LLM."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from haystack import Document, Pipeline
from haystack.dataclasses import ChatMessage
from haystack_integrations.components.retrievers.pinecone import (
    PineconeEmbeddingRetriever,
)
from haystack_integrations.document_stores.pinecone import PineconeDocumentStore

from hay_v2_bot.components.chat_responder import ContextChatResponder
from hay_v2_bot.components.config import AppConfig
from hay_v2_bot.components.constants import (
    MAX_SUMMARY_SOURCE_CHARS,
    MAX_SUMMARY_TOKENS,
    MEMORY_TOP_K,
    MIN_COSINE_SCORE,
    SUMMARY_TEMPERATURE,
)
from hay_v2_bot.components.errors import BotError
from hay_v2_bot.components.memory_utils import (
    first_sentence,
    select_documents,
    user_memory_filter,
)
from hay_v2_bot.components.openai_factory import (
    build_chat_generator,
    build_text_embedder,
)
from hay_v2_bot.components.texts import EMPTY_ANSWER_TEXT, SUMMARY_PROMPT, SYSTEM_PROMPT


def build_generation_pipeline(
    document_store: PineconeDocumentStore,
    config: AppConfig,
) -> Pipeline:
    """Собирает RAG-пайплайн ответа на текстовый вопрос.

    Граф: openai text_embedder → retriever → responder.
    И эмбеддинг, и LLM идут через ``OPENAI_BASE_URL``.

    Args:
        document_store: Общее хранилище с памятью и файлами.
        config: Настройки приложения.

    Returns:
        Пайплайн. Входы: ``text_embedder.text``, ``retriever.filters``,
        ``retriever.top_k``, ``responder.query``, ``responder.history``.
    """
    pipeline = Pipeline()
    pipeline.add_component('text_embedder', build_text_embedder(config))
    pipeline.add_component(
        'retriever',
        PineconeEmbeddingRetriever(
            document_store=document_store,
            top_k=MEMORY_TOP_K,
        ),
    )
    pipeline.add_component(
        'responder',
        ContextChatResponder(build_chat_generator(config)),
    )
    pipeline.connect('text_embedder.embedding', 'retriever.query_embedding')
    pipeline.connect('retriever.documents', 'responder.documents')
    return pipeline


def run_generation(
    pipeline: Pipeline,
    *,
    user_id: int,
    query: str,
    history: Sequence[Mapping[str, str]],
) -> str:
    """Запускает generation_pipeline для пользователя.

    Args:
        pipeline: Пайплайн из ``build_generation_pipeline``.
        user_id: Идентификатор Telegram.
        query: Текст вопроса.
        history: История сеанса.

    Returns:
        Текст ответа.

    Raises:
        BotError: Если генерация не удалась.
    """
    try:
        result = pipeline.run(
            {
                'text_embedder': {'text': query},
                'retriever': {
                    'filters': user_memory_filter(user_id),
                    'top_k': MEMORY_TOP_K,
                },
                'responder': {
                    'query': query,
                    'history': [dict(item) for item in history],
                    'system_prompt': SYSTEM_PROMPT,
                },
            },
        )
    except BotError:
        raise
    except Exception as error:
        raise BotError(
            code='chat_error',
            message='Не удалось подготовить ответ.',
            details={'error_type': type(error).__name__},
        ) from error
    answer = result.get('responder', {}).get('answer')
    if not isinstance(answer, str) or answer.strip() == '':
        return EMPTY_ANSWER_TEXT
    return answer.strip()


def build_retrieval_pipeline(
    document_store: PineconeDocumentStore,
    config: AppConfig,
) -> Pipeline:
    """Собирает короткий пайплайн поиска (дедуп памяти) на OpenAI embeddings.

    Args:
        document_store: Хранилище.
        config: Настройки с ``OPENAI_BASE_URL``.

    Returns:
        Пайплайн ``text_embedder → retriever``.
    """
    pipeline = Pipeline()
    pipeline.add_component('text_embedder', build_text_embedder(config))
    pipeline.add_component(
        'retriever',
        PineconeEmbeddingRetriever(document_store=document_store, top_k=1),
    )
    pipeline.connect('text_embedder.embedding', 'retriever.query_embedding')
    return pipeline


def run_retrieval(
    pipeline: Pipeline,
    *,
    user_id: int,
    text: str,
    top_k: int = MEMORY_TOP_K,
) -> list[Document]:
    """Ищет документы пользователя по тексту запроса.

    Args:
        pipeline: Retrieval-пайплайн.
        user_id: Владелец.
        text: Текст запроса.
        top_k: Сколько документов вернуть.

    Returns:
        Список документов выше порога сходства.
    """
    try:
        result = pipeline.run(
            {
                'text_embedder': {'text': text},
                'retriever': {
                    'filters': user_memory_filter(user_id),
                    'top_k': top_k,
                },
            },
            include_outputs_from={'retriever'},
        )
    except Exception as error:
        raise BotError(
            code='recall_error',
            message='Не удалось прочитать память.',
            details={'error_type': type(error).__name__},
        ) from error
    documents = result.get('retriever', {}).get('documents', [])
    if not isinstance(documents, list):
        return []
    return select_documents(documents, MIN_COSINE_SCORE)


def summarize_file_content(
    config: AppConfig,
    chunk_texts: Sequence[str],
    filename: str,
) -> str:
    """Строит одно предложение-резюме по тексту чанков файла.

    Args:
        config: Настройки приложения (LLM через ``OPENAI_BASE_URL``).
        chunk_texts: Тексты сохранённых чанков.
        filename: Имя файла для контекста.

    Returns:
        Одно предложение на русском.

    Raises:
        BotError: Если модель не ответила.
    """
    joined = '\n'.join(
        text.strip() for text in chunk_texts if text.strip() != ''
    )
    source = joined[:MAX_SUMMARY_SOURCE_CHARS]
    if source == '':
        return f'Файл «{filename}» сохранён, но извлечь текст для резюме не удалось.'
    generator = build_chat_generator(
        config,
        temperature=SUMMARY_TEMPERATURE,
        max_tokens=MAX_SUMMARY_TOKENS,
    )
    messages = [
        ChatMessage.from_system(SUMMARY_PROMPT),
        ChatMessage.from_user(
            f'Имя файла: {filename}\n\nСодержимое:\n{source}',
        ),
    ]
    try:
        result = generator.run(messages=messages)
    except Exception as error:
        raise BotError(
            code='summary_error',
            message='Не удалось сформулировать резюме файла.',
            details={'error_type': type(error).__name__},
        ) from error
    replies = result.get('replies', [])
    raw = ''
    if isinstance(replies, list) and len(replies) > 0:
        first = replies[0]
        if isinstance(first, ChatMessage) and isinstance(first.text, str):
            raw = first.text
    summary = first_sentence(raw)
    if summary == '':
        return f'Файл «{filename}» изучен и сохранён в базу знаний.'
    return summary
