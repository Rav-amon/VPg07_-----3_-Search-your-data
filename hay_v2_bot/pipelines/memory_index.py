"""Пайплайн записи текстовой памяти: OpenAI embed → Pinecone."""

from __future__ import annotations

from collections.abc import Sequence

from haystack import Document, Pipeline
from haystack.components.writers import DocumentWriter
from haystack.document_stores.types import DuplicatePolicy
from haystack_integrations.document_stores.pinecone import PineconeDocumentStore

from hay_v2_bot.components.config import AppConfig
from hay_v2_bot.components.errors import BotError
from hay_v2_bot.components.openai_factory import build_document_embedder


def build_memory_index_pipeline(
    document_store: PineconeDocumentStore,
    config: AppConfig,
) -> Pipeline:
    """Собирает конвейер записи реплик через удалённые OpenAI-эмбеддинги.

    Args:
        document_store: Хранилище Pinecone.
        config: Настройки с ``OPENAI_BASE_URL`` и ``text-embedding-3-small``.

    Returns:
        Пайплайн с входом ``embedder.documents``.
    """
    pipeline = Pipeline()
    pipeline.add_component('embedder', build_document_embedder(config))
    pipeline.add_component(
        'writer',
        DocumentWriter(
            document_store=document_store,
            policy=DuplicatePolicy.OVERWRITE,
        ),
    )
    pipeline.connect('embedder.documents', 'writer.documents')
    return pipeline


def run_memory_index(
    pipeline: Pipeline,
    documents: Sequence[Document],
) -> int:
    """Считает эмбеддинги через OpenAI API и пишет документы в Pinecone.

    Args:
        pipeline: Пайплайн из ``build_memory_index_pipeline``.
        documents: Документы без векторов.

    Returns:
        Число записанных документов.

    Raises:
        BotError: Если запись не удалась.
    """
    if len(documents) == 0:
        return 0
    try:
        result = pipeline.run({'embedder': {'documents': list(documents)}})
    except Exception as error:
        raise BotError(
            code='remember_error',
            message='Не удалось записать память.',
            details={'error_type': type(error).__name__},
        ) from error
    writer_result = result.get('writer', {})
    written = writer_result.get('documents_written', 0)
    return written if isinstance(written, int) else 0
