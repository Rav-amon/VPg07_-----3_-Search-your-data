"""Пайплайн загрузки файлов: DocLing → метаданные → OpenAI embed → Pinecone."""

from __future__ import annotations

from haystack import Pipeline
from haystack.components.writers import DocumentWriter
from haystack.document_stores.types import DuplicatePolicy
from haystack_integrations.document_stores.pinecone import PineconeDocumentStore

from hay_v2_bot.components.config import AppConfig
from hay_v2_bot.components.constants import DEFAULT_CHUNK_MAX_TOKENS
from hay_v2_bot.components.errors import BotError
from hay_v2_bot.components.meta_enricher import DocumentMetaEnricher
from hay_v2_bot.components.openai_factory import build_document_embedder


def _build_openai_chunker(embedding_model: str):
    """Собирает HybridChunker на tiktoken OpenAI, без локальных HF-моделей.

    Args:
        embedding_model: Имя модели эмбеддингов OpenAI для tiktoken.

    Returns:
        Экземпляр ``HybridChunker``.

    Raises:
        BotError: Если нет docling / tiktoken.
    """
    try:
        import tiktoken
        from docling.chunking import HybridChunker
        from docling_core.transforms.chunker.tokenizer.openai import OpenAITokenizer
    except ImportError as error:
        raise BotError(
            code='dependency_error',
            message=(
                'Для чанкинга нужны docling и tiktoken. Установите зависимости '
                'из hay_v2_bot/requirements.txt.'
            ),
            details={'error_type': type(error).__name__},
        ) from error
    try:
        encoding = tiktoken.encoding_for_model(embedding_model)
    except KeyError:
        encoding = tiktoken.get_encoding('cl100k_base')
    tokenizer = OpenAITokenizer(
        tokenizer=encoding,
        max_tokens=DEFAULT_CHUNK_MAX_TOKENS,
    )
    return HybridChunker(tokenizer=tokenizer)


def _build_document_converter():
    """Создаёт быстрый Docling DocumentConverter для Telegram.

    - PDF через ``PyPdfium``: обходит поломку ``docling_parse`` на путях
      с кириллицей.
    - ``NativePdfPipeline``: без layout/OCR/TableFormer на CPU. Полный
      StandardPdfPipeline на больших файлах выглядит как зависание бота.

    Returns:
        Настроенный ``DocumentConverter``.

    Raises:
        BotError: Если пакет ``docling`` не установлен.
    """
    try:
        from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.pipeline_options import NativePdfPipelineOptions
        from docling.document_converter import (
            DocumentConverter,
            NativePdfFormatOption,
        )
    except ImportError as error:
        raise BotError(
            code='dependency_error',
            message=(
                'Не установлен docling. Установите зависимости из '
                'hay_v2_bot/requirements.txt.'
            ),
            details={'error_type': type(error).__name__},
        ) from error
    pdf_option = NativePdfFormatOption(
        backend=PyPdfiumDocumentBackend,
        pipeline_options=NativePdfPipelineOptions(
            generate_page_images=False,
            generate_picture_images=False,
        ),
    )
    return DocumentConverter(
        format_options={
            InputFormat.PDF: pdf_option,
        },
    )


def _build_docling_converter(embedding_model: str):
    """Создаёт DoclingConverter с чанкером под OpenAI tokenizer.

    Args:
        embedding_model: Имя модели ``text-embedding-3-small`` и т.п.

    Returns:
        Экземпляр ``DoclingConverter``.

    Raises:
        BotError: Если пакет ``docling-haystack`` не установлен.
    """
    try:
        from haystack_integrations.components.converters.docling import (
            DoclingConverter,
            ExportType,
        )
    except ImportError as error:
        raise BotError(
            code='dependency_error',
            message=(
                'Не установлен docling-haystack. Установите зависимости из '
                'hay_v2_bot/requirements.txt.'
            ),
            details={'error_type': type(error).__name__},
        ) from error
    return DoclingConverter(
        converter=_build_document_converter(),
        export_type=ExportType.DOC_CHUNKS,
        chunker=_build_openai_chunker(embedding_model),
    )


def build_ingestion_pipeline(
    document_store: PineconeDocumentStore,
    config: AppConfig,
) -> Pipeline:
    """Собирает пайплайн индексации файлов через DocLing и OpenAI embeddings.

    Граф: converter → enricher → openai embedder → writer.

    Args:
        document_store: Хранилище Pinecone.
        config: Настройки с ``OPENAI_BASE_URL`` и моделью small.

    Returns:
        Готовый пайплайн. Входы ``converter.sources``, ``enricher.user_id``,
        ``enricher.filename``.
    """
    pipeline = Pipeline()
    pipeline.add_component(
        'converter',
        _build_docling_converter(config.embedding_model),
    )
    pipeline.add_component('enricher', DocumentMetaEnricher())
    pipeline.add_component('embedder', build_document_embedder(config))
    pipeline.add_component(
        'writer',
        DocumentWriter(
            document_store=document_store,
            policy=DuplicatePolicy.OVERWRITE,
        ),
    )
    pipeline.connect('converter.documents', 'enricher.documents')
    pipeline.connect('enricher.documents', 'embedder.documents')
    pipeline.connect('embedder.documents', 'writer.documents')
    return pipeline


def run_ingestion(
    pipeline: Pipeline,
    file_path: str,
    user_id: str,
    filename: str,
) -> int:
    """Запускает ingestion для одного локального файла.

    Args:
        pipeline: Пайплайн из ``build_ingestion_pipeline``.
        file_path: Локальный путь к файлу.
        user_id: Идентификатор владельца.
        filename: Отображаемое имя файла.

    Returns:
        Число записанных документов.

    Raises:
        BotError: Если конвертация или запись не удались.
    """
    try:
        result = pipeline.run(
            {
                'converter': {'sources': [file_path]},
                'enricher': {
                    'user_id': user_id,
                    'filename': filename,
                },
            },
        )
    except BotError:
        raise
    except Exception as error:
        raise BotError(
            code='ingestion_error',
            message='Не удалось проанализировать и сохранить файл.',
            details={'error_type': type(error).__name__},
        ) from error
    writer_result = result.get('writer', {})
    written = writer_result.get('documents_written', 0)
    if not isinstance(written, int):
        return 0
    if written <= 0:
        raise BotError(
            code='ingestion_empty',
            message='DocLing не извлёк текстовых чанков из файла.',
            details={'filename': filename},
        )
    return written
