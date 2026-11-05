"""Haystack-пайплайны индексации, поиска и генерации."""

from hay_v2_bot.pipelines.generation import build_generation_pipeline
from hay_v2_bot.pipelines.ingestion import build_ingestion_pipeline
from hay_v2_bot.pipelines.memory_index import build_memory_index_pipeline
from hay_v2_bot.pipelines.store import build_document_store, ensure_pinecone_index

__all__ = [
    'build_document_store',
    'build_generation_pipeline',
    'build_ingestion_pipeline',
    'build_memory_index_pipeline',
    'ensure_pinecone_index',
]
