"""Переиспользуемые компоненты и чистые утилиты бота."""

from hay_v2_bot.components.chat_responder import ContextChatResponder
from hay_v2_bot.components.config import AppConfig, load_config
from hay_v2_bot.components.errors import BotError
from hay_v2_bot.components.meta_enricher import DocumentMetaEnricher
from hay_v2_bot.components.openai_factory import (
    build_chat_generator,
    build_document_embedder,
    build_text_embedder,
)

__all__ = [
    'AppConfig',
    'BotError',
    'ContextChatResponder',
    'DocumentMetaEnricher',
    'build_chat_generator',
    'build_document_embedder',
    'build_text_embedder',
    'load_config',
]
