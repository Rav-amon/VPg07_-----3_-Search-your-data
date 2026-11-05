"""Фабрика OpenAI-клиентов Haystack: всегда через OPENAI_BASE_URL."""

from __future__ import annotations

from haystack.components.embedders import OpenAIDocumentEmbedder, OpenAITextEmbedder
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.utils import Secret

from hay_v2_bot.components.config import AppConfig
from hay_v2_bot.components.constants import (
    CHAT_TEMPERATURE,
    MAX_REPLY_TOKENS,
    OPENAI_MAX_RETRIES,
    OPENAI_TIMEOUT_SECONDS,
)
from hay_v2_bot.components.errors import BotError


def require_base_url(config: AppConfig) -> str:
    """Возвращает обязательный base URL OpenAI API.

    Args:
        config: Настройки приложения.

    Returns:
        Непустой ``openai_base_url``.

    Raises:
        BotError: Если base URL не задан.
    """
    base_url = config.openai_base_url
    if base_url is None or base_url.strip() == '':
        raise BotError(
            code='config_error',
            message=(
                'Не задан OPENAI_BASE_URL. Ожидается значение из .env, '
                'например https://api.openai.com/v1.'
            ),
            details={'env': 'OPENAI_BASE_URL'},
        )
    return base_url.strip()


def build_document_embedder(config: AppConfig) -> OpenAIDocumentEmbedder:
    """Создаёт удалённый эмбеддер документов через OpenAI API.

    Args:
        config: Настройки с моделью ``text-embedding-3-small`` и base URL.

    Returns:
        ``OpenAIDocumentEmbedder`` без локальных моделей.
    """
    return OpenAIDocumentEmbedder(
        api_key=Secret.from_token(config.openai_api_key),
        model=config.embedding_model,
        dimensions=config.embedding_dimension,
        api_base_url=require_base_url(config),
        timeout=OPENAI_TIMEOUT_SECONDS,
        max_retries=OPENAI_MAX_RETRIES,
    )


def build_text_embedder(config: AppConfig) -> OpenAITextEmbedder:
    """Создаёт удалённый эмбеддер текста запроса через OpenAI API.

    Args:
        config: Настройки с моделью ``text-embedding-3-small`` и base URL.

    Returns:
        ``OpenAITextEmbedder`` без локальных моделей.
    """
    return OpenAITextEmbedder(
        api_key=Secret.from_token(config.openai_api_key),
        model=config.embedding_model,
        dimensions=config.embedding_dimension,
        api_base_url=require_base_url(config),
        timeout=OPENAI_TIMEOUT_SECONDS,
        max_retries=OPENAI_MAX_RETRIES,
    )


def build_chat_generator(
    config: AppConfig,
    *,
    temperature: float = CHAT_TEMPERATURE,
    max_tokens: int = MAX_REPLY_TOKENS,
) -> OpenAIChatGenerator:
    """Создаёт chat-генератор OpenAI через тот же base URL.

    Args:
        config: Настройки приложения.
        temperature: Температура выборки.
        max_tokens: Лимит токенов ответа.

    Returns:
        ``OpenAIChatGenerator`` с ``api_base_url`` из конфига.
    """
    generation_kwargs = {
        'temperature': temperature,
        'max_completion_tokens': max_tokens,
    }
    return OpenAIChatGenerator(
        api_key=Secret.from_token(config.openai_api_key),
        model=config.chat_model,
        api_base_url=require_base_url(config),
        generation_kwargs=generation_kwargs,
        timeout=OPENAI_TIMEOUT_SECONDS,
        max_retries=OPENAI_MAX_RETRIES,
    )
