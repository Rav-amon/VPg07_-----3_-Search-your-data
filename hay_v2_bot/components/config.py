"""Чтение и проверка настроек из окружения."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

from hay_v2_bot.components.constants import (
    ALLOWED_CLOUDS,
    DEFAULT_CHAT_MODEL,
    DEFAULT_CLOUD,
    DEFAULT_EMBEDDING_DIMENSION,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_INDEX_NAME,
    DEFAULT_NAMESPACE,
    DEFAULT_OPENAI_BASE_URL,
    DEFAULT_REGION,
    MODEL_DIMENSIONS,
)
from hay_v2_bot.components.errors import BotError

TELEGRAM_BOT_TOKEN_ENV = 'TELEGRAM_BOT_TOKEN'
OPENAI_API_KEY_ENV = 'OPENAI_API_KEY'
OPENAI_BASE_URL_ENV = 'OPENAI_BASE_URL'
OPENAI_CHAT_MODEL_ENV = 'OPENAI_CHAT_MODEL'
OPENAI_EMBEDDING_MODEL_ENV = 'OPENAI_EMBEDDING_MODEL'
OPENAI_EMBEDDING_DIMENSIONS_ENV = 'OPENAI_EMBEDDING_DIMENSIONS'
PINECONE_API_KEY_ENV = 'PINECONE_API_KEY'
HAYSTACK_PINECONE_INDEX_ENV = 'HAYSTACK_PINECONE_INDEX'
HAYSTACK_PINECONE_NAMESPACE_ENV = 'HAYSTACK_PINECONE_NAMESPACE'
PINECONE_CLOUD_ENV = 'PINECONE_CLOUD'
PINECONE_REGION_ENV = 'PINECONE_REGION'

INDEX_NAME_PATTERN = re.compile(r'^[a-z0-9][a-z0-9-]{0,44}$')
PROJECT_ROOT = Path(__file__).resolve().parents[2]
ENV_PATH = PROJECT_ROOT / '.env'


@dataclass(frozen=True)
class AppConfig:
    """Собранные настройки приложения.

    Attributes:
        telegram_token: Токен Telegram-бота.
        openai_api_key: Ключ OpenAI.
        openai_base_url: Base URL OpenAI API из ``OPENAI_BASE_URL``.
        chat_model: Имя диалоговой модели.
        pinecone_api_key: Ключ Pinecone.
        index_name: Имя dense-индекса под OpenAI-эмбеддинги.
        namespace: Пространство имён индекса.
        embedding_model: Удалённая модель, обычно ``text-embedding-3-small``.
        embedding_dimension: Размерность векторов (должна совпадать с индексом).
        pinecone_cloud: Облако serverless-индекса.
        pinecone_region: Регион serverless-индекса.
    """

    telegram_token: str
    openai_api_key: str
    openai_base_url: str
    chat_model: str
    pinecone_api_key: str
    index_name: str
    namespace: str
    embedding_model: str
    embedding_dimension: int
    pinecone_cloud: str
    pinecone_region: str


def read_env(name: str) -> str | None:
    """Читает непустую переменную окружения.

    Args:
        name: Имя переменной.

    Returns:
        Обрезанное значение или ``None``.
    """
    raw = os.getenv(name)
    if raw is None:
        return None
    value = raw.strip()
    if value == '':
        return None
    return value


def require_config(value: str | None, name: str) -> str:
    """Проверяет, что обязательная настройка задана.

    Args:
        value: Уже прочитанное значение.
        name: Имя переменной для текста ошибки.

    Returns:
        То же значение.

    Raises:
        BotError: Если значение отсутствует.
    """
    if value is None or value.strip() == '':
        raise BotError(
            code='config_error',
            message=f'Не задана переменная {name}.',
            details={'env': name},
        )
    return value


def read_positive_int(name: str, default: int) -> int:
    """Читает положительное целое из окружения.

    Args:
        name: Имя переменной.
        default: Значение по умолчанию.

    Returns:
        Целое больше нуля.

    Raises:
        BotError: Если строка не является положительным целым.
    """
    raw = read_env(name)
    if raw is None:
        return default
    if not raw.isdecimal():
        raise BotError(
            code='config_error',
            message=f'{name} должно быть положительным целым.',
            details={'env': name},
        )
    value = int(raw)
    if value <= 0:
        raise BotError(
            code='config_error',
            message=f'{name} должно быть положительным целым.',
            details={'env': name},
        )
    return value


def require_embedding_dimension(model: str, dimension: int) -> None:
    """Проверяет, что размерность не больше полной для модели OpenAI.

    Для ``text-embedding-3-*`` API допускает укорочение вектора, но не
    увеличение сверх полной размерности модели.

    Args:
        model: Имя модели эмбеддингов OpenAI.
        dimension: Запрошенная размерность индекса.

    Raises:
        BotError: Если размерность больше полной для известной модели.
    """
    full_size = MODEL_DIMENSIONS.get(model)
    if full_size is None:
        return
    if dimension > full_size:
        raise BotError(
            code='config_error',
            message=(
                f'Модель {model} даёт максимум {full_size} измерений, '
                f'а {OPENAI_EMBEDDING_DIMENSIONS_ENV} равен {dimension}.'
            ),
            details={'model': model, 'dimension': dimension},
        )


def require_index_name(name: str) -> str:
    """Проверяет имя индекса по правилам Pinecone.

    Args:
        name: Имя индекса.

    Returns:
        То же имя.

    Raises:
        BotError: Если имя недопустимо.
    """
    if INDEX_NAME_PATTERN.fullmatch(name) is None:
        raise BotError(
            code='config_error',
            message=(
                f'{HAYSTACK_PINECONE_INDEX_ENV} должно состоять из строчных '
                'латинских букв, цифр и дефисов и быть не длиннее 45 символов.'
            ),
            details={'index': name},
        )
    return name


def require_cloud(name: str) -> str:
    """Проверяет облако для serverless-индекса.

    Args:
        name: Имя облака.

    Returns:
        То же имя.

    Raises:
        BotError: Если облако не поддерживается.
    """
    if name not in ALLOWED_CLOUDS:
        raise BotError(
            code='config_error',
            message=f'{PINECONE_CLOUD_ENV} должно быть aws, gcp или azure.',
            details={'cloud': name},
        )
    return name


def load_config() -> AppConfig:
    """Загружает ``.env`` и собирает конфигурацию приложения.

    Returns:
        Неизменяемый объект настроек.

    Raises:
        BotError: Если обязательные переменные не заданы или некорректны.
    """
    load_dotenv(ENV_PATH)
    load_dotenv()
    embedding_model = (
        read_env(OPENAI_EMBEDDING_MODEL_ENV) or DEFAULT_EMBEDDING_MODEL
    )
    embedding_dimension = read_positive_int(
        OPENAI_EMBEDDING_DIMENSIONS_ENV,
        DEFAULT_EMBEDDING_DIMENSION,
    )
    require_embedding_dimension(embedding_model, embedding_dimension)
    index_name = require_index_name(
        read_env(HAYSTACK_PINECONE_INDEX_ENV) or DEFAULT_INDEX_NAME,
    )
    cloud = require_cloud(read_env(PINECONE_CLOUD_ENV) or DEFAULT_CLOUD)
    return AppConfig(
        telegram_token=require_config(
            read_env(TELEGRAM_BOT_TOKEN_ENV),
            TELEGRAM_BOT_TOKEN_ENV,
        ),
        openai_api_key=require_config(
            read_env(OPENAI_API_KEY_ENV),
            OPENAI_API_KEY_ENV,
        ),
        openai_base_url=(
            read_env(OPENAI_BASE_URL_ENV) or DEFAULT_OPENAI_BASE_URL
        ),
        chat_model=read_env(OPENAI_CHAT_MODEL_ENV) or DEFAULT_CHAT_MODEL,
        pinecone_api_key=require_config(
            read_env(PINECONE_API_KEY_ENV),
            PINECONE_API_KEY_ENV,
        ),
        index_name=index_name,
        namespace=read_env(HAYSTACK_PINECONE_NAMESPACE_ENV) or DEFAULT_NAMESPACE,
        embedding_model=embedding_model,
        embedding_dimension=embedding_dimension,
        pinecone_cloud=cloud,
        pinecone_region=read_env(PINECONE_REGION_ENV) or DEFAULT_REGION,
    )
