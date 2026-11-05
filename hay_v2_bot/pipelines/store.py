"""Фабрика PineconeDocumentStore и создание serverless-индекса."""

from __future__ import annotations

import logging
import time

from haystack.utils import Secret
from haystack_integrations.document_stores.pinecone import PineconeDocumentStore
from pinecone import Pinecone, ServerlessSpec

from hay_v2_bot.components.config import AppConfig
from hay_v2_bot.components.errors import BotError

logger = logging.getLogger(__name__)

INDEX_READY_ATTEMPTS = 30
INDEX_READY_SLEEP_SECONDS = 2.0


def metric_name(raw: object) -> str:
    """Приводит метрику индекса Pinecone к строке в нижнем регистре.

    Args:
        raw: Поле description.metric.

    Returns:
        Строковое имя метрики.
    """
    value = getattr(raw, 'value', raw)
    return str(value).lower()


def ensure_pinecone_index(config: AppConfig) -> None:
    """Создаёт dense-индекс с метрикой cosine, если его ещё нет.

    Args:
        config: Настройки приложения.

    Raises:
        BotError: Если индекс существует с другой размерностью/метрикой
            или создание не удалось.
    """
    client = Pinecone(api_key=config.pinecone_api_key)
    existing = {item['name'] for item in client.list_indexes()}
    if config.index_name in existing:
        description = client.describe_index(config.index_name)
        dimension = int(description.dimension)
        metric = metric_name(description.metric)
        if dimension != config.embedding_dimension:
            raise BotError(
                code='index_error',
                message=(
                    f'Индекс {config.index_name} имеет размерность {dimension}, '
                    f'ожидалась {config.embedding_dimension}.'
                ),
                details={
                    'index': config.index_name,
                    'dimension': dimension,
                },
            )
        if metric != 'cosine':
            raise BotError(
                code='index_error',
                message=(
                    f'Индекс {config.index_name} использует метрику {metric}, '
                    'нужна cosine.'
                ),
                details={'index': config.index_name, 'metric': metric},
            )
        return
    logger.info('Создаю индекс Pinecone: %s', config.index_name)
    try:
        client.create_index(
            name=config.index_name,
            dimension=config.embedding_dimension,
            metric='cosine',
            spec=ServerlessSpec(
                cloud=config.pinecone_cloud,
                region=config.pinecone_region,
            ),
        )
    except Exception as error:
        raise BotError(
            code='index_error',
            message='Не удалось создать индекс Pinecone.',
            details={'error_type': type(error).__name__},
        ) from error
    for _attempt in range(INDEX_READY_ATTEMPTS):
        description = client.describe_index(config.index_name)
        status = getattr(description, 'status', None)
        ready = False
        if isinstance(status, dict):
            ready = bool(status.get('ready', False))
        elif status is not None:
            ready = bool(getattr(status, 'ready', False))
        if ready:
            return
        time.sleep(INDEX_READY_SLEEP_SECONDS)
    raise BotError(
        code='index_error',
        message='Индекс Pinecone создан, но не перешёл в ready вовремя.',
        details={'index': config.index_name},
    )


def build_document_store(config: AppConfig) -> PineconeDocumentStore:
    """Создаёт document store Haystack поверх dense-индекса Pinecone.

    Args:
        config: Настройки приложения.

    Returns:
        Готовый ``PineconeDocumentStore``.
    """
    ensure_pinecone_index(config)
    return PineconeDocumentStore(
        api_key=Secret.from_token(config.pinecone_api_key),
        index=config.index_name,
        namespace=config.namespace,
        dimension=config.embedding_dimension,
        metric='cosine',
        spec={
            'serverless': {
                'cloud': config.pinecone_cloud,
                'region': config.pinecone_region,
            },
        },
    )
