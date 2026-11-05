"""Скачивание и проверка файлов из Telegram."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import telebot

from hay_v2_bot.components.constants import (
    MAX_FILE_BYTES,
    SUPPORTED_FILE_EXTENSIONS,
    SUPPORTED_MIME_TYPES,
)
from hay_v2_bot.components.errors import BotError
from hay_v2_bot.components.texts import (
    FILE_EMPTY_TEXT,
    FILE_TOO_LARGE_TEXT,
    UNSUPPORTED_FILE_TEXT,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DownloadedFile:
    """Локальная копия файла из Telegram.

    Attributes:
        path: Путь во временном каталоге.
        filename: Оригинальное имя файла.
        extension: Расширение в нижнем регистре с точкой.
    """

    path: Path
    filename: str
    extension: str


def resolve_extension(filename: str | None, mime_type: str | None) -> str | None:
    """Определяет расширение файла по имени или MIME.

    Args:
        filename: Имя из Telegram.
        mime_type: MIME-тип вложения.

    Returns:
        Расширение с точкой или ``None``.
    """
    if isinstance(filename, str) and '.' in filename:
        extension = Path(filename).suffix.lower()
        if extension in SUPPORTED_FILE_EXTENSIONS:
            return extension
    mime_map = {
        'application/pdf': '.pdf',
        'application/vnd.openxmlformats-officedocument.wordprocessingml.document': (
            '.docx'
        ),
        'application/vnd.openxmlformats-officedocument.presentationml.presentation': (
            '.pptx'
        ),
        'text/html': '.html',
        'text/markdown': '.md',
        'text/plain': '.txt',
        'text/csv': '.csv',
        'image/png': '.png',
        'image/jpeg': '.jpg',
        'image/tiff': '.tiff',
        'image/webp': '.webp',
    }
    if isinstance(mime_type, str):
        mapped = mime_map.get(mime_type.lower())
        if mapped is not None:
            return mapped
    return None


def is_supported_document(document: telebot.types.Document) -> bool:
    """Проверяет, поддерживается ли вложение DocLing.

    Args:
        document: Объект document из Telegram.

    Returns:
        ``True``, если формат в белом списке.
    """
    extension = resolve_extension(document.file_name, document.mime_type)
    if extension is not None:
        return True
    mime = document.mime_type
    return isinstance(mime, str) and mime.lower() in SUPPORTED_MIME_TYPES


def download_telegram_document(
    bot: telebot.TeleBot,
    document: telebot.types.Document,
    target_dir: Path,
) -> DownloadedFile:
    """Скачивает документ Telegram во временный каталог.

    Args:
        bot: Экземпляр TeleBot.
        document: Вложение пользователя.
        target_dir: Каталог для сохранения.

    Returns:
        Описание локального файла.

    Raises:
        BotError: Если файл пустой, слишком большой или формат не поддержан.
    """
    filename = document.file_name if isinstance(document.file_name, str) else ''
    if filename.strip() == '':
        filename = f'document_{document.file_id}'
    extension = resolve_extension(document.file_name, document.mime_type)
    if extension is None:
        raise BotError(
            code='unsupported_file',
            message=UNSUPPORTED_FILE_TEXT,
            details={'mime_type': document.mime_type or ''},
        )
    file_size = document.file_size if isinstance(document.file_size, int) else 0
    if file_size <= 0 and document.file_size is not None:
        raise BotError(
            code='empty_file',
            message=FILE_EMPTY_TEXT,
        )
    if file_size > MAX_FILE_BYTES:
        raise BotError(
            code='file_too_large',
            message=FILE_TOO_LARGE_TEXT,
            details={'file_size': file_size},
        )
    target_dir.mkdir(parents=True, exist_ok=True)
    safe_stem = Path(filename).stem.replace('/', '_').replace('\\', '_')[:80]
    if safe_stem.strip() == '':
        safe_stem = 'document'
    local_path = target_dir / f'{safe_stem}{extension}'
    try:
        file_info = bot.get_file(document.file_id)
        payload = bot.download_file(file_info.file_path)
    except Exception as error:
        raise BotError(
            code='download_error',
            message='Не удалось скачать файл из Telegram.',
            details={'error_type': type(error).__name__},
        ) from error
    if not isinstance(payload, (bytes, bytearray)) or len(payload) == 0:
        raise BotError(code='empty_file', message=FILE_EMPTY_TEXT)
    if len(payload) > MAX_FILE_BYTES:
        raise BotError(
            code='file_too_large',
            message=FILE_TOO_LARGE_TEXT,
            details={'file_size': len(payload)},
        )
    local_path.write_bytes(bytes(payload))
    logger.info(
        'Файл скачан: name=%s bytes=%s',
        filename,
        len(payload),
    )
    return DownloadedFile(
        path=local_path,
        filename=Path(filename).name,
        extension=extension,
    )
