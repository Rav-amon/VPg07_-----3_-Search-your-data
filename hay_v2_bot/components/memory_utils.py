"""Чистые функции для истории диалога, фильтров и дедупликации памяти."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from uuid import uuid4

import telebot
from haystack import Document

from hay_v2_bot.components.constants import (
    ASSISTANT_ROLE,
    COSINE_DEDUP_THRESHOLD,
    DOC_TYPE_CHAT,
    DOC_TYPE_PROFILE,
    MAX_MEMORY_CHARS,
    TELEGRAM_MESSAGE_LIMIT,
    USER_ROLE,
)
from hay_v2_bot.components.texts import EMPTY_ANSWER_TEXT

MEMORY_ACTION_REMEMBER = 'remember'
MEMORY_ACTION_SKIP = 'skip'
MEMORY_ACTION_UPDATE = 'update'


def user_memory_filter(user_id: int) -> dict[str, object]:
    """Собирает фильтр Pinecone только по записям одного пользователя.

    Args:
        user_id: Числовой идентификатор Telegram.

    Returns:
        Фильтр метаданных для поиска и удаления.
    """
    return {
        'field': 'meta.user_id',
        'operator': '==',
        'value': str(user_id),
    }


def new_memory_id() -> str:
    """Генерирует идентификатор слота памяти.

    Returns:
        UUID без дефисов.
    """
    return uuid4().hex


def profile_text(user: telebot.types.User) -> str:
    """Формулирует устойчивый факт о том, кто пишет боту.

    Args:
        user: Отправитель команды /start.

    Returns:
        Текст для долговременной памяти.
    """
    name = user.first_name.strip() if isinstance(user.first_name, str) else ''
    if name == '':
        name = 'без имени'
    parts = [f'Имя пользователя в Telegram: {name}.', f'Идентификатор: {user.id}.']
    if isinstance(user.username, str) and user.username.strip() != '':
        parts.append(f'Имя пользователя: @{user.username.strip()}.')
    return ' '.join(parts)


def build_chat_document(
    user: telebot.types.User,
    text: str,
    doc_type: str = DOC_TYPE_CHAT,
    document_id: str | None = None,
) -> Document:
    """Собирает Haystack-документ для записи реплики или профиля.

    Args:
        user: Владелец факта.
        text: Текст для эмбеддинга и хранения.
        doc_type: Тип записи в метаданных.
        document_id: Готовый идентификатор или ``None`` для нового.

    Returns:
        Документ без вектора.
    """
    cleaned = text.strip()[:MAX_MEMORY_CHARS]
    meta: dict[str, object] = {
        'user_id': str(user.id),
        'doc_type': doc_type,
        'role': USER_ROLE,
    }
    if isinstance(user.first_name, str) and user.first_name.strip() != '':
        meta['first_name'] = user.first_name.strip()
    if isinstance(user.username, str) and user.username.strip() != '':
        meta['username'] = user.username.strip()
    return Document(
        id=document_id or new_memory_id(),
        content=cleaned,
        meta=meta,
    )


def build_profile_document(user: telebot.types.User) -> Document:
    """Собирает документ профиля пользователя.

    Args:
        user: Отправитель /start.

    Returns:
        Документ профиля.
    """
    return build_chat_document(
        user,
        profile_text(user),
        doc_type=DOC_TYPE_PROFILE,
        document_id=f'profile-{user.id}',
    )


def format_documents(documents: Sequence[Document]) -> str:
    """Склеивает найденные фрагменты в маркированный список.

    Args:
        documents: Документы поиска.

    Returns:
        Текст для подсказки или пустая строка.
    """
    lines: list[str] = []
    for document in documents:
        content = document.content
        if not isinstance(content, str):
            continue
        cleaned = content.strip()
        if cleaned == '' or cleaned in lines:
            continue
        filename = document.meta.get('filename') if document.meta else None
        if isinstance(filename, str) and filename.strip() != '':
            lines.append(f'[{filename}] {cleaned}')
        else:
            lines.append(cleaned)
    return '\n'.join(f'- {line}' for line in lines)


def select_documents(
    documents: Sequence[Document],
    min_score: float,
) -> list[Document]:
    """Оставляет документы не ниже порога косинусного сходства.

    Args:
        documents: Результат retriever.
        min_score: Нижняя граница score.

    Returns:
        Отфильтрованный список.
    """
    selected: list[Document] = []
    for document in documents:
        score = document.score
        if score is None or score >= min_score:
            selected.append(document)
    return selected


def decide_memory_action(
    similarity: float | None,
    new_text: str,
    stored_text: str | None,
    threshold: float = COSINE_DEDUP_THRESHOLD,
) -> str:
    """Решает, создавать, обновлять или пропускать слот памяти.

    Args:
        similarity: Сходство с ближайшим фрагментом.
        new_text: Новый текст.
        stored_text: Текст ближайшего слота.
        threshold: Порог дедупликации.

    Returns:
        ``remember``, ``skip`` или ``update``.
    """
    if similarity is None or similarity < threshold:
        return MEMORY_ACTION_REMEMBER
    if (
        isinstance(stored_text, str)
        and stored_text.strip() == new_text.strip()
    ):
        return MEMORY_ACTION_SKIP
    return MEMORY_ACTION_UPDATE


def trim_history(
    turns: Sequence[Mapping[str, str]],
    limit: int,
) -> list[dict[str, str]]:
    """Оставляет только последние реплики сеанса.

    Args:
        turns: Полная история.
        limit: Максимум сообщений.

    Returns:
        Новый список не длиннее ``limit``.
    """
    kept = [
        {'role': item['role'], 'content': item['content']}
        for item in turns
        if item.get('role') in (USER_ROLE, ASSISTANT_ROLE)
        and isinstance(item.get('content'), str)
        and item.get('content') != ''
    ]
    if len(kept) <= limit:
        return kept
    return kept[-limit:]


def format_history(history: Sequence[Mapping[str, str]]) -> str:
    """Превращает историю сеанса в текст для промпта.

    Args:
        history: Реплики с полями ``role`` и ``content``.

    Returns:
        Многострочный текст или пометку об отсутствии истории.
    """
    lines: list[str] = []
    for item in history:
        role = item.get('role', '')
        content = item.get('content', '')
        if role not in (USER_ROLE, ASSISTANT_ROLE) or content == '':
            continue
        label = 'Пользователь' if role == USER_ROLE else 'Помощник'
        lines.append(f'{label}: {content}')
    if len(lines) == 0:
        return 'Истории сеанса пока нет.'
    return '\n'.join(lines)


def split_telegram_text(
    text: str,
    limit: int = TELEGRAM_MESSAGE_LIMIT,
) -> list[str]:
    """Делит длинный ответ на части, которые принимает Telegram.

    Args:
        text: Исходный ответ.
        limit: Максимальная длина одной части.

    Returns:
        Непустой список частей.
    """
    if text.strip() == '':
        return [EMPTY_ANSWER_TEXT]
    if len(text) <= limit:
        return [text]
    parts: list[str] = []
    rest = text
    while len(rest) > limit:
        cut = rest.rfind('\n', 0, limit)
        if cut < 1:
            cut = limit
        parts.append(rest[:cut].strip())
        rest = rest[cut:].lstrip()
    if rest.strip() != '':
        parts.append(rest.strip())
    return [part for part in parts if part != '']


def first_sentence(text: str) -> str:
    """Оставляет только первое предложение ответа.

    Args:
        text: Сырой ответ модели.

    Returns:
        Одно предложение без лишних пробелов.
    """
    cleaned = ' '.join(text.strip().split())
    if cleaned == '':
        return ''
    for marker in ('. ', '! ', '? '):
        index = cleaned.find(marker)
        if index != -1:
            return cleaned[: index + 1].strip()
    if cleaned[-1] not in '.!?':
        return f'{cleaned}.'
    return cleaned
