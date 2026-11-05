"""Компонент генерации ответа по найденным документам и истории."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from haystack import Document, component
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.dataclasses import ChatMessage

from hay_v2_bot.components.constants import (
    ASSISTANT_ROLE,
    MIN_COSINE_SCORE,
    USER_ROLE,
)
from hay_v2_bot.components.errors import BotError
from hay_v2_bot.components.memory_utils import (
    format_documents,
    format_history,
    select_documents,
)
from hay_v2_bot.components.texts import EMPTY_ANSWER_TEXT, SYSTEM_PROMPT


def build_chat_messages(
    documents: Sequence[Document],
    history: Sequence[Mapping[str, str]],
    query: str,
    system_prompt: str = SYSTEM_PROMPT,
) -> list[ChatMessage]:
    """Собирает сообщения для Chat Completions.

    Args:
        documents: Найденные фрагменты памяти и файлов.
        history: Недавние реплики сеанса.
        query: Текущий вопрос пользователя.
        system_prompt: Базовая системная инструкция.

    Returns:
        Список ``ChatMessage`` для OpenAIChatGenerator.
    """
    memory_text = format_documents(documents)
    history_text = format_history(history)
    system_parts = [system_prompt]
    if memory_text != '':
        system_parts.append(
            'Что уже известно из памяти и загруженных документов:\n'
            f'{memory_text}',
        )
    system_parts.append(f'Недавний диалог:\n{history_text}')
    messages = [ChatMessage.from_system('\n\n'.join(system_parts))]
    for item in history:
        role = item.get('role', '')
        content = item.get('content', '')
        if role == USER_ROLE and content != '':
            messages.append(ChatMessage.from_user(content))
        elif role == ASSISTANT_ROLE and content != '':
            messages.append(ChatMessage.from_assistant(content))
    messages.append(ChatMessage.from_user(query))
    return messages


def extract_reply_text(replies: object) -> str:
    """Достаёт текст первого ответа генератора.

    Args:
        replies: Поле ``replies`` из результата генератора.

    Returns:
        Текст ответа или запасная фраза.
    """
    if not isinstance(replies, list) or len(replies) == 0:
        return EMPTY_ANSWER_TEXT
    first = replies[0]
    if isinstance(first, ChatMessage) and isinstance(first.text, str):
        cleaned = first.text.strip()
        if cleaned != '':
            return cleaned
    if isinstance(first, str) and first.strip() != '':
        return first.strip()
    return EMPTY_ANSWER_TEXT


@component
class ContextChatResponder:
    """Строит промпт из документов и истории, затем вызывает LLM."""

    def __init__(self, chat_generator: OpenAIChatGenerator) -> None:
        """Сохраняет генератор чата.

        Args:
            chat_generator: OpenAI Chat Completions через Haystack.
        """
        self._chat_generator = chat_generator

    @component.output_types(answer=str, replies=list[ChatMessage])
    def run(
        self,
        documents: list[Document],
        query: str,
        history: list[dict[str, str]] | None = None,
        system_prompt: str = SYSTEM_PROMPT,
    ) -> dict[str, object]:
        """Генерирует ответ с опорой на найденный контекст.

        Args:
            documents: Результат retriever.
            query: Вопрос пользователя.
            history: История сеанса. ``None`` трактуется как пустой список.
            system_prompt: Системная инструкция.

        Returns:
            Словарь с ``answer`` и сырыми ``replies``.

        Raises:
            BotError: Если модель не ответила.
        """
        filtered = select_documents(documents, MIN_COSINE_SCORE)
        messages = build_chat_messages(
            documents=filtered,
            history=history or [],
            query=query,
            system_prompt=system_prompt,
        )
        try:
            result = self._chat_generator.run(messages=messages)
        except Exception as error:
            raise BotError(
                code='chat_error',
                message='Диалоговая модель не ответила.',
                details={'error_type': type(error).__name__},
            ) from error
        replies = result.get('replies', [])
        return {
            'answer': extract_reply_text(replies),
            'replies': replies if isinstance(replies, list) else [],
        }
