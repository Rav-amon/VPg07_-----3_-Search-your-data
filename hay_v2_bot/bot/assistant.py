"""Оркестрация Telegram-бота: текст, файлы, память и пайплайны Haystack."""

from __future__ import annotations

import logging
import tempfile
import threading
from pathlib import Path

import telebot
import telebot.apihelper as telegram_api
from haystack import Document, Pipeline
from haystack_integrations.document_stores.pinecone import PineconeDocumentStore

from hay_v2_bot.bot.files import (
    download_telegram_document,
    is_supported_document,
)
from hay_v2_bot.components.config import AppConfig, load_config
from hay_v2_bot.components.constants import (
    ASSISTANT_ROLE,
    COSINE_DEDUP_THRESHOLD,
    DOC_TYPE_CHAT,
    FORGET_ROUNDS,
    LONG_POLLING_TIMEOUT,
    MAX_HISTORY_MESSAGES,
    POLLING_HTTP_TIMEOUT,
    POLLING_SESSION_SECONDS,
    TYPING_INTERVAL_SECONDS,
    TYPING_JOIN_SECONDS,
    USER_ROLE,
)
from hay_v2_bot.components.errors import BotError
from hay_v2_bot.components.memory_utils import (
    MEMORY_ACTION_SKIP,
    MEMORY_ACTION_UPDATE,
    build_chat_document,
    build_profile_document,
    decide_memory_action,
    new_memory_id,
    split_telegram_text,
    trim_history,
    user_memory_filter,
)
from hay_v2_bot.components.texts import (
    CHAT_ERROR_TEXT,
    EMPTY_TEXT,
    FILE_DONE_TEXT,
    FILE_ERROR_TEXT,
    FILE_RECEIVED_TEXT,
    FORGET_ERROR_TEXT,
    FORGET_TEXT,
    MEMORY_ERROR_SUFFIX,
    START_TEXT,
    UNKNOWN_COMMAND_TEXT,
    UNSUPPORTED_FILE_TEXT,
    UNSUPPORTED_MEDIA_TEXT,
)
from hay_v2_bot.pipelines.generation import (
    build_generation_pipeline,
    build_retrieval_pipeline,
    run_generation,
    run_retrieval,
    summarize_file_content,
)
from hay_v2_bot.pipelines.ingestion import build_ingestion_pipeline, run_ingestion
from hay_v2_bot.pipelines.memory_index import (
    build_memory_index_pipeline,
    run_memory_index,
)
from hay_v2_bot.pipelines.store import build_document_store

logger = logging.getLogger(__name__)

UNSUPPORTED_MEDIA_TYPES = (
    'audio',
    'photo',
    'sticker',
    'video',
    'voice',
    'location',
    'contact',
    'animation',
    'video_note',
)


class TelegramAssistant:
    """Принимает сообщения Telegram и обслуживает память с файлами через Haystack."""

    def __init__(
        self,
        config: AppConfig,
        document_store: PineconeDocumentStore,
        generation_pipeline: Pipeline,
        ingestion_pipeline: Pipeline,
        memory_index_pipeline: Pipeline,
        retrieval_pipeline: Pipeline,
    ) -> None:
        """Связывает бота, store и готовые пайплайны.

        Args:
            config: Настройки приложения.
            document_store: Общий Pinecone store.
            generation_pipeline: RAG-ответ на текст.
            ingestion_pipeline: DocLing → Pinecone.
            memory_index_pipeline: Запись реплик.
            retrieval_pipeline: Поиск для дедупликации памяти.
        """
        self._config = config
        self._store = document_store
        self._generation = generation_pipeline
        self._ingestion = ingestion_pipeline
        self._memory_index = memory_index_pipeline
        self._retrieval = retrieval_pipeline
        self._bot = telebot.TeleBot(config.telegram_token)
        self._history: dict[int, list[dict[str, str]]] = {}
        self._lock = threading.Lock()
        self._register_handlers()

    @classmethod
    def from_env(cls) -> TelegramAssistant:
        """Создаёт помощника из переменных окружения.

        Returns:
            Готовый ассистент.

        Raises:
            BotError: Если конфигурация или зависимости неверны.
        """
        config = load_config()
        # Дублируем ключ в окружение на случай компонентов, читающих только env.
        import os

        os.environ['OPENAI_API_KEY'] = config.openai_api_key
        os.environ['OPENAI_BASE_URL'] = config.openai_base_url
        document_store = build_document_store(config)
        generation = build_generation_pipeline(document_store, config)
        ingestion = build_ingestion_pipeline(document_store, config)
        memory_index = build_memory_index_pipeline(document_store, config)
        retrieval = build_retrieval_pipeline(document_store, config)
        for pipeline in (generation, ingestion, memory_index, retrieval):
            pipeline.warm_up()
        return cls(
            config=config,
            document_store=document_store,
            generation_pipeline=generation,
            ingestion_pipeline=ingestion,
            memory_index_pipeline=memory_index,
            retrieval_pipeline=retrieval,
        )

    def run(self) -> None:
        """Запускает long polling Telegram."""
        logger.info('Telegram-бот v2 (Haystack + DocLing) запущен.')
        telegram_api.RETRY_ON_ERROR = True
        telegram_api.SESSION_TIME_TO_LIVE = POLLING_SESSION_SECONDS
        self._bot.infinity_polling(
            skip_pending=True,
            timeout=POLLING_HTTP_TIMEOUT,
            long_polling_timeout=LONG_POLLING_TIMEOUT,
        )

    def _register_handlers(self) -> None:
        """Подключает команды, текст и документы."""
        self._bot.message_handler(commands=['start'])(self.on_start)
        self._bot.message_handler(commands=['forget'])(self.on_forget)
        self._bot.message_handler(content_types=['text'])(self.on_text)
        self._bot.message_handler(content_types=['document'])(self.on_document)
        self._bot.message_handler(
            content_types=list(UNSUPPORTED_MEDIA_TYPES),
        )(self.on_unsupported)

    def on_start(self, message: telebot.types.Message) -> None:
        """Знакомится с пользователем и сохраняет профиль.

        Args:
            message: Команда /start.
        """
        user = message.from_user
        if user is None:
            return
        self._try_index_documents([build_profile_document(user)], user.id)
        self._reply(message, START_TEXT)

    def on_forget(self, message: telebot.types.Message) -> None:
        """Удаляет долговременную память и файлы пользователя.

        Args:
            message: Команда /forget.
        """
        user = message.from_user
        if user is None:
            return
        try:
            self._delete_user_memory(user.id)
        except BotError as error:
            self._log_error('forget', user.id, error)
            self._reply(message, FORGET_ERROR_TEXT)
            return
        with self._lock:
            self._history.pop(user.id, None)
        self._reply(message, FORGET_TEXT)

    def on_unsupported(self, message: telebot.types.Message) -> None:
        """Отвечает на неподдерживаемые вложения.

        Args:
            message: Сообщение другого типа.
        """
        self._reply(message, UNSUPPORTED_MEDIA_TEXT)

    def on_text(self, message: telebot.types.Message) -> None:
        """Отвечает на реплику через generation_pipeline и пишет её в память.

        Args:
            message: Текстовое сообщение.
        """
        user = message.from_user
        raw_text = message.text or ''
        text = raw_text.strip()
        if user is None:
            return
        if text.startswith('/'):
            self._reply(message, UNKNOWN_COMMAND_TEXT)
            return
        if text == '':
            self._reply(message, EMPTY_TEXT)
            return
        stop_typing = threading.Event()
        typing_worker = threading.Thread(
            target=self._keep_typing,
            args=(message.chat.id, stop_typing),
            daemon=True,
        )
        typing_worker.start()
        with self._lock:
            history = list(self._history.get(user.id, []))
        try:
            answer = run_generation(
                self._generation,
                user_id=user.id,
                query=text,
                history=history,
            )
        except BotError as error:
            self._log_error('chat', user.id, error)
            self._try_remember(user, text)
            self._reply(message, CHAT_ERROR_TEXT)
            return
        finally:
            stop_typing.set()
            typing_worker.join(timeout=TYPING_JOIN_SECONDS)
        saved = self._try_remember(user, text)
        if not saved:
            answer = f'{answer}{MEMORY_ERROR_SUFFIX}'
        with self._lock:
            self._history[user.id] = trim_history(
                [
                    *history,
                    {'role': USER_ROLE, 'content': text},
                    {'role': ASSISTANT_ROLE, 'content': answer},
                ],
                MAX_HISTORY_MESSAGES,
            )
        self._reply(message, answer)

    def on_document(self, message: telebot.types.Message) -> None:
        """Принимает файл и запускает анализ в фоне, не блокируя polling.

        Args:
            message: Сообщение с документом.
        """
        user = message.from_user
        document = message.document
        if user is None or document is None:
            return
        if not is_supported_document(document):
            self._reply(message, UNSUPPORTED_FILE_TEXT)
            return
        self._reply(message, FILE_RECEIVED_TEXT)
        worker = threading.Thread(
            target=self._process_document,
            args=(message, user, document),
            daemon=True,
            name=f'doc-ingest-{user.id}',
        )
        worker.start()

    def _process_document(
        self,
        message: telebot.types.Message,
        user: telebot.types.User,
        document: telebot.types.Document,
    ) -> None:
        """Скачивает файл, прогоняет ingestion_pipeline и шлёт резюме.

        Args:
            message: Исходное сообщение с документом.
            user: Отправитель.
            document: Вложение Telegram.
        """
        stop_typing = threading.Event()
        typing_worker = threading.Thread(
            target=self._keep_typing,
            args=(message.chat.id, stop_typing),
            daemon=True,
        )
        typing_worker.start()
        temp_dir: tempfile.TemporaryDirectory[str] | None = None
        summary = ''
        try:
            temp_dir = tempfile.TemporaryDirectory(prefix='hay_v2_doc_')
            downloaded = download_telegram_document(
                self._bot,
                document,
                Path(temp_dir.name),
            )
            run_ingestion(
                self._ingestion,
                file_path=str(downloaded.path),
                user_id=str(user.id),
                filename=downloaded.filename,
            )
            chunk_texts = self._load_file_chunk_texts(user.id, downloaded.filename)
            summary = summarize_file_content(
                self._config,
                chunk_texts,
                downloaded.filename,
            )
        except BotError as error:
            self._log_error('ingestion', user.id, error)
            user_facing_codes = {
                'unsupported_file',
                'empty_file',
                'file_too_large',
                'download_error',
            }
            reply_text = (
                error.message if error.code in user_facing_codes else FILE_ERROR_TEXT
            )
            self._reply(message, reply_text)
            return
        except Exception as error:
            self._log_error(
                'ingestion',
                user.id,
                BotError(
                    code='ingestion_error',
                    message='Неожиданная ошибка обработки файла.',
                    details={'error_type': type(error).__name__},
                ),
            )
            self._reply(message, FILE_ERROR_TEXT)
            return
        finally:
            stop_typing.set()
            typing_worker.join(timeout=TYPING_JOIN_SECONDS)
            if temp_dir is not None:
                temp_dir.cleanup()
        self._reply(message, FILE_DONE_TEXT)
        if summary != '':
            self._reply(message, summary)

    def _try_remember(self, user: telebot.types.User, text: str) -> bool:
        """Пишет реплику в память с дедупликацией по косинусному сходству.

        Args:
            user: Владелец факта.
            text: Текст реплики.

        Returns:
            ``True``, если запись/пропуск/обновление прошли без ошибки.
        """
        try:
            nearest = run_retrieval(
                self._retrieval,
                user_id=user.id,
                text=text,
                top_k=1,
            )
            matched = nearest[0] if len(nearest) > 0 else None
            similarity = matched.score if matched is not None else None
            stored_text = matched.content if matched is not None else None
            action = decide_memory_action(
                similarity,
                text,
                stored_text if isinstance(stored_text, str) else None,
                COSINE_DEDUP_THRESHOLD,
            )
            if action == MEMORY_ACTION_SKIP:
                return True
            if action == MEMORY_ACTION_UPDATE and matched is not None:
                memory_id = matched.id
            else:
                memory_id = new_memory_id()
            document = build_chat_document(
                user,
                text,
                doc_type=DOC_TYPE_CHAT,
                document_id=memory_id,
            )
            run_memory_index(self._memory_index, [document])
        except BotError as error:
            self._log_error('remember', user.id, error)
            return False
        return True

    def _try_index_documents(
        self,
        documents: list[Document],
        user_id: int,
    ) -> bool:
        """Индексирует готовые документы без дедупликации.

        Args:
            documents: Документы без векторов.
            user_id: Владелец для журнала.

        Returns:
            ``True`` при успехе.
        """
        try:
            run_memory_index(self._memory_index, documents)
        except BotError as error:
            self._log_error('remember', user_id, error)
            return False
        return True

    def _load_file_chunk_texts(self, user_id: int, filename: str) -> list[str]:
        """Читает сохранённые чанки файла для резюме.

        Args:
            user_id: Владелец.
            filename: Имя файла в метаданных.

        Returns:
            Тексты чанков по порядку chunk_index.
        """
        filters = {
            'operator': 'AND',
            'conditions': [
                {
                    'field': 'meta.user_id',
                    'operator': '==',
                    'value': str(user_id),
                },
                {
                    'field': 'meta.filename',
                    'operator': '==',
                    'value': filename,
                },
                {
                    'field': 'meta.doc_type',
                    'operator': '==',
                    'value': 'file_chunk',
                },
            ],
        }
        try:
            documents = self._store.filter_documents(filters=filters)
        except Exception as error:
            logger.warning(
                'Не удалось прочитать чанки для резюме: error_type=%s',
                type(error).__name__,
            )
            return []
        def chunk_sort_key(item: Document) -> int:
            raw_index = item.meta.get('chunk_index', 0)
            if isinstance(raw_index, bool):
                return 0
            if isinstance(raw_index, (int, float)):
                return int(raw_index)
            if isinstance(raw_index, str) and raw_index.lstrip('-').isdecimal():
                return int(raw_index)
            return 0

        ordered = sorted(documents, key=chunk_sort_key)
        texts: list[str] = []
        for document in ordered:
            if isinstance(document.content, str) and document.content.strip() != '':
                texts.append(document.content.strip())
        return texts

    def _delete_user_memory(self, user_id: int) -> None:
        """Удаляет документы пользователя пакетами.

        Args:
            user_id: Идентификатор Telegram.

        Raises:
            BotError: Если удаление не удалось.
        """
        filters = user_memory_filter(user_id)
        try:
            for _round in range(FORGET_ROUNDS):
                deleted = self._store.delete_by_filter(filters=filters)
                if deleted == 0:
                    return
        except Exception as error:
            raise BotError(
                code='forget_error',
                message='Не удалось удалить память пользователя.',
                details={'error_type': type(error).__name__},
            ) from error

    def _keep_typing(self, chat_id: int, stop: threading.Event) -> None:
        """Повторяет статус «печатает», пока готовится ответ.

        Args:
            chat_id: Чат Telegram.
            stop: Событие остановки.
        """
        while not stop.is_set():
            try:
                self._bot.send_chat_action(chat_id, 'typing')
            except Exception as error:
                logger.warning(
                    'Не удалось отправить статус набора: error_type=%s',
                    type(error).__name__,
                )
                return
            if stop.wait(TYPING_INTERVAL_SECONDS):
                return

    def _reply(self, message: telebot.types.Message, text: str) -> None:
        """Отправляет ответ, разрезая его на допустимые части.

        Args:
            message: Исходное сообщение пользователя.
            text: Текст ответа.
        """
        parts = split_telegram_text(text)
        self._bot.reply_to(message, parts[0])
        for part in parts[1:]:
            self._bot.send_message(message.chat.id, part)

    def _log_error(self, operation: str, user_id: int, error: BotError) -> None:
        """Пишет в журнал тип сбоя без текста сообщения и секретов.

        Args:
            operation: Имя шага.
            user_id: Идентификатор пользователя.
            error: Пойманная ошибка.
        """
        logger.error(
            'Ошибка бота: operation=%s user_id=%s code=%s error_type=%s',
            operation,
            user_id,
            error.code,
            error.details.get('error_type', ''),
        )
