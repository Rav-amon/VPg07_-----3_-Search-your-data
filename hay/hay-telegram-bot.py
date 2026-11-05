"""Телеграм-помощник на Haystack: агент с инструментами и память в Pinecone.

Долговременный контекст пишется в ``PineconeDocumentStore`` с метрикой cosine,
как в интеграции Pinecone. Перед ответом тот же индекс ищется по эмбеддингу
реплики и подмешивается в системную подсказку. Текущий сеанс дополнительно
держится в сообщениях агента, чтобы диалог не обрывался между соседними фразами.

Инструменты ходят во внешние API: случайный факт о собаках и фото породы.
Если пользователь назвал породу, снимок берётся именно этой породы.
Фото загружается в модель OpenAI, которая рассказывает об этой породе.
"""

from __future__ import annotations

import base64
import json
import logging
import os
import re
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Annotated
from uuid import uuid4

import filetype
import telebot
import telebot.apihelper as telegram_api
from dotenv import load_dotenv
from haystack import Document, Pipeline
from haystack.components.agents import Agent
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.components.writers import DocumentWriter
from haystack.dataclasses import ChatMessage, ChatRole, ImageContent
from haystack.document_stores.types import DuplicatePolicy
from haystack.tools import Tool, tool
from haystack_integrations.components.embedders.sentence_transformers import (
    SentenceTransformersDocumentEmbedder,
    SentenceTransformersTextEmbedder,
)
from haystack_integrations.components.retrievers.pinecone import (
    PineconeEmbeddingRetriever,
)
from haystack_integrations.document_stores.pinecone import PineconeDocumentStore
from pinecone import Pinecone

logger = logging.getLogger(__name__)

# Каталог проекта на уровень выше папки hay, там лежит .env.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENV_PATH = PROJECT_ROOT / '.env'

TELEGRAM_BOT_TOKEN_ENV = 'TELEGRAM_BOT_TOKEN'
OPENAI_API_KEY_ENV = 'OPENAI_API_KEY'
OPENAI_BASE_URL_ENV = 'OPENAI_BASE_URL'
OPENAI_CHAT_MODEL_ENV = 'OPENAI_CHAT_MODEL'
PINECONE_API_KEY_ENV = 'PINECONE_API_KEY'
HAYSTACK_PINECONE_INDEX_ENV = 'HAYSTACK_PINECONE_INDEX'
HAYSTACK_PINECONE_NAMESPACE_ENV = 'HAYSTACK_PINECONE_NAMESPACE'
HAYSTACK_EMBEDDING_MODEL_ENV = 'HAYSTACK_EMBEDDING_MODEL'
HAYSTACK_EMBEDDING_DIMENSION_ENV = 'HAYSTACK_EMBEDDING_DIMENSION'
PINECONE_CLOUD_ENV = 'PINECONE_CLOUD'
PINECONE_REGION_ENV = 'PINECONE_REGION'

DEFAULT_CHAT_MODEL = 'gpt-4o-mini'
DEFAULT_INDEX_NAME = 'vpg06-personal-assistant'
DEFAULT_NAMESPACE = 'default'
DEFAULT_EMBEDDING_MODEL = 'sentence-transformers/all-mpnet-base-v2'
DEFAULT_EMBEDDING_DIMENSION = 768
DEFAULT_CLOUD = 'aws'
DEFAULT_REGION = 'us-east-1'
# Размерность известных моделей. Индекс Pinecone обязан совпадать с ней.
MODEL_DIMENSIONS = {
    'sentence-transformers/all-mpnet-base-v2': 768,
    'sentence-transformers/all-MiniLM-L6-v2': 384,
}
ALLOWED_CLOUDS = frozenset({'aws', 'gcp', 'azure'})
INDEX_NAME_PATTERN = re.compile(r'^[a-z0-9][a-z0-9-]{0,44}$')

# Ниже этого косинусного сходства фрагмент считается чужим и в подсказку не идёт.
MIN_COSINE_SCORE = 0.35
MEMORY_TOP_K = 5
# Сколько последних реплик сеанса отдавать модели вместе с долговременной памятью.
MAX_HISTORY_MESSAGES = 10
# Метаданные Pinecone ограничены по размеру, поэтому реплика обрезается.
MAX_MEMORY_CHARS = 8000
FORGET_ROUNDS = 20
TELEGRAM_MESSAGE_LIMIT = 4000
POLLING_HTTP_TIMEOUT = 30
LONG_POLLING_TIMEOUT = 10
POLLING_SESSION_SECONDS = 120
CHAT_TEMPERATURE = 0.7
VISION_TEMPERATURE = 0.3
MAX_REPLY_TOKENS = 800
MAX_VISION_TOKENS = 700
OPENAI_TIMEOUT_SECONDS = 90.0
OPENAI_MAX_RETRIES = 2
AGENT_STEPS = 6
HTTP_TIMEOUT_SECONDS = 20.0
HTTP_RETRY_ATTEMPTS = 3
HTTP_RETRY_BASE_SECONDS = 0.4
MAX_JSON_BYTES = 65_536
MAX_IMAGE_BYTES = 4 * 1024 * 1024
TRANSIENT_STATUS_CODES = frozenset({429, 500, 502, 503, 504})
USER_AGENT = 'VPg06PersonalAssistant/1.0'
DOG_FACT_API_URL = 'https://dogapi.dog/api/v2/facts'
DOG_IMAGE_API_URL = 'https://dog.ceo/api/breeds/image/random'
DOG_BREEDS_API_URL = 'https://dog.ceo/api/breeds/list/all'
DOG_BREED_IMAGE_API = 'https://dog.ceo/api/breed'
DOG_API_HOSTS = frozenset({'dogapi.dog', 'dog.ceo'})
IMAGE_HOSTS = frozenset({'images.dog.ceo'})
SUPPORTED_IMAGE_MIME_TYPES = frozenset({
    'image/jpeg',
    'image/png',
    'image/gif',
    'image/webp',
})
DOG_IMAGE_LINE_PREFIX = 'DOG_IMAGE_URL:'
DOG_BREED_LINE_PREFIX = 'DOG_BREED:'
PHOTO_CAPTION = 'Случайная собака'
TELEGRAM_CAPTION_LIMIT = 1024
# Окончания снимаются с длинных на короткие, чтобы «ого» не резалось как «о».
BREED_INFLECTION_SUFFIXES = (
    'ами',
    'ями',
    'ого',
    'ему',
    'ому',
    'ыми',
    'ими',
    'ов',
    'ев',
    'ей',
    'ой',
    'ий',
    'ый',
    'ая',
    'яя',
    'ое',
    'ее',
    'ые',
    'ие',
    'ам',
    'ям',
    'ах',
    'ях',
    'ом',
    'ем',
    'ую',
    'юю',
    'а',
    'я',
    'у',
    'ю',
    'е',
    'ы',
    'и',
    'о',
)
ROLE_USER_LABEL = 'Пользователь'
ROLE_ASSISTANT_LABEL = 'Помощник'
TYPING_INTERVAL_SECONDS = 4.0
TYPING_JOIN_SECONDS = 1.0

SYSTEM_PROMPT = (
    'Ты личный помощник в Telegram. Отвечай по-русски, тепло и по делу, '
    'как человек, который помнит собеседника и продолжает разговор.\n'
    'Собеседник: {{ speaker }}\n'
    'Местное время сервера: {{ current_time }}\n\n'
    'Ниже фрагменты памяти этого же пользователя. Они найдены в Pinecone '
    'по косинусному сходству с текущей репликой. Опирайся на них и на '
    'сообщения этого сеанса. Не приписывай человеку факты, которых нет '
    'ни в памяти, ни в диалоге. Если сведения неполные, спроси.\n'
    '{% if memory %}'
    'Память:\n'
    '{% for document in memory %}'
    '- {{ document.content }}\n'
    '{% endfor %}'
    '{% else %}'
    'Близких записей в памяти пока нет.\n'
    '{% endif %}\n'
    'Инструменты вызывай только по просьбе о собаках.\n'
    'get_random_dog_fact — один новый факт. Перескажи его по-русски.\n'
    'describe_random_dog — фото собаки и справка о породе. В аргумент breed '
    'передавай название породы дословно из просьбы пользователя. '
    'Если породу не называли, передай пустую строку: тогда снимок случайный. '
    'Инструмент находит фото именно этой породы, загружает его в модель и '
    'возвращает сведения о ней: что видно на снимке, как порода появилась, '
    'характер и для какой работы её выводили. Перескажи эти сведения. '
    'Не подменяй породу другой. Фото в чат отправит бот. Служебные строки '
    'с адресом не повторяй. Повторно инструмент не вызывай, если не просили '
    'ещё одно фото или другую породу.\n'
    'Если порода уже была в этом диалоге или в памяти и новое фото не просят, '
    'отвечай по уже известным сведениям.'
)
VISION_PROMPT = (
    'На фотографии собака. Определи породу по внешности. '
    'Ответь по-русски, без приветствия, в трёх коротких абзацах:\n'
    '1) какая порода на фото и почему ты так решил;\n'
    '2) как эта порода появилась: от каких собак её выводили и зачем;\n'
    '3) две-три узнаваемые черты характера или работы.\n'
    'Если порода неочевидна, назови ближайшие варианты и не выдумывай кличку.'
)
START_TEXT = (
    'Привет. Я помощник: помню наш разговор и продолжаю его с того места, '
    'где остановились. Можно просто писать.\n'
    'Попросите факт о собаках, случайное фото или снимок конкретной породы. '
    'Я покажу собаку и расскажу, как эта порода появилась.\n'
    'Команды: /fact, /dog, /forget.'
)
FORGET_TEXT = 'Я стёр долговременную память об этом чате.'
FORGET_ERROR_TEXT = 'Не удалось очистить память. Попробуйте ещё раз.'
UNSUPPORTED_TEXT = 'Пока я понимаю только текстовые сообщения.'
EMPTY_TEXT = 'Напишите сообщение текстом.'
UNKNOWN_COMMAND_TEXT = 'Такой команды нет. Доступны /start, /fact, /dog и /forget.'
CHAT_ERROR_TEXT = 'Сейчас не получилось подготовить ответ. Попробуйте ещё раз.'
MEMORY_ERROR_SUFFIX = '\n\nЭту реплику не удалось записать в память.'
EMPTY_ANSWER_TEXT = 'Я не смог сформулировать ответ. Напишите ещё раз.'
REQUEST_FILLER_WORDS = (
    'покажи',
    'покажите',
    'показать',
    'фото',
    'фотографию',
    'фотку',
    'картинку',
    'снимок',
    'собаку',
    'собака',
    'порода',
    'породу',
    'породы',
    'хочу',
    'мне',
    'пожалуйста',
    'изображение',
    'расскажи',
    'про',
    'об',
    'о',
    'этой',
    'эту',
    'это',
    'какой',
    'какая',
    'какую',
    'дай',
    'пришли',
    'найди',
    'информацию',
    'историю',
    'и',
    'а',
    'с',
    'на',
    'для',
)
FACT_REQUEST = 'Расскажи один случайный факт о собаках.'
DOG_REQUEST = 'Покажи одну случайную собаку и расскажи, что это за порода.'
# Путь каталога dog.ceo: русское название и дополнительные формы запроса.
BREED_LABELS: dict[str, tuple[str, tuple[str, ...]]] = {
    'labrador': ('лабрадор', ('лабрадор ретривер', 'labrador retriever')),
    'retriever/golden': (
        'золотистый ретривер',
        ('голден', 'голден ретривер', 'golden retriever'),
    ),
    'retriever/chesapeake': ('чесапик бей ретривер', ('чесапик',)),
    'retriever/flatcoated': ('прямошерстный ретривер', ('флэт коатед ретривер',)),
    'retriever/curly': ('курчавошёрстный ретривер', ()),
    'german/shepherd': (
        'немецкая овчарка',
        ('овчарка', 'немецкая овчарка', 'german shepherd'),
    ),
    'husky': ('хаски', ('сибирский хаски', 'siberian husky')),
    'pembroke': (
        'вельш-корги пемброк',
        ('корги', 'вельш корги', 'вельш корги пемброк', 'пемброк'),
    ),
    'corgi/cardigan': (
        'вельш-корги кардиган',
        ('кардиган', 'корги кардиган'),
    ),
    'terrier/yorkshire': (
        'йоркширский терьер',
        ('йорк', 'йоркшир', 'йоркширский терьер'),
    ),
    'beagle': ('бигль', ()),
    'dachshund': ('такса', ()),
    'chihuahua': ('чихуахуа', ('чихуа',)),
    'pug': ('мопс', ()),
    'rottweiler': ('ротвейлер', ()),
    'doberman': ('доберман', ('доберман пинчер',)),
    'shiba': ('сиба-ину', ('сиба', 'шиба', 'шиба ину', 'shiba inu')),
    'samoyed': ('самоед', ('самоедская собака',)),
    'malamute': ('маламут', ('аляскинский маламут',)),
    'akita': ('акита-ину', ('акита',)),
    'boxer': ('боксер', ()),
    'dalmatian': ('далматин', ('далматинец',)),
    'spaniel/cocker': ('кокер-спаниель', ('кокер', 'кокер спаниель')),
    'bulldog/french': ('французский бульдог', ('французский бульдог', 'французик')),
    'bulldog/english': ('английский бульдог', ()),
    'bulldog/boston': ('бостонский бульдог', ()),
    'pitbull': ('питбуль', ('питбультерьер',)),
    'chow': ('чау-чау', ('чау чау',)),
    'sharpei': ('шарпей', ('шар пей',)),
    'shihtzu': ('ши-тцу', ('ши тцу',)),
    'pomeranian': (
        'померанский шпиц',
        ('шпиц', 'немецкий шпиц', 'померан', 'карликовый шпиц'),
    ),
    'spitz/japanese': ('японский шпиц', ()),
    'maltese': ('мальтийская болонка', ('мальтезе',)),
    'papillon': ('папийон', ('папильон',)),
    'havanese': ('хаванез', ()),
    'pekinese': ('пекинес', ()),
    'basenji': ('басенджи', ()),
    'borzoi': ('русская псовая борзая', ('борзая',)),
    'ovcharka/caucasian': ('кавказская овчарка', ('кавказец',)),
    'mastiff/tibetan': ('тибетский мастиф', ()),
    'mastiff/english': ('английский мастиф', ()),
    'mastiff/bull': ('бульмастиф', ()),
    'ridgeback/rhodesian': ('родезийский риджбек', ('риджбек',)),
    'weimaraner': ('веймаранер', ()),
    'vizsla': ('венгерская выжла', ('выжла',)),
    'whippet': ('уиппет', ()),
    'greyhound/italian': ('левретка', ('итальянская борзая',)),
    'stbernard': ('сенбернар', ('сен бернар',)),
    'newfoundland': ('ньюфаундленд', ()),
    'leonberg': ('леонбергер', ()),
    'mountain/bernese': (
        'бернский зенненхунд',
        ('бернская овчарка', 'зенненхунд'),
    ),
    'mountain/swiss': ('большой швейцарский зенненхунд', ()),
    'collie/border': ('бордер-колли', ('бордер колли',)),
    'rough/collie': ('колли', ('длинношёрстный колли',)),
    'australian/shepherd': ('австралийская овчарка', ('аусси',)),
    'malinois': ('малинуа', ('бельгийская овчарка малинуа',)),
    'groenendael': ('грюнендаль', ()),
    'tervuren': ('тервюрен', ()),
    'cattledog/australian': ('австралийский хилер', ('австралийская пастушья',)),
    'sheepdog/shetland': ('шелти', ('шетландская овчарка',)),
    'sheepdog/english': ('бобтейл', ('староанглийская овчарка',)),
    'schnauzer/miniature': ('цвергшнауцер', ('миниатюрный шнауцер',)),
    'schnauzer/giant': ('ризеншнауцер', ('гигантский шнауцер',)),
    'pinscher/miniature': ('цвергпинчер', ('миниатюрный пинчер',)),
    'dane/great': ('немецкий дог', ('дог', 'большой дог')),
    'bullterrier/staffordshire': (
        'стаффордширский бультерьер',
        ('стаффорд', 'стафф'),
    ),
    'terrier/russell': ('джек-рассел-терьер', ('джек рассел',)),
    'terrier/westhighland': (
        'вест-хайленд-уайт-терьер',
        ('вести', 'вест хайленд'),
    ),
    'terrier/scottish': ('скотч-терьер', ('шотландский терьер',)),
    'terrier/border': ('бордер-терьер', ()),
    'terrier/fox': ('фокстерьер', ()),
    'terrier/boston': ('бостон-терьер', ('бостон терьер',)),
    'springer/english': ('английский спрингер-спаниель', ('спрингер',)),
    'setter/irish': ('ирландский сеттер', ()),
    'setter/english': ('английский сеттер', ()),
    'setter/gordon': ('шотландский сеттер', ('гордон',)),
    'pointer/german': ('немецкий курцхаар', ('курцхаар',)),
    'pointer/germanlonghair': ('немецкий лангхаар', ('лангхаар',)),
    'hound/basset': ('бассет-хаунд', ('бассет',)),
    'hound/blood': ('бладхаунд', ()),
    'hound/afghan': ('афганская борзая', ('афган',)),
    'labradoodle': ('лабрадудль', ()),
    'cockapoo': ('кокапу', ()),
    'cavapoo': ('кавапу', ()),
    'puggle': ('пагль', ()),
    'frise/bichon': ('бишон фризе', ('бишон',)),
    'cotondetulear': ('котон-де-тулеар', ()),
    'lhasa': ('лхаса апсо', ('лхаса',)),
    'poodle/toy': ('той-пудель', ('той пудель',)),
    'poodle/miniature': ('миниатюрный пудель', ('карликовый пудель',)),
    'poodle/medium': ('средний пудель', ()),
    'poodle/standard': ('королевский пудель', ('стандартный пудель',)),
    'airedale': ('эрдельтерьер', ('эрдель',)),
    'saluki': ('салюки', ()),
    'wolfhound/irish': ('ирландский волкодав', ()),
    'deerhound/scottish': ('шотландский дирхаунд', ('дирхаунд',)),
    'kelpie': ('австралийский келпи', ('келпи',)),
    'keeshond': ('кеесхонд', ()),
    'eskimo': ('американская эскимосская собака', ()),
    'finnish/lapphund': ('финский лаппхунд', ()),
    'briard': ('бриар', ()),
    'appenzeller': ('аппенцеллер', ()),
    'entlebucher': ('энтлебухер', ()),
}


@dataclass(frozen=True)
class BreedMatch:
    """Точное соответствие запроса записи каталога dog.ceo.

    Attributes:
        api_path: Путь породы в API, например ``retriever/golden``.
        url_token: Фрагмент адреса снимка, например ``retriever-golden``.
        title: Русское или читаемое название для ответа.
    """

    api_path: str
    url_token: str
    title: str


_breed_matches: dict[str, BreedMatch] | None = None
_breed_aliases: dict[str, str] | None = None
_breed_catalog_lock = threading.Lock()

UNSUPPORTED_CONTENT_TYPES = (
    'audio',
    'document',
    'photo',
    'sticker',
    'video',
    'voice',
    'location',
    'contact',
)


class AssistantError(Exception):
    """Ошибка помощника с кодом, текстом и подробностями без секретов.

    Attributes:
        code: Машинный код ошибки.
        message: Текст для журнала и, где уместно, для модели.
        details: Дополнительные поля.
    """

    def __init__(
        self,
        code: str,
        message: str,
        details: Mapping[str, object] | None = None,
    ) -> None:
        """Сохраняет код и подробности.

        Args:
            code: Машинный код.
            message: Текст исключения.
            details: Дополнительные поля. ``None`` становится пустым словарём.
        """
        self.code = code
        self.message = message
        self.details = dict(details or {})
        super().__init__(message)


def read_env(name: str) -> str | None:
    """Читает непустую переменную окружения.

    Args:
        name: Имя переменной.

    Returns:
        Обрезанное значение или ``None``, если переменной нет или она пустая.
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
        То же значение, если оно не пустое.

    Raises:
        AssistantError: Если значение отсутствует.
    """
    if value is None or value.strip() == '':
        raise AssistantError(
            code='config_error',
            message=f'Не задана переменная {name}.',
            details={'env': name},
        )
    return value


def read_positive_int(name: str, default: int) -> int:
    """Читает положительное целое из окружения.

    Args:
        name: Имя переменной.
        default: Значение, если переменная не задана.

    Returns:
        Целое больше нуля.

    Raises:
        AssistantError: Если строка не является положительным целым.
    """
    raw = read_env(name)
    if raw is None:
        return default
    if not raw.isdecimal():
        raise AssistantError(
            code='config_error',
            message=f'{name} должно быть положительным целым.',
            details={'env': name},
        )
    value = int(raw)
    if value <= 0:
        raise AssistantError(
            code='config_error',
            message=f'{name} должно быть положительным целым.',
            details={'env': name},
        )
    return value


def require_model_dimension(model: str, dimension: int) -> None:
    """Сверяет размерность индекса с известной моделью эмбеддингов.

    Args:
        model: Имя модели Sentence Transformers.
        dimension: Размерность, с которой создаётся индекс.

    Raises:
        AssistantError: Если для этой модели размерность другая.
    """
    expected = MODEL_DIMENSIONS.get(model)
    if expected is None or expected == dimension:
        return
    raise AssistantError(
        code='config_error',
        message=(
            f'Модель {model} строит векторы размера {expected}, '
            f'а {HAYSTACK_EMBEDDING_DIMENSION_ENV} равен {dimension}.'
        ),
        details={'model': model, 'dimension': dimension},
    )


def require_index_name(name: str) -> str:
    """Проверяет имя индекса по правилам Pinecone.

    Args:
        name: Имя индекса.

    Returns:
        То же имя, если оно допустимо.

    Raises:
        AssistantError: Если имя не подходит для Pinecone.
    """
    if INDEX_NAME_PATTERN.fullmatch(name) is None:
        raise AssistantError(
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
        То же имя, если оно поддерживается.

    Raises:
        AssistantError: Если облако не из списка Pinecone.
    """
    if name not in ALLOWED_CLOUDS:
        raise AssistantError(
            code='config_error',
            message=f'{PINECONE_CLOUD_ENV} должно быть aws, gcp или azure.',
            details={'cloud': name},
        )
    return name


def user_memory_filter(user_id: int) -> dict[str, str]:
    """Собирает фильтр Haystack только по записям одного пользователя.

    Префикс ``meta.`` интеграция Pinecone снимает сама: в индексе поле лежит
    как обычный ключ метаданных.

    Args:
        user_id: Числовой идентификатор Telegram.

    Returns:
        Условие равенства для поиска и удаления.
    """
    return {
        'field': 'meta.user_id',
        'operator': '==',
        'value': str(user_id),
    }


def speaker_line(user: telebot.types.User) -> str:
    """Собирает короткую строку о том, кто сейчас пишет.

    Args:
        user: Отправитель сообщения.

    Returns:
        Имя, необязательный username и идентификатор.
    """
    name = user.first_name.strip() if isinstance(user.first_name, str) else ''
    if name == '':
        name = 'без имени'
    parts = [f'имя {name}', f'id {user.id}']
    if isinstance(user.username, str) and user.username.strip() != '':
        parts.append(f'@{user.username.strip()}')
    return ', '.join(parts)


def profile_text(user: telebot.types.User) -> str:
    """Формулирует устойчивый факт профиля для долговременной памяти.

    Args:
        user: Отправитель команды /start.

    Returns:
        Текст, который потом можно найти по смыслу.
    """
    name = user.first_name.strip() if isinstance(user.first_name, str) else ''
    if name == '':
        name = 'без имени'
    parts = [
        f'Пользователя зовут {name}.',
        f'Идентификатор Telegram: {user.id}.',
    ]
    if isinstance(user.username, str) and user.username.strip() != '':
        parts.append(f'Имя пользователя: @{user.username.strip()}.')
    return ' '.join(parts)


def current_time_label() -> str:
    """Возвращает местное время сервера для подсказки модели.

    Returns:
        Дата и время без буквенного названия пояса.
    """
    return datetime.now().astimezone().strftime('%Y-%m-%d %H:%M')


def limit_memory_text(text: str, limit: int = MAX_MEMORY_CHARS) -> str:
    """Обрезает текст памяти до лимита метаданных.

    Args:
        text: Исходная реплика.
        limit: Максимум символов.

    Returns:
        Исходный текст или усечённый текст с пометкой.
    """
    cleaned = text.strip()
    if len(cleaned) <= limit:
        return cleaned
    return f'{cleaned[:limit].rstrip()} …'


def build_memory_document(
    user_id: int,
    user_text: str,
    answer_text: str | None,
) -> Document:
    """Собирает документ одного хода диалога.

    Args:
        user_id: Владелец памяти.
        user_text: Реплика пользователя.
        answer_text: Ответ помощника. ``None``, если ответа не было.

    Returns:
        Документ без эмбеддинга. Вектор добавляет индексация.
    """
    created_at = datetime.now().astimezone().strftime('%Y-%m-%dT%H:%M:%S%z')
    if answer_text is None or answer_text.strip() == '':
        content = f'{ROLE_USER_LABEL}: {user_text.strip()}'
        role = 'user'
    else:
        content = (
            f'{ROLE_USER_LABEL}: {user_text.strip()}\n'
            f'{ROLE_ASSISTANT_LABEL}: {answer_text.strip()}'
        )
        role = 'turn'
    return Document(
        id=f'{user_id}-{uuid4().hex}',
        content=limit_memory_text(content),
        meta={
            'user_id': str(user_id),
            'role': role,
            'created_at': created_at,
        },
    )


def build_profile_document(user: telebot.types.User) -> Document:
    """Собирает документ профиля.

    Args:
        user: Отправитель /start.

    Returns:
        Документ с именем и идентификатором.
    """
    created_at = datetime.now().astimezone().strftime('%Y-%m-%dT%H:%M:%S%z')
    return Document(
        id=f'{user.id}-profile-{uuid4().hex}',
        content=profile_text(user),
        meta={
            'user_id': str(user.id),
            'role': 'profile',
            'created_at': created_at,
        },
    )


def select_memory(
    documents: Sequence[Document],
    minimum_score: float,
) -> list[Document]:
    """Оставляет непустые фрагменты не ниже порога косинусного сходства.

    Args:
        documents: Результат поиска, ближайшие первыми.
        minimum_score: Минимальный score. Документ без score сохраняется.

    Returns:
        Новый список подходящих документов.
    """
    selected: list[Document] = []
    for document in documents:
        if not isinstance(document.content, str) or document.content.strip() == '':
            continue
        if document.score is not None and document.score < minimum_score:
            continue
        selected.append(document)
    return selected


def documents_from_retrieval(result: Mapping[str, object]) -> list[Document]:
    """Достаёт документы из ответа конвейера поиска.

    Args:
        result: Словарь, который вернул ``Pipeline.run``.

    Returns:
        Список документов или пустой список, если формы ответа нет.
    """
    retriever = result.get('retriever')
    if not isinstance(retriever, Mapping):
        return []
    documents = retriever.get('documents')
    if not isinstance(documents, list):
        return []
    return [item for item in documents if isinstance(item, Document)]


def dialogue_tail(messages: Sequence[ChatMessage], limit: int) -> list[ChatMessage]:
    """Оставляет последние текстовые реплики пользователя и помощника.

    Args:
        messages: Сообщения сеанса, включая служебные вызовы инструментов.
        limit: Максимум сообщений.

    Returns:
        Новые ``ChatMessage`` без вызовов инструментов.
    """
    rebuilt: list[ChatMessage] = []
    for message in messages:
        if message.tool_calls:
            continue
        text = message.text
        if not isinstance(text, str) or text.strip() == '':
            continue
        cleaned = text.strip()
        if message.role == ChatRole.USER:
            rebuilt.append(ChatMessage.from_user(cleaned))
        elif message.role == ChatRole.ASSISTANT:
            rebuilt.append(ChatMessage.from_assistant(cleaned))
    if len(rebuilt) <= limit:
        return rebuilt
    return rebuilt[-limit:]


def answer_from_agent(result: Mapping[str, object]) -> str:
    """Берёт последний текстовый ответ агента без вызова инструмента.

    Args:
        result: Ответ ``Agent.run``.

    Returns:
        Текст ответа или запасная фраза.
    """
    messages = result.get('messages')
    if not isinstance(messages, list):
        return EMPTY_ANSWER_TEXT
    for item in reversed(messages):
        if not isinstance(item, ChatMessage):
            continue
        if item.role != ChatRole.ASSISTANT or item.tool_calls:
            continue
        text = item.text
        if isinstance(text, str) and text.strip() != '':
            return text.strip()
    return EMPTY_ANSWER_TEXT


def messages_from_agent(result: Mapping[str, object]) -> list[ChatMessage]:
    """Достаёт все сообщения прогона агента.

    Args:
        result: Ответ ``Agent.run``.

    Returns:
        Только объекты ``ChatMessage``.
    """
    messages = result.get('messages')
    if not isinstance(messages, list):
        return []
    return [item for item in messages if isinstance(item, ChatMessage)]


def tool_result_text(result: object) -> str:
    """Превращает результат инструмента в строку, если это текст.

    Args:
        result: Поле ``ToolCallResult.result``.

    Returns:
        Строка или пустая строка для других типов.
    """
    if isinstance(result, str):
        return result
    return ''


def is_allowed_image_url(url: str) -> bool:
    """Проверяет, что ссылка ведёт на разрешённый хост фотографий.

    Args:
        url: Адрес из API.

    Returns:
        ``True``, если схема https и хост в списке.
    """
    parsed = urllib.parse.urlparse(url)
    return parsed.scheme == 'https' and parsed.hostname in IMAGE_HOSTS


def collect_dog_photos(messages: Sequence[ChatMessage]) -> list[tuple[str, str]]:
    """Находит фото и подпись породы, которые инструмент вернул агенту.

    Args:
        messages: Полный прогон, включая сообщения роли tool.

    Returns:
        Пары из разрешённого адреса и подписи. Подпись — название породы.
    """
    photos: list[tuple[str, str]] = []
    seen: set[str] = set()
    image_prefix = f'{DOG_IMAGE_LINE_PREFIX} '
    breed_prefix = f'{DOG_BREED_LINE_PREFIX} '
    for message in messages:
        for item in message.tool_call_results:
            url = ''
            breed = ''
            for line in tool_result_text(item.result).splitlines():
                if line.startswith(image_prefix):
                    url = line[len(image_prefix):].strip()
                elif line.startswith(breed_prefix):
                    breed = line[len(breed_prefix):].strip()
            if not is_allowed_image_url(url) or url in seen:
                continue
            seen.add(url)
            caption = breed if breed != '' else PHOTO_CAPTION
            photos.append((url, caption[:TELEGRAM_CAPTION_LIMIT]))
    return photos


def strip_service_lines(text: str) -> str:
    """Убирает служебные строки с адресом и названием породы.

    Args:
        text: Ответ модели.

    Returns:
        Текст без служебных строк. Если ничего не осталось — запасная фраза.
    """
    kept = [
        line
        for line in text.splitlines()
        if not line.strip().startswith(
            (DOG_IMAGE_LINE_PREFIX, DOG_BREED_LINE_PREFIX),
        )
    ]
    cleaned = '\n'.join(kept).strip()
    if cleaned == '':
        return EMPTY_ANSWER_TEXT
    return cleaned


def split_telegram_text(text: str, limit: int = TELEGRAM_MESSAGE_LIMIT) -> list[str]:
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


def parse_dog_fact(payload: Mapping[str, object]) -> str:
    """Достаёт текст факта из ответа dogapi.dog.

    Args:
        payload: Разобранный JSON.

    Returns:
        Текст факта.

    Raises:
        AssistantError: Если в ответе нет текста факта.
    """
    data = payload.get('data')
    if not isinstance(data, list) or len(data) == 0:
        raise AssistantError(
            code='dog_fact_error',
            message='Сервис фактов не вернул запись.',
        )
    first = data[0]
    if not isinstance(first, Mapping):
        raise AssistantError(
            code='dog_fact_error',
            message='Сервис фактов вернул неожиданный формат.',
        )
    attributes = first.get('attributes')
    if not isinstance(attributes, Mapping):
        raise AssistantError(
            code='dog_fact_error',
            message='Сервис фактов вернул запись без текста.',
        )
    body = attributes.get('body')
    if not isinstance(body, str) or body.strip() == '':
        raise AssistantError(
            code='dog_fact_error',
            message='Сервис фактов вернул пустой текст.',
        )
    return body.strip()


def parse_dog_image_url(payload: Mapping[str, object]) -> str:
    """Достаёт адрес случайного фото из ответа dog.ceo.

    Args:
        payload: Разобранный JSON.

    Returns:
        Адрес https на разрешённом хосте.

    Raises:
        AssistantError: Если адреса нет или хост чужой.
    """
    status = payload.get('status')
    message = payload.get('message')
    if status != 'success' or not isinstance(message, str):
        raise AssistantError(
            code='dog_image_error',
            message='Сервис фотографий не вернул снимок.',
        )
    url = message.strip()
    if not is_allowed_image_url(url):
        raise AssistantError(
            code='dog_image_error',
            message='Адрес фотографии не из списка разрешённых.',
        )
    return url


def mime_from_header(header: str | None) -> str | None:
    """Оставляет MIME изображения из заголовка Content-Type.

    Args:
        header: Значение заголовка или ``None``.

    Returns:
        MIME или ``None``, если это не изображение.
    """
    if header is None:
        return None
    mime = header.split(';', 1)[0].strip().lower()
    if mime == 'image/jpg':
        return 'image/jpeg'
    if mime.startswith('image/'):
        return mime
    return None


def guess_image_mime(payload: bytes, header: str | None) -> str:
    """Определяет MIME снимка по заголовку, а затем по байтам.

    Args:
        payload: Тело ответа.
        header: Заголовок Content-Type.

    Returns:
        MIME из списка поддерживаемых.

    Raises:
        AssistantError: Если файл не похож на поддерживаемое изображение.
    """
    mime = mime_from_header(header)
    if mime is None:
        kind = filetype.guess(payload)
        if kind is not None and kind.mime.startswith('image/'):
            mime = 'image/jpeg' if kind.mime == 'image/jpg' else kind.mime
    if mime not in SUPPORTED_IMAGE_MIME_TYPES:
        raise AssistantError(
            code='dog_image_error',
            message='Случайный файл не оказался поддерживаемым изображением.',
        )
    return mime


def request_bytes(
    url: str,
    accept: str,
    max_bytes: int,
    allowed_hosts: frozenset[str],
) -> tuple[bytes, str | None]:
    """Скачивает тело ответа с повтором временных сбоев.

    Args:
        url: Адрес запроса.
        accept: Заголовок Accept.
        max_bytes: Верхняя граница размера.
        allowed_hosts: Допустимые хосты исходного и конечного адреса.

    Returns:
        Байты и заголовок Content-Type, если сервер его прислал.

    Raises:
        AssistantError: Если запрос не удался или ответ слишком большой.
    """
    last_error: AssistantError | None = None
    for attempt in range(HTTP_RETRY_ATTEMPTS):
        try:
            return read_bytes_once(url, accept, max_bytes, allowed_hosts)
        except AssistantError as error:
            last_error = error
            if error.code != 'transient_http_error':
                raise
            if attempt == HTTP_RETRY_ATTEMPTS - 1:
                raise
            time.sleep(HTTP_RETRY_BASE_SECONDS * (2 ** attempt))
    raise last_error or AssistantError(
        code='http_error',
        message='Не удалось выполнить внешний запрос.',
    )


def read_bytes_once(
    url: str,
    accept: str,
    max_bytes: int,
    allowed_hosts: frozenset[str],
) -> tuple[bytes, str | None]:
    """Выполняет один HTTP GET.

    Args:
        url: Адрес запроса.
        accept: Заголовок Accept.
        max_bytes: Верхняя граница размера.
        allowed_hosts: Допустимые хосты.

    Returns:
        Байты и Content-Type.

    Raises:
        AssistantError: При сетевой ошибке, чужом хосте или слишком большом теле.
    """
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme != 'https' or parsed.hostname not in allowed_hosts:
        raise AssistantError(
            code='unsafe_url',
            message='Внешний адрес не из списка разрешённых.',
        )
    request = urllib.request.Request(
        url,
        headers={'User-Agent': USER_AGENT, 'Accept': accept},
    )
    try:
        with urllib.request.urlopen(request, timeout=HTTP_TIMEOUT_SECONDS) as response:
            final = urllib.parse.urlparse(response.geturl())
            if final.scheme != 'https' or final.hostname not in allowed_hosts:
                raise AssistantError(
                    code='unsafe_url',
                    message='Внешний сервис перенаправил запрос на чужой адрес.',
                )
            content_type = response.headers.get('Content-Type')
            chunks: list[bytes] = []
            total = 0
            while True:
                block = response.read(64 * 1024)
                if not block:
                    break
                total += len(block)
                if total > max_bytes:
                    raise AssistantError(
                        code='payload_too_large',
                        message='Ответ внешнего сервиса слишком большой.',
                    )
                chunks.append(block)
    except AssistantError:
        raise
    except urllib.error.HTTPError as error:
        transient = error.code in TRANSIENT_STATUS_CODES
        raise AssistantError(
            code='transient_http_error' if transient else 'http_error',
            message=(
                'Внешний сервис временно недоступен.'
                if transient
                else 'Внешний сервис отклонил запрос.'
            ),
            details={'status': error.code},
        ) from error
    except (urllib.error.URLError, TimeoutError, ConnectionError) as error:
        raise AssistantError(
            code='transient_http_error',
            message='Не удалось связаться с внешним сервисом.',
            details={'error_type': type(error).__name__},
        ) from error
    header = content_type if isinstance(content_type, str) else None
    return b''.join(chunks), header


def request_json(url: str, allowed_hosts: frozenset[str]) -> dict[str, object]:
    """Скачивает JSON-объект.

    Args:
        url: Адрес запроса.
        allowed_hosts: Допустимые хосты.

    Returns:
        Словарь верхнего уровня.

    Raises:
        AssistantError: Если тело не является JSON-объектом.
    """
    payload, _header = request_bytes(
        url,
        'application/json',
        MAX_JSON_BYTES,
        allowed_hosts,
    )
    try:
        parsed = json.loads(payload.decode('utf-8'))
    except (UnicodeError, json.JSONDecodeError) as error:
        raise AssistantError(
            code='invalid_json',
            message='Внешний сервис вернул неразборчивый ответ.',
        ) from error
    if not isinstance(parsed, dict):
        raise AssistantError(
            code='invalid_json',
            message='Внешний сервис вернул не объект.',
        )
    return parsed


def fetch_dog_fact() -> str:
    """Запрашивает один случайный факт о собаках.

    Returns:
        Текст факта на языке источника.

    Raises:
        AssistantError: Если сервис не ответил или формат другой.
    """
    payload = request_json(DOG_FACT_API_URL, DOG_API_HOSTS)
    return parse_dog_fact(payload)


def normalize_breed_query(text: str) -> str:
    """Приводит название породы к словам в нижнем регистре.

    Args:
        text: Строка из запроса пользователя или каталога.

    Returns:
        Слова через пробел. Буква «ё» заменяется на «е».
    """
    lowered = text.lower().replace('ё', 'е')
    cleaned = re.sub(r'[^a-zа-я0-9]+', ' ', lowered)
    return ' '.join(cleaned.split())


def token_stem(token: str) -> str:
    """Снимает одно падежное окончание, если основа остаётся узнаваемой.

    Args:
        token: Одно слово уже нормализованного запроса.

    Returns:
        Основа или исходное слово, если окончание снимать нельзя.
    """
    for suffix in BREED_INFLECTION_SUFFIXES:
        if token.endswith(suffix) and len(token) - len(suffix) >= 3:
            return token[:-len(suffix)]
    return token


def stems_of(text: str) -> tuple[str, ...]:
    """Делит нормализованную строку на основы слов.

    Args:
        text: Результат ``normalize_breed_query``.

    Returns:
        Основы в исходном порядке.
    """
    return tuple(token_stem(part) for part in text.split() if part != '')


def leftover_stems(query: str, alias: str) -> tuple[str, ...] | None:
    """Возвращает основы запроса, которые не вошли в имя породы.

    Args:
        query: Нормализованный запрос.
        alias: Нормализованное имя из каталога.

    Returns:
        Основы вокруг совпадения. ``None``, если имя в запрос не входит.
    """
    query_stems = stems_of(query)
    alias_stems = stems_of(alias)
    if len(alias_stems) == 0 or len(alias_stems) > len(query_stems):
        return None
    width = len(alias_stems)
    last_start = len(query_stems) - width
    for start in range(last_start + 1):
        if query_stems[start:start + width] != alias_stems:
            continue
        return query_stems[:start] + query_stems[start + width:]
    return None


def filler_stems() -> frozenset[str]:
    """Собирает основы слов, которые не являются названием породы.

    Returns:
        Основы вроде «покажи» и «фото».
    """
    stems: set[str] = set()
    for word in REQUEST_FILLER_WORDS:
        normalized = normalize_breed_query(word)
        if normalized == '':
            continue
        stems.add(token_stem(normalized))
    return frozenset(stems)


def default_breed_title(path: str) -> str:
    """Собирает читаемое английское имя из пути каталога.

    Args:
        path: Путь вида ``retriever/golden`` или ``husky``.

    Returns:
        Имя, где подпорода стоит перед породой.
    """
    parts = path.split('/')
    if len(parts) == 1:
        return parts[0]
    return f'{parts[1]} {parts[0]}'


def english_breed_labels(path: str) -> tuple[str, ...]:
    """Возвращает английские имена, по которым путь ищется точно.

    Args:
        path: Путь каталога.

    Returns:
        Полные имена. Короткое имя родителя с подпородами не добавляется,
        чтобы «ретривер» не выбирал случайный подвид.
    """
    parts = path.split('/')
    if len(parts) == 1:
        return (parts[0],)
    breed, sub = parts
    return (f'{sub} {breed}', f'{breed} {sub}')


def add_breed_alias(aliases: dict[str, str], label: str, path: str) -> None:
    """Кладёт имя в указатель, не затирая уже занятое другим путём.

    Args:
        aliases: Нормализованное имя к пути каталога.
        label: Человеческое имя.
        path: Путь dog.ceo.
    """
    key = normalize_breed_query(label)
    if key == '':
        return
    current = aliases.get(key)
    if current is not None and current != path:
        return
    aliases[key] = path


def build_breed_index(
    payload: Mapping[str, object],
) -> tuple[dict[str, BreedMatch], dict[str, str]]:
    """Строит каталог пород и имена, по которым их можно найти.

    Args:
        payload: Ответ ``https://dog.ceo/api/breeds/list/all``.

    Returns:
        Словарь совпадений по пути и словарь имён.

    Raises:
        AssistantError: Если в ответе нет списка пород.
    """
    message = payload.get('message')
    if not isinstance(message, Mapping):
        raise AssistantError(
            code='breed_catalog_error',
            message='Каталог пород не прочитан.',
        )
    matches: dict[str, BreedMatch] = {}
    for breed, subs in message.items():
        if not isinstance(breed, str) or breed == '' or breed == 'mix':
            continue
        sub_names = [
            item
            for item in subs
            if isinstance(item, str) and item != ''
        ] if isinstance(subs, list) else []
        paths = (
            [breed]
            if len(sub_names) == 0
            else [f'{breed}/{name}' for name in sub_names]
        )
        for path in paths:
            labels = BREED_LABELS.get(path)
            title = labels[0] if labels is not None else default_breed_title(path)
            matches[path] = BreedMatch(
                api_path=path,
                url_token=path.replace('/', '-'),
                title=title,
            )
    aliases: dict[str, str] = {}
    for path, match in matches.items():
        labels = BREED_LABELS.get(path)
        if labels is not None:
            add_breed_alias(aliases, labels[0], path)
            for extra in labels[1]:
                add_breed_alias(aliases, extra, path)
        for label in english_breed_labels(path):
            add_breed_alias(aliases, label, path)
        add_breed_alias(aliases, match.title, path)
    return matches, aliases


def load_breed_index() -> tuple[dict[str, BreedMatch], dict[str, str]]:
    """Один раз читает каталог пород и запоминает его до перезапуска.

    Returns:
        Совпадения и имена.

    Raises:
        AssistantError: Если каталог не скачался.
    """
    global _breed_matches, _breed_aliases
    with _breed_catalog_lock:
        if _breed_matches is not None and _breed_aliases is not None:
            return _breed_matches, _breed_aliases
        payload = request_json(DOG_BREEDS_API_URL, DOG_API_HOSTS)
        matches, aliases = build_breed_index(payload)
        _breed_matches = matches
        _breed_aliases = aliases
        return matches, aliases


def shown_breed_query(query: str) -> str:
    """Оставляет короткую одну строку из запроса для текста ошибки.

    Args:
        query: То, что передал агент.

    Returns:
        Не длиннее 80 символов.
    """
    compact = ' '.join(query.split())
    if len(compact) <= 80:
        return compact
    return compact[:80].rstrip()


def find_breed(query: str) -> BreedMatch:
    """Находит одну породу каталога по запросу.

    Берётся самое длинное имя, чьи основы совпали с запросом. Одинаково
    длинные имена разных пород считаются неоднозначными.

    Args:
        query: Название породы словами пользователя.

    Returns:
        Совпадение с путём API и названием для ответа.

    Raises:
        AssistantError: Если порода не найдена или запрос неоднозначен.
    """
    normalized = normalize_breed_query(query)
    shown = shown_breed_query(query)
    if normalized == '':
        raise AssistantError(
            code='breed_not_found',
            message='Название породы пустое.',
        )
    matches, aliases = load_breed_index()
    fillers = filler_stems()
    best_length = -1
    best_paths: set[str] = set()
    for alias, path in aliases.items():
        leftover = leftover_stems(normalized, alias)
        if leftover is None:
            continue
        if any(stem not in fillers for stem in leftover):
            continue
        if len(alias) > best_length:
            best_length = len(alias)
            best_paths = {path}
        elif len(alias) == best_length:
            best_paths.add(path)
    if len(best_paths) != 1:
        if len(best_paths) > 1:
            message = (
                f'По запросу «{shown}» подходит несколько пород. '
                'Назовите породу точнее.'
            )
        else:
            message = (
                f'Порода «{shown}» не найдена. Назовите её точнее, '
                'например «лабрадор» или «золотистый ретривер».'
            )
        raise AssistantError(code='breed_not_found', message=message)
    path = next(iter(best_paths))
    match = matches.get(path)
    if match is None:
        raise AssistantError(
            code='breed_not_found',
            message=f'Порода «{shown}» не найдена в каталоге фотографий.',
        )
    return match


def breed_image_api_url(api_path: str) -> str:
    """Собирает адрес случайного снимка конкретной породы.

    Args:
        api_path: Путь вида ``retriever/golden``.

    Returns:
        URL dog.ceo.
    """
    return f'{DOG_BREED_IMAGE_API}/{api_path}/images/random'


def image_matches_breed(url: str, url_token: str) -> bool:
    """Проверяет, что адрес снимка принадлежит запрошенной породе.

    Args:
        url: Адрес файла изображения.
        url_token: Ожидаемый фрагмент пути.

    Returns:
        ``True``, если в пути есть ``/breeds/<токен>/``.
    """
    path = urllib.parse.urlparse(url).path.lower()
    return f'/breeds/{url_token.lower()}/' in path


def vision_prompt_for_breed(title: str) -> str:
    """Просит модель рассказать именно о подтверждённой породе.

    Args:
        title: Название из каталога, не сырой ввод пользователя.

    Returns:
        Текст запроса к модели со снимком.
    """
    return (
        f'На фото собака породы «{title}». Порода уже подтверждена каталогом, '
        'не подменяй её другой и не выдумывай кличку.\n'
        'Ответь по-русски, без приветствия, тремя короткими абзацами:\n'
        f'1) что видно на фото и почему это {title};\n'
        f'2) как появилась порода {title}: от каких собак её выводили и зачем;\n'
        f'3) характер, примерный размер и для какой работы она создавалась.'
    )


def fetch_breed_image_url(match: BreedMatch) -> str:
    """Скачивает адрес снимка и проверяет, что это запрошенная порода.

    Args:
        match: Найденная порода.

    Returns:
        Адрес изображения.

    Raises:
        AssistantError: Если сервис не вернул снимок этой породы.
    """
    api_url = breed_image_api_url(match.api_path)
    last_url = ''
    for _attempt in range(2):
        last_url = parse_dog_image_url(request_json(api_url, DOG_API_HOSTS))
        if image_matches_breed(last_url, match.url_token):
            return last_url
    raise AssistantError(
        code='dog_image_error',
        message=f'Снимок не совпал с породой «{match.title}».',
        details={'url_token': match.url_token},
    )


def describe_dog_photo(vision: OpenAIChatGenerator, breed_query: str) -> str:
    """Скачивает фото породы, загружает его в модель и возвращает справку.

    Пустой запрос оставляет случайное фото. Непустой ищется в каталоге
    dog.ceo и не подменяется другой породой. Адрес и название остаются
    служебными строками, чтобы бот отправил снимок с подписью.

    Args:
        vision: Отдельный генератор с поддержкой изображений.
        breed_query: Название породы или пустая строка.

    Returns:
        Служебные строки и текст о породе.

    Raises:
        AssistantError: Если фото или описание получить не удалось.
    """
    match: BreedMatch | None = None
    if breed_query.strip() != '':
        match = find_breed(breed_query)
        image_url = fetch_breed_image_url(match)
        prompt = vision_prompt_for_breed(match.title)
    else:
        image_url = parse_dog_image_url(
            request_json(DOG_IMAGE_API_URL, DOG_API_HOSTS),
        )
        prompt = VISION_PROMPT
    payload, header = request_bytes(
        image_url,
        'image/*',
        MAX_IMAGE_BYTES,
        IMAGE_HOSTS,
    )
    mime_type = guess_image_mime(payload, header)
    image = ImageContent(
        base64_image=base64.standard_b64encode(payload).decode('ascii'),
        mime_type=mime_type,
        detail='high',
    )
    message = ChatMessage.from_user(content_parts=[prompt, image])
    try:
        result = vision.run(messages=[message])
    except Exception as error:
        raise AssistantError(
            code='vision_error',
            message='Не удалось загрузить фото в модель и получить описание породы.',
            details={'error_type': type(error).__name__},
        ) from error
    replies = result.get('replies')
    description = ''
    if isinstance(replies, list) and len(replies) > 0:
        first = replies[0]
        if isinstance(first, ChatMessage) and isinstance(first.text, str):
            description = first.text.strip()
    if description == '':
        raise AssistantError(
            code='vision_empty',
            message='Модель не описала фотографию собаки.',
        )
    breed_line = ''
    if match is not None:
        breed_line = f'{DOG_BREED_LINE_PREFIX} {match.title}\n'
    return f'{DOG_IMAGE_LINE_PREFIX} {image_url}\n{breed_line}{description}'


def build_dog_tools(vision: OpenAIChatGenerator) -> list[Tool]:
    """Собирает инструменты агента вокруг внешних API о собаках.

    Args:
        vision: Генератор, которому передаются байты фотографии.

    Returns:
        Инструмент факта и инструмент описания фото.
    """

    @tool
    def get_random_dog_fact() -> str:
        """Возвращает один случайный факт о собаках из бесплатного API.

        Вызывай, когда пользователь просит факт или что-нибудь интересное
        о собаках и не просит фотографию.
        """
        try:
            fact = fetch_dog_fact()
        except AssistantError as error:
            logger.error('Факт о собаках: code=%s', error.code)
            return error.message
        except Exception as error:
            logger.error('Факт о собаках: error_type=%s', type(error).__name__)
            return 'Не удалось получить факт о собаках.'
        return f'Случайный факт о собаках: {fact}'

    @tool
    def describe_random_dog(
        breed: Annotated[
            str,
            'Порода из просьбы пользователя, дословно. '
            'Пустая строка, только если породу не называли.',
        ],
    ) -> str:
        """Находит фото названной породы и возвращает справку о ней.

        Снимок скачивается из каталога этой породы и загружается в модель.
        В ответе есть подтверждённое название, что видно на фото, как порода
        появилась, характер и для какой работы её выводили. Если порода не
        названа, снимок случайный. Вызывай один раз на одну просьбу.
        """
        try:
            described = describe_dog_photo(vision, breed)
        except AssistantError as error:
            logger.error('Фото собаки: code=%s', error.code)
            return error.message
        except Exception as error:
            logger.error('Фото собаки: error_type=%s', type(error).__name__)
            return 'Не удалось получить и описать фотографию собаки.'
        return described

    return [get_random_dog_fact, describe_random_dog]


def build_chat_generator(
    model: str,
    base_url: str | None,
    temperature: float,
    max_tokens: int,
) -> OpenAIChatGenerator:
    """Создаёт генератор OpenAI с общими настройками таймаута.

    Args:
        model: Имя модели.
        base_url: Необязательный базовый адрес API.
        temperature: Температура выборки.
        max_tokens: Верхняя граница длины ответа.

    Returns:
        Готовый ``OpenAIChatGenerator``. Ключ берётся из окружения.
    """
    generation_kwargs = {
        'temperature': temperature,
        'max_completion_tokens': max_tokens,
    }
    if base_url is None:
        return OpenAIChatGenerator(
            model=model,
            generation_kwargs=generation_kwargs,
            timeout=OPENAI_TIMEOUT_SECONDS,
            max_retries=OPENAI_MAX_RETRIES,
        )
    return OpenAIChatGenerator(
        model=model,
        api_base_url=base_url,
        generation_kwargs=generation_kwargs,
        timeout=OPENAI_TIMEOUT_SECONDS,
        max_retries=OPENAI_MAX_RETRIES,
    )


def build_indexing_pipeline(
    document_store: PineconeDocumentStore,
    model: str,
) -> Pipeline:
    """Собирает конвейер записи: эмбеддинг документа и писатель Pinecone.

    Args:
        document_store: Хранилище с метрикой cosine.
        model: Модель Sentence Transformers. Та же, что у поиска.

    Returns:
        Конвейер с входами эмбеддера.
    """
    pipeline = Pipeline()
    pipeline.add_component(
        'embedder',
        SentenceTransformersDocumentEmbedder(model=model),
    )
    pipeline.add_component(
        'writer',
        DocumentWriter(
            document_store=document_store,
            policy=DuplicatePolicy.OVERWRITE,
        ),
    )
    pipeline.connect('embedder.documents', 'writer.documents')
    return pipeline


def build_retrieval_pipeline(
    document_store: PineconeDocumentStore,
    model: str,
) -> Pipeline:
    """Собирает конвейер поиска по примеру RAG и Pinecone.

    Эмбеддинг реплики считается той же моделью, что и векторы документов,
    затем ``PineconeEmbeddingRetriever`` сравнивает их косинусной метрикой
    индекса.

    Args:
        document_store: Хранилище контекста.
        model: Модель Sentence Transformers.

    Returns:
        Конвейер с входом ``text_embedder.text``.
    """
    pipeline = Pipeline()
    pipeline.add_component(
        'text_embedder',
        SentenceTransformersTextEmbedder(model=model),
    )
    pipeline.add_component(
        'retriever',
        PineconeEmbeddingRetriever(document_store=document_store, top_k=MEMORY_TOP_K),
    )
    pipeline.connect('text_embedder.embedding', 'retriever.query_embedding')
    return pipeline


def build_agent(
    chat_generator: OpenAIChatGenerator,
    tools: Sequence[Tool],
) -> Agent:
    """Собирает агента, который отвечает текстом и при необходимости зовёт инструменты.

    Args:
        chat_generator: Диалоговая модель.
        tools: Инструменты факта и фото.

    Returns:
        Агент с системной подсказкой-шаблоном.
    """
    return Agent(
        chat_generator=chat_generator,
        tools=list(tools),
        system_prompt=SYSTEM_PROMPT,
        exit_conditions=['text'],
        max_agent_steps=AGENT_STEPS,
        raise_on_tool_invocation_failure=False,
        tool_concurrency_limit=1,
    )


def metric_name(raw: object) -> str:
    """Приводит метрику индекса Pinecone к строке в нижнем регистре.

    Args:
        raw: Поле description.metric.

    Returns:
        Строковое имя метрики.
    """
    value = getattr(raw, 'value', raw)
    return str(value).lower()


def existing_index_names(client: Pinecone) -> set[str]:
    """Собирает имена индексов проекта.

    Args:
        client: Клиент Pinecone.

    Returns:
        Множество имён. Пустое, если список индексов пуст.
    """
    listed = client.list_indexes()
    indexes = getattr(listed, 'indexes', listed)
    names: set[str] = set()
    if indexes is None:
        return names
    for item in indexes:
        name = getattr(item, 'name', None)
        if isinstance(name, str):
            names.add(name)
        elif isinstance(item, str):
            names.add(item)
    return names


def ensure_index_compatible(index_name: str, dimension: int) -> None:
    """Проверяет уже существующий индекс: cosine и нужная размерность.

    Если индекса ещё нет, ``PineconeDocumentStore`` создаст его сам.

    Args:
        index_name: Имя индекса.
        dimension: Ожидаемая размерность эмбеддинга.

    Raises:
        AssistantError: Если индекс есть, но метрика или размерность другие.
    """
    api_key = require_config(read_env(PINECONE_API_KEY_ENV), PINECONE_API_KEY_ENV)
    try:
        client = Pinecone(api_key=api_key)
        names = existing_index_names(client)
    except Exception as error:
        raise AssistantError(
            code='pinecone_error',
            message='Не удалось получить список индексов Pinecone.',
            details={'error_type': type(error).__name__},
        ) from error
    if index_name not in names:
        return
    try:
        description = client.describe_index(index_name)
    except Exception as error:
        raise AssistantError(
            code='pinecone_error',
            message='Не удалось прочитать параметры индекса Pinecone.',
            details={'error_type': type(error).__name__},
        ) from error
    actual_dimension = getattr(description, 'dimension', None)
    if actual_dimension != dimension or not metric_name(
        getattr(description, 'metric', ''),
    ).endswith('cosine'):
        raise AssistantError(
            code='index_mismatch',
            message=(
                f'Индекс {index_name} уже существует и не совпадает '
                f'с cosine и размерностью {dimension}. Задайте новое имя '
                f'в {HAYSTACK_PINECONE_INDEX_ENV}.'
            ),
            details={
                'index': index_name,
                'dimension': actual_dimension,
            },
        )


class TelegramAssistant:
    """Принимает сообщения Telegram и отвечает агентом с памятью в Pinecone."""

    def __init__(
        self,
        token: str,
        document_store: PineconeDocumentStore,
        indexing: Pipeline,
        retrieval: Pipeline,
        agent: Agent,
        embedding_model: str,
    ) -> None:
        """Связывает бота, конвейеры памяти и агента.

        Args:
            token: Токен Telegram-бота.
            document_store: Хранилище контекста.
            indexing: Конвейер записи эмбеддингов.
            retrieval: Конвейер поиска по косинусу.
            agent: Агент с инструментами.
            embedding_model: Имя модели для журнала прогрева.
        """
        self._store = document_store
        self._indexing = indexing
        self._retrieval = retrieval
        self._agent = agent
        self._embedding_model = embedding_model
        self._bot = telebot.TeleBot(token)
        self._history: dict[int, list[ChatMessage]] = {}
        self._lock = threading.Lock()
        self._register_handlers()

    @classmethod
    def from_env(cls) -> TelegramAssistant:
        """Создаёт помощника из ``.env`` и переменных окружения.

        Returns:
            Готовый, но ещё не прогретый помощник.

        Raises:
            AssistantError: Если настройки пустые или индекс несовместим.
        """
        load_dotenv(ENV_PATH)
        load_dotenv()
        token = require_config(read_env(TELEGRAM_BOT_TOKEN_ENV), TELEGRAM_BOT_TOKEN_ENV)
        require_config(read_env(OPENAI_API_KEY_ENV), OPENAI_API_KEY_ENV)
        require_config(read_env(PINECONE_API_KEY_ENV), PINECONE_API_KEY_ENV)
        base_url = read_env(OPENAI_BASE_URL_ENV)
        chat_model = read_env(OPENAI_CHAT_MODEL_ENV) or DEFAULT_CHAT_MODEL
        index_name = require_index_name(
            read_env(HAYSTACK_PINECONE_INDEX_ENV) or DEFAULT_INDEX_NAME,
        )
        namespace = read_env(HAYSTACK_PINECONE_NAMESPACE_ENV) or DEFAULT_NAMESPACE
        embedding_model = read_env(HAYSTACK_EMBEDDING_MODEL_ENV) or DEFAULT_EMBEDDING_MODEL
        dimension = read_positive_int(
            HAYSTACK_EMBEDDING_DIMENSION_ENV,
            DEFAULT_EMBEDDING_DIMENSION,
        )
        cloud = require_cloud(read_env(PINECONE_CLOUD_ENV) or DEFAULT_CLOUD)
        region = read_env(PINECONE_REGION_ENV) or DEFAULT_REGION
        require_model_dimension(embedding_model, dimension)
        ensure_index_compatible(index_name, dimension)
        try:
            document_store = PineconeDocumentStore(
                index=index_name,
                namespace=namespace,
                dimension=dimension,
                metric='cosine',
                spec={'serverless': {'region': region, 'cloud': cloud}},
                show_progress=False,
            )
        except Exception as error:
            raise AssistantError(
                code='pinecone_error',
                message='Не удалось открыть или создать индекс Pinecone.',
                details={'error_type': type(error).__name__},
            ) from error
        chat_generator = build_chat_generator(
            chat_model,
            base_url,
            CHAT_TEMPERATURE,
            MAX_REPLY_TOKENS,
        )
        vision = build_chat_generator(
            chat_model,
            base_url,
            VISION_TEMPERATURE,
            MAX_VISION_TOKENS,
        )
        return cls(
            token=token,
            document_store=document_store,
            indexing=build_indexing_pipeline(document_store, embedding_model),
            retrieval=build_retrieval_pipeline(document_store, embedding_model),
            agent=build_agent(chat_generator, build_dog_tools(vision)),
            embedding_model=embedding_model,
        )

    def run(self) -> None:
        """Прогревает модели и принимает сообщения, пока процесс не остановлен."""
        os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
        logger.info('Загружаю модель эмбеддингов %s.', self._embedding_model)
        try:
            self._retrieval.warm_up()
            self._indexing.warm_up()
            self._agent.warm_up()
        except Exception as error:
            raise AssistantError(
                code='warmup_error',
                message='Не удалось подготовить модель эмбеддингов или агента.',
                details={'error_type': type(error).__name__},
            ) from error
        logger.info('Телеграм-бот запущен.')
        telegram_api.RETRY_ON_ERROR = True
        telegram_api.SESSION_TIME_TO_LIVE = POLLING_SESSION_SECONDS
        self._bot.infinity_polling(
            skip_pending=True,
            timeout=POLLING_HTTP_TIMEOUT,
            long_polling_timeout=LONG_POLLING_TIMEOUT,
        )

    def _register_handlers(self) -> None:
        """Подключает команды и обработчик обычного текста."""
        self._bot.message_handler(commands=['start'])(self.on_start)
        self._bot.message_handler(commands=['forget'])(self.on_forget)
        self._bot.message_handler(commands=['fact'])(self.on_fact)
        self._bot.message_handler(commands=['dog'])(self.on_dog)
        self._bot.message_handler(content_types=['text'])(self.on_text)
        self._bot.message_handler(
            content_types=list(UNSUPPORTED_CONTENT_TYPES),
        )(self.on_unsupported)

    def on_start(self, message: telebot.types.Message) -> None:
        """Знакомится с пользователем и сохраняет профиль в Pinecone.

        Args:
            message: Команда /start.
        """
        user = message.from_user
        if user is None:
            return
        self._try_index([build_profile_document(user)], user.id)
        self._reply(message, START_TEXT)

    def on_forget(self, message: telebot.types.Message) -> None:
        """Удаляет долговременную память пользователя и текущий сеанс.

        Args:
            message: Команда /forget.
        """
        user = message.from_user
        if user is None:
            return
        try:
            self._delete_user_memory(user.id)
        except AssistantError as error:
            self._log_error('forget', user.id, error)
            self._reply(message, FORGET_ERROR_TEXT)
            return
        with self._lock:
            self._history.pop(user.id, None)
        self._reply(message, FORGET_TEXT)

    def on_fact(self, message: telebot.types.Message) -> None:
        """Просит агента принести случайный факт о собаках.

        Args:
            message: Команда /fact.
        """
        self._answer(message, FACT_REQUEST)

    def on_dog(self, message: telebot.types.Message) -> None:
        """Просит агента показать собаку и описать породу.

        Args:
            message: Команда /dog.
        """
        self._answer(message, DOG_REQUEST)

    def on_unsupported(self, message: telebot.types.Message) -> None:
        """Отвечает, если прислали не текст.

        Args:
            message: Сообщение другого типа.
        """
        self._reply(message, UNSUPPORTED_TEXT)

    def on_text(self, message: telebot.types.Message) -> None:
        """Отвечает на обычную реплику с учётом памяти.

        Args:
            message: Текстовое сообщение.
        """
        text = (message.text or '').strip()
        if text.startswith('/'):
            self._reply(message, UNKNOWN_COMMAND_TEXT)
            return
        if text == '':
            self._reply(message, EMPTY_TEXT)
            return
        self._answer(message, text)

    def _answer(self, message: telebot.types.Message, text: str) -> None:
        """Ищет память, запускает агента и сохраняет ход диалога.

        Args:
            message: Исходное сообщение Telegram.
            text: Реплика, которую нужно передать агенту.
        """
        user = message.from_user
        if user is None:
            return
        stop_typing = threading.Event()
        typing_worker = threading.Thread(
            target=self._keep_typing,
            args=(message.chat.id, stop_typing),
            daemon=True,
        )
        typing_worker.start()
        try:
            with self._lock:
                memory = self._try_recall(user.id, text)
                history = list(self._history.get(user.id, []))
                try:
                    result = self._agent.run(
                        messages=[*history, ChatMessage.from_user(text)],
                        memory=memory,
                        speaker=speaker_line(user),
                        current_time=current_time_label(),
                    )
                except Exception as error:
                    wrapped = AssistantError(
                        code='chat_error',
                        message='Агент не подготовил ответ.',
                        details={'error_type': type(error).__name__},
                    )
                    self._log_error('chat', user.id, wrapped)
                    self._try_index(
                        [build_memory_document(user.id, text, None)],
                        user.id,
                    )
                    self._reply(message, CHAT_ERROR_TEXT)
                    return
                answer = strip_service_lines(answer_from_agent(result))
                photos = collect_dog_photos(messages_from_agent(result))
                saved = self._try_index(
                    [build_memory_document(user.id, text, answer)],
                    user.id,
                )
                if not saved:
                    answer = f'{answer}{MEMORY_ERROR_SUFFIX}'
                self._history[user.id] = dialogue_tail(
                    [
                        *history,
                        ChatMessage.from_user(text),
                        ChatMessage.from_assistant(answer),
                    ],
                    MAX_HISTORY_MESSAGES,
                )
        finally:
            stop_typing.set()
            typing_worker.join(timeout=TYPING_JOIN_SECONDS)
        self._send_photos(message.chat.id, photos)
        self._reply(message, answer)

    def _try_recall(self, user_id: int, text: str) -> list[Document]:
        """Ищет близкий контекст пользователя. Сбой поиска не обрывает диалог.

        Args:
            user_id: Идентификатор Telegram.
            text: Реплика, по которой строится запрос.

        Returns:
            Фрагменты выше порога сходства или пустой список.
        """
        try:
            result = self._retrieval.run(
                {
                    'text_embedder': {'text': text},
                    'retriever': {
                        'filters': user_memory_filter(user_id),
                        'top_k': MEMORY_TOP_K,
                    },
                },
            )
        except Exception as error:
            self._log_error(
                'recall',
                user_id,
                AssistantError(
                    code='recall_error',
                    message='Не удалось прочитать память.',
                    details={'error_type': type(error).__name__},
                ),
            )
            return []
        return select_memory(documents_from_retrieval(result), MIN_COSINE_SCORE)

    def _try_index(self, documents: Sequence[Document], user_id: int) -> bool:
        """Считает эмбеддинги и записывает документы в Pinecone.

        Args:
            documents: Документы без векторов.
            user_id: Владелец, нужен только для журнала.

        Returns:
            ``True``, если запись прошла без ошибки.
        """
        if len(documents) == 0:
            return True
        try:
            self._indexing.run({'embedder': {'documents': list(documents)}})
        except Exception as error:
            self._log_error(
                'remember',
                user_id,
                AssistantError(
                    code='remember_error',
                    message='Не удалось записать память.',
                    details={'error_type': type(error).__name__},
                ),
            )
            return False
        return True

    def _delete_user_memory(self, user_id: int) -> None:
        """Удаляет документы пользователя пакетами, пока фильтр что-то находит.

        Args:
            user_id: Идентификатор Telegram.

        Raises:
            AssistantError: Если Pinecone не принял удаление.
        """
        filters = user_memory_filter(user_id)
        try:
            for _round in range(FORGET_ROUNDS):
                deleted = self._store.delete_by_filter(filters=filters)
                if deleted == 0:
                    return
        except Exception as error:
            raise AssistantError(
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

    def _send_photos(
        self,
        chat_id: int,
        photos: Sequence[tuple[str, str]],
    ) -> None:
        """Отправляет фотографии с подписью породы.

        Args:
            chat_id: Чат Telegram.
            photos: Пары из адреса снимка и подписи.
        """
        for url, caption in photos:
            try:
                self._bot.send_photo(chat_id, url, caption=caption)
            except Exception as error:
                logger.error(
                    'Не удалось отправить фото: error_type=%s',
                    type(error).__name__,
                )

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

    def _log_error(self, operation: str, user_id: int, error: AssistantError) -> None:
        """Пишет в журнал тип сбоя без текста реплики и секретов.

        Args:
            operation: Имя шага.
            user_id: Идентификатор пользователя.
            error: Пойманная ошибка помощника.
        """
        logger.error(
            'Ошибка бота: operation=%s user_id=%s code=%s error_type=%s',
            operation,
            user_id,
            error.code,
            error.details.get('error_type', ''),
        )


def main() -> None:
    """Читает настройки и запускает бота."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(name)s %(message)s',
    )
    try:
        assistant = TelegramAssistant.from_env()
        assistant.run()
    except AssistantError as error:
        logger.error(
            'Бот не запущен: code=%s message=%s',
            error.code,
            error.message,
        )
        raise SystemExit(1) from error


if __name__ == '__main__':
    main()
