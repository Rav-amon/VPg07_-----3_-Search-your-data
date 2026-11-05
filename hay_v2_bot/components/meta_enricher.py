"""Компонент Haystack: обогащение чанков DocLing плоскими метаданными."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any
from uuid import uuid4

from haystack import Document, component

from hay_v2_bot.components.constants import DOC_TYPE_FILE, MAX_MEMORY_CHARS


def _as_mapping(value: object) -> Mapping[str, Any] | None:
    """Возвращает словарь, если значение похоже на mapping.

    Args:
        value: Произвольное значение.

    Returns:
        Mapping или ``None``.
    """
    if isinstance(value, Mapping):
        return value
    return None


def extract_page_number(meta: Mapping[str, Any]) -> int | None:
    """Достаёт номер страницы из метаданных DocLing, если он есть.

    Args:
        meta: Метаданные Haystack-документа.

    Returns:
        Номер страницы или ``None``.
    """
    page_number = meta.get('page_number')
    if isinstance(page_number, int):
        return page_number
    if isinstance(page_number, float):
        return int(page_number)
    if isinstance(page_number, str) and page_number.isdecimal():
        return int(page_number)
    dl_meta = _as_mapping(meta.get('dl_meta'))
    if dl_meta is None:
        return None
    inner_meta = _as_mapping(dl_meta.get('meta'))
    if inner_meta is None:
        return None
    doc_items = inner_meta.get('doc_items')
    if not isinstance(doc_items, list) or len(doc_items) == 0:
        return None
    first_item = _as_mapping(doc_items[0])
    if first_item is None:
        return None
    prov = first_item.get('prov')
    if not isinstance(prov, list) or len(prov) == 0:
        return None
    first_prov = _as_mapping(prov[0])
    if first_prov is None:
        return None
    page_no = first_prov.get('page_no')
    if isinstance(page_no, int):
        return page_no
    if isinstance(page_no, float):
        return int(page_no)
    if isinstance(page_no, str) and page_no.isdecimal():
        return int(page_no)
    return None


def extract_section(meta: Mapping[str, Any]) -> str | None:
    """Достаёт путь заголовков секции из метаданных DocLing.

    Args:
        meta: Метаданные Haystack-документа.

    Returns:
        Строка секции или ``None``.
    """
    dl_meta = _as_mapping(meta.get('dl_meta'))
    if dl_meta is None:
        return None
    inner_meta = _as_mapping(dl_meta.get('meta'))
    if inner_meta is None:
        return None
    headings = inner_meta.get('headings')
    if not isinstance(headings, list):
        return None
    parts = [str(item).strip() for item in headings if str(item).strip() != '']
    if len(parts) == 0:
        return None
    return ' / '.join(parts)[:200]


def flatten_file_meta(
    source_meta: Mapping[str, Any],
    *,
    user_id: str,
    filename: str,
    chunk_index: int,
) -> dict[str, object]:
    """Собирает плоские метаданные, допустимые для Pinecone.

    Args:
        source_meta: Исходные meta от DocLingConverter.
        user_id: Идентификатор владельца.
        filename: Имя загруженного файла.
        chunk_index: Порядковый номер чанка с нуля.

    Returns:
        Плоский словарь метаданных.
    """
    meta: dict[str, object] = {
        'user_id': user_id,
        'filename': filename,
        'chunk_index': chunk_index,
        'doc_type': DOC_TYPE_FILE,
    }
    page = extract_page_number(source_meta)
    if page is not None:
        meta['page'] = page
    section = extract_section(source_meta)
    if section is not None:
        meta['section'] = section
    return meta


@component
class DocumentMetaEnricher:
    """Добавляет user_id, filename, chunk_index и page к чанкам DocLing."""

    @component.output_types(documents=list[Document])
    def run(
        self,
        documents: list[Document],
        user_id: str,
        filename: str,
    ) -> dict[str, list[Document]]:
        """Обогащает документы и обрезает слишком длинный текст.

        Args:
            documents: Чанки от DocLingConverter.
            user_id: Владелец файла.
            filename: Имя файла для метаданных и цитирования.

        Returns:
            Словарь с ключом ``documents``.
        """
        enriched: list[Document] = []
        for index, document in enumerate(documents):
            content = document.content if isinstance(document.content, str) else ''
            cleaned = content.strip()
            if cleaned == '':
                continue
            source_meta = document.meta if isinstance(document.meta, dict) else {}
            meta = flatten_file_meta(
                source_meta,
                user_id=user_id,
                filename=filename,
                chunk_index=index,
            )
            enriched.append(
                Document(
                    id=f'file-{user_id}-{uuid4().hex}',
                    content=cleaned[:MAX_MEMORY_CHARS],
                    meta=meta,
                ),
            )
        return {'documents': enriched}
