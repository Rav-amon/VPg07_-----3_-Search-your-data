"""Единый формат ошибок бота."""

from __future__ import annotations

from collections.abc import Mapping


class BotError(Exception):
    """Ошибка с машинным кодом, текстом и подробностями без секретов.

    Attributes:
        code: Короткий код, например ``config_error``.
        message: Текст для журнала.
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
