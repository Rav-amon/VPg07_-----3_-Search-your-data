"""Точка входа Telegram-бота v2 на Haystack и DocLing."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

# Позволяет запускать ``python hay_v2_bot/main.py`` из корня проекта.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from hay_v2_bot.bot.assistant import TelegramAssistant
from hay_v2_bot.components.errors import BotError


def main() -> None:
    """Читает настройки и запускает бота."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(name)s %(message)s',
    )
    logger = logging.getLogger(__name__)
    try:
        assistant = TelegramAssistant.from_env()
    except BotError as error:
        logger.error(
            'Бот не запущен: error_type=%s code=%s message=%s',
            type(error).__name__,
            error.code,
            error.message,
        )
        raise SystemExit(1) from error
    assistant.run()


if __name__ == '__main__':
    main()
