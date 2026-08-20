"""Настройка логирования (NF-19).

Весь вывод идёт через `logging`, а не `print`: интерфейс подключает свой
обработчик и показывает сообщения в панели прогресса, не трогая код модулей.
"""

from __future__ import annotations

import logging
import sys
from collections.abc import Callable
from pathlib import Path

__all__ = ["CallbackHandler", "get_logger", "setup_logging"]

_FORMAT = "%(asctime)s %(levelname)-7s %(name)-24s %(message)s"
_DATE_FORMAT = "%H:%M:%S"


def setup_logging(
    level: int = logging.INFO,
    log_file: Path | None = None,
    quiet_libraries: bool = True,
) -> None:
    """Настроить корневой логгер.

    Args:
        level: Порог для собственных сообщений проекта.
        log_file: Файл для полного лога. Пишется всегда на уровне DEBUG,
            независимо от `level`.
        quiet_libraries: Приглушить болтливые зависимости.
    """
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    for handler in root.handlers[:]:
        root.removeHandler(handler)

    console = logging.StreamHandler(sys.stderr)
    console.setLevel(level)
    console.setFormatter(logging.Formatter(_FORMAT, _DATE_FORMAT))
    root.addHandler(console)

    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(logging.Formatter(_FORMAT, _DATE_FORMAT))
        root.addHandler(file_handler)

    if quiet_libraries:
        for name in ("yt_dlp", "urllib3", "matplotlib", "PIL", "numba"):
            logging.getLogger(name).setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """Логгер модуля. Вызывать как `get_logger(__name__)`."""
    return logging.getLogger(name)


class CallbackHandler(logging.Handler):
    """Обработчик, передающий записи во внешнюю функцию.

    Нужен интерфейсу: Qt подключает его и получает сообщения
    без изменения кода модулей.

    Example:
        >>> messages: list[str] = []
        >>> handler = CallbackHandler(messages.append)
        >>> logging.getLogger().addHandler(handler)
    """

    def __init__(self, callback: Callable[[str], None], level: int = logging.INFO):
        super().__init__(level)
        self._callback = callback
        self.setFormatter(logging.Formatter("%(message)s"))

    def emit(self, record: logging.LogRecord) -> None:
        """Передать запись во внешнюю функцию."""
        try:
            self._callback(self.format(record))
        except Exception:
            self.handleError(record)
