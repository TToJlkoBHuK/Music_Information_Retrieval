"""Иерархия исключений.

Каждая ошибка несёт два текста: технический (в лог) и `user_message`
для показа в интерфейсе. Разделение нужно, чтобы музыкант не видел
traceback вместо объяснения, что пошло не так (F-35).
"""

from __future__ import annotations

__all__ = [
    "CancelledError",
    "ConfigError",
    "DemuxError",
    "DownloadError",
    "ExportError",
    "KeyboardNotFoundError",
    "MirError",
    "TranscriptionError",
    "UnsupportedSourceError",
]


class MirError(Exception):
    """Базовое исключение проекта."""

    default_user_message = "Произошла ошибка при обработке"

    def __init__(self, technical: str, user_message: str | None = None) -> None:
        super().__init__(technical)
        self.technical = technical
        self.user_message = user_message or self.default_user_message

    def __str__(self) -> str:
        return self.technical


class ConfigError(MirError):
    """Некорректная конфигурация: неизвестный ключ или значение вне диапазона."""

    default_user_message = "Ошибка в настройках приложения"


class UnsupportedSourceError(MirError):
    """Ссылка не распознана или файл имеет неподдерживаемый формат."""

    default_user_message = (
        "Не удалось распознать ссылку. Укажите ссылку на видео "
        "с YouTube, VK Видео или Rutube либо выберите файл на диске"
    )


class DownloadError(MirError):
    """Не удалось получить видео."""

    default_user_message = "Не удалось загрузить видео"


class DemuxError(MirError):
    """Не удалось разделить файл на видеоряд и аудиодорожку."""

    default_user_message = (
        "Не удалось обработать видеофайл. Возможно, он повреждён "
        "или записан в неподдерживаемом формате"
    )


class KeyboardNotFoundError(MirError):
    """В кадре не найдена фортепианная клавиатура."""

    default_user_message = (
        "В этом видео не найдена фортепианная клавиатура. "
        "Программа работает с роликами формата piano visualizer"
    )


class TranscriptionError(MirError):
    """Сбой нейросетевой транскрипции аудио."""

    default_user_message = "Не удалось распознать звуковую дорожку"


class ExportError(MirError):
    """Сбой при формировании выходных файлов."""

    default_user_message = "Не удалось сохранить результат"


class CancelledError(MirError):
    """Обработка прервана пользователем."""

    default_user_message = "Обработка отменена"
