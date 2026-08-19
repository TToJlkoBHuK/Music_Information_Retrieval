"""Перечисления, общие для всех этапов конвейера."""

from __future__ import annotations

from enum import Enum, IntEnum, IntFlag

__all__ = ["Clef", "Hand", "Platform", "Source", "Stage", "StrEnum"]


class StrEnum(str, Enum):
    """Строковое перечисление.

    Собственная реализация вместо `enum.StrEnum`: тот появился в Python 3.11,
    а проект поддерживает 3.10 (см. NF-11).
    """

    def __str__(self) -> str:
        return str(self.value)


class Hand(IntEnum):
    """Партия руки. `UNKNOWN` — рука ещё не определена."""

    LEFT = 0
    RIGHT = 1
    UNKNOWN = 2


class Source(IntFlag):
    """Канал, подтвердивший событие.

    Флаги, а не обычное перечисление: `VIDEO | AUDIO` даёт `BOTH`,
    что используется при слиянии каналов.
    """

    VIDEO = 1
    AUDIO = 2
    BOTH = VIDEO | AUDIO


class Clef(IntEnum):
    """Нотный ключ. Определяет, на каком стане записана нота."""

    TREBLE = 0
    BASS = 1


class Platform(StrEnum):
    """Источник видео."""

    YOUTUBE = "youtube"
    VK_VIDEO = "vk"
    RUTUBE = "rutube"
    LOCAL_FILE = "local"
    UNKNOWN = "unknown"

    @property
    def needs_proxy_in_russia(self) -> bool:
        """YouTube из России без прокси недоступен, отечественные площадки — доступны."""
        return self is Platform.YOUTUBE


class Stage(StrEnum):
    """Этап конвейера. Используется в колбэках прогресса и в логах."""

    DOWNLOAD = "download"
    DEMUX = "demux"
    VISION = "vision"
    AUDIO = "audio"
    FUSION = "fusion"
    NOTATION = "notation"
    EXPORT = "export"

    @property
    def title_ru(self) -> str:
        """Название для показа в интерфейсе."""
        return _STAGE_TITLES[self]


_STAGE_TITLES: dict[Stage, str] = {
    Stage.DOWNLOAD: "Загрузка видео",
    Stage.DEMUX: "Подготовка данных",
    Stage.VISION: "Анализ видеоряда",
    Stage.AUDIO: "Анализ звука",
    Stage.FUSION: "Сверка каналов",
    Stage.NOTATION: "Сборка нот",
    Stage.EXPORT: "Экспорт файлов",
}
