"""Модель данных проекта.

Единица времени везде — секунда от начала ролика (`float`). Кадры и тики MIDI
остаются внутри модулей и наружу не протекают: иначе на границах постоянна
путаница с единицами измерения.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypeAlias

import numpy.typing as npt

from mir.common.enums import Hand, Platform, Source, Stage

__all__ = [
    "PITCH_MAX",
    "PITCH_MIN",
    "Frame",
    "KeySlot",
    "KeyboardLayout",
    "MediaBundle",
    "NoteEvent",
    "ProgressCallback",
    "VideoInfo",
]

PITCH_MIN = 21
"""MIDI-номер ноты ля субконтроктавы (A0) — нижняя клавиша фортепиано."""

PITCH_MAX = 108
"""MIDI-номер ноты до пятой октавы (C8) — верхняя клавиша фортепиано."""

ProgressCallback = Callable[[Stage, float], None]
"""Колбэк прогресса: этап и доля выполнения в диапазоне 0..1."""

Frame: TypeAlias = "npt.NDArray[Any]"
"""Кадр видео: плотный массив `uint8`, обычно BGR или HSV.

Точный тип элемента не указан намеренно. OpenCV объявляет возвращаемое
значение как `Mat | ndarray[Any, dtype[integer | floating]]`, и любая
попытка сузить его до `uint8` заставила бы расставлять `cast` вокруг
каждого вызова, ничего не проверяя по существу.
"""


@dataclass(frozen=True, slots=True)
class NoteEvent:
    """Одна нота. Атом, проходящий через весь конвейер.

    Неизменяем осознанно: каждый этап создаёт новый список, а не правит
    существующий. Это позволяет сравнить состояние «до» и «после»
    при отладке и исключает скрытые побочные эффекты.

    Attributes:
        pitch: MIDI-номер, 21..108.
        onset: Начало звучания, секунды от начала ролика.
        offset: Конец звучания, секунды.
        velocity: Сила нажатия в шкале MIDI, 0..127.
        hand: Партия руки.
        source: Канал, подтвердивший событие.
        confidence: Уверенность в событии, 0..1.
    """

    pitch: int
    onset: float
    offset: float
    velocity: int
    hand: Hand = Hand.UNKNOWN
    source: Source = Source.VIDEO
    confidence: float = 1.0

    def __post_init__(self) -> None:
        if not PITCH_MIN <= self.pitch <= PITCH_MAX:
            raise ValueError(
                f"pitch={self.pitch} вне диапазона фортепиано ({PITCH_MIN}..{PITCH_MAX})"
            )
        if self.offset <= self.onset:
            raise ValueError(f"offset={self.offset} должен быть строго больше onset={self.onset}")
        if not 0 <= self.velocity <= 127:
            raise ValueError(f"velocity={self.velocity} вне диапазона 0..127")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"confidence={self.confidence} вне диапазона 0..1")

    @property
    def duration(self) -> float:
        """Длительность звучания в секундах."""
        return self.offset - self.onset


@dataclass(frozen=True, slots=True)
class KeySlot:
    """Одна клавиша на изображении.

    Attributes:
        pitch: MIDI-номер ноты этой клавиши.
        x_min: Левая граница в пикселях.
        x_max: Правая граница в пикселях.
        is_black: Чёрная клавиша.
    """

    pitch: int
    x_min: int
    x_max: int
    is_black: bool

    @property
    def center_x(self) -> float:
        """Горизонтальный центр клавиши."""
        return (self.x_min + self.x_max) / 2


@dataclass(frozen=True, slots=True)
class KeyboardLayout:
    """Разметка клавиатуры в кадре. Python-зеркало структуры из C++ ядра.

    Attributes:
        bbox: Область клавиатуры: x, y, ширина, высота.
        keys: Клавиши слева направо.
        lowest_pitch: Нижняя видимая нота.
        highest_pitch: Верхняя видимая нота.
        is_cropped: Видны не все 88 клавиш.
        confidence: Качество детекции, 0..1.
        octave_candidates: Другие возможные значения `lowest_pitch`.

            Узор чёрных клавиш повторяется каждую октаву, поэтому при
            обрезанной клавиатуре видеоканал определяет позицию внутри
            октавы, но не саму октаву. Здесь перечислены равновозможные
            варианты; выбрать верный помогает аудиоканал, где высоты
            абсолютны. Пусто, если неоднозначности нет.
    """

    bbox: tuple[int, int, int, int]
    keys: tuple[KeySlot, ...]
    lowest_pitch: int
    highest_pitch: int
    is_cropped: bool
    confidence: float
    octave_candidates: tuple[int, ...] = ()

    @property
    def octave_ambiguous(self) -> bool:
        """Октава определена неоднозначно и требует проверки по звуку."""
        return len(self.octave_candidates) > 1

    def pitch_at(self, x: int) -> int | None:
        """Определить ноту по горизонтальной координате.

        Чёрные клавиши перекрывают белые, поэтому проверяются первыми.

        Args:
            x: Координата в пикселях.

        Returns:
            MIDI-номер или `None`, если координата вне клавиатуры.
        """
        for key in self.keys:
            if key.is_black and key.x_min <= x <= key.x_max:
                return key.pitch
        for key in self.keys:
            if not key.is_black and key.x_min <= x <= key.x_max:
                return key.pitch
        return None

    @property
    def visible_range(self) -> tuple[int, int]:
        """Диапазон видимых нот."""
        return self.lowest_pitch, self.highest_pitch

    def covers(self, pitch: int) -> bool:
        """Попадает ли нота в видимый диапазон."""
        return self.lowest_pitch <= pitch <= self.highest_pitch


@dataclass(frozen=True, slots=True)
class VideoInfo:
    """Метаданные ролика, полученные без скачивания.

    Attributes:
        title: Название.
        duration: Длительность в секундах.
        platform: Источник.
        url: Нормализованная ссылка.
        width: Ширина кадра лучшего доступного формата.
        height: Высота кадра.
        fps: Частота кадров.
        filesize_approx: Оценка размера в байтах.
    """

    title: str
    duration: float
    platform: Platform
    url: str
    width: int | None = None
    height: int | None = None
    fps: float | None = None
    filesize_approx: int | None = None


@dataclass(frozen=True, slots=True)
class MediaBundle:
    """Подготовленный материал: видеоряд и аудиодорожка.

    Результат этапа [`mir.ingest`][] и вход для этапов `vision` и `audio`.

    Attributes:
        video_path: Путь к видеофайлу.
        audio_path: Путь к извлечённому WAV.
        fps: Частота кадров, дробная.
        duration: Длительность в секундах.
        width: Ширина кадра.
        height: Высота кадра.
        title: Название произведения.
        source_url: Исходная ссылка, если видео скачано.
        platform: Источник.
    """

    video_path: Path
    audio_path: Path
    fps: float
    duration: float
    width: int
    height: int
    title: str | None = None
    source_url: str | None = None
    platform: Platform = Platform.LOCAL_FILE

    @property
    def frame_count(self) -> int:
        """Оценка числа кадров."""
        return int(self.duration * self.fps)


@dataclass
class QualityReport:
    """Достоверность результата. Показывается пользователю и идёт в метрики.

    Attributes:
        keyboard_confidence: Качество детекции клавиатуры.
        keyboard_cropped: Клавиатура была обрезана.
        notes_confirmed: Ноты, подтверждённые обоими каналами.
        notes_from_video_only: Ноты только из видео.
        notes_from_audio_only: Ноты только из аудио.
        av_offset_ms: Найденный рассинхрон аудио и видео.
        warnings: Предупреждения для показа пользователю.
    """

    keyboard_confidence: float = 0.0
    keyboard_cropped: bool = False
    notes_confirmed: int = 0
    notes_from_video_only: int = 0
    notes_from_audio_only: int = 0
    av_offset_ms: float = 0.0
    warnings: list[str] = field(default_factory=list)

    @property
    def agreement_rate(self) -> float:
        """Доля нот, подтверждённых обоими каналами.

        Значение ниже 0.6 означает, что каналы сильно разошлись
        и к результату стоит отнестись критически.
        """
        total = self.notes_confirmed + self.notes_from_video_only + self.notes_from_audio_only
        return self.notes_confirmed / total if total else 0.0
