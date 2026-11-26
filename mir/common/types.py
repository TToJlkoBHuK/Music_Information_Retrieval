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

import numpy as np
import numpy.typing as npt

from mir.common.enums import Hand, Platform, Source, Stage

__all__ = [
    "BLACK_PITCH_CLASSES",
    "PITCH_MAX",
    "PITCH_MIN",
    "CancelCheck",
    "Frame",
    "KeySlot",
    "KeyboardLayout",
    "MediaBundle",
    "NoteEvent",
    "PedalEvent",
    "ProgressCallback",
    "TempoMap",
    "VideoInfo",
    "is_black_key",
]

PITCH_MIN = 21
"""MIDI-номер ноты ля субконтроктавы (A0) — нижняя клавиша фортепиано."""

PITCH_MAX = 108
"""MIDI-номер ноты до пятой октавы (C8) — верхняя клавиша фортепиано."""

BLACK_PITCH_CLASSES = frozenset({1, 3, 6, 8, 10})
"""Ступени внутри октавы, соответствующие чёрным клавишам (до-диез, ре-диез, ...)."""


def is_black_key(pitch: int) -> bool:
    """Чёрная ли клавиша у ноты с этим MIDI-номером."""
    return pitch % 12 in BLACK_PITCH_CLASSES


ProgressCallback = Callable[[Stage, float], None]
"""Колбэк прогресса: этап и доля выполнения в диапазоне 0..1."""

CancelCheck = Callable[[], bool]
"""Опрос отмены: `True` — пользователь нажал «Отмена», работу пора прервать.

Функция, а не флаг: интерфейс держит `threading.Event`, CLI — ничего,
и обоим удобно передать то, что у них есть, не подстраиваясь под тип.
"""

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
class PedalEvent:
    """Нажатие правой педали (F-13).

    Attributes:
        onset: Момент нажатия, секунды.
        offset: Момент отпускания, секунды.
    """

    onset: float
    offset: float

    def __post_init__(self) -> None:
        if self.offset <= self.onset:
            raise ValueError(f"offset={self.offset} должен быть строго больше onset={self.onset}")

    @property
    def duration(self) -> float:
        """Сколько педаль была нажата, секунды."""
        return self.offset - self.onset


def _frozen_array(values: npt.ArrayLike) -> npt.NDArray[np.float64]:
    """Копия массива, защищённая от записи.

    `frozen=True` у датакласса запрещает заменить поле, но не изменить
    содержимое массива на месте. Без этой защиты сетку долей можно было бы
    незаметно испортить из любого этапа.
    """
    array = np.array(values, dtype=np.float64)
    array.flags.writeable = False
    return array


@dataclass(frozen=True, eq=False)
class TempoMap:
    """Временна́я сетка: где лежат доли.

    Перевод секунд в доли идёт интерполяцией по найденным долям, а не
    делением на средний темп. У живого исполнения темп плавает, и линейная
    формула накапливает ошибку: при отклонении темпа на 3 % к сотой доле
    сетка уезжает на три доли.

    Attributes:
        beats: Моменты долей, секунды. Строго возрастают, не меньше двух.
        downbeats: Моменты сильных долей — начала тактов.
        beats_per_bar: Долей в такте, по которому расставлены `downbeats`.
            Предварительное значение: размер определяется на этапе 5 (#33).
        confidence: Насколько сетка совпадает с атаками звука, 0..1.
            Низкое значение означает rubato или сложный ритм.
        is_manual: Темп задан пользователем (F-31), а не найден.
    """

    beats: npt.NDArray[np.float64]
    downbeats: npt.NDArray[np.float64] = field(default_factory=lambda: _frozen_array([]))
    beats_per_bar: int = 4
    confidence: float = 1.0
    is_manual: bool = False

    def __post_init__(self) -> None:
        beats = _frozen_array(self.beats)
        if beats.ndim != 1 or len(beats) < 2:
            raise ValueError("сетке нужно не меньше двух долей")
        if not np.all(np.diff(beats) > 0):
            raise ValueError("моменты долей должны строго возрастать")
        if self.beats_per_bar < 1:
            raise ValueError(f"beats_per_bar={self.beats_per_bar} должен быть положительным")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"confidence={self.confidence} вне диапазона 0..1")
        object.__setattr__(self, "beats", beats)
        object.__setattr__(self, "downbeats", _frozen_array(self.downbeats))

    @classmethod
    def constant(
        cls,
        bpm: float,
        duration: float,
        start: float = 0.0,
        beats_per_bar: int = 4,
        is_manual: bool = True,
    ) -> TempoMap:
        """Ровная сетка с постоянным темпом.

        Args:
            bpm: Темп, долей в минуту.
            duration: До какого момента строить сетку, секунды.
            start: Момент первой доли.
            beats_per_bar: Долей в такте; первая доля считается сильной.
            is_manual: Темп задан пользователем.
        """
        if bpm <= 0:
            raise ValueError(f"bpm={bpm} должен быть положительным")
        period = 60.0 / bpm
        count = max(int(np.ceil((duration - start) / period)) + 1, 2)
        beats = (start + np.arange(count) * period).astype(np.float64)
        return cls(
            beats=beats,
            downbeats=beats[::beats_per_bar],
            beats_per_bar=beats_per_bar,
            confidence=1.0,
            is_manual=is_manual,
        )

    @property
    def bpm(self) -> float:
        """Средний темп по всей сетке, долей в минуту.

        Берётся по крайним долям, а не по медиане интервалов: интервалы
        квантованы шагом анализа (8 мс), и медиана по ним даёт ошибку
        до процента, тогда как среднее по сотне долей — сотые доли процента.
        """
        span = float(self.beats[-1] - self.beats[0])
        return 60.0 * (len(self.beats) - 1) / span

    def seconds_to_beats_array(self, times: npt.ArrayLike) -> npt.NDArray[np.float64]:
        """Секунды → позиция в долях для массива моментов.

        Вне сетки позиция продолжается с темпом крайнего интервала.
        """
        t = np.asarray(times, dtype=np.float64)
        beats = self.beats
        index = np.arange(len(beats), dtype=np.float64)
        result = np.interp(t, beats, index)
        first = float(beats[1] - beats[0])
        last = float(beats[-1] - beats[-2])
        result = np.where(t < beats[0], (t - beats[0]) / first, result)
        result = np.where(t > beats[-1], len(beats) - 1 + (t - beats[-1]) / last, result)
        return result

    def seconds_to_beats(self, t: float) -> float:
        """Секунды → позиция в долях."""
        return float(self.seconds_to_beats_array([t])[0])

    def beats_to_seconds_array(self, positions: npt.ArrayLike) -> npt.NDArray[np.float64]:
        """Позиция в долях → секунды для массива позиций."""
        b = np.asarray(positions, dtype=np.float64)
        beats = self.beats
        index = np.arange(len(beats), dtype=np.float64)
        result = np.interp(b, index, beats)
        first = float(beats[1] - beats[0])
        last = float(beats[-1] - beats[-2])
        result = np.where(b < 0, beats[0] + b * first, result)
        result = np.where(b > len(beats) - 1, beats[-1] + (b - len(beats) + 1) * last, result)
        return result

    def beats_to_seconds(self, position: float) -> float:
        """Позиция в долях → секунды."""
        return float(self.beats_to_seconds_array([position])[0])

    def local_bpm(self, t: float) -> float:
        """Темп в окрестности момента `t` — по интервалу, в который он попал."""
        index = int(np.clip(np.searchsorted(self.beats, t) - 1, 0, len(self.beats) - 2))
        return 60.0 / float(self.beats[index + 1] - self.beats[index])

    def nearest_grid_point(self, t: float, subdivision: int = 4) -> float:
        """Ближайший узел сетки.

        Args:
            t: Момент, секунды.
            subdivision: Узлов на долю: 4 — шестнадцатые при доле-четверти.
        """
        if subdivision < 1:
            raise ValueError(f"subdivision={subdivision} должен быть положительным")
        position = round(self.seconds_to_beats(t) * subdivision) / subdivision
        return self.beats_to_seconds(position)

    def shifted(self, offset: float) -> TempoMap:
        """Та же сетка, сдвинутая на `offset` секунд.

        Нужна при компенсации рассинхрона: сетку, найденную по звуку,
        переносят на шкалу времени видео вместе с нотами.
        """
        return TempoMap(
            beats=self.beats + offset,
            downbeats=self.downbeats + offset,
            beats_per_bar=self.beats_per_bar,
            confidence=self.confidence,
            is_manual=self.is_manual,
        )


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
