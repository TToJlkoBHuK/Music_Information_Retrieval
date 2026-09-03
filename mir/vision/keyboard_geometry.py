"""Геометрия фортепианной клавиатуры.

Раскладка нужна и генератору тестовых роликов, и детектору: первый по ней
рисует, второй — сверяет найденное. Держать её в одном месте обязательно,
иначе тесты начнут проверять сами себя.
"""

from __future__ import annotations

from dataclasses import dataclass

from mir.common.types import PITCH_MAX, PITCH_MIN, KeyboardLayout, KeySlot

__all__ = [
    "BLACK_KEY_HEIGHT_RATIO",
    "BLACK_KEY_WIDTH_RATIO",
    "BLACK_PITCH_CLASSES",
    "KeyboardGeometry",
    "black_pattern",
    "is_black_key",
    "white_keys_in_range",
]

BLACK_PITCH_CLASSES = frozenset({1, 3, 6, 8, 10})
"""Ступени внутри октавы, соответствующие чёрным клавишам (до-диез, ре-диез, ...)."""

BLACK_KEY_WIDTH_RATIO = 0.6
"""Ширина чёрной клавиши относительно белой."""

BLACK_KEY_HEIGHT_RATIO = 0.62
"""Высота чёрной клавиши относительно белой."""


def is_black_key(pitch: int) -> bool:
    """Чёрная ли клавиша у ноты с этим MIDI-номером."""
    return pitch % 12 in BLACK_PITCH_CLASSES


def white_keys_in_range(lowest: int, highest: int) -> list[int]:
    """Белые клавиши диапазона, слева направо."""
    return [p for p in range(lowest, highest + 1) if not is_black_key(p)]


def black_pattern(lowest: int, highest: int) -> list[bool]:
    """Последовательность «чёрная/белая» для диапазона.

    Именно этот узор — группы из двух и трёх чёрных клавиш — позволяет
    привязать найденную в кадре клавиатуру к абсолютным нотам, даже когда
    видна лишь её часть.
    """
    return [is_black_key(p) for p in range(lowest, highest + 1)]


@dataclass(frozen=True)
class KeyboardGeometry:
    """Расчёт координат клавиш по ширине области.

    Attributes:
        x: Левая граница области клавиатуры.
        width: Ширина области в пикселях.
        lowest_pitch: Нижняя нота.
        highest_pitch: Верхняя нота.
    """

    x: int
    width: int
    lowest_pitch: int = PITCH_MIN
    highest_pitch: int = PITCH_MAX

    def __post_init__(self) -> None:
        if self.lowest_pitch > self.highest_pitch:
            raise ValueError("lowest_pitch должен быть не больше highest_pitch")
        if is_black_key(self.lowest_pitch) or is_black_key(self.highest_pitch):
            raise ValueError("границы диапазона должны приходиться на белые клавиши")

    @property
    def white_pitches(self) -> list[int]:
        """Белые клавиши диапазона."""
        return white_keys_in_range(self.lowest_pitch, self.highest_pitch)

    @property
    def white_width(self) -> float:
        """Ширина одной белой клавиши в пикселях."""
        return self.width / len(self.white_pitches)

    def white_index(self, pitch: int) -> int:
        """Порядковый номер белой клавиши слева направо."""
        return sum(1 for p in range(self.lowest_pitch, pitch) if not is_black_key(p))

    def bounds(self, pitch: int) -> tuple[float, float]:
        """Левая и правая границы клавиши в пикселях.

        Чёрная клавиша центрируется на стыке соседних белых и уже их —
        так же, как на настоящем инструменте.
        """
        w = self.white_width
        if not is_black_key(pitch):
            left = self.x + self.white_index(pitch) * w
            return left, left + w

        boundary = self.x + self.white_index(pitch) * w
        half = w * BLACK_KEY_WIDTH_RATIO / 2
        return boundary - half, boundary + half

    def build_layout(self, y: int, height: int, confidence: float = 1.0) -> KeyboardLayout:
        """Собрать [`KeyboardLayout`][mir.common.types.KeyboardLayout].

        Args:
            y: Верхняя граница клавиатуры.
            height: Высота области.
            confidence: Уверенность детекции.
        """
        slots: list[KeySlot] = []
        for pitch in range(self.lowest_pitch, self.highest_pitch + 1):
            left, right = self.bounds(pitch)
            slots.append(
                KeySlot(
                    pitch=pitch,
                    x_min=round(left),
                    x_max=round(right),
                    is_black=is_black_key(pitch),
                )
            )
        return KeyboardLayout(
            bbox=(self.x, y, self.width, height),
            keys=tuple(slots),
            lowest_pitch=self.lowest_pitch,
            highest_pitch=self.highest_pitch,
            is_cropped=(self.lowest_pitch > PITCH_MIN or self.highest_pitch < PITCH_MAX),
            confidence=confidence,
        )
