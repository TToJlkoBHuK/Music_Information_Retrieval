"""Переводы между системами отсчёта времени.

В проекте их четыре: кадры видео, секунды, доли такта и тики MIDI.
Формулы собраны здесь, чтобы не расползались по коду магическими числами.
"""

from __future__ import annotations

from fractions import Fraction

__all__ = [
    "DEFAULT_PPQ",
    "beats_to_note_value",
    "frame_to_seconds",
    "seconds_to_frame",
    "seconds_to_ticks",
    "ticks_to_seconds",
]

DEFAULT_PPQ = 480
"""Тиков на четвертную ноту.

480 делится на 2, 3, 4 и 5, поэтому обычные длительности, триоли
и квинтоли выражаются целыми числами тиков без округления.
"""


def frame_to_seconds(frame_idx: int, fps: float) -> float:
    """Номер кадра → секунды.

    Args:
        frame_idx: Индекс кадра, начиная с нуля.
        fps: Частота кадров. Дробная, например 29.97 — округлять нельзя:
            на часовом ролике ошибка накопится в несколько секунд.

    Raises:
        ValueError: Если `fps` не положителен.
    """
    if fps <= 0:
        raise ValueError(f"fps={fps} должен быть положительным")
    return frame_idx / fps


def seconds_to_frame(t: float, fps: float) -> int:
    """Секунды → ближайший номер кадра."""
    if fps <= 0:
        raise ValueError(f"fps={fps} должен быть положительным")
    return round(t * fps)


def seconds_to_ticks(t: float, bpm: float, ppq: int = DEFAULT_PPQ) -> int:
    """Секунды → тики MIDI.

    Args:
        t: Время в секундах.
        bpm: Темп, ударов в минуту.
        ppq: Тиков на четверть.
    """
    if bpm <= 0:
        raise ValueError(f"bpm={bpm} должен быть положительным")
    return round(t * bpm / 60.0 * ppq)


def ticks_to_seconds(ticks: int, bpm: float, ppq: int = DEFAULT_PPQ) -> float:
    """Тики MIDI → секунды."""
    if bpm <= 0:
        raise ValueError(f"bpm={bpm} должен быть положительным")
    return ticks / ppq * 60.0 / bpm


def beats_to_note_value(beats: float, beat_unit: int = 4) -> Fraction:
    """Длительность в долях → нотная длительность.

    Результат — доля целой ноты: 1 — целая, 1/4 — четвертная.
    Используется `Fraction`, а не `float`: три триоли должны дать ровно
    одну долю, тогда как 0.333 * 3 = 0.999 ломает проверку заполненности такта.

    Args:
        beats: Длительность в долях.
        beat_unit: Знаменатель размера: 4 — доля равна четверти.

    Example:
        >>> beats_to_note_value(1.0)
        Fraction(1, 4)
        >>> beats_to_note_value(2.0)
        Fraction(1, 2)
    """
    return Fraction(beats).limit_denominator(64) / beat_unit
