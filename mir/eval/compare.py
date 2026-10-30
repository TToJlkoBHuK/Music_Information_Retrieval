"""Сверка двух каналов без эталона.

Ручной разметки реальных роликов пока нет (#13), но два канала проверяют
друг друга: если видео и звук описывают одну и ту же музыку, их
фортепианные валики совпадают — с точностью до сдвига по времени.
Если же лучшее совпадение получается при сдвиге **по высоте**, значит
один из каналов неверно определяет ноты.

Сравниваются валики, а не начала нот: на реальных роликах оба канала
иногда сливают повторённую ноту в одну длинную, и начала тогда не
совпадают, а звучание по времени — совпадает.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from mir.common.types import PITCH_MAX, PITCH_MIN, NoteEvent

__all__ = ["ChannelAgreement", "compare_channels", "piano_roll"]

_PITCHES = PITCH_MAX - PITCH_MIN + 1


def piano_roll(notes: list[NoteEvent], step: float, frames: int) -> npt.NDArray[np.float32]:
    """Валик: 1 там, где нота звучит. Строки — высоты 21..108, столбцы — шаги времени."""
    roll = np.zeros((_PITCHES, frames), dtype=np.float32)
    for note in notes:
        first = int(note.onset / step)
        last = max(int(note.offset / step), first + 1)
        if first < frames:
            roll[note.pitch - PITCH_MIN, max(first, 0) : min(last, frames)] = 1.0
    return roll


@dataclass(frozen=True)
class ChannelAgreement:
    """Насколько видео и звук описывают одну музыку.

    Attributes:
        similarity: Косинусная мера совпадения валиков без сдвига
            по высоте, при лучшем сдвиге по времени, 0..1.
        time_shift: Этот лучший сдвиг: звук относительно видео, секунды.
        best_pitch_shift: На сколько полутонов поднять ноты видео, чтобы
            совпадение стало наибольшим. 0 — высоты видео верны.
        best_similarity: Совпадение при этом сдвиге высоты.
        pitch_shifts: Лучшее совпадение для каждого сдвига высоты.
    """

    similarity: float
    time_shift: float
    best_pitch_shift: int
    best_similarity: float
    pitch_shifts: dict[int, float]

    @property
    def pitch_consistent(self) -> bool:
        """Лучшее совпадение — без сдвига высоты."""
        return self.best_pitch_shift == 0


def compare_channels(
    video: list[NoteEvent],
    audio: list[NoteEvent],
    step: float = 0.05,
    max_time_shift: float = 1.0,
    max_pitch_shift: int = 12,
) -> ChannelAgreement:
    """Сравнить валики видео и звука.

    Сдвиги по времени и по высоте ищутся совместно: если сначала искать
    время без учёта высоты, при ошибочных высотах видео лучший сдвиг
    по времени выходит случайным, и верный сдвиг высоты уже не находится.
    Взаимная корреляция по времени для всех сдвигов сразу считается через
    преобразование Фурье — построчно, с суммой по высотам.

    Args:
        video: Ноты видеоканала.
        audio: Ноты звукового канала.
        step: Шаг валика по времени, секунды.
        max_time_shift: Предел поиска сдвига по времени.
        max_pitch_shift: Предел поиска сдвига по высоте, полутоны.
    """
    end = max((n.offset for n in [*video, *audio]), default=0.0)
    frames = int(end / step) + 2
    v_roll = piano_roll(video, step, frames)
    a_roll = piano_roll(audio, step, frames)

    size = 1 << int(np.ceil(np.log2(2 * frames)))
    v_spec = np.fft.rfft(v_roll, size, axis=1)
    a_spec = np.fft.rfft(a_roll, size, axis=1)
    v_energy = np.sum(v_roll, axis=1)
    a_norm = float(np.sqrt(np.sum(a_roll)))
    reach = round(max_time_shift / step)
    lags = np.r_[np.arange(0, reach + 1), np.arange(-reach, 0)]

    by_pitch: dict[int, float] = {}
    lag_of: dict[int, int] = {}
    for k in range(-max_pitch_shift, max_pitch_shift + 1):
        # Строка видео r переходит в строку r + k; ушедшие за край отбрасываются.
        rows = np.arange(max(-k, 0), _PITCHES - max(k, 0))
        product = np.sum(a_spec[rows + k] * np.conj(v_spec[rows]), axis=0)
        correlation = np.fft.irfft(product, size)[lags]
        norm = float(np.sqrt(np.sum(v_energy[rows]))) * a_norm
        best = int(np.argmax(correlation))
        by_pitch[k] = float(correlation[best]) / norm if norm else 0.0
        lag_of[k] = int(lags[best])

    best_pitch = max(by_pitch, key=lambda k: by_pitch[k])
    return ChannelAgreement(
        similarity=by_pitch[0],
        time_shift=lag_of[0] * step,
        best_pitch_shift=best_pitch,
        best_similarity=by_pitch[best_pitch],
        pitch_shifts=by_pitch,
    )
