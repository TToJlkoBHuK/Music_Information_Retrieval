"""Компенсация рассинхрона звука и картинки (#29, F-11).

В роликах звук часто сдвинут относительно картинки — от десятков
миллисекунд до полусекунды: задержка рендера визуализатора, монтаж,
перекодирование площадкой. Слияние каналов без компенсации только
испортит результат: ноты одного аккорда не сопоставятся, если сдвиг
больше допуска в 80 мс.

**Метод.** Для каждой пары «нота видео — нота звука» одной высоты, чьи
начала ближе `max_av_offset`, берётся разность моментов. Если каналы
сдвинуты на Δ, у большинства пар разность близка к Δ, у прочих —
случайна. Плотность разностей, сглаженная гауссианой, даёт пик на Δ.
Это ровно взаимная корреляция двух последовательностей атак, только
посчитанная по парам, а не через спектр.

**Почему по высотам.** Если коррелировать все атаки без учёта высоты,
ровный ритм даёт пики на каждой доле: сдвиг на одну долю «совпадает»
почти так же хорошо, как верный. Учёт высоты оставляет в основном
настоящие пары — у мелодии высоты повторяются не в такт.

Если транскрипции звука нет, есть запасной путь по одним атакам —
[`estimate_av_offset_from_onsets`][mir.fusion.aligner.estimate_av_offset_from_onsets].
Он слабее именно из-за ритма, и уверенность у него честно ниже.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, replace

import numpy as np
import numpy.typing as npt

from mir.common.logging import get_logger
from mir.common.types import NoteEvent
from mir.config import FusionConfig

__all__ = [
    "OffsetEstimate",
    "apply_offset",
    "estimate_av_offset",
    "estimate_av_offset_from_onsets",
]

_log = get_logger(__name__)

_PEAK_EXCLUSION_SIGMAS = 4.0
"""Насколько далеко от главного пика искать соперника, в сигмах сглаживания.

Ближе четырёх сигм соседние значения — склон того же пика, а не другой
кандидат: гауссиана на 4σ спадает до 0.03 % от вершины.
"""

_SUPPORT_SIGMAS = 2.0
"""Пара «поддерживает» сдвиг, если её разность в пределах 2σ от него."""

_CHORD_WINDOW = 0.03
"""Атаки видео ближе 30 мс считаются одной — для метода без высот."""


@dataclass(frozen=True)
class OffsetEstimate:
    """Найденный сдвиг звука относительно картинки.

    Attributes:
        offset: Сдвиг в секундах. Положительный — звук отстаёт: нота
            слышна позже, чем видна.
        confidence: Насколько пик выделяется среди других кандидатов, 0..1.
            `1 - вершина_соперника / вершина_пика` над общим фоном.
        pairs: Сколько пар поддерживает найденный сдвиг.
        method: `notes` — по нотам с учётом высоты, `onsets` — по атакам.
    """

    offset: float
    confidence: float
    pairs: int
    method: str

    def is_reliable(self, min_confidence: float) -> bool:
        """Можно ли применять сдвиг."""
        return self.confidence >= min_confidence and self.pairs > 0

    @property
    def offset_ms(self) -> float:
        """Сдвиг в миллисекундах — для отчёта (`QualityReport.av_offset_ms`)."""
        return self.offset * 1000.0


def _pitch_differences(
    video: list[NoteEvent], audio: list[NoteEvent], window: float
) -> npt.NDArray[np.float64]:
    """Разности «звук − видео» для всех пар одной высоты в пределах окна."""
    by_pitch: dict[int, list[float]] = defaultdict(list)
    for note in audio:
        by_pitch[note.pitch].append(note.onset)
    sorted_audio = {pitch: np.sort(times) for pitch, times in by_pitch.items()}

    chunks: list[npt.NDArray[np.float64]] = []
    for note in video:
        times = sorted_audio.get(note.pitch)
        if times is None:
            continue
        lo = np.searchsorted(times, note.onset - window)
        hi = np.searchsorted(times, note.onset + window, side="right")
        if hi > lo:
            chunks.append(times[lo:hi] - note.onset)
    return np.concatenate(chunks) if chunks else np.empty(0)


def _onset_differences(
    video: npt.NDArray[np.float64], audio: npt.NDArray[np.float64], window: float
) -> npt.NDArray[np.float64]:
    """Разности «звук − видео» для всех атак в пределах окна."""
    audio = np.sort(audio)
    chunks: list[npt.NDArray[np.float64]] = []
    for time in video:
        lo = np.searchsorted(audio, time - window)
        hi = np.searchsorted(audio, time + window, side="right")
        if hi > lo:
            chunks.append(audio[lo:hi] - time)
    return np.concatenate(chunks) if chunks else np.empty(0)


def _locate_peak(
    differences: npt.NDArray[np.float64], config: FusionConfig, method: str
) -> OffsetEstimate:
    """Найти пик плотности разностей и оценить его надёжность."""
    if len(differences) < config.min_offset_pairs:
        return OffsetEstimate(0.0, 0.0, len(differences), method)

    window = config.max_av_offset
    step = config.offset_resolution
    sigma = config.offset_smoothing
    edges = np.arange(-window - step / 2, window + step, step)
    counts, _ = np.histogram(differences, bins=edges)
    centers = (edges[:-1] + edges[1:]) / 2

    radius = int(np.ceil(4 * sigma / step))
    kernel = np.exp(-0.5 * (np.arange(-radius, radius + 1) * step / sigma) ** 2)
    density = np.convolve(counts.astype(np.float64), kernel, mode="same")

    peak = int(np.argmax(density))
    offset = float(centers[peak])
    if 0 < peak < len(density) - 1:
        # Парабола по трём точкам: пик между узлами сетки.
        left, top, right = density[peak - 1 : peak + 2]
        curvature = left - 2 * top + right
        if curvature < 0:
            offset += float(0.5 * (left - right) / curvature) * step

    background = float(np.median(density))
    far = np.abs(centers - offset) > _PEAK_EXCLUSION_SIGMAS * sigma
    rival = float(np.max(density[far])) if np.any(far) else background
    height = float(density[peak]) - background
    confidence = 1.0 - (rival - background) / height if height > 0 else 0.0

    support = int(np.sum(np.abs(differences - offset) <= _SUPPORT_SIGMAS * sigma))
    if support < config.min_offset_pairs:
        confidence = 0.0
    return OffsetEstimate(offset, float(np.clip(confidence, 0.0, 1.0)), support, method)


def estimate_av_offset(
    video_notes: list[NoteEvent], audio_notes: list[NoteEvent], config: FusionConfig
) -> OffsetEstimate:
    """Оценить сдвиг звука по нотам обоих каналов с учётом высоты.

    Args:
        video_notes: Ноты из видеоканала.
        audio_notes: Ноты из транскрипции звука.
        config: Параметры из секции `[fusion]`.

    Returns:
        Сдвиг и уверенность. Применять ли его — решает вызывающий,
        сравнивая уверенность с `min_offset_confidence`.
    """
    differences = _pitch_differences(video_notes, audio_notes, config.max_av_offset)
    estimate = _locate_peak(differences, config, "notes")
    _log.info(
        "рассинхрон по нотам: %+.0f мс, уверенность %.2f, пар %d",
        estimate.offset_ms,
        estimate.confidence,
        estimate.pairs,
    )
    return estimate


def estimate_av_offset_from_onsets(
    video_notes: list[NoteEvent], audio_onsets: npt.NDArray[np.float64], config: FusionConfig
) -> OffsetEstimate:
    """Оценить сдвиг по атакам звука, без транскрипции.

    Запасной путь, когда нейросеть не установлена или отключена. Высоты
    звука здесь неизвестны, поэтому на ровном ритме соседняя доля даёт
    почти такой же пик — уверенность на такой музыке низкая, и сдвиг
    не применится. Это правильное поведение: неуверенный сдвиг хуже
    нулевого.
    """
    video = np.sort(np.array([n.onset for n in video_notes], dtype=np.float64))
    if len(video):
        video = video[np.concatenate([[True], np.diff(video) > _CHORD_WINDOW])]
    differences = _onset_differences(video, audio_onsets, config.max_av_offset)
    estimate = _locate_peak(differences, config, "onsets")
    _log.info(
        "рассинхрон по атакам: %+.0f мс, уверенность %.2f, пар %d",
        estimate.offset_ms,
        estimate.confidence,
        estimate.pairs,
    )
    return estimate


def apply_offset(notes: list[NoteEvent], offset: float) -> list[NoteEvent]:
    """Сдвинуть все ноты на `offset` секунд.

    Чтобы перенести звук на шкалу видео, сдвиг берётся со знаком минус:
    `apply_offset(audio_notes, -estimate.offset)`.

    Ноты, целиком ушедшие в отрицательное время, отбрасываются; у
    попавших на ноль начало обрезается до нуля.

    Returns:
        Новый список — `NoteEvent` неизменяем.
    """
    shifted: list[NoteEvent] = []
    for note in notes:
        onset, offset_time = note.onset + offset, note.offset + offset
        if offset_time <= 0:
            continue
        shifted.append(replace(note, onset=max(onset, 0.0), offset=offset_time))
    return shifted
