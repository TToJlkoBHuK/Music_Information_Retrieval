"""Атаки звука: точные моменты начала нот (#28).

Видео при 30 fps видит мир с шагом 33 мс, при 25 fps — 40 мс. Аудио здесь
разбирается с шагом 8 мс, а разброс найденных атак около 3 мс. Эти метки
нужны, чтобы подтянуть время нот, найденных по видео, к тому, что звучало
на самом деле.

Почему окно 1024, а не 2048, как у librosa по умолчанию, — в
`config/README.md`: замер на синтетике показал вдвое меньшую задержку
и разброс при той же полноте.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import numpy.typing as npt

from mir.audio.signal import AudioSignal
from mir.common.logging import get_logger
from mir.common.types import NoteEvent
from mir.config import AudioConfig

__all__ = [
    "OnsetEnvelope",
    "detect_onsets",
    "merge_close",
    "onset_envelope",
    "refine_note_onsets",
]

_log = get_logger(__name__)


@dataclass(frozen=True, eq=False)
class OnsetEnvelope:
    """Функция новизны: насколько резко меняется спектр в каждый момент.

    Пики функции — кандидаты в атаки. Ту же функцию использует и
    beat-tracker, поэтому она считается один раз и передаётся дальше.

    Attributes:
        values: Значение на каждом шаге анализа. Поток считается по
            спектру в децибелах: так тихая атака на фоне тишины видна
            не хуже громкой — для поиска атак это и нужно.
        sample_rate: Частота дискретизации исходного сигнала.
        hop_length: Шаг анализа в отсчётах.
        start: Момент ролика, соответствующий первому шагу.
        accent: Тот же поток, но по спектру мощности, без логарифма.
            Здесь громкая атака весит больше тихой — это мера акцента.
            Для сильных долей нужна именно она: в децибелах сильная доля
            метронома выходила даже слабее прочих (замер — в README модуля).
    """

    values: npt.NDArray[np.float32]
    sample_rate: int
    hop_length: int
    start: float = 0.0
    accent: npt.NDArray[np.float32] | None = None

    @property
    def frame_rate(self) -> float:
        """Шагов анализа в секунду."""
        return self.sample_rate / self.hop_length

    def times(self) -> npt.NDArray[np.float64]:
        """Момент каждого шага, секунды от начала ролика."""
        return self.start + np.arange(len(self.values)) / self.frame_rate

    def frame_of(self, t: float) -> int:
        """Ближайший шаг анализа к моменту `t`."""
        index = round((t - self.start) * self.frame_rate)
        return int(np.clip(index, 0, len(self.values) - 1))


def onset_envelope(signal: AudioSignal, config: AudioConfig) -> OnsetEnvelope:
    """Посчитать функцию новизны по сигналу.

    Используется спектральный поток в мел-шкале — стандарт для
    фортепиано: у рояля резкая широкополосная атака, и поток на ней
    даёт чёткий пик даже внутри аккорда.
    """
    import librosa

    sr, hop, n_fft = signal.sample_rate, config.hop_length, config.onset_n_fft
    # Спектр считается один раз на обе функции. `n_fft` передаётся и сюда:
    # по нему librosa компенсирует задержку центрированного окна, и без него
    # атаки уезжали на 32 мс.
    power = librosa.feature.melspectrogram(y=signal.samples, sr=sr, hop_length=hop, n_fft=n_fft)
    values = librosa.onset.onset_strength(
        S=librosa.power_to_db(power), sr=sr, hop_length=hop, n_fft=n_fft
    )
    accent = librosa.onset.onset_strength(S=power, sr=sr, hop_length=hop, n_fft=n_fft)
    return OnsetEnvelope(
        values=values.astype(np.float32),
        sample_rate=signal.sample_rate,
        hop_length=config.hop_length,
        start=signal.start,
        accent=accent.astype(np.float32),
    )


def detect_onsets(envelope: OnsetEnvelope, config: AudioConfig) -> npt.NDArray[np.float64]:
    """Найти атаки по функции новизны.

    Returns:
        Моменты атак, секунды от начала ролика, по возрастанию.
    """
    import librosa

    if len(envelope.values) < 3 or float(np.max(envelope.values)) <= 0:
        return np.empty(0)
    frames = librosa.onset.onset_detect(
        onset_envelope=envelope.values,
        sr=envelope.sample_rate,
        hop_length=envelope.hop_length,
        delta=config.onset_delta,
        wait=max(round(config.onset_wait * envelope.frame_rate), 1),
        units="frames",
    )
    times: npt.NDArray[np.float64] = envelope.start + np.asarray(frames) / envelope.frame_rate
    return np.sort(times)


def merge_close(times: npt.ArrayLike, window: float) -> npt.NDArray[np.float64]:
    """Слить моменты ближе `window` в один — первый из группы.

    Аккорд звучит как одна атака, но транскрипция даёт по событию на
    каждую ноту с разбросом в несколько миллисекунд. Для темпа и для
    подсчёта межнотных интервалов это должна быть одна точка, иначе
    медианный интервал выходит почти нулевым.
    """
    values = np.sort(np.asarray(times, dtype=np.float64))
    if len(values) == 0:
        return values
    keep = np.concatenate([[True], np.diff(values) > window])
    # Сравнение с соседом, а не с началом группы: так арпеджированный
    # аккорд, где ноты вступают с шагом 10-15 мс, сливается целиком.
    # Трель не пострадает — у неё шаг 60 мс и больше, далеко за окном.
    result: npt.NDArray[np.float64] = values[keep]
    return result


def refine_note_onsets(
    notes: list[NoteEvent],
    onsets: npt.NDArray[np.float64],
    max_shift: float,
) -> list[NoteEvent]:
    """Подтянуть начала нот к ближайшим атакам звука.

    Нота сдвигается, только если атака не дальше `max_shift`. Значение по
    умолчанию равно допуску MIREX (50 мс): уточнение не должно превращать
    правильную по меркам оценки ноту в неправильную. Конец ноты не
    меняется — затухание звука с отпусканием клавиши связано слабо.

    Сдвигается только время начала: высоту по звуку без транскрипции
    не узнать, и все ноты аккорда притягиваются к одной атаке — что
    и требуется, аккорд должен быть одновременным.

    Args:
        notes: Ноты, например из видеоканала.
        onsets: Моменты атак из [`detect_onsets`][mir.audio.onsets.detect_onsets].
        max_shift: Наибольший допустимый сдвиг, секунды.

    Returns:
        Новый список той же длины и в том же порядке.
    """
    if len(onsets) == 0:
        return list(notes)

    sorted_onsets = np.sort(onsets)
    result: list[NoteEvent] = []
    moved = 0
    for note in notes:
        index = int(np.searchsorted(sorted_onsets, note.onset))
        candidates = sorted_onsets[max(index - 1, 0) : index + 1]
        nearest = float(candidates[np.argmin(np.abs(candidates - note.onset))])
        if abs(nearest - note.onset) <= max_shift and nearest < note.offset:
            result.append(replace(note, onset=nearest))
            moved += int(nearest != note.onset)
        else:
            result.append(note)
    _log.debug("уточнено начало %d нот из %d", moved, len(notes))
    return result
