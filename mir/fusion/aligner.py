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

**Сдвиг не постоянен.** На трёх проверенных роликах он вёл себя по-разному:
у «Лунной сонаты» 114 мс первые пять минут и 86 к концу, у Clair de lune
плавно растёт от 94 до 120 мс (часы двух устройств расходятся примерно на
сто миллионных), у Interstellar ровный (`docs/known-issues.md`, В-07).
Поэтому для слияния сдвиг оценивается в скользящих окнах —
[`estimate_offset_curve`][mir.fusion.aligner.estimate_offset_curve].

**Октава.** У обрезанной клавиатуры видео знает положение ноты внутри
октавы, но не саму октаву. Звук знает высоту точно, и верная октава — та,
при которой каналы совмещаются: [`estimate_pitch_shift`][mir.fusion.aligner.estimate_pitch_shift].
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, replace

import numpy as np
import numpy.typing as npt

from mir.common.logging import get_logger
from mir.common.types import PITCH_MAX, PITCH_MIN, NoteEvent
from mir.config import FusionConfig

__all__ = [
    "OffsetCurve",
    "OffsetEstimate",
    "apply_offset",
    "estimate_av_offset",
    "estimate_av_offset_from_onsets",
    "estimate_offset_curve",
    "estimate_pitch_shift",
    "transpose",
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

_OCTAVE_SHIFTS = (-24, -12, 12, 24)
"""Какие сдвиги высоты видео пробовать, полутоны.

Узор чёрных клавиш повторяется через октаву, поэтому ошибиться видео может
только на целое число октав. Две октавы — с запасом: обрезанная клавиатура
короче трёх октав уже не даёт детектору опознать узор.
"""

_SHIFT_ADVANTAGE = 2.0
"""Во сколько раз больше пар должен собрать сдвиг высоты, чтобы его принять.

Музыка сама повторяется через октаву: при сдвиге на 12 совпадает заметная
часть нот (на «Лунной сонате» сходство валиков 0.30 против 0.68 при нуле).
Двукратный перевес отделяет настоящую ошибку октавы от этого эха.
"""

_ENOUGH_SUPPORT = 0.5
"""Каналы совмещены, если пару нашла хотя бы половина нот видео.

При нулевом сдвиге высоты этого достаточно, чтобы другие не пробовать; от
ненулевого сдвига требуется то же самое, чтобы его принять.
"""


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


def transpose(notes: Sequence[NoteEvent], shift: int) -> list[NoteEvent]:
    """Сдвинуть высоту нот; вышедшие за клавиатуру фортепиано пропадают."""
    if not shift:
        return list(notes)
    return [
        replace(note, pitch=note.pitch + shift)
        for note in notes
        if PITCH_MIN <= note.pitch + shift <= PITCH_MAX
    ]


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


@dataclass(frozen=True, eq=False)
class OffsetCurve:
    """Сдвиг звука относительно картинки как функция времени.

    Между узлами сдвиг интерполируется линейно, за краями держится крайнее
    значение.

    Attributes:
        times: Узлы — время видео, секунды, по возрастанию.
        offsets: Сдвиг в узлах, секунды. Положительный — звук отстаёт.
        overall: Оценка одного сдвига на весь ролик — от неё отсчитаны окна.
        reliable: Каналы удалось совместить. Если нет, кривая нулевая и
            переводить по ней время нельзя.
    """

    times: npt.NDArray[np.float64]
    offsets: npt.NDArray[np.float64]
    overall: OffsetEstimate
    reliable: bool

    def at(self, video_time: float) -> float:
        """Сдвиг в момент `video_time` по шкале видео."""
        return float(np.interp(video_time, self.times, self.offsets))

    def to_audio(self, video_time: float) -> float:
        """Перевести время видео в шкалу звука."""
        return video_time + self.at(video_time)

    def to_video(self, audio_time: float) -> float:
        """Перевести время звука в шкалу видео.

        Сдвиг задан от времени видео, поэтому обратный перевод — два шага
        последовательных приближений. Сдвиг меняется на миллисекунды за
        десятки секунд, и второй шаг уже точен до микросекунд.
        """
        video_time = audio_time - self.at(audio_time)
        return audio_time - self.at(video_time)

    @property
    def drift(self) -> float:
        """Размах сдвига по ролику, секунды."""
        return float(np.ptp(self.offsets)) if len(self.offsets) else 0.0

    @classmethod
    def constant(cls, overall: OffsetEstimate, reliable: bool) -> OffsetCurve:
        """Один сдвиг на весь ролик."""
        offset = overall.offset if reliable else 0.0
        return cls(np.zeros(1), np.full(1, offset), overall, reliable)


def estimate_offset_curve(
    video_notes: list[NoteEvent], audio_notes: list[NoteEvent], config: FusionConfig
) -> OffsetCurve:
    """Оценить сдвиг звука в скользящих окнах по ходу ролика.

    Сначала находится один сдвиг на весь ролик. Если он ненадёжен, каналы
    совместить нельзя, и окна не считаются. Иначе в каждом окне длиной
    `offset_window` (с шагом в половину окна) тем же методом ищется свой
    сдвиг — но только среди разностей не дальше `max_offset_drift` от общего:
    на коротком отрезке ровный ритм легко даёт пик на соседней доле, а так
    далеко от общего сдвига настоящий уйти не может.

    Общий сдвиг считается найденным, если оценка уверенна или если при нём
    пару находит половина нот видео.

    Окно, где пар меньше `min_offset_pairs`, пропускается — его значение
    даёт интерполяция по соседям. Одиночный выброс убирает медиана по трём
    соседним окнам; ступенька (как у «Лунной сонаты») при этом сохраняется.

    Returns:
        Кривую. `reliable=False`, если совместить каналы не удалось.
    """
    overall = estimate_av_offset(video_notes, audio_notes, config)
    # Уверенность оценки низка у ровной, «машинной» музыки: сдвиг на период
    # совпадает почти так же хорошо. Но если при найденном сдвиге пару нашла
    # половина нот видео, каналы описывают одну музыку, и для слияния этого
    # достаточно: даже ошибка на период сопоставит ноту с её же повтором.
    supported = overall.pairs >= _ENOUGH_SUPPORT * len(video_notes)
    if not (overall.is_reliable(config.min_offset_confidence) or supported):
        return OffsetCurve.constant(overall, reliable=False)

    onsets = np.array([note.onset for note in video_notes])
    order = np.argsort(onsets)
    ordered = [video_notes[i] for i in order]
    onsets = onsets[order]
    step = config.offset_window / 2
    times: list[float] = []
    offsets: list[float] = []
    for start in np.arange(onsets[0], onsets[-1] + step, step):
        lo, hi = np.searchsorted(onsets, [start, start + config.offset_window])
        differences = _pitch_differences(ordered[lo:hi], audio_notes, config.max_av_offset)
        differences = differences[np.abs(differences - overall.offset) <= config.max_offset_drift]
        local = _locate_peak(differences, config, "notes")
        if local.pairs < config.min_offset_pairs:
            continue
        near = np.abs(differences - local.offset) <= _SUPPORT_SIGMAS * config.offset_smoothing
        times.append(float(start + config.offset_window / 2))
        offsets.append(float(np.median(differences[near])))

    if not times:
        return OffsetCurve.constant(overall, reliable=True)
    smooth = np.array(offsets)
    if len(smooth) >= 3:
        padded = np.concatenate(([smooth[0]], smooth, [smooth[-1]]))
        smooth = np.median(np.stack([padded[:-2], padded[1:-1], padded[2:]]), axis=0)
    curve = OffsetCurve(np.array(times), smooth, overall, reliable=True)
    _log.info(
        "рассинхрон по окнам: от %+.0f до %+.0f мс, окон %d",
        float(smooth.min()) * 1000,
        float(smooth.max()) * 1000,
        len(times),
    )
    return curve


def estimate_pitch_shift(
    video_notes: Sequence[NoteEvent], audio_notes: list[NoteEvent], config: FusionConfig
) -> int:
    """На сколько полутонов сдвинуть ноты видео, чтобы каналы совместились.

    Обычно ноль. Другие сдвиги — на одну и две октавы — пробуются, только
    если при нуле каналы не совмещаются или пару нашло меньше половины нот
    видео: у обрезанной клавиатуры детектор знает узор чёрных клавиш, но
    октаву угадывает (`KeyboardLayout.octave_candidates`).

    Returns:
        Сдвиг, кратный 12. Ненулевой — только если при нём пару находит
        половина нот видео и вдвое больше, чем без сдвига.
    """
    if not video_notes or not audio_notes:
        return 0
    base = estimate_av_offset(list(video_notes), audio_notes, config)
    reliable = base.is_reliable(config.min_offset_confidence)
    if reliable and base.pairs >= _ENOUGH_SUPPORT * len(video_notes):
        return 0

    best_shift, best_pairs = 0, base.pairs if reliable else 0
    for shift in _OCTAVE_SHIFTS:
        # Ноты, уходящие за клавиатуру, при пробе просто пропадают: одна
        # ложная нота на краю не должна запрещать верный сдвиг.
        estimate = estimate_av_offset(transpose(video_notes, shift), audio_notes, config)
        if estimate.is_reliable(config.min_offset_confidence) and estimate.pairs > best_pairs:
            best_shift, best_pairs = shift, estimate.pairs
    # Сдвиг принимается, только если он совмещает каналы по-настоящему: пару
    # находит половина нот видео и вдвое больше, чем без сдвига. Иначе это
    # случайные совпадения — у двух неродственных мелодий с общей сеткой
    # ритма их набирается на «надёжную» оценку.
    floor = max(
        _SHIFT_ADVANTAGE * (base.pairs if reliable else 0), _ENOUGH_SUPPORT * len(video_notes)
    )
    if best_shift and best_pairs >= floor:
        _log.warning(
            "ноты видео сдвинуты на %+d полутонов: октава клавиатуры определена по звуку",
            best_shift,
        )
        return best_shift
    return 0
