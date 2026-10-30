"""Темп и сетка долей (#27, F-14).

Без темпа секунды не перевести в длительности: нота в 0.5 с — четверть
при 120 BPM и половинная при 60. Сетка долей нужна этапу 5, где к ней
привязывается квантизация.

Основа — динамическое программирование librosa (Эллис, 2007) поверх той
же функции новизны, что и для атак. Своё сверху — три вещи:

* **переменный темп.** У librosa по умолчанию жёсткость 100, и на
  ускорении 90→130 BPM она теряет каждую десятую долю (F = 0.90).
  При 50 доли находятся все, а на ровном темпе результат тот же;
* **ошибка вдвое.** Beat-tracker путает 60 и 120 BPM. Это не ошибка
  алгоритма, а неоднозначность самой музыки: восьмые при 60 и четверти
  при 120 звучат одинаково. Выбор делается по нотной традиции — см.
  [`TempoAnalyzer.analyze`][mir.audio.tempo.TempoAnalyzer.analyze];
* **уверенность.** Доля долей, совпавших с реальными атаками. Низкая
  уверенность — признак rubato, и интерфейс предлагает задать темп руками.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

from mir.audio.onsets import OnsetEnvelope, merge_close
from mir.common.logging import get_logger
from mir.common.types import TempoMap
from mir.config import AudioConfig

__all__ = ["TempoAnalysis", "TempoAnalyzer", "beat_match_rate", "estimate_downbeats"]

_log = get_logger(__name__)

_MIN_INTERVALS = 8
"""Меньше межнотных интервалов — статистика для проверки темпа ненадёжна."""

_NOTE_VALUE_MARGIN = float(np.sqrt(2.0))
"""Середина между соседними длительностями в логарифмической шкале.

Соседние длительности отличаются вдвое, поэтому граница «ближе к этой,
чем к соседней» лежит на √2. Это не подобранное число, а геометрия шкалы.
"""

_ACCENT_WINDOW = 0.05
"""Где искать атаку доли при оценке её силы, ± секунды.

Пик функции новизны отстаёт от атаки на 10-20 мс (см. `config/README.md`),
поэтому значение берётся как максимум в окрестности, а не в точке.
"""

_REFINE_TOLERANCE = 0.01
"""Расхождение оценок темпа, при котором трекинг повторяется.

Половина допуска NF-08 (2 %): меньшее расхождение на результат
не влияет, а лишний проход стоит времени.
"""

_FALLBACK_BPM = 120.0
"""Темп пустой сетки, когда определить его не из чего (тишина, одна нота).

Сетка при этом помечается уверенностью 0, и интерфейс обязан попросить
задать темп вручную — число здесь лишь не даёт конвейеру упасть.
"""


@dataclass
class TempoAnalysis:
    """Результат определения темпа с подробностями для отчёта.

    Attributes:
        tempo: Сетка долей.
        raw_bpm: Темп до исправления ошибки вдвое.
        correction: Во сколько раз темп исправлен: 1, 2 или 0.5.
        median_ioi_beats: Типичный межнотный интервал в долях после
            исправления. По нему видно, почему выбран этот метрический
            уровень.
        variation: Разброс длительности долей относительно средней.
            На ровном темпе — около 0.01, на rubato — 0.1 и больше.
            Только для отчёта: сетка, следующая за rubato, не ошибочна,
            квантизация по ней работает и при плавающем темпе.
        warnings: Предупреждения для пользователя.
    """

    tempo: TempoMap
    raw_bpm: float
    correction: float = 1.0
    median_ioi_beats: float | None = None
    variation: float = 0.0
    warnings: list[str] = field(default_factory=list)


def beat_match_rate(beats: npt.ArrayLike, onsets: npt.ArrayLike, tolerance: float) -> float:
    """Доля долей, рядом с которыми есть атака.

    Мера того, насколько сетка описывает реально звучащее. На ровной
    музыке близка к 1. Падает на rubato (сетка уезжает с атак), на
    синкопах и на длинных паузах — во всех этих случаях автоматическому
    темпу стоит доверять меньше.
    """
    beat_times = np.asarray(beats, dtype=np.float64)
    sorted_onsets = np.sort(np.asarray(onsets, dtype=np.float64))
    if len(beat_times) == 0 or len(sorted_onsets) == 0:
        return 0.0
    last = len(sorted_onsets) - 1
    index = np.searchsorted(sorted_onsets, beat_times)
    left = sorted_onsets[np.clip(index - 1, 0, last)]
    right = sorted_onsets[np.clip(index, 0, last)]
    nearest = np.minimum(np.abs(beat_times - left), np.abs(beat_times - right))
    return float(np.mean(nearest <= tolerance))


def estimate_downbeats(
    beats: npt.ArrayLike, envelope: OnsetEnvelope, beats_per_bar: int
) -> npt.NDArray[np.float64]:
    """Расставить сильные доли при известном числе долей в такте.

    Из `beats_per_bar` возможных сдвигов выбирается тот, при котором на
    первую долю такта в среднем приходится самая сильная атака. Сильная
    доля в фортепианной музыке обычно отмечена басом и аккордом.

    Это предварительная расстановка: размер такта и затакт определяет
    этап 5 (#33, F-16, F-21) и при необходимости пересчитывает сетку.
    """
    beat_times = np.asarray(beats, dtype=np.float64)
    if beats_per_bar <= 1 or len(beat_times) < beats_per_bar:
        return beat_times.copy()
    strength = envelope.accent if envelope.accent is not None else envelope.values
    reach = max(round(_ACCENT_WINDOW * envelope.frame_rate), 1)
    accent = np.array(
        [
            float(np.max(strength[max(f - reach, 0) : f + reach + 1]))
            for f in (envelope.frame_of(t) for t in beat_times)
        ]
    )
    scores = [float(np.mean(accent[phase::beats_per_bar])) for phase in range(beats_per_bar)]
    phase = int(np.argmax(scores))
    result: npt.NDArray[np.float64] = beat_times[phase::beats_per_bar]
    return result


class TempoAnalyzer:
    """Определение темпа и сетки долей.

    Args:
        config: Параметры из секции `[audio]`.
    """

    def __init__(self, config: AudioConfig) -> None:
        self.config = config

    def _track(self, envelope: OnsetEnvelope, bpm: float | None) -> npt.NDArray[np.float64]:
        import librosa

        _, frames = librosa.beat.beat_track(
            onset_envelope=envelope.values,
            sr=envelope.sample_rate,
            hop_length=envelope.hop_length,
            tightness=self.config.beat_tightness,
            bpm=bpm,
            units="frames",
        )
        beats: npt.NDArray[np.float64] = envelope.start + np.asarray(frames) / envelope.frame_rate
        return beats

    def _refine(
        self, envelope: OnsetEnvelope, beats: npt.NDArray[np.float64], used_bpm: float
    ) -> npt.NDArray[np.float64]:
        """Повторить трекинг с темпом, который дали сами найденные доли.

        Начальная оценка темпа берётся по автокорреляции и квантована
        её шагом; на шестнадцатых при 140 BPM она давала 112. Трекер
        с допуском всё равно выходил на верные доли, но часть пути шёл
        с неверным шагом, и средний темп отличался на 1.7 %. Второй проход
        со средним темпом по найденным долям снимает это расхождение.
        Если оценки уже согласованы, второй проход не делается.
        """
        if len(beats) < 3:
            return beats
        observed = 60.0 * (len(beats) - 1) / float(beats[-1] - beats[0])
        if abs(observed / used_bpm - 1.0) <= _REFINE_TOLERANCE:
            return beats
        _log.debug("уточнение темпа: %.1f → %.1f BPM", used_bpm, observed)
        refined = self._track(envelope, observed)
        return refined if len(refined) >= 2 else beats

    @staticmethod
    def _fixed_grid(envelope: OnsetEnvelope, bpm: float) -> npt.NDArray[np.float64]:
        """Ровная сетка с заданным шагом, сдвинутая на самые сильные атаки.

        Для темпа, заданного вручную, доли не должны «плыть»: пользователь
        ввёл 50 BPM и ждёт ровно 50. Трекер с допуском на переменный темп
        утягивал такую сетку к 51.6 BPM вслед за атаками. Поэтому шаг здесь
        жёсткий, а по звуку выбирается только сдвиг.
        """
        period = envelope.frame_rate * 60.0 / bpm
        count = len(envelope.values)
        if count < 2 or period >= count:
            return np.empty(0)
        best_phase, best_score = 0.0, -1.0
        for phase in np.arange(0.0, period, 1.0):
            index = np.round(np.arange(phase, count - 0.5, period)).astype(int)
            score = float(np.mean(envelope.values[np.minimum(index, count - 1)]))
            if score > best_score:
                best_phase, best_score = float(phase), score
        positions = np.arange(best_phase, count - 0.5, period, dtype=np.float64)
        grid: npt.NDArray[np.float64] = envelope.start + positions / envelope.frame_rate
        return grid

    def _global_tempo(self, envelope: OnsetEnvelope) -> float:
        import librosa

        estimate = librosa.feature.tempo(
            onset_envelope=envelope.values,
            sr=envelope.sample_rate,
            hop_length=envelope.hop_length,
            max_tempo=self.config.max_bpm,
        )
        bpm = float(np.atleast_1d(estimate)[0])
        while 0 < bpm < self.config.min_bpm:
            bpm *= 2
        return bpm

    def _octave_correction(self, bpm: float, attacks: npt.ArrayLike) -> tuple[float, float | None]:
        """Выбрать метрический уровень по типичной длительности ноты.

        Межнотный интервал в долях: `m = IOI · BPM / 60`. Удвоение темпа
        удваивает `m`. Если типичная нота при найденном темпе ближе
        к тридцать второй, чем к шестнадцатой, — темп занижен вдвое; если
        ближе к половинной, чем к четверти, — завышен. Граница проходит
        посередине между соседними длительностями в логарифмической шкале,
        то есть с множителем √2. Граница ровно на 1 не годится: метроном
        с щелчком на каждую долю даёт `m ≈ 1.00 ± 0.01`, и решение
        зависело бы от шума.

        Внутри диапазона найденный темп не трогается: и 120 BPM
        с шестнадцатыми, и 240 с восьмыми попадают в него, и предпочесть
        одно другому по звуку нельзя.

        Это нотная традиция, а не измерение, и в записке она так и
        называется. Ошибку, которую она не ловит, правит пользователь
        кнопками «×2» и «÷2» (F-31).

        Returns:
            Исправленный темп и типичный интервал в долях (`None`, если
            интервалов слишком мало для вывода).
        """
        chords = merge_close(attacks, self.config.chord_window)
        intervals = np.diff(chords)
        if len(intervals) < _MIN_INTERVALS:
            return bpm, None
        ioi = float(np.median(intervals))

        low = self.config.tempo_ioi_min_beats / _NOTE_VALUE_MARGIN
        high = self.config.tempo_ioi_max_beats * _NOTE_VALUE_MARGIN
        corrected = bpm
        for _ in range(3):
            beats_per_ioi = ioi * corrected / 60.0
            if beats_per_ioi < low and corrected * 2 <= self.config.max_bpm:
                corrected *= 2
            elif beats_per_ioi > high and corrected / 2 >= self.config.min_bpm:
                corrected /= 2
            else:
                break
        return corrected, ioi * corrected / 60.0

    def analyze(
        self,
        envelope: OnsetEnvelope,
        onsets: npt.NDArray[np.float64],
        attacks: npt.NDArray[np.float64] | None = None,
        force_bpm: float | None = None,
    ) -> TempoAnalysis:
        """Найти темп и доли.

        Args:
            envelope: Функция новизны из
                [`onset_envelope`][mir.audio.onsets.onset_envelope].
            onsets: Найденные атаки — для оценки уверенности.
            attacks: Начала нот для проверки метрического уровня. Лучше
                всего подходят ноты транскрипции: у них меньше ложных атак.
                По умолчанию берутся `onsets`.
            force_bpm: Темп, заданный пользователем (F-31). Сетка всё равно
                выравнивается по атакам, задаётся лишь её шаг.

        Returns:
            Сетка с уверенностью и подробности определения.
        """
        attacks = onsets if attacks is None else attacks
        warnings: list[str] = []

        if force_bpm is not None:
            if force_bpm <= 0:
                raise ValueError(f"force_bpm={force_bpm} должен быть положительным")
            raw_bpm = corrected = float(force_bpm)
            ioi_beats = None
        else:
            raw_bpm = self._global_tempo(envelope) if len(onsets) >= 2 else 0.0
            corrected, ioi_beats = self._octave_correction(raw_bpm, attacks)

        if force_bpm is not None:
            beats = self._fixed_grid(envelope, corrected) if len(onsets) else np.empty(0)
        else:
            beats = self._track(envelope, corrected) if corrected > 0 else np.empty(0)
            beats = self._refine(envelope, beats, corrected)
        if len(beats) < 2:
            return self._fallback(envelope, force_bpm, raw_bpm)

        if corrected != raw_bpm:
            _log.info("темп исправлен: %.1f → %.1f BPM", raw_bpm, corrected)

        confidence = beat_match_rate(beats, onsets, self.config.beat_match_tolerance)
        if confidence < self.config.min_tempo_confidence and force_bpm is None:
            warnings.append(
                f"Темп определён неуверенно ({confidence:.0%} долей совпали с атаками). "
                "Возможно, исполнение свободное (rubato). Проверьте темп и при "
                "необходимости задайте его вручную"
            )

        per_bar = self.config.default_beats_per_bar
        tempo = TempoMap(
            beats=beats,
            downbeats=estimate_downbeats(beats, envelope, per_bar),
            beats_per_bar=per_bar,
            confidence=confidence,
            is_manual=force_bpm is not None,
        )
        _log.info(
            "темп %.1f BPM (сетка из %d долей), уверенность %.2f",
            tempo.bpm,
            len(beats),
            confidence,
        )
        intervals = np.diff(beats)
        return TempoAnalysis(
            tempo=tempo,
            raw_bpm=raw_bpm,
            correction=corrected / raw_bpm if raw_bpm else 1.0,
            median_ioi_beats=ioi_beats,
            variation=float(np.std(intervals) / np.mean(intervals)),
            warnings=warnings,
        )

    def _fallback(
        self, envelope: OnsetEnvelope, force_bpm: float | None, raw_bpm: float
    ) -> TempoAnalysis:
        """Сетка, когда долей не нашлось: тишина или одиночные звуки."""
        duration = len(envelope.values) / envelope.frame_rate
        bpm = force_bpm or _FALLBACK_BPM
        tempo = TempoMap.constant(
            bpm,
            duration=envelope.start + duration,
            start=envelope.start,
            beats_per_bar=self.config.default_beats_per_bar,
            is_manual=force_bpm is not None,
        )
        if force_bpm is None:
            tempo = TempoMap(
                beats=tempo.beats,
                downbeats=tempo.downbeats,
                beats_per_bar=tempo.beats_per_bar,
                confidence=0.0,
                is_manual=False,
            )
        message = (
            "Не удалось определить темп: в записи слишком мало атак. Задайте темп вручную"
            if force_bpm is None
            else "Доли не выровнены по звуку: в записи слишком мало атак"
        )
        _log.warning(message)
        return TempoAnalysis(tempo=tempo, raw_bpm=raw_bpm, warnings=[message])
