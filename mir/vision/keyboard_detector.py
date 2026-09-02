"""Автоматическая детекция клавиатуры в кадре (F-06, F-07).

Ручная калибровка — главный барьер открытых аналогов: пользователь обязан
сам выровнять сетку клавиш по картинке. Здесь клавиатура находится сама.

Порядок работы:

1. медианный кадр по выборке — подсветка и падающие блоки исчезают,
   остаётся «пустая» клавиатура;
2. поиск горизонтальной полосы с регулярным вертикальным узором;
3. выделение чёрных клавиш по яркости;
4. привязка к абсолютным нотам по узору групп из двух и трёх чёрных клавиш.

Четвёртый шаг — ключевой: он даёт правильные ноты даже когда в кадре
видна лишь часть клавиатуры (F-07).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import cv2
import numpy as np
import numpy.typing as npt

from mir.common.errors import KeyboardNotFoundError
from mir.common.logging import get_logger
from mir.common.types import PITCH_MAX, PITCH_MIN, Frame, KeyboardLayout
from mir.config import KeyboardConfig
from mir.vision import accel
from mir.vision.keyboard_geometry import KeyboardGeometry, is_black_key

__all__ = ["KeyboardDetector", "build_median_frame", "match_black_pattern"]

_log = get_logger(__name__)

_WHITE_PER_OCTAVE = 7
_BLACK_AFTER_WHITE: tuple[bool, ...] = (True, True, False, True, True, True, False)
"""Есть ли чёрная клавиша справа от белой ступени: до, ре, ми, фа, соль, ля, си.

Именно этот узор — 2 чёрные, пропуск, 3 чёрные, пропуск — позволяет
однозначно определить, какая из найденных белых клавиш является «до».
"""

_EXTENT_WINDOW_PERIODS = 2.5
"""Ширина окна поиска краёв клавиатуры в шагах решётки.

Должна заведомо превышать ширину клавиши, иначе окно, не накрывшее
ни одного стыка, обнуляет плато и рвёт клавиатуру на куски.
"""

_HARMONIC_TOLERANCE = 0.95
"""Насколько составляющая спектра должна быть близка к сильнейшей,
чтобы считаться той же периодичностью, а не случайным всплеском."""

_EXTENT_GAP_PERIODS = 12.0
"""Разрыв какой ширины считается перекрытием, а не краем клавиатуры.

Руки исполнителя закрывают до десятка клавиш подряд. Разрыв уже этого
затягивается, шире — принимается за настоящий край.
"""

_BLACK_KEY_DARKNESS = 0.8
"""Во сколько раз граница должна быть темнее соседних белых клавиш.

Сравнение относительное и локальное: тёмная тема, виньетка и неравномерная
подсветка меняют абсолютную яркость вдоль клавиатуры в разы, а отношение
«граница к соседям» остаётся тем же.
"""

_MIN_PATTERN_MARGIN = 0.1
"""Минимальный отрыв верного сдвига от остальных.

У чистой клавиатуры отрыв около 0.57. Порог занижен намеренно: узор
TTFTTTF заметно похож на себя, сдвинутый на четыре позиции, и одна
ошибка сегментации сокращает отрыв до 0.19. Задача проверки — отсечь
вырожденные узоры с нулевым отрывом, а не наказывать за одну ошибку.
"""


def build_median_frame(
    frames: Sequence[Frame],
) -> Frame:
    """Попиксельная медиана по кадрам.

    Подсветка клавиш и падающие блоки появляются лишь в части кадров,
    поэтому медиана оставляет неподвижную клавиатуру.

    Raises:
        ValueError: Список кадров пуст.
    """
    return accel.median_frame(frames)


def match_black_pattern(has_black_after: list[bool]) -> tuple[int, float]:
    """Определить, какая белая клавиша соответствует ноте «до».

    Args:
        has_black_after: Для каждой найденной белой клавиши — есть ли
            справа от неё чёрная.

    Returns:
        Индекс белой клавиши, являющейся «до», и доля совпавших позиций.

    Example:
        >>> match_black_pattern([True, True, False, True, True, True, False])
        (0, 1.0)
    """
    if not has_black_after:
        return 0, 0.0

    scores = [
        sum(
            observed == _BLACK_AFTER_WHITE[(index + shift) % _WHITE_PER_OCTAVE]
            for index, observed in enumerate(has_black_after)
        )
        / len(has_black_after)
        for shift in range(_WHITE_PER_OCTAVE)
    ]
    best_shift = max(range(_WHITE_PER_OCTAVE), key=scores.__getitem__)

    # Одной доли совпадений мало: узор из одних «есть чёрная справа»
    # совпадает с эталоном на 5/7 при любом сдвиге и прошёл бы порог.
    # Признак настоящей клавиатуры — отрыв лучшего сдвига от остальных:
    # у верного он около 0.57, у вырожденного ноль.
    runner_up = max(score for shift, score in enumerate(scores) if shift != best_shift)
    margin = scores[best_shift] - runner_up
    confidence = scores[best_shift] * min(1.0, margin / _MIN_PATTERN_MARGIN)

    # shift — позиция первой белой клавиши внутри октавы; «до» отстоит
    # от неё на столько же шагов назад
    return (_WHITE_PER_OCTAVE - best_shift) % _WHITE_PER_OCTAVE, confidence


@dataclass(frozen=True)
class _Band:
    """Найденная полоса клавиатуры."""

    top: int
    bottom: int
    confidence: float


class KeyboardDetector:
    """Поиск клавиатуры и построение карты клавиш.

    Args:
        config: Пороговые значения детекции.

    Example:
        >>> detector = KeyboardDetector()      # doctest: +SKIP
        >>> layout = detector.detect(frames)   # doctest: +SKIP
    """

    def __init__(self, config: KeyboardConfig | None = None) -> None:
        self.config = config or KeyboardConfig()

    def detect(self, frames: Sequence[Frame]) -> KeyboardLayout:
        """Найти клавиатуру и разметить клавиши.

        Args:
            frames: Кадры из разных мест ролика.

        Returns:
            Разметка с привязкой к абсолютным нотам.

        Raises:
            KeyboardNotFoundError: Клавиатура не найдена — вероятно,
                ролик не является piano visualizer.
        """
        median = build_median_frame(frames)
        gray = cv2.cvtColor(median, cv2.COLOR_BGR2GRAY)

        band = self._find_band(gray)
        boundaries = self._find_white_boundaries(gray, band)
        black_flags = self._detect_black_keys(gray, band, boundaries)

        shift, pattern_score = match_black_pattern(black_flags)
        _log.info(
            "клавиатура: y=%d..%d, белых клавиш %d, узор совпал на %.0f%%",
            band.top,
            band.bottom,
            len(boundaries) - 1,
            pattern_score * 100,
        )

        if pattern_score < 0.7:
            raise KeyboardNotFoundError(
                f"узор чёрных клавиш совпал лишь на {pattern_score:.0%}",
                "Не удалось разобрать клавиатуру. Возможно, ролик снят "
                "под углом или это не piano visualizer",
            )

        return self._build_layout(band, boundaries, shift, pattern_score)

    @staticmethod
    def _runs(marked: npt.NDArray[np.bool_]) -> list[tuple[int, int]]:
        """Непрерывные участки подряд идущих True как пары (начало, конец)."""
        padded = np.concatenate(([False], marked, [False]))
        edges = np.flatnonzero(padded[1:] != padded[:-1])
        return list(zip(edges[::2], edges[1::2], strict=True))

    def _find_band(self, gray: Frame) -> _Band:
        """Найти горизонтальную полосу клавиатуры.

        Признак — обилие вертикальных границ: у клавиатуры их десятки
        на строку, у зоны падающих блоков единицы.

        Полоса ищется как самый длинный непрерывный участок таких строк,
        а не отсчитывается от нижнего края кадра. Упереться в низ она
        не обязана: у реальных роликов там оказываются то тень от клавиш,
        то полоса педали, то чёрные поля от несовпадения пропорций.
        Прежний вариант на таком кадре не находил ничего вовсе.
        """
        height = gray.shape[0]
        edges = np.abs(cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3))
        per_row = edges.mean(axis=1)

        threshold = per_row.mean() + per_row.std() * 0.5
        rows_with_pattern = per_row > threshold

        min_height = max(int(height * 0.05), 4)
        runs = [(a, b) for a, b in self._runs(rows_with_pattern) if b - a >= min_height]

        if not runs:
            raise KeyboardNotFoundError(
                f"нет ни одной полосы с регулярным узором выше {min_height} пикселей",
                "В этом видео не найдена фортепианная клавиатура",
            )

        # Клавиатура у визуализаторов всегда в нижней части кадра, поэтому
        # верхние участки рассматриваются только когда других нет: сверху
        # такой же плотный узор дают титры и ряды падающих блоков.
        lower = [(a, b) for a, b in runs if (a + b) / 2 > height * 0.5]
        top, bottom = max(lower or runs, key=lambda run: run[1] - run[0])

        confidence = float(np.clip(per_row[top:bottom].mean() / (threshold + 1e-6) - 1.0, 0.0, 1.0))
        return _Band(top=int(top), bottom=int(bottom), confidence=confidence)

    @staticmethod
    def _gap_profile(gray: Frame, band: _Band) -> Frame:
        """Профиль темноты по колонкам: пик приходится на стык клавиш.

        Для оценки шага решётки берётся именно темнота, а не модуль
        градиента: у стыка шириной больше пикселя градиент даёт два
        всплеска — на входе в тёмную линию и на выходе, — и период
        определяется вдвое меньше настоящего.
        """
        strip = gray[band.bottom - max(2, (band.bottom - band.top) // 3) : band.bottom, :]
        brightness = strip.astype(np.float32).mean(axis=0)
        darkness: Frame = brightness.max() - brightness
        return darkness

    @staticmethod
    def _edge_profile(gray: Frame, band: _Band) -> Frame:
        """Насыщенность вертикальными границами по колонкам.

        Берётся нижняя треть полосы: чёрные клавиши туда не достают,
        поэтому всплески отмечают только стыки между белыми.
        """
        strip = gray[band.bottom - max(2, (band.bottom - band.top) // 3) : band.bottom, :]
        edges: Frame = np.abs(cv2.Sobel(strip.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3))
        profile: Frame = edges.mean(axis=0)
        return profile

    def _keyboard_extent(self, profile: Frame, period: float) -> tuple[int, int]:
        """Левый и правый край клавиатуры по горизонтали.

        Границей служит не яркость, а насыщенность вертикальными краями:
        внутри клавиатуры они идут регулярно, за её пределами фон ровный.
        Яркость для этого не годится — тонкие тёмные стыки между белыми
        клавишами рвут светлую область на полсотни кусков, и самый длинный
        из них оказывается шириной в одну клавишу.

        Окно расширения привязано к найденному шагу клавиш, а не к ширине
        кадра: у обрезанной клавиатуры клавиши широкие, и окно постоянной
        доли кадра оказывалось уже одной клавиши — плато не складывалось.
        """
        window = max(int(period * _EXTENT_WINDOW_PERIODS) | 1, 9)
        plateau = cv2.dilate(profile.reshape(1, -1), np.ones((1, window), dtype=np.uint8)).ravel()

        # Разрывы шириной в несколько клавиш закрываются: руки исполнителя
        # перекрывают стыки, и без этого самым длинным участком оказывался
        # кусок клавиатуры сбоку от рук.
        gap = max(int(period * _EXTENT_GAP_PERIODS) | 1, window)
        mask = (plateau > plateau.max() * 0.15).astype(np.uint8).reshape(1, -1)
        closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((1, gap), dtype=np.uint8))

        runs = self._runs(closed.ravel() > 0)
        if not runs:
            return 0, len(profile)
        left, right = max(runs, key=lambda run: run[1] - run[0])
        return int(left), int(right)

    def _find_white_boundaries(self, gray: Frame, band: _Band) -> list[int]:
        """Построить сетку границ белых клавиш.

        Границы не отбираются по порогу яркости, а восстанавливаются как
        периодическая решётка. Пороговый отбор на реальном ролике терял
        треть клавиш: руки исполнителя, блики подсветки и сжатие съедают
        часть границ, и разброс ширины доходил до шестикратного. Клавиши
        же равномерны по построению, поэтому период ищется преобразованием
        Фурье, а сетка достраивается на всю ширину клавиатуры — включая
        участки, закрытые руками.
        """
        profile = self._edge_profile(gray, band)
        period, phase = self._estimate_period(self._gap_profile(gray, band))
        if period <= 0:
            raise KeyboardNotFoundError(
                "не удалось определить шаг клавиш",
                "В этом видео не найдена фортепианная клавиатура",
            )

        left, right = self._keyboard_extent(profile, period)

        # Счёт ведётся с запасом в одну позицию по обе стороны: фаза может
        # оказаться близка к целому периоду, и крайняя клавиша тогда
        # выпадала бы из сетки, укорачивая диапазон на ноту.
        count = int((right - left) / period) + 2
        first = int(np.floor((left - phase) / period))
        candidates = (phase + (first + np.arange(-1, count + 1)) * period).round().astype(int)
        margin = period * 0.25
        grid = [
            int(np.clip(x, 0, gray.shape[1] - 1))
            for x in candidates
            if left - margin <= x <= right + margin
        ]

        if len(grid) < 8:
            raise KeyboardNotFoundError(
                f"найдено лишь {len(grid)} границ клавиш",
                "В этом видео не найдена фортепианная клавиатура",
            )

        _log.debug("сетка клавиш: шаг %.1f px, границ %d", period, len(grid))
        return grid

    def _estimate_period(self, profile: Frame) -> tuple[float, float]:
        """Шаг и смещение решётки клавиш.

        Профиль вертикальных границ — почти чистая периодическая волна:
        всплеск на каждом стыке белых клавиш. Её частота находится как
        наибольшая по амплитуде составляющая спектра в диапазоне
        допустимых ширин клавиши, а фаза — как аргумент той же
        составляющей. Отдельные пропущенные или лишние всплески на такую
        оценку почти не влияют: они размазываются по всему спектру.
        """
        centered = profile - profile.mean()
        spectrum = np.fft.rfft(centered)
        freqs = np.fft.rfftfreq(len(centered))

        min_period = max(self.config.min_key_width_px, 4.0)
        max_period = len(centered) / 6
        usable = (freqs >= 1.0 / max_period) & (freqs <= 1.0 / min_period)
        if not usable.any():
            return 0.0, 0.0

        magnitudes = np.where(usable, np.abs(spectrum), 0.0)
        peak = magnitudes.max()
        if peak <= 0:
            return 0.0, 0.0

        # Профиль стыков — почти гребёнка узких импульсов, а у неё все
        # гармоники сопоставимы по амплитуде. Простой максимум спектра
        # поэтому случайно попадает на вторую или третью гармонику и
        # занижает шаг вдвое-втрое. Из сопоставимых по силе составляющих
        # берётся самая низкая по частоте — она и есть основная.
        candidates = np.flatnonzero(magnitudes >= peak * _HARMONIC_TOLERANCE)
        index = int(candidates.min())
        period = float(1.0 / freqs[index])

        # Волна имеет вид cos(2*pi*x/period + arg), максимумы приходятся
        # на x = -arg * period / (2*pi) с шагом period.
        phase = float(-np.angle(spectrum[index]) * period / (2 * np.pi)) % period
        return period, phase

    def _detect_black_keys(self, gray: Frame, band: _Band, boundaries: list[int]) -> list[bool]:
        """Для каждой белой клавиши определить, есть ли справа чёрная.

        Яркость на границе сравнивается с яркостью в центрах двух соседних
        белых клавиш, а не с общим порогом по всей клавиатуре. Общий порог
        разваливается на тёмной теме: одна подсвеченная клавиша задирает
        максимум, и почти каждая граница оказывается «темнее среднего» —
        на реальном ролике узор выходил из одних сплошных чёрных.
        """
        strip_bottom = band.top + int(
            (band.bottom - band.top) * self.config.black_key_height_ratio * 0.5
        )
        strip = gray[band.top : max(strip_bottom, band.top + 2), :]
        column_brightness = strip.mean(axis=0).astype(np.float32)
        width = len(column_brightness)

        def around(center: float, half: float) -> float:
            low = max(int(center - half), 0)
            high = min(int(center + half) + 1, width)
            window = column_brightness[low:high]
            return float(window.mean()) if window.size else 0.0

        flags: list[bool] = []
        for index in range(len(boundaries) - 1):
            edge = boundaries[index + 1]
            key_width = boundaries[index + 1] - boundaries[index]
            half = max(key_width * 0.15, 1.0)

            at_edge = around(edge, half)
            neighbours = (
                around(edge - key_width * 0.5, half) + around(edge + key_width * 0.5, half)
            ) / 2
            flags.append(at_edge < neighbours * _BLACK_KEY_DARKNESS)
        return flags

    def _build_layout(
        self, band: _Band, boundaries: list[int], shift: int, score: float
    ) -> KeyboardLayout:
        """Собрать разметку, привязав найденные клавиши к нотам."""
        white_count = len(boundaries) - 1
        octave = self._guess_octave(white_count)
        lowest = octave * 12 + [0, 2, 4, 5, 7, 9, 11][(_WHITE_PER_OCTAVE - shift) % 7]
        lowest = max(PITCH_MIN, min(lowest, PITCH_MAX))

        while is_black_key(lowest):
            lowest += 1

        highest = lowest
        remaining = white_count - 1
        while remaining > 0 and highest < PITCH_MAX:
            highest += 1
            if not is_black_key(highest):
                remaining -= 1

        geometry = KeyboardGeometry(
            x=boundaries[0],
            width=boundaries[-1] - boundaries[0],
            lowest_pitch=lowest,
            highest_pitch=highest,
        )
        return geometry.build_layout(
            y=band.top,
            height=band.bottom - band.top,
            confidence=min(band.confidence, score),
        )

    @staticmethod
    def _guess_octave(white_count: int) -> int:
        """Оценить октаву нижней клавиши по числу видимых белых.

        Полная клавиатура — 52 белые клавиши от ля субконтроктавы.
        Обрезанные ролики обычно центрированы, поэтому недостающие клавиши
        делятся поровну между краями.
        """
        if white_count >= 50:
            return 1
        missing = 52 - white_count
        return 1 + max(0, missing // 2) // _WHITE_PER_OCTAVE + 1
