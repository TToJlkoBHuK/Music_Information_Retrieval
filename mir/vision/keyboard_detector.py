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

    def _find_band(self, gray: Frame) -> _Band:
        """Найти горизонтальную полосу клавиатуры.

        Признак — обилие вертикальных границ: у клавиатуры их десятки
        на строку, у зоны падающих блоков единицы.
        """
        height = gray.shape[0]
        edges = np.abs(cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3))
        per_row = edges.mean(axis=1)

        threshold = per_row.mean() + per_row.std() * 0.5
        rows_with_pattern = per_row > threshold

        bottom = height - 1
        top = bottom
        while top > 0 and rows_with_pattern[top]:
            top -= 1

        if bottom - top < height * 0.05:
            raise KeyboardNotFoundError(
                f"полоса клавиатуры слишком узкая: {bottom - top} пикселей",
                "В этом видео не найдена фортепианная клавиатура",
            )

        confidence = float(rows_with_pattern[top:bottom].mean())
        return _Band(top=top + 1, bottom=bottom, confidence=confidence)

    def _find_white_boundaries(self, gray: Frame, band: _Band) -> list[int]:
        """Найти вертикальные границы белых клавиш.

        Берётся нижняя треть клавиатуры: чёрные клавиши туда не достают,
        поэтому видны только границы между белыми.
        """
        strip_top = band.bottom - max(2, (band.bottom - band.top) // 3)
        strip = gray[strip_top : band.bottom, :].astype(np.float32)
        profile = np.abs(cv2.Sobel(strip, cv2.CV_32F, 1, 0, ksize=3)).mean(axis=0)

        threshold = profile.max() * 0.3
        peaks: list[int] = []
        for x in range(1, len(profile) - 1):
            if profile[x] < threshold:
                continue
            if (
                profile[x] >= profile[x - 1]
                and profile[x] >= profile[x + 1]
                and (not peaks or x - peaks[-1] > self.config.min_key_width_px)
            ):
                peaks.append(x)

        if len(peaks) < 8:
            raise KeyboardNotFoundError(
                f"найдено лишь {len(peaks)} границ клавиш",
                "В этом видео не найдена фортепианная клавиатура",
            )
        return peaks

    def _detect_black_keys(self, gray: Frame, band: _Band, boundaries: list[int]) -> list[bool]:
        """Для каждой белой клавиши определить, есть ли справа чёрная.

        Проверяется верхняя часть клавиатуры на границе между белыми:
        тёмное пятно означает чёрную клавишу.
        """
        strip_bottom = band.top + int(
            (band.bottom - band.top) * self.config.black_key_height_ratio * 0.5
        )
        strip = gray[band.top : max(strip_bottom, band.top + 2), :]
        column_brightness = strip.mean(axis=0)
        dark_threshold = (column_brightness.max() + column_brightness.min()) / 2

        flags: list[bool] = []
        for index in range(len(boundaries) - 1):
            edge = boundaries[index + 1]
            window = column_brightness[max(edge - 2, 0) : edge + 3]
            flags.append(bool(window.size and window.mean() < dark_threshold))
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
