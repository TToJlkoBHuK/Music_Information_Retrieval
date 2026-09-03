"""Автокалибровка под конкретный визуализатор (F-12).

Ролики делают в разных программах: цветовые схемы, свечение, частицы,
водяные знаки. Жёсткие пороги, подогнанные под один Synthesia, на соседнем
ролике развалятся, поэтому параметры выводятся из первых секунд записи.

Определяются четыре вещи: цвет фона, палитра блоков, скорость падения
и линия касания клавиатуры.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from itertools import pairwise

import cv2
import numpy as np
import numpy.typing as npt

from mir.common.logging import get_logger
from mir.common.types import Frame, KeyboardLayout

__all__ = ["VisualizerProfile", "calibrate"]

_log = get_logger(__name__)

_MAX_BLOCK_COLORS = 4
"""Больше четырёх цветов блоков не встречается: обычно две руки, редко голоса."""

_MIN_CLUSTER_SHARE = 0.05
"""Кластер меньше этой доли пикселей — шум или эффект, а не партия."""

_MIN_HUE_SEPARATION = 12.0
"""Насколько должны различаться тона, чтобы считаться разными партиями.

В шкале OpenCV это 24° настоящего тона. Кластеризация всегда возвращает
запрошенное число центров, поэтому на одноцветной схеме она разделит
блок и его сглаженный край и создаст видимость двух рук. Сжатие сдвигает
тон на единицы, а типичная пара «зелёный и оранжевый» расходится на 45 —
запас достаточный.
"""

_MIN_HUE_SATURATION = 40.0
"""Ниже этой насыщенности тон не несёт сведений и сравнивать его нельзя."""


@dataclass(frozen=True)
class VisualizerProfile:
    """Параметры конкретного ролика.

    Attributes:
        background_hsv: Цвет фона зоны блоков.
        block_colors_hsv: Цвета блоков, по убыванию занимаемой площади.
        fall_speed: Скорость падения блоков, пикселей в секунду.
        hit_line_y: Y-координата касания клавиатуры.
        confidence: Насколько уверенно определены параметры.
    """

    background_hsv: tuple[float, float, float]
    block_colors_hsv: tuple[tuple[float, float, float], ...] = ()
    fall_speed: float = 0.0
    hit_line_y: int = 0
    confidence: float = 0.0
    warnings: tuple[str, ...] = field(default_factory=tuple)

    @property
    def has_hand_colors(self) -> bool:
        """Можно ли разделить руки по цвету.

        При одном цвете на весь ролик разделение придётся делать
        эвристиками по высоте и голосоведению.
        """
        return len(self.block_colors_hsv) >= 2


def _background_color(
    hsv_frames: Sequence[Frame], region_height: int
) -> tuple[float, float, float]:
    """Мода цвета в верхней части зоны блоков.

    Сверху блоки появляются реже всего, поэтому там почти всегда фон.
    """
    samples = np.concatenate([f[: max(region_height // 4, 1)].reshape(-1, 3) for f in hsv_frames])
    quantized = (samples // 8).astype(np.int32)
    codes = quantized[:, 0] * 1024 + quantized[:, 1] * 32 + quantized[:, 2]
    values, counts = np.unique(codes, return_counts=True)
    dominant = values[counts.argmax()]
    return (
        float((dominant // 1024) * 8),
        float(((dominant % 1024) // 32) * 8),
        float((dominant % 32) * 8),
    )


def _circular_distance(first: float, second: float) -> float:
    """Расстояние между тонами с учётом замыкания шкалы."""
    diff = abs(first - second)
    return min(diff, 180.0 - diff)


def _hue_peaks(hues: npt.NDArray[np.int32]) -> list[int]:
    """Найти вершины гистограммы тонов, разнесённые достаточно далеко.

    K-средних здесь не годится: метод стохастический и на одних и тех же
    кадрах от запуска к запуску то разделял две руки, то сливал их
    в один усреднённый цвет. Гистограмма по тону детерминирована,
    объяснима и опирается на то, чем руки в визуализаторах и отличаются.
    """
    histogram = np.bincount(hues, minlength=180).astype(np.float64)

    # Сглаживание круговое: тон замкнут, и вершина у нуля не должна
    # обрезаться краем массива.
    kernel = np.ones(5) / 5.0
    smoothed = np.convolve(np.concatenate([histogram] * 3), kernel, mode="same")[180:360]

    order = np.argsort(-smoothed)
    peaks: list[int] = []
    for candidate in order:
        if smoothed[candidate] <= 0:
            break
        if all(
            _circular_distance(float(candidate), float(p)) >= _MIN_HUE_SEPARATION for p in peaks
        ):
            peaks.append(int(candidate))
        if len(peaks) == _MAX_BLOCK_COLORS:
            break
    return peaks


def _block_colors(
    hsv_frames: Sequence[Frame],
    background: tuple[float, float, float],
) -> tuple[tuple[tuple[float, float, float], ...], float]:
    """Выделить цвета блоков среди непохожих на фон пикселей.

    Returns:
        Цвета по убыванию занимаемой площади и доля крупнейшего из них.
    """
    samples = np.concatenate([f.reshape(-1, 3) for f in hsv_frames]).astype(np.float32)
    distance = np.abs(samples - np.array(background, dtype=np.float32)).sum(axis=1)
    foreground = samples[distance > 60]

    if len(foreground) < 100:
        return (), 0.0

    # Тон различает партии только у насыщенных пикселей; сглаженные края
    # и полупрозрачное свечение сюда не попадают.
    coloured = foreground[foreground[:, 1] >= _MIN_HUE_SATURATION]
    if len(coloured) < 100:
        mean = foreground.mean(axis=0)
        return ((float(mean[0]), float(mean[1]), float(mean[2])),), 1.0

    hues = coloured[:, 0].astype(np.int32) % 180
    peaks = _hue_peaks(hues)
    if not peaks:
        return (), 0.0

    # Каждый пиксель отходит ближайшей вершине; редкие группы отсеиваются
    # как блики и остатки фона, а не как партия.
    peak_array = np.array(peaks, dtype=np.float32)
    diff = np.abs(hues[:, None] - peak_array[None, :])
    labels = np.minimum(diff, 180.0 - diff).argmin(axis=1)
    shares = np.bincount(labels, minlength=len(peaks)) / len(labels)

    colors: list[tuple[float, float, float]] = []
    kept_shares: list[float] = []
    for index in np.argsort(-shares):
        if shares[index] < _MIN_CLUSTER_SHARE:
            continue
        group = coloured[labels == index]
        # Тон усредняется относительно вершины: около нуля прямое
        # среднее дало бы 90 вместо 0.
        offsets = (group[:, 0] - peaks[index] + 90.0) % 180.0 - 90.0
        colors.append(
            (
                float((peaks[index] + offsets.mean()) % 180.0),
                float(group[:, 1].mean()),
                float(group[:, 2].mean()),
            )
        )
        kept_shares.append(float(shares[index]))

    if not colors:
        return (), 0.0
    return tuple(colors), kept_shares[0]


def _fall_speed(gray_frames: Sequence[Frame], fps: float) -> tuple[float, float]:
    """Оценить скорость падения фазовой корреляцией соседних кадров.

    Блоки движутся строго вертикально с постоянной скоростью, поэтому
    сдвиг между кадрами — прямая мера скорости. Берётся медиана по парам:
    отдельные кадры могут дать выброс из-за появления новых блоков.
    """
    shifts: list[float] = []
    for previous, current in pairwise(gray_frames):
        (_, dy), response = cv2.phaseCorrelate(
            previous.astype(np.float64), current.astype(np.float64)
        )
        if response > 0.05 and dy > 0:
            shifts.append(dy)

    if not shifts:
        return 0.0, 0.0

    speed = float(np.median(shifts)) * fps
    spread = float(np.std(shifts) / (np.median(shifts) + 1e-6))
    return speed, max(0.0, 1.0 - spread)


def calibrate(
    frames: Sequence[Frame],
    layout: KeyboardLayout,
    fps: float,
) -> VisualizerProfile:
    """Определить параметры визуализатора.

    Args:
        frames: Подряд идущие кадры из начала ролика. Именно подряд —
            иначе не измерить скорость падения.
        layout: Разметка клавиатуры: её верхняя граница и есть линия касания.
        fps: Частота кадров.

    Returns:
        Профиль визуализатора.
    """
    hit_line = layout.bbox[1]
    block_region = [f[:hit_line] for f in frames if hit_line > 1]

    if not block_region:
        return VisualizerProfile(
            background_hsv=(0.0, 0.0, 0.0),
            hit_line_y=hit_line,
            warnings=("зона падающих блоков пуста",),
        )

    hsv_frames = [cv2.cvtColor(f, cv2.COLOR_BGR2HSV) for f in block_region]
    gray_frames = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in block_region]

    background = _background_color(hsv_frames, hit_line)
    colors, colour_share = _block_colors(hsv_frames, background)
    speed, speed_confidence = _fall_speed(gray_frames, fps)

    warnings: list[str] = []
    if speed <= 0:
        warnings.append("не удалось измерить скорость падения блоков")
    if len(colors) < 2:
        warnings.append("блоки одного цвета: руки придётся разделять по высоте")

    _log.info(
        "профиль: фон HSV%s, цветов блоков %d, скорость %.0f px/с",
        tuple(round(c) for c in background),
        len(colors),
        speed,
    )

    return VisualizerProfile(
        background_hsv=background,
        block_colors_hsv=colors,
        fall_speed=speed,
        hit_line_y=hit_line,
        confidence=min(colour_share + 0.3, 1.0) * max(speed_confidence, 0.1),
        warnings=tuple(warnings),
    )
