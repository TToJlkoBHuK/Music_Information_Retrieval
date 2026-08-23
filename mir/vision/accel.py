"""Мост к нативному ядру `mir_core` с запасной реализацией на numpy.

Ядро на C++ ускоряет покадровый разбор, но требует сборки. Обязательным
его делать нельзя: тогда проект не запустится там, где нет компилятора.
Поэтому здесь единый интерфейс, а выбор реализации происходит при импорте.

Обе ветки обязаны давать совпадающий результат — это проверяется тестом
`tests/unit/test_accel.py`, который сравнивает их на общих данных.

Сборка ядра:

```
cmake -B core/build -S core -DCMAKE_BUILD_TYPE=Release
cmake --build core/build --config Release
```
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Final, cast

import numpy as np
import numpy.typing as npt

from mir.common.logging import get_logger
from mir.common.types import Frame

__all__ = [
    "HAS_NATIVE",
    "DeviationWeights",
    "backend_name",
    "key_deviations",
    "median_frame",
    "sample_regions",
]

_log = get_logger(__name__)

SAMPLE_INSET: Final = 0.25
"""Отступ от краёв клавиши при взятии пробы цвета, доля ширины."""

try:
    from mir.vision import mir_core  # type: ignore[attr-defined]

    _native: Any = mir_core
    HAS_NATIVE = True
except ImportError:  # pragma: no cover - зависит от наличия собранного ядра
    _native = None
    HAS_NATIVE = False


class DeviationWeights:
    """Веса каналов HSV при сравнении цвета с эталоном.

    Тон весит больше яркости: подсветка меняет именно цвет, тогда как блики
    и затемнение трогают только яркость.
    """

    __slots__ = ("hue", "saturation", "value")

    def __init__(self, hue: float = 0.5, saturation: float = 0.35, value: float = 0.15) -> None:
        self.hue = hue
        self.saturation = saturation
        self.value = value


def backend_name() -> str:
    """Какая реализация используется: `mir_core (C++)` или `numpy`."""
    return "mir_core (C++)" if HAS_NATIVE else "numpy"


def median_frame(frames: Sequence[Frame]) -> Frame:
    """Попиксельная медиана стопки кадров.

    Args:
        frames: Кадры одинакового размера.

    Returns:
        Кадр, где подсветка и падающие блоки убраны: они не задерживаются
        на месте, а клавиатура неподвижна.

    Raises:
        ValueError: Список кадров пуст.
    """
    if not frames:
        raise ValueError("нужен хотя бы один кадр")
    if HAS_NATIVE:
        return cast(Frame, _native.median_frame(frames))
    median: Frame = np.median(np.stack(frames), axis=0).astype(np.uint8)
    return median


def sample_regions(
    hsv: Frame,
    regions: npt.NDArray[np.int32],
    inset: float = SAMPLE_INSET,
) -> npt.NDArray[np.float32]:
    """Средние цвета HSV по прямоугольным пробам.

    Args:
        hsv: Кадр в HSV формы `(height, width, 3)`.
        regions: Области формы `(N, 4)`: `x_min, x_max, y_min, y_max`.
        inset: Отступ от боковых краёв области, доля ширины.

    Returns:
        Массив `(N, 3)` со средними значениями каналов.
    """
    if HAS_NATIVE:
        return cast("npt.NDArray[np.float32]", _native.sample_regions(hsv, regions, inset))
    return _sample_regions_numpy(hsv, regions, inset)


def key_deviations(
    hsv: Frame,
    regions: npt.NDArray[np.int32],
    references: npt.NDArray[np.float32],
    weights: DeviationWeights | None = None,
    inset: float = SAMPLE_INSET,
) -> npt.NDArray[np.float32]:
    """Отклонения цвета клавиш от эталона, 0..1.

    Основной горячий вызов: один проход по кадру вместо 88 срезов numpy.

    Args:
        hsv: Кадр в HSV.
        regions: Области клавиш формы `(N, 4)`.
        references: Эталонные цвета формы `(N, 3)`.
        weights: Веса каналов.
        inset: Отступ от краёв области.

    Returns:
        Массив `(N,)`: 0 — цвет совпал с эталоном, 1 — полное расхождение.
    """
    w = weights or DeviationWeights()
    if HAS_NATIVE:
        return cast(
            "npt.NDArray[np.float32]",
            _native.key_deviations(hsv, regions, references, w.hue, w.saturation, w.value, inset),
        )
    current = _sample_regions_numpy(hsv, regions, inset)
    return deviation_from_colors(current, references, w)


def deviation_from_colors(
    current: npt.NDArray[np.float32],
    references: npt.NDArray[np.float32],
    weights: DeviationWeights | None = None,
) -> npt.NDArray[np.float32]:
    """Расстояние между наборами цветов.

    Два обстоятельства делают наивную формулу непригодной:

    * тон замкнут по кругу — 179 и 0 соседние оттенки, а не
      противоположные, иначе красная подсветка давала бы ложное
      срабатывание на каждом кадре;
    * у бесцветного пикселя тона нет — белая клавиша при малейшем шуме
      даёт произвольный оттенок. Поэтому вес тона умножается на
      насыщенность менее насыщенного из двух цветов, а высвободившийся
      вес уходит насыщенности: в паре «белая клавиша против цветной
      подсветки» весь сигнал несёт именно она.
    """
    w = weights or DeviationWeights()
    hue = np.abs(current[:, 0] - references[:, 0])
    hue = np.minimum(hue, 180.0 - hue) / 90.0
    sat = np.abs(current[:, 1] - references[:, 1]) / 255.0
    val = np.abs(current[:, 2] - references[:, 2]) / 255.0

    hue_confidence = np.minimum(current[:, 1], references[:, 1]) / 255.0
    sat_weight = w.saturation + w.hue * (1.0 - hue_confidence)

    score = hue * w.hue * hue_confidence + sat * sat_weight + val * w.value
    bounded: npt.NDArray[np.float32] = np.minimum(score, 1.0).astype(np.float32)
    return bounded


def _sample_regions_numpy(
    hsv: Frame,
    regions: npt.NDArray[np.int32],
    inset: float,
) -> npt.NDArray[np.float32]:
    """Запасная реализация усреднения проб.

    Повторяет арифметику ядра построчно, включая обработку областей,
    выходящих за границы кадра: расхождение здесь означало бы, что
    результат зависит от наличия компилятора.
    """
    height, width = hsv.shape[:2]
    out = np.zeros((len(regions), 3), dtype=np.float32)

    for index, (x_min, x_max, y_min, y_max) in enumerate(regions):
        pad = int((int(x_max) - int(x_min)) * inset)
        x0 = min(max(int(x_min) + pad, 0), width)
        x1 = min(max(max(int(x_max) - pad, x0 + 1), 0), width)
        y0 = min(max(int(y_min), 0), height)
        y1 = min(max(max(int(y_max), y0 + 1), 0), height)
        if x1 <= x0 or y1 <= y0:
            continue
        patch = hsv[y0:y1, x0:x1]
        out[index] = patch.reshape(-1, 3).mean(axis=0)

    return out


if not HAS_NATIVE:  # pragma: no cover - зависит от наличия собранного ядра
    _log.debug("нативное ядро mir_core не найдено, используется numpy")
