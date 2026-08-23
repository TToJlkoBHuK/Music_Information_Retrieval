"""Разбор видеоряда: клавиатура, подсветка клавиш и падающие блоки.

Точка входа — [`analyze_video`][mir.vision.analyzer.analyze_video]: она
проводит ролик через все шаги этапа и возвращает список нот. Отдельные
классы нужны для отладки и тестов.

Тяжёлые вычисления идут через [`accel`][mir.vision.accel], который
использует нативное ядро `mir_core`, а без него — numpy.

Подробное описание устройства — в README.md рядом с этим файлом.
"""

from __future__ import annotations

from mir.vision.accel import HAS_NATIVE, backend_name
from mir.vision.analyzer import VisionResult, analyze_file, analyze_video
from mir.vision.block_tracker import BlockTracker
from mir.vision.calibration import VisualizerProfile, calibrate
from mir.vision.key_tracker import KeyTracker
from mir.vision.keyboard_detector import KeyboardDetector, build_median_frame
from mir.vision.keyboard_geometry import KeyboardGeometry

__all__ = [
    "HAS_NATIVE",
    "BlockTracker",
    "KeyTracker",
    "KeyboardDetector",
    "KeyboardGeometry",
    "VisionResult",
    "VisualizerProfile",
    "analyze_file",
    "analyze_video",
    "backend_name",
    "build_median_frame",
    "calibrate",
]
