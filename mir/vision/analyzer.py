"""Разбор видеоряда целиком — точка входа этапа `vision`.

Внутри видеоканала два независимых источника сведений, и они дополняют
друг друга:

* **подсветка клавиш** даёт точные моменты начала и конца звучания;
* **падающие блоки** дают партию руки и видны заранее.

Оба обходят ролик за один проход, после чего результаты сводятся:
время берётся от подсветки, рука — от блоков.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from mir.common.enums import Hand, Stage
from mir.common.errors import KeyboardNotFoundError
from mir.common.logging import get_logger
from mir.common.types import Frame, KeyboardLayout, MediaBundle, NoteEvent, ProgressCallback
from mir.config import MirConfig, load_config
from mir.vision.block_tracker import BlockTracker
from mir.vision.calibration import VisualizerProfile, calibrate
from mir.vision.key_tracker import KeyTracker
from mir.vision.keyboard_detector import KeyboardDetector, build_median_frame

__all__ = ["VisionResult", "analyze_video"]

_log = get_logger(__name__)

_CALIBRATION_FRAMES = 25
"""Сколько подряд идущих кадров нужно для замера скорости падения."""

_HAND_MATCH_TOLERANCE = 0.12
"""Допуск сопоставления событий подсветки и блоков, секунды."""


@dataclass
class VisionResult:
    """Результат разбора видеоряда.

    Attributes:
        notes: События с `source=VIDEO`.
        layout: Разметка клавиатуры.
        profile: Профиль визуализатора.
        frames_processed: Сколько кадров прочитано.
        elapsed_sec: Время обработки — идёт в таблицы производительности.
        notes_from_keys: Событий от подсветки клавиш.
        notes_from_blocks: Событий от падающих блоков.
    """

    notes: list[NoteEvent]
    layout: KeyboardLayout
    profile: VisualizerProfile
    frames_processed: int = 0
    elapsed_sec: float = 0.0
    notes_from_keys: int = 0
    notes_from_blocks: int = 0
    warnings: list[str] = field(default_factory=list)


def _sample_frames(capture: cv2.VideoCapture, count: int) -> list[Frame]:
    """Взять кадры равномерно по всему ролику.

    Именно равномерно: подряд идущие кадры почти одинаковы, и медиана
    по ним не убрала бы подсветку.
    """
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        return []

    frames: list[Frame] = []
    for index in np.linspace(0, total - 1, min(count, total), dtype=int):
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(index))
        ok, frame = capture.read()
        if ok:
            frames.append(frame)
    return frames


def _consecutive_frames(capture: cv2.VideoCapture, start: int, count: int) -> list[Frame]:
    """Взять подряд идущие кадры — нужны для замера скорости падения."""
    capture.set(cv2.CAP_PROP_POS_FRAMES, start)
    frames: list[Frame] = []
    for _ in range(count):
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(frame)
    return frames


def _merge_hands(key_notes: list[NoteEvent], block_notes: list[NoteEvent]) -> list[NoteEvent]:
    """Перенести партию руки с блоков на события подсветки.

    Время берётся от подсветки: она даёт момент нажатия напрямую, тогда как
    у блоков он вычисляется интерполяцией и накапливает ошибку скорости
    падения. Рука есть только у блоков — цвет подсветки её не кодирует.
    """
    if not block_notes:
        return key_notes

    by_pitch: dict[int, list[NoteEvent]] = {}
    for note in block_notes:
        by_pitch.setdefault(note.pitch, []).append(note)

    merged: list[NoteEvent] = []
    for note in key_notes:
        candidates = by_pitch.get(note.pitch, [])
        nearest = min(
            (c for c in candidates if abs(c.onset - note.onset) < _HAND_MATCH_TOLERANCE),
            key=lambda c: abs(c.onset - note.onset),
            default=None,
        )
        if nearest is None or nearest.hand is Hand.UNKNOWN:
            merged.append(note)
        else:
            merged.append(
                NoteEvent(
                    pitch=note.pitch,
                    onset=note.onset,
                    offset=note.offset,
                    velocity=note.velocity,
                    hand=nearest.hand,
                    source=note.source,
                    confidence=min(1.0, note.confidence * 1.1),
                )
            )
    return merged


def analyze_video(
    media: MediaBundle,
    config: MirConfig | None = None,
    progress: ProgressCallback | None = None,
) -> VisionResult:
    """Разобрать видеоряд и вернуть найденные ноты.

    Args:
        media: Подготовленный материал с этапа `ingest`.
        config: Конфигурация. По умолчанию загружается стандартная.
        progress: Колбэк прогресса.

    Returns:
        События с указанием руки, разметка клавиатуры и профиль ролика.

    Raises:
        KeyboardNotFoundError: Клавиатура не найдена — ролик не является
            piano visualizer.
    """
    import time

    config = config or load_config()
    started = time.perf_counter()

    capture = cv2.VideoCapture(str(media.video_path))
    if not capture.isOpened():
        raise KeyboardNotFoundError(
            f"не удалось открыть {media.video_path}",
            "Не удалось прочитать видеофайл",
        )

    try:
        sample = _sample_frames(capture, config.vision.keyboard.sample_frames)
        if not sample:
            raise KeyboardNotFoundError(
                "не удалось прочитать ни одного кадра",
                "Видеофайл пуст или повреждён",
            )

        layout = KeyboardDetector(config.vision.keyboard).detect(sample)
        reference = build_median_frame(sample)

        calibration_start = min(int(media.fps * 2), max(media.frame_count - _CALIBRATION_FRAMES, 0))
        profile = calibrate(
            _consecutive_frames(capture, calibration_start, _CALIBRATION_FRAMES),
            layout,
            media.fps,
        )

        key_tracker = KeyTracker(layout, reference, config.vision.tracker)
        block_tracker = BlockTracker(layout, profile, media.fps, config.vision.tracker)

        capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
        key_notes: list[NoteEvent] = []
        index = 0
        total = max(media.frame_count, 1)

        while True:
            ok, frame = capture.read()
            if not ok:
                break
            timestamp = index / media.fps
            key_notes.extend(key_tracker.process_frame(frame, timestamp))
            block_tracker.process_frame(frame, timestamp)
            index += 1
            if progress and index % 30 == 0:
                progress(Stage.VISION, min(index / total, 1.0))

        key_notes.extend(key_tracker.flush(index / media.fps))
        block_notes = block_tracker.flush()
    finally:
        capture.release()

    notes = sorted(_merge_hands(key_notes, block_notes), key=lambda n: (n.onset, n.pitch))
    elapsed = time.perf_counter() - started

    _log.info(
        "видеоряд разобран: %d кадров за %.1f с, нот %d (подсветка %d, блоки %d)",
        index,
        elapsed,
        len(notes),
        len(key_notes),
        len(block_notes),
    )

    if progress:
        progress(Stage.VISION, 1.0)

    return VisionResult(
        notes=notes,
        layout=layout,
        profile=profile,
        frames_processed=index,
        elapsed_sec=elapsed,
        notes_from_keys=len(key_notes),
        notes_from_blocks=len(block_notes),
        warnings=list(profile.warnings),
    )


def analyze_file(
    video_path: Path,
    config: MirConfig | None = None,
    progress: ProgressCallback | None = None,
) -> VisionResult:
    """Разобрать видеофайл без прохода через `ingest`.

    Удобно для отладки: аудиодорожка на этом этапе не нужна.
    """
    capture = cv2.VideoCapture(str(video_path))
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    capture.release()

    bundle = MediaBundle(
        video_path=video_path,
        audio_path=video_path,
        fps=fps,
        duration=frames / fps if fps else 0.0,
        width=width,
        height=height,
    )
    return analyze_video(bundle, config, progress)
