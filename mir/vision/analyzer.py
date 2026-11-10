"""Разбор видеоряда целиком — точка входа этапа `vision`.

Внутри видеоканала два источника нот:

* **падающие блоки** — линия-датчик над клавиатурой (`line_sensor`). Блоки
  падают выше клавиатуры, и руки исполнителя их не закрывают;
* **подсветка клавиш** (`key_tracker`) — для визуализаторов без блоков.

Основной источник — блоки. На самых популярных роликах, где поверх
нарисованной клавиатуры сняты настоящие руки, подсветка непригодна:
руку трекер принимает за нажатие (`docs/known-issues.md`, В-05). Подсветка
остаётся запасным путём, когда блоков в ролике нет.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import pairwise
from pathlib import Path

import cv2
import numpy as np

from mir.common.enums import Stage
from mir.common.errors import KeyboardNotFoundError
from mir.common.logging import get_logger
from mir.common.types import Frame, KeyboardLayout, MediaBundle, NoteEvent, ProgressCallback
from mir.config import MirConfig, load_config
from mir.vision.calibration import VisualizerProfile, calibrate
from mir.vision.key_tracker import KeyTracker
from mir.vision.keyboard_detector import KeyboardDetector, build_median_frame
from mir.vision.line_sensor import SensorBank, SensorReport

__all__ = ["VisionResult", "analyze_video"]

_log = get_logger(__name__)

_CALIBRATION_FRAMES = 25
"""Сколько подряд идущих кадров нужно для замера скорости падения."""

_CALIBRATION_AT = 0.4
"""В какой точке ролика замерять скорость падения, доля длины.

Не в начале: там заставка, где блоки не движутся вовсе.
"""

_MIN_SENSOR_NOTES = 8
"""Меньше нот от датчика блоков — считать, что блоков в ролике нет.

Несколько случайных событий даёт и ролик без блоков: титры, вспышки.
Музыкальный фрагмент даже в полминуты содержит десятки нот.
"""


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
        notes_from_blocks: Нот от падающих блоков (линия-датчик).
        source: Какой источник дал итоговые ноты: `blocks` или `keys`.
        sensor: Подробности линии-датчика: скорость падения, линия касания.
        unconfirmed: Блоки-кандидаты: то, что датчик видел, но не смог
            доказать. Сами по себе не ноты; при слиянии каналов кандидат
            вместе с нотой звука той же высоты — уже нота (`mir.fusion`).
    """

    notes: list[NoteEvent]
    layout: KeyboardLayout
    profile: VisualizerProfile
    frames_processed: int = 0
    elapsed_sec: float = 0.0
    notes_from_keys: int = 0
    notes_from_blocks: int = 0
    source: str = "keys"
    sensor: SensorReport | None = None
    unconfirmed: list[NoteEvent] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


_SAMPLE_FROM = 0.15
_SAMPLE_TO = 0.92
"""Доля ролика, из которой берутся кадры для детекции клавиатуры.

Края отброшены намеренно: ролики почти всегда начинаются заставкой
с обложкой и названием, а заканчиваются титрами. Клавиатуры там нет,
и попавшие в выборку кадры смещают медиану.
"""


def _sample_frames(capture: cv2.VideoCapture, count: int) -> list[Frame]:
    """Взять кадры равномерно по основной части ролика.

    Именно равномерно: подряд идущие кадры почти одинаковы, и медиана
    по ним не убрала бы подсветку.
    """
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        return []

    first = int(total * _SAMPLE_FROM) if total > 100 else 0
    last = int(total * _SAMPLE_TO) if total > 100 else total - 1

    frames: list[Frame] = []
    for index in np.linspace(first, last, min(count, last - first + 1), dtype=int):
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


_SPAN_PROBE_STEP = 0.5
"""Шаг сканирования при поиске границ фрагмента с клавиатурой, секунды."""

_SPAN_SEARCH_SHARE = 0.3
"""Какую долю ролика с каждого края обыскивать на заставку и титры.

Заставка длиннее трети ролика — это уже не заставка.
"""

_SPAN_SIMILARITY = 0.3
"""Насколько профиль стыков должен совпасть с эталонным.

У кадра с клавиатурой совпадение около единицы даже при яркой подсветке:
она меняет яркость клавиш, но не положение стыков. У заставки с обложкой
регулярного узора нет вовсе, и совпадение падает к нулю.

Порог был 0.5, но руки исполнителя закрывают часть клавиатуры: на
синтетике с руками кадры с клавиатурой давали 0.48, и разбор начинался
позже первой ноты. Замер на ролике Rousseau: заставка 0.00–0.02, переход
0.71, игра 0.95–0.97. Порог 0.3 лежит с запасом между ними.
"""

_SPAN_SEGMENTS = 8
"""На сколько участков делить клавиатуру при сравнении профилей.

Совпадение считается по каждому участку отдельно, и берётся медиана. Рука
портит два-три участка из восьми, а медиана этого не замечает; общая
корреляция по всей ширине падала вдвое от одной ладони.
"""


def _keyboard_signature(frame: Frame, layout: KeyboardLayout) -> Frame:
    """Профиль стыков клавиш — признак того, что клавиатура в кадре.

    Берётся нижняя треть области клавиатуры: подсветка меняет там яркость,
    но стыки остаются на местах, поэтому признак устойчив к игре.
    """
    x, y, width, height = layout.bbox
    strip = frame[y + 2 * height // 3 : y + height, x : x + width]
    if strip.size == 0:
        return np.zeros(1, dtype=np.float32)

    gray = cv2.cvtColor(strip, cv2.COLOR_BGR2GRAY).astype(np.float32)
    profile = np.abs(cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)).mean(axis=0)
    centered: Frame = profile - profile.mean()
    return centered


def _correlation(first: Frame, second: Frame) -> float:
    """Нормированная корреляция двух профилей, 0..1."""
    first = first - first.mean()
    second = second - second.mean()
    norm = float(np.linalg.norm(first) * np.linalg.norm(second))
    if norm <= 0:
        return 0.0
    return max(0.0, float(np.dot(first, second) / norm))


def _similarity(first: Frame, second: Frame) -> float:
    """Совпадение профилей стыков: медиана корреляций по участкам, 0..1."""
    size = min(len(first), len(second))
    if size < _SPAN_SEGMENTS:
        return _correlation(first[:size], second[:size])
    bounds = np.linspace(0, size, _SPAN_SEGMENTS + 1).astype(int)
    scores = [_correlation(first[a:b], second[a:b]) for a, b in pairwise(bounds)]
    return float(np.median(scores))


def find_keyboard_span(
    capture: cv2.VideoCapture,
    layout: KeyboardLayout,
    reference: Frame,
    fps: float,
    total_frames: int,
) -> tuple[float, float]:
    """Найти отрезок ролика, где клавиатура действительно в кадре.

    Ролики открываются заставкой с обложкой и названием, а заканчиваются
    титрами; их длительность у каждого автора своя, и просить пользователя
    указывать её вручную — ровно то неудобство, которое проект и должен
    устранить.

    Каждый пробный кадр сравнивается по профилю стыков с эталонным
    (медианным) кадром. Признак выбран так, чтобы не зависеть от игры:
    подсветка меняет яркость клавиш, но не положение стыков.

    Returns:
        Начало и конец отрезка в секундах.
    """
    expected = _keyboard_signature(reference, layout)
    step = max(int(fps * _SPAN_PROBE_STEP), 1)
    limit = int(total_frames * _SPAN_SEARCH_SHARE)

    def has_keyboard(index: int) -> bool:
        capture.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = capture.read()
        if not ok:
            return False
        return _similarity(_keyboard_signature(frame, layout), expected) >= _SPAN_SIMILARITY

    def refine(absent: int, present: int) -> int:
        """Уточнить границу перехода двоичным поиском.

        Грубого шага мало: между последней проверкой и настоящей границей
        остаётся до полусекунды заставки, а резкая смена кадра выглядит для
        трекера как одновременное нажатие всех клавиш сразу.
        """
        while abs(present - absent) > 1:
            middle = (absent + present) // 2
            if has_keyboard(middle):
                present = middle
            else:
                absent = middle
        return present

    start = 0
    previous = 0
    for index in range(0, limit, step):
        if has_keyboard(index):
            start = index if index == 0 else refine(previous, index)
            break
        previous = index

    end = total_frames
    previous = total_frames - 1
    for index in range(total_frames - 1, total_frames - limit, -step):
        if has_keyboard(index):
            end = index if index == total_frames - 1 else refine(previous, index)
            break
        previous = index

    # Отступ внутрь отрезка: переход между заставкой и клавиатурой часто
    # плавный, и пограничные кадры дают всплеск ложных нажатий.
    if start > 0:
        start = min(start + step, total_frames - 1)
    if end < total_frames:
        end = max(end - step, start + 1)

    if start > 0 or end < total_frames:
        _log.info(
            "клавиатура в кадре с %.1f с по %.1f с из %.1f с",
            start / fps,
            end / fps,
            total_frames / fps,
        )
    return start / fps, end / fps


def analyze_video(
    media: MediaBundle,
    config: MirConfig | None = None,
    progress: ProgressCallback | None = None,
    max_seconds: float | None = None,
    start_seconds: float = 0.0,
) -> VisionResult:
    """Разобрать видеоряд и вернуть найденные ноты.

    Args:
        media: Подготовленный материал с этапа `ingest`.
        config: Конфигурация. По умолчанию загружается стандартная.
        progress: Колбэк прогресса.
        max_seconds: Разобрать только отрезок ролика. Клавиатура и профиль
            всё равно определяются по всей длине: медиана по кадрам из
            одного места хуже убирает подсветку.
        start_seconds: С какой секунды начинать разбор. Нужен, когда ролик
            открывается заставкой.

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

        # Калибровка идёт по середине ролика, а не по его началу. В первые
        # секунды у роликов заставка с обложкой: блоки не движутся, и
        # скорость падения выходила нулевой. Без неё длительность ноты
        # не вычислить, и трекер блоков терял три четверти событий.
        calibration_start = min(
            int(media.frame_count * _CALIBRATION_AT),
            max(media.frame_count - _CALIBRATION_FRAMES, 0),
        )
        profile = calibrate(
            _consecutive_frames(capture, calibration_start, _CALIBRATION_FRAMES),
            layout,
            media.fps,
        )

        key_tracker = KeyTracker(layout, reference, config.vision.tracker)
        sensor = SensorBank(layout, sample, media.fps, config.vision.sensor)

        span_start, span_end = find_keyboard_span(
            capture, layout, reference, media.fps, media.frame_count
        )
        if start_seconds <= 0:
            start_seconds = span_start
        if max_seconds is None:
            max_seconds = max(span_end - start_seconds, 0.0) or None

        first = int(start_seconds * media.fps)
        capture.set(cv2.CAP_PROP_POS_FRAMES, first)
        key_notes: list[NoteEvent] = []
        index = first
        limit = first + int(max_seconds * media.fps) if max_seconds else None
        total = max(min(media.frame_count, limit or media.frame_count) - first, 1)

        while True:
            if limit is not None and index >= limit:
                break
            ok, frame = capture.read()
            if not ok:
                break
            timestamp = index / media.fps
            key_notes.extend(key_tracker.process_frame(frame, timestamp))
            sensor.process_frame(frame, timestamp)
            index += 1
            if progress and index % 30 == 0:
                progress(Stage.VISION, min((index - first) / total, 1.0))

        key_notes.extend(key_tracker.flush(index / media.fps))
        block_notes = sensor.finish()
    finally:
        capture.release()

    warnings = list(profile.warnings)
    if len(block_notes) >= _MIN_SENSOR_NOTES:
        source, notes = "blocks", block_notes
        # Скорость падения датчик измерил сам, по задержке между полосами.
        warnings = [w for w in warnings if "скорость падения" not in w]
    else:
        source, notes = "keys", key_notes
        reason = sensor.report.reason or f"по блокам нашлось лишь {len(block_notes)} нот"
        warnings.append(f"ноты взяты с подсветки клавиш: {reason}")
    notes = sorted(notes, key=lambda n: (n.onset, n.pitch))
    elapsed = time.perf_counter() - started

    _log.info(
        "видеоряд разобран: %d кадров за %.1f с, нот %d (источник %s; подсветка %d, блоки %d)",
        index,
        elapsed,
        len(notes),
        source,
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
        source=source,
        sensor=sensor.report,
        unconfirmed=sensor.unconfirmed if source == "blocks" else [],
        warnings=warnings,
    )


def analyze_file(
    video_path: Path,
    config: MirConfig | None = None,
    progress: ProgressCallback | None = None,
    max_seconds: float | None = None,
    start_seconds: float = 0.0,
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
    return analyze_video(bundle, config, progress, max_seconds, start_seconds)
