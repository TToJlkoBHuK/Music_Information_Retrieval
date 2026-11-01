"""Разбор звука целиком — точка входа этапа `audio`.

Порядок: транскрипция нейросетью → атаки → темп. Атаки и темп считаются
по одной функции новизны. Начала нот из транскрипции, если она была,
помогают выбрать метрический уровень темпа: у них меньше ложных атак,
чем у детектора по спектру.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import numpy.typing as npt

from mir.audio.models import ModelRegistry, select_device
from mir.audio.onsets import OnsetEnvelope, detect_onsets, onset_envelope
from mir.audio.signal import load_audio
from mir.audio.tempo import TempoAnalysis, TempoAnalyzer
from mir.audio.transcriber import AudioTranscriber, SegmentModel
from mir.common.enums import Stage
from mir.common.errors import CancelledError
from mir.common.logging import get_logger
from mir.common.types import CancelCheck, NoteEvent, PedalEvent, ProgressCallback, TempoMap
from mir.config import MirConfig, load_config

__all__ = ["AudioResult", "analyze_audio", "load_model"]

_log = get_logger(__name__)

_SHARE_MODEL = 0.05
_SHARE_TRANSCRIPTION = 0.80
_SHARE_ONSETS = 0.07
"""Доли общего прогресса этапа. Остаток уходит на темп.

Распределение по замеру на CPU: нейросеть — подавляющая часть времени,
атаки и темп на семиминутном ролике считаются за секунды.
"""


@dataclass
class AudioResult:
    """Результат разбора звука.

    Attributes:
        notes: Ноты с `source=AUDIO`. Пусто, если транскрипция отключена.
        pedals: Нажатия правой педали.
        tempo: Сетка долей.
        tempo_details: Как получен темп: до и после исправления ошибки вдвое,
            разброс долей.
        onsets: Моменты атак, секунды.
        envelope: Функция новизны — нужна для оценки рассинхрона, когда
            транскрипции нет.
        device: На чём считалась нейросеть; `None`, если не считалась.
        duration: Длительность разобранного звука, секунды.
        elapsed_sec: Время работы этапа.
        warnings: Предупреждения для пользователя.
    """

    notes: list[NoteEvent]
    pedals: list[PedalEvent]
    tempo: TempoMap
    tempo_details: TempoAnalysis
    onsets: npt.NDArray[np.float64]
    envelope: OnsetEnvelope
    device: str | None
    duration: float
    elapsed_sec: float
    warnings: list[str] = field(default_factory=list)


def load_model(
    config: MirConfig,
    progress: Callable[[float], None] | None = None,
    cancel: CancelCheck | None = None,
) -> tuple[SegmentModel, list[str]]:
    """Подготовить нейросеть: найти или скачать веса, выбрать устройство.

    Returns:
        Модель и предупреждения (например, о переходе на процессор).

    Raises:
        TranscriptionError: Нет библиотек, весов или они не подходят.
    """
    from mir.audio.bytedance import ByteDanceModel

    registry = ModelRegistry(config.ingest.cache_path, config.audio.model_path)

    def on_bytes(done: int, total: int) -> None:
        if progress is not None and total:
            progress(done / total)

    path, official = registry.ensure(on_bytes, cancel)
    device, warning = select_device(config.audio.device)
    return ByteDanceModel(path, device, official), [warning] if warning else []


def analyze_audio(
    audio_path: Path,
    config: MirConfig | None = None,
    *,
    transcribe: bool = True,
    force_bpm: float | None = None,
    start_seconds: float = 0.0,
    max_seconds: float | None = None,
    progress: ProgressCallback | None = None,
    cancel: CancelCheck | None = None,
    model: SegmentModel | None = None,
) -> AudioResult:
    """Разобрать звуковую дорожку.

    Args:
        audio_path: WAV из `ingest` или любой звуковой файл.
        config: Конфигурация. По умолчанию загружается стандартная.
        transcribe: Распознавать ноты нейросетью. Без неё считаются только
            атаки и темп — секунды вместо минут.
        force_bpm: Темп, заданный пользователем (F-31).
        start_seconds: С какой секунды разбирать.
        max_seconds: Сколько секунд разобрать.
        progress: Колбэк прогресса.
        cancel: Опрос отмены.
        model: Готовая модель — чтобы не загружать веса повторно
            при разборе нескольких роликов подряд.

    Raises:
        TranscriptionError: Нет библиотек или весов, файл не читается.
        CancelledError: Пользователь отменил обработку.
    """
    config = config or load_config()
    started = time.perf_counter()
    warnings: list[str] = []

    def report(fraction: float) -> None:
        if progress is not None:
            progress(Stage.AUDIO, min(max(fraction, 0.0), 1.0))

    def check_cancel() -> None:
        if cancel is not None and cancel():
            raise CancelledError("разбор звука отменён")

    report(0.0)
    signal = load_audio(audio_path, start_seconds, max_seconds)
    _log.info("звук: %.1f с, %d Гц", signal.duration, signal.sample_rate)

    notes: list[NoteEvent] = []
    pedals: list[PedalEvent] = []
    device: str | None = None
    if transcribe:
        if model is None:
            model, model_warnings = load_model(config, lambda f: report(f * _SHARE_MODEL), cancel)
            warnings.extend(model_warnings)
        report(_SHARE_MODEL)
        result = AudioTranscriber(model, config.audio).transcribe(
            signal,
            progress=lambda f: report(_SHARE_MODEL + f * _SHARE_TRANSCRIPTION),
            cancel=cancel,
        )
        notes, pedals, device = result.notes, result.pedals, result.device
        warnings.extend(result.warnings)

    check_cancel()
    share = _SHARE_MODEL + _SHARE_TRANSCRIPTION if transcribe else 0.0
    envelope = onset_envelope(signal, config.audio)
    onsets = detect_onsets(envelope, config.audio)
    report(share + _SHARE_ONSETS)

    check_cancel()
    attacks = np.array([n.onset for n in notes]) if notes else None
    details = TempoAnalyzer(config.audio).analyze(envelope, onsets, attacks, force_bpm)
    warnings.extend(details.warnings)
    report(1.0)

    return AudioResult(
        notes=notes,
        pedals=pedals,
        tempo=details.tempo,
        tempo_details=details,
        onsets=onsets,
        envelope=envelope,
        device=device,
        duration=signal.duration,
        elapsed_sec=time.perf_counter() - started,
        warnings=warnings,
    )
