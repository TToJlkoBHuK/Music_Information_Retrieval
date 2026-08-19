"""Модель данных и утилиты, общие для всех этапов.

Модуль ни от кого не зависит — только стандартная библиотека. Это правило
удерживает конвейер однонаправленным: `notation` может импортировать
`common`, но не наоборот и не через соседей.
"""

from __future__ import annotations

from mir.common.enums import Clef, Hand, Platform, Source, Stage
from mir.common.errors import (
    CancelledError,
    ConfigError,
    DemuxError,
    DownloadError,
    ExportError,
    KeyboardNotFoundError,
    MirError,
    TranscriptionError,
    UnsupportedSourceError,
)
from mir.common.logging import CallbackHandler, get_logger, setup_logging
from mir.common.timeutil import (
    DEFAULT_PPQ,
    beats_to_note_value,
    frame_to_seconds,
    seconds_to_frame,
    seconds_to_ticks,
    ticks_to_seconds,
)
from mir.common.types import (
    PITCH_MAX,
    PITCH_MIN,
    KeyboardLayout,
    KeySlot,
    MediaBundle,
    NoteEvent,
    ProgressCallback,
    QualityReport,
    VideoInfo,
)

__all__ = [
    "DEFAULT_PPQ",
    "PITCH_MAX",
    "PITCH_MIN",
    "CallbackHandler",
    "CancelledError",
    "Clef",
    "ConfigError",
    "DemuxError",
    "DownloadError",
    "ExportError",
    "Hand",
    "KeySlot",
    "KeyboardLayout",
    "KeyboardNotFoundError",
    "MediaBundle",
    "MirError",
    "NoteEvent",
    "Platform",
    "ProgressCallback",
    "QualityReport",
    "Source",
    "Stage",
    "TranscriptionError",
    "UnsupportedSourceError",
    "VideoInfo",
    "beats_to_note_value",
    "frame_to_seconds",
    "get_logger",
    "seconds_to_frame",
    "seconds_to_ticks",
    "setup_logging",
    "ticks_to_seconds",
]
