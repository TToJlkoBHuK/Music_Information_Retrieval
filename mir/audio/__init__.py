"""Анализ аудиодорожки: транскрипция, атаки, темп (этап 4).

Второй, независимый от видео источник истины. Подробное описание
устройства — в README.md рядом с этим файлом.

PyTorch здесь не импортируется: он нужен только нейросетевой
транскрипции и подгружается при создании модели. Атаки и темп
работают и без него.
"""

from __future__ import annotations

from mir.audio.analyzer import AudioResult, analyze_audio, load_model
from mir.audio.onsets import OnsetEnvelope, detect_onsets, onset_envelope, refine_note_onsets
from mir.audio.signal import AudioSignal, load_audio
from mir.audio.tempo import TempoAnalysis, TempoAnalyzer

__all__ = [
    "AudioResult",
    "AudioSignal",
    "OnsetEnvelope",
    "TempoAnalysis",
    "TempoAnalyzer",
    "analyze_audio",
    "detect_onsets",
    "load_audio",
    "load_model",
    "onset_envelope",
    "refine_note_onsets",
]
