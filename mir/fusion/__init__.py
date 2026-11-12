"""Слияние видео- и аудиоканалов с взаимной проверкой.

Ядро научной новизны работы. Точка входа — [`fuse`][mir.fusion.merger.fuse]:
из нот видео и нот звука она собирает один список и по каждой исходной
ноте записывает решение с причиной.

Подробное описание устройства — в README.md рядом с этим файлом.
"""

from __future__ import annotations

from mir.fusion.aligner import (
    OffsetCurve,
    OffsetEstimate,
    apply_offset,
    estimate_av_offset,
    estimate_av_offset_from_onsets,
    estimate_offset_curve,
    estimate_pitch_shift,
)
from mir.fusion.decisions import Decision, FusionResult, Reason
from mir.fusion.matcher import MatchResult, match_notes
from mir.fusion.merger import fuse

__all__ = [
    "Decision",
    "FusionResult",
    "MatchResult",
    "OffsetCurve",
    "OffsetEstimate",
    "Reason",
    "apply_offset",
    "estimate_av_offset",
    "estimate_av_offset_from_onsets",
    "estimate_offset_curve",
    "estimate_pitch_shift",
    "fuse",
    "match_notes",
]
