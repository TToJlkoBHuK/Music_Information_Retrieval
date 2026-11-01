"""Слияние видео- и аудиоканалов с взаимной проверкой.

Ядро научной новизны работы. Компенсация рассинхрона (`aligner`) готова
на этапе 4 (#29): она нужна до слияния. Остальное — этап 5.

Подробное описание устройства — в README.md рядом с этим файлом.
"""

from __future__ import annotations

from mir.fusion.aligner import (
    OffsetEstimate,
    apply_offset,
    estimate_av_offset,
    estimate_av_offset_from_onsets,
)

__all__ = [
    "OffsetEstimate",
    "apply_offset",
    "estimate_av_offset",
    "estimate_av_offset_from_onsets",
]
