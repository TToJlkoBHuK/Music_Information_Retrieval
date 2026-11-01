"""Оценка качества распознавания по правилам MIREX и сверка каналов."""

from mir.eval.compare import ChannelAgreement, compare_channels
from mir.eval.metrics import TranscriptionScore, evaluate, match_notes
from mir.eval.notes_io import read_notes_csv, write_notes_csv

__all__ = [
    "ChannelAgreement",
    "TranscriptionScore",
    "compare_channels",
    "evaluate",
    "match_notes",
    "read_notes_csv",
    "write_notes_csv",
]
