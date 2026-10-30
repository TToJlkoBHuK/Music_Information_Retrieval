"""Оценка качества распознавания по правилам MIREX."""

from mir.eval.metrics import TranscriptionScore, evaluate, match_notes
from mir.eval.notes_io import read_notes_csv, write_notes_csv

__all__ = ["TranscriptionScore", "evaluate", "match_notes", "read_notes_csv", "write_notes_csv"]
