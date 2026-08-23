"""Оценка качества распознавания по правилам MIREX."""

from mir.eval.metrics import TranscriptionScore, evaluate, match_notes

__all__ = ["TranscriptionScore", "evaluate", "match_notes"]
