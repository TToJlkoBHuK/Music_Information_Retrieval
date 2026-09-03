"""Метрики MIREX: сопоставление нот и подсчёт F-меры."""

from __future__ import annotations

import pytest

from mir.common.enums import Hand
from mir.common.types import NoteEvent
from mir.eval import evaluate, match_notes


def note(pitch: int, onset: float, duration: float = 0.4, hand: Hand = Hand.UNKNOWN) -> NoteEvent:
    return NoteEvent(pitch=pitch, onset=onset, offset=onset + duration, velocity=80, hand=hand)


def test_perfect_match_gives_unit_f1():
    truth = [note(60, 0.0), note(62, 0.5), note(64, 1.0)]

    score = evaluate(truth, list(truth))

    assert score.f1 == pytest.approx(1.0)
    assert score.precision == pytest.approx(1.0)
    assert score.recall == pytest.approx(1.0)
    assert score.onset_error_max == pytest.approx(0.0)


def test_onset_within_tolerance_counts_as_hit():
    """Стандарт MIREX — 50 мс; 40 мс должны засчитаться."""
    truth = [note(60, 1.0)]
    found = [note(60, 1.04)]

    assert evaluate(truth, found).matched == 1


def test_onset_beyond_tolerance_is_a_miss():
    truth = [note(60, 1.0)]
    found = [note(60, 1.09)]

    score = evaluate(truth, found)

    assert score.matched == 0
    assert score.f1 == pytest.approx(0.0)


def test_wrong_pitch_never_matches():
    """Октавная ошибка — не «почти верно», а промах."""
    score = evaluate([note(60, 1.0)], [note(72, 1.0)])

    assert score.matched == 0


def test_duplicates_count_as_false_positives():
    """Одна эталонная нота закрывается ровно одним найденным событием."""
    truth = [note(60, 1.0)]
    found = [note(60, 1.0), note(60, 1.01)]

    score = evaluate(truth, found)

    assert score.matched == 1
    assert score.recall == pytest.approx(1.0)
    assert score.precision == pytest.approx(0.5)
    assert score.f1 == pytest.approx(2 / 3)


def test_greedy_matching_prefers_nearest():
    truth = [note(60, 1.0)]
    found = [note(60, 1.03), note(60, 1.005)]

    pairs = match_notes(truth, found)

    assert pairs == [(0, 1)]


def test_missing_notes_lower_recall_only():
    truth = [note(60, 0.0), note(62, 0.5)]
    found = [note(60, 0.0)]

    score = evaluate(truth, found)

    assert score.precision == pytest.approx(1.0)
    assert score.recall == pytest.approx(0.5)


def test_offset_checked_separately_from_onset():
    """Конец звучания проверяется отдельно от начала.

    Верное начало при вдвое большей длительности — совпадение
    по onset, но не по offset.
    """
    truth = [note(60, 1.0, duration=0.4)]
    found = [note(60, 1.0, duration=0.8)]

    score = evaluate(truth, found)

    assert score.matched == 1
    assert score.offset_matched == 0
    assert score.f1 == pytest.approx(1.0)
    assert score.f1_with_offset == pytest.approx(0.0)


def test_hand_accuracy_ignores_unknown():
    truth = [note(60, 0.0, hand=Hand.LEFT), note(62, 0.5, hand=Hand.RIGHT)]
    found = [note(60, 0.0, hand=Hand.LEFT), note(62, 0.5, hand=Hand.UNKNOWN)]

    score = evaluate(truth, found)

    assert score.hand_known == 1
    assert score.hand_accuracy == pytest.approx(1.0)


def test_empty_input_does_not_divide_by_zero():
    score = evaluate([], [])

    assert score.f1 == 0.0
    assert score.precision == 0.0
    assert score.recall == 0.0


def test_report_is_printable():
    score = evaluate([note(60, 0.0)], [note(60, 0.0)])

    report = score.format_report()

    assert "F1" in report
    assert "precision" in report
