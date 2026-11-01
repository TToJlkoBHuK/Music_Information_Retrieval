"""Сверка двух каналов без эталона."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from mir.common.types import NoteEvent
from mir.eval.compare import compare_channels, piano_roll


def melody(count: int = 80, seed: int = 3) -> list[NoteEvent]:
    rng = np.random.default_rng(seed)
    notes, t = [], 0.5
    for _ in range(count):
        duration = float(rng.uniform(0.2, 0.8))
        notes.append(NoteEvent(int(rng.integers(40, 80)), t, t + duration, 80))
        t += float(rng.choice([0.25, 0.5, 0.75]))
    return notes


def test_identical_channels_agree():
    notes = melody()
    result = compare_channels(notes, notes)
    assert result.similarity == pytest.approx(1.0)
    assert result.time_shift == 0.0
    assert result.pitch_consistent


def test_time_shift_found():
    notes = melody()
    late = [replace(n, onset=n.onset + 0.3, offset=n.offset + 0.3) for n in notes]
    result = compare_channels(notes, late)
    assert result.time_shift == pytest.approx(0.3, abs=0.051)
    assert result.pitch_consistent


def test_wrong_video_pitches_detected():
    """Видео ошиблось на кварту: лучшее совпадение — при сдвиге на +5."""
    audio = melody()
    video = [replace(n, pitch=n.pitch - 5) for n in audio]
    result = compare_channels(video, audio)
    assert result.best_pitch_shift == 5
    assert not result.pitch_consistent
    assert result.best_similarity > result.similarity + 0.5


def test_roll_marks_sounding_time():
    roll = piano_roll([NoteEvent(60, 1.0, 1.5, 80)], step=0.1, frames=30)
    assert roll[60 - 21, 10:15].tolist() == [1.0] * 5
    assert roll.sum() == 5
