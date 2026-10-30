"""Детекция атак и уточнение времени нот (#28)."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pytest

from mir.audio.onsets import detect_onsets, merge_close, onset_envelope, refine_note_onsets
from mir.audio.signal import AudioSignal
from mir.common.enums import Hand, Source
from mir.common.types import NoteEvent
from mir.config import load_config
from tests.fixtures.audio_synth import SAMPLE_RATE, render_notes, scale_melody


def random_notes(count: int = 120, seed: int = 1) -> list[NoteEvent]:
    rng = np.random.default_rng(seed)
    notes, t = [], 0.5
    for _ in range(count):
        duration = float(rng.uniform(0.1, 0.6))
        notes.append(
            NoteEvent(int(rng.integers(36, 90)), t, t + duration, int(rng.integers(30, 120)))
        )
        t += float(rng.choice([0.12, 0.25, 0.37, 0.5]))
    return notes


def match(
    estimated: npt.NDArray[np.float64], truth: npt.NDArray[np.float64], tolerance: float = 0.05
):
    """Полнота и ошибки времени при сопоставлении один к одному."""
    used: set[int] = set()
    errors = []
    for t in truth:
        distance = np.abs(estimated - t)
        index = int(np.argmin(distance))
        if distance[index] <= tolerance and index not in used:
            used.add(index)
            errors.append(estimated[index] - t)
    return len(errors) / len(truth), len(errors) / len(estimated), np.array(errors)


@pytest.fixture(scope="module")
def audio_config():
    return load_config(use_user_config=False, use_env=False).audio


@pytest.mark.parametrize("material", ["melody", "random"])
def test_onsets_found_precisely(audio_config, material):
    """Полнота, точность и время атак на синтетике с известным ответом.

    Пороги — с запасом от замеренного (задержка 10 мс, разброс 2-3 мс),
    чтобы тест ловил поломку, а не дрожание чисел.
    """
    notes = scale_melody(100, 8) if material == "melody" else random_notes()
    truth = np.unique(np.round([n.onset for n in notes], 4))
    signal = AudioSignal(render_notes(notes), SAMPLE_RATE)

    onsets = detect_onsets(onset_envelope(signal, audio_config), audio_config)
    recall, precision, errors = match(onsets, truth)

    assert recall >= 0.95
    assert precision >= 0.9
    assert abs(float(np.mean(errors))) < 0.015
    assert float(np.std(errors)) < 0.006


def test_onsets_respect_signal_start(audio_config):
    """Атаки отрезка возвращаются во времени ролика, а не отрезка."""
    notes = [NoteEvent(60, 0.5 + i * 0.5, 0.8 + i * 0.5, 90) for i in range(6)]
    signal = AudioSignal(render_notes(notes), SAMPLE_RATE, start=100.0)

    onsets = detect_onsets(onset_envelope(signal, audio_config), audio_config)

    assert np.all(onsets >= 100.0)
    assert onsets[0] == pytest.approx(100.5, abs=0.03)


def test_silence_has_no_onsets(audio_config):
    signal = AudioSignal(np.zeros(SAMPLE_RATE * 2, dtype=np.float32), SAMPLE_RATE)
    assert len(detect_onsets(onset_envelope(signal, audio_config), audio_config)) == 0


class TestMergeClose:
    def test_chord_becomes_one_attack(self):
        assert np.allclose(merge_close([1.0, 1.004, 1.011, 2.0], 0.03), [1.0, 2.0])

    def test_rolled_chord_merges_whole(self):
        """Арпеджиато с шагом 15 мс — один аккорд, а не четыре ноты."""
        assert len(merge_close([1.0, 1.015, 1.030, 1.045], 0.03)) == 1

    def test_trill_is_kept(self):
        trill = np.arange(0, 1, 0.07)
        assert len(merge_close(trill, 0.03)) == len(trill)

    def test_empty(self):
        assert len(merge_close([], 0.03)) == 0


def note(pitch: int, onset: float, offset: float | None = None) -> NoteEvent:
    return NoteEvent(
        pitch, onset, offset or onset + 0.4, 70, Hand.RIGHT, Source.VIDEO, confidence=0.8
    )


class TestRefine:
    def test_moves_to_nearest_attack(self):
        result = refine_note_onsets([note(60, 1.03)], np.array([0.5, 1.0, 1.6]), 0.05)
        assert result[0].onset == pytest.approx(1.0)

    def test_never_beyond_max_shift(self):
        result = refine_note_onsets([note(60, 1.08)], np.array([1.0]), 0.05)
        assert result[0].onset == pytest.approx(1.08)

    def test_chord_snaps_together(self):
        chord = [note(60, 2.00), note(64, 2.033), note(67, 1.98)]
        result = refine_note_onsets(chord, np.array([2.01]), 0.05)
        assert {n.onset for n in result} == {2.01}

    def test_keeps_order_offset_and_attributes(self):
        notes = [note(62, 3.04, 3.5), note(60, 1.02, 1.4)]
        result = refine_note_onsets(notes, np.array([1.0, 3.0]), 0.05)
        assert [n.pitch for n in result] == [62, 60]
        assert result[0].offset == 3.5
        assert result[0].hand is Hand.RIGHT
        assert result[0].source is Source.VIDEO
        assert result[0].confidence == 0.8

    def test_does_not_create_inverted_note(self):
        """Атака после конца очень короткой ноты не должна её вывернуть."""
        short = note(60, 1.00, 1.02)
        result = refine_note_onsets([short], np.array([1.03]), 0.05)
        assert result[0].onset == 1.00

    def test_no_onsets_returns_copy(self):
        notes = [note(60, 1.0)]
        result = refine_note_onsets(notes, np.empty(0), 0.05)
        assert result == notes
        assert result is not notes
