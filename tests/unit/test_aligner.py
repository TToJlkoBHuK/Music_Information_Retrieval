"""Оценка и компенсация рассинхрона звука и картинки (#29, F-11)."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from mir.audio.onsets import detect_onsets, onset_envelope
from mir.audio.signal import AudioSignal
from mir.common.types import NoteEvent
from mir.config import FusionConfig, MirConfig, load_config
from mir.fusion.aligner import (
    apply_offset,
    estimate_av_offset,
    estimate_av_offset_from_onsets,
)
from tests.fixtures.audio_synth import SAMPLE_RATE, render_notes, scale_melody


@pytest.fixture(scope="module")
def full_config():
    return load_config(use_user_config=False, use_env=False)


@pytest.fixture(scope="module")
def config(full_config: MirConfig) -> FusionConfig:
    return full_config.fusion


def random_melody(count: int = 300, seed: int = 5) -> list[NoteEvent]:
    rng = np.random.default_rng(seed)
    notes, t = [], 0.5
    for _ in range(count):
        duration = float(rng.uniform(0.1, 0.6))
        notes.append(NoteEvent(int(rng.integers(40, 85)), t, t + duration, 80))
        t += float(rng.choice([0.12, 0.25, 0.37, 0.5]))
    return notes


def as_video(notes: list[NoteEvent], fps: float | None) -> list[NoteEvent]:
    """Как ноты видит видеоканал: подсветка — с первого кадра после атаки."""
    if fps is None:
        return notes
    result = []
    for note in notes:
        onset = float(np.ceil(note.onset * fps) / fps)
        result.append(replace(note, onset=onset, offset=max(note.offset, onset + 1 / fps)))
    return result


def as_audio(notes: list[NoteEvent], delta: float, seed: int = 0) -> list[NoteEvent]:
    rng = np.random.default_rng(seed)
    return [
        replace(n, onset=n.onset + delta + float(rng.normal(0, 0.003)), offset=n.offset + delta)
        for n in notes
    ]


@pytest.mark.parametrize("delta", [-0.3, 0.0, 0.12, 0.45])
def test_shift_found(config, delta):
    notes = random_melody()
    estimate = estimate_av_offset(notes, as_audio(notes, delta), config)
    assert estimate.offset == pytest.approx(delta, abs=0.003)
    assert estimate.is_reliable(config.min_offset_confidence)


@pytest.mark.parametrize("fps", [25, 60])
def test_video_frame_lag_is_absorbed(config, fps):
    """Подсветка появляется на кадре после атаки, в среднем на полкадра позже.

    Найденный сдвиг включает это запаздывание — так и нужно: после
    компенсации звук ложится на шкалу видео.
    """
    notes = random_melody()
    estimate = estimate_av_offset(as_video(notes, fps), as_audio(notes, 0.2), config)
    frame = 1 / fps
    assert 0.2 - frame <= estimate.offset <= 0.2 + 0.003


def test_melody_with_repeating_pattern(config):
    notes = scale_melody(110, 16, subdivision=4)
    estimate = estimate_av_offset(as_video(notes, 30), as_audio(notes, 0.15), config)
    assert estimate.offset == pytest.approx(0.15, abs=1 / 30)
    assert estimate.is_reliable(config.min_offset_confidence)


def test_periodic_material_is_not_trusted(config):
    """Одна нота с ровным шагом: любой сдвиг на шаг совпадает так же хорошо."""
    notes = [NoteEvent(60, 0.5 + i * 0.5, 0.8 + i * 0.5, 80) for i in range(80)]
    estimate = estimate_av_offset(notes, as_audio(notes, 0.1), config)
    assert not estimate.is_reliable(config.min_offset_confidence)


def test_too_few_pairs(config):
    notes = random_melody(count=4)
    estimate = estimate_av_offset(notes, as_audio(notes, 0.1), config)
    assert estimate.confidence == 0.0


def test_no_common_pitches(config):
    video = [NoteEvent(60, 1.0 + i, 1.5 + i, 80) for i in range(20)]
    audio = [NoteEvent(72, 1.1 + i, 1.5 + i, 80) for i in range(20)]
    assert estimate_av_offset(video, audio, config).confidence == 0.0


class TestFromOnsets:
    """Запасной путь без транскрипции — по атакам звука."""

    def onsets(self, notes, delay, audio_config):
        samples = render_notes(notes)
        padded = np.concatenate([np.zeros(int(delay * SAMPLE_RATE), np.float32), samples])
        envelope = onset_envelope(AudioSignal(padded, SAMPLE_RATE), audio_config)
        return detect_onsets(envelope, audio_config)

    def test_irregular_rhythm(self, config, full_config):
        notes = random_melody(count=120)
        onsets = self.onsets(notes, 0.2, full_config.audio)
        estimate = estimate_av_offset_from_onsets(as_video(notes, 30), onsets, config)
        assert estimate.offset == pytest.approx(0.2, abs=1 / 30)
        assert estimate.is_reliable(config.min_offset_confidence)

    def test_even_rhythm_not_trusted(self, config, full_config):
        """Ровные шестнадцатые без высот неоднозначны — сдвиг не применяется."""
        notes = scale_melody(110, 8, subdivision=4)
        onsets = self.onsets(notes, 0.2, full_config.audio)
        estimate = estimate_av_offset_from_onsets(as_video(notes, 30), onsets, config)
        assert not estimate.is_reliable(config.min_offset_confidence)


class TestApplyOffset:
    def test_moves_both_ends(self):
        shifted = apply_offset([NoteEvent(60, 1.0, 1.5, 80)], -0.25)
        assert (shifted[0].onset, shifted[0].offset) == (0.75, 1.25)

    def test_negative_time_clipped_or_dropped(self):
        notes = [NoteEvent(60, 0.1, 0.3, 80), NoteEvent(62, 0.1, 0.6, 80)]
        shifted = apply_offset(notes, -0.4)
        assert len(shifted) == 1
        assert shifted[0].onset == 0.0
        assert shifted[0].offset == pytest.approx(0.2)

    def test_roundtrip_with_estimate(self, config):
        notes = random_melody()
        audio = as_audio(notes, 0.3)
        estimate = estimate_av_offset(notes, audio, config)
        aligned = apply_offset(audio, -estimate.offset)
        errors = [a.onset - v.onset for a, v in zip(aligned, notes, strict=True)]
        assert abs(float(np.mean(errors))) < 0.002
