"""Темп и сетка долей (#27, NF-08).

NF-08 требует ошибки темпа не больше 2 %. Темп метронома должен
находиться точно. У музыки темп проверяется с точностью до метрического
уровня — множителя 2: восьмые при 60 BPM и четверти при 120 звучат
одинаково, и по звуку выбрать нельзя (так же считает MIREX, «Accuracy 2»).
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pytest

from mir.audio.onsets import detect_onsets, onset_envelope
from mir.audio.signal import AudioSignal
from mir.audio.tempo import TempoAnalysis, TempoAnalyzer, beat_match_rate, estimate_downbeats
from mir.config import AudioConfig, load_config
from tests.fixtures.audio_synth import (
    SAMPLE_RATE,
    metronome,
    render_notes,
    scale_melody,
    tempo_ramp_clicks,
)

NF08 = 0.02


@pytest.fixture(scope="module")
def config() -> AudioConfig:
    return load_config(use_user_config=False, use_env=False).audio


def analyze(samples: npt.NDArray[np.float32], config: AudioConfig, **kwargs) -> TempoAnalysis:
    signal = AudioSignal(samples, SAMPLE_RATE)
    envelope = onset_envelope(signal, config)
    onsets = detect_onsets(envelope, config)
    return TempoAnalyzer(config).analyze(envelope, onsets, **kwargs)


def error_modulo_level(found: float, true: float) -> float:
    return min(abs(found - true * k) / (true * k) for k in (0.5, 1.0, 2.0))


@pytest.mark.parametrize("bpm", [60, 90, 120, 150])
def test_metronome_exact(config, bpm):
    """Щелчок — это доля: здесь метрический уровень однозначен."""
    samples, _ = metronome(bpm, 30)
    result = analyze(samples, config)
    assert abs(result.tempo.bpm - bpm) / bpm < NF08
    assert result.tempo.confidence > 0.9


@pytest.mark.parametrize("bpm", [70, 100, 140])
@pytest.mark.parametrize("subdivision", [2, 4])
def test_music_within_nf08(config, bpm, subdivision):
    result = analyze(render_notes(scale_melody(bpm, 8, subdivision=subdivision)), config)
    assert error_modulo_level(result.tempo.bpm, bpm) < NF08


def test_beats_land_on_beats(config):
    """Не только темп, но и положение долей: допуск 70 мс, как в MIREX."""
    bpm = 100
    result = analyze(render_notes(scale_melody(bpm, 8)), config)
    true_beats = 0.5 + np.arange(32) * 60 / bpm
    assert beat_match_rate(true_beats, result.tempo.beats, 0.07) > 0.95


def test_accelerando_followed(config):
    """Ускорение 90 → 130 BPM.

    При жёсткости librosa по умолчанию терялась каждая десятая доля,
    при 50 — ни одной.
    """
    samples, clicks = tempo_ramp_clicks(90, 130, 40)
    result = analyze(samples, config)
    assert beat_match_rate(clicks, result.tempo.beats, 0.07) > 0.97
    assert result.tempo.local_bpm(5.0) < result.tempo.local_bpm(35.0)


def test_manual_tempo_is_exact(config):
    """Темп, введённый пользователем, не должен «плыть» за атаками."""
    result = analyze(render_notes(scale_melody(100, 8)), config, force_bpm=50.0)
    assert result.tempo.bpm == pytest.approx(50.0, rel=1e-3)
    assert result.tempo.is_manual
    assert result.warnings == []


def test_manual_tempo_aligned_to_attacks(config):
    result = analyze(render_notes(scale_melody(100, 8)), config, force_bpm=100.0)
    true_beats = 0.5 + np.arange(32) * 0.6
    assert beat_match_rate(true_beats, result.tempo.beats, 0.05) > 0.95


def test_silence_falls_back_with_warning(config):
    result = analyze(np.zeros(SAMPLE_RATE * 4, dtype=np.float32), config)
    assert result.tempo.confidence == 0.0
    assert result.warnings
    assert "вручную" in result.warnings[0]


class TestOctaveCorrection:
    """Выбор метрического уровня по типичной длительности ноты."""

    @staticmethod
    def correct(config: AudioConfig, bpm: float, ioi: float) -> float:
        attacks = np.arange(40) * ioi
        corrected, _ = TempoAnalyzer(config)._octave_correction(bpm, attacks)
        return corrected

    def test_half_notes_mean_tempo_too_high(self, config):
        # Интервал 1 с при 120 BPM — половинные: вероятнее, это четверти при 60.
        assert self.correct(config, 120.0, 1.0) == pytest.approx(60.0)

    def test_thirty_seconds_mean_tempo_too_low(self, config):
        # Интервал 0.125 с при 60 BPM — тридцать вторые: скорее шестнадцатые при 120.
        assert self.correct(config, 60.0, 0.125) == pytest.approx(120.0)

    @pytest.mark.parametrize(("bpm", "ioi"), [(120.0, 0.5), (120.0, 0.25), (120.0, 0.125)])
    def test_reasonable_reading_untouched(self, config, bpm, ioi):
        assert self.correct(config, bpm, ioi) == bpm

    def test_quarter_with_jitter_not_flipped(self, config):
        """Метроном: интервал ровно в долю ± шум не должен менять уровень."""
        assert self.correct(config, 90.4, 60 / 90) == 90.4

    def test_respects_bpm_limits(self, config):
        assert self.correct(config, config.max_bpm, 0.01) == config.max_bpm

    def test_too_few_intervals_left_alone(self, config):
        corrected, ioi = TempoAnalyzer(config)._octave_correction(120.0, np.array([0.0, 1.0]))
        assert corrected == 120.0
        assert ioi is None


def test_beat_match_rate():
    beats = np.array([1.0, 2.0, 3.0, 4.0])
    assert beat_match_rate(beats, np.array([1.01, 2.5, 3.0]), 0.05) == pytest.approx(0.5)
    assert beat_match_rate(beats, np.empty(0), 0.05) == 0.0


@pytest.mark.parametrize("material", ["metronome", "melody"])
def test_downbeat_phase_follows_accent(config, material):
    """Сетка начинается со второй доли такта: сильная должна сдвинуться.

    По потоку в децибелах это не работало: щелчок с акцентом и без давали
    почти одинаковый пик. Мера акцента — поток по мощности.
    """
    if material == "metronome":
        samples, _ = metronome(120, 12, beats_per_bar=4, start=0.5)
    else:
        samples = render_notes(scale_melody(120, 6))
    envelope = onset_envelope(AudioSignal(samples, SAMPLE_RATE), config)
    beats = 1.0 + np.arange(20) * 0.5
    downbeats = estimate_downbeats(beats, envelope, 4)
    assert downbeats[0] == pytest.approx(2.5)
