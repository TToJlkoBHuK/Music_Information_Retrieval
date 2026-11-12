"""Рассинхрон как функция времени и октава обрезанной клавиатуры (#30)."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace

import numpy as np
import pytest

from mir.common.types import NoteEvent
from mir.config import FusionConfig, load_config
from mir.fusion.aligner import estimate_offset_curve, estimate_pitch_shift


@pytest.fixture(scope="module")
def config() -> FusionConfig:
    return load_config(use_user_config=False, use_env=False).fusion


def melody(seconds: float, seed: int = 3) -> list[NoteEvent]:
    """Неровная мелодия на всю длину: около четырёх нот в секунду."""
    rng = np.random.default_rng(seed)
    notes, t = [], 1.0
    while t < seconds:
        duration = float(rng.uniform(0.1, 0.5))
        notes.append(NoteEvent(int(rng.integers(45, 85)), t, t + duration, 70))
        t += float(rng.choice([0.12, 0.25, 0.37]))
    return notes


def heard(
    notes: list[NoteEvent], offset: Callable[[float], float], seed: int = 1
) -> list[NoteEvent]:
    """Те же ноты в звуке: сдвиг зависит от времени, начало с разбросом 5 мс."""
    rng = np.random.default_rng(seed)
    result = []
    for note in notes:
        onset = note.onset + offset(note.onset) + float(rng.normal(0, 0.005))
        result.append(replace(note, onset=onset, offset=onset + note.duration))
    return result


def test_constant_offset_gives_flat_curve(config: FusionConfig):
    video = melody(200)
    curve = estimate_offset_curve(video, heard(video, lambda t: 0.150), config)

    assert curve.reliable
    assert curve.drift < 0.004
    assert curve.at(100.0) == pytest.approx(0.150, abs=0.003)


def test_slow_drift_is_tracked(config: FusionConfig):
    """Часы двух устройств расходятся на сто миллионных — как у Clair de lune."""

    def drifting(t: float) -> float:
        return 0.090 + 1e-4 * t

    video = melody(300)
    curve = estimate_offset_curve(video, heard(video, drifting), config)

    for t in (30.0, 150.0, 270.0):
        assert curve.at(t) == pytest.approx(drifting(t), abs=0.004)
    assert curve.drift == pytest.approx(0.027, abs=0.006)


def test_step_is_tracked(config: FusionConfig):
    """Сдвиг меняется ступенькой — как у «Лунной сонаты» на 300-й секунде."""

    def stepping(t: float) -> float:
        return 0.114 if t < 150 else 0.086

    video = melody(300)
    curve = estimate_offset_curve(video, heard(video, stepping), config)

    assert curve.at(60.0) == pytest.approx(0.114, abs=0.004)
    assert curve.at(240.0) == pytest.approx(0.086, abs=0.004)


def test_time_conversion_is_invertible(config: FusionConfig):
    video = melody(300)
    curve = estimate_offset_curve(video, heard(video, lambda t: 0.090 + 1e-4 * t), config)

    for t in (5.0, 123.4, 290.0):
        assert curve.to_video(curve.to_audio(t)) == pytest.approx(t, abs=1e-5)


def test_one_wrong_window_is_removed(config: FusionConfig):
    """В одном окне половина нот звука уехала на 60 мс: медиана по соседям это гасит."""
    video = melody(300)
    audio = heard(video, lambda t: 0.100)
    audio = [
        replace(n, onset=n.onset + 0.06, offset=n.offset + 0.06) if 120 < n.onset < 135 else n
        for n in audio
    ]

    curve = estimate_offset_curve(video, audio, config)

    assert curve.drift < 0.02


def test_unrelated_channels_are_not_aligned(config: FusionConfig):
    """Разная музыка в каналах: кривая помечена ненадёжной и равна нулю."""
    curve = estimate_offset_curve(melody(120, seed=1), melody(120, seed=2), config)

    assert not curve.reliable
    assert curve.at(60.0) == 0.0


def test_short_piece_falls_back_to_overall_offset(config: FusionConfig):
    """Нот меньше, чем нужно окну: остаётся один сдвиг на весь ролик."""
    video = melody(6)
    curve = estimate_offset_curve(video, heard(video, lambda t: 0.2), config)

    assert curve.reliable
    assert curve.at(3.0) == pytest.approx(0.2, abs=0.01)


def test_octave_error_is_found(config: FusionConfig):
    """Обрезанная клавиатура: видео ошиблось на октаву вниз."""
    truth = melody(120)
    video = [replace(n, pitch=n.pitch - 12) for n in truth]

    assert estimate_pitch_shift(video, heard(truth, lambda t: 0.1), config) == 12


def test_no_shift_when_channels_agree(config: FusionConfig):
    truth = melody(120)
    assert estimate_pitch_shift(truth, heard(truth, lambda t: 0.1), config) == 0


def test_octave_doubling_is_not_an_octave_error(config: FusionConfig):
    """В звуке к каждой третьей ноте добавлена октава: сдвиг высоты не нужен."""
    truth = melody(120)
    audio = heard(truth, lambda t: 0.1)
    audio += [replace(n, pitch=n.pitch + 12) for n in audio[::3]]

    assert estimate_pitch_shift(truth, audio, config) == 0


def test_no_shift_for_unrelated_channels(config: FusionConfig):
    assert estimate_pitch_shift(melody(120, seed=1), melody(120, seed=2), config) == 0
