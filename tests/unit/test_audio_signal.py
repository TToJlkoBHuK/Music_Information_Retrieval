"""Чтение звука с диска."""

from __future__ import annotations

import wave
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pytest

from mir.audio.signal import AudioSignal, load_audio
from mir.common.errors import TranscriptionError
from tests.fixtures.audio_synth import write_wav

SR = 16000


def tone(seconds: float = 2.0, freq: float = 440.0) -> npt.NDArray[np.float32]:
    t = np.arange(int(seconds * SR)) / SR
    return (0.5 * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def test_reads_what_was_written(tmp_path: Path):
    signal = tone()
    path = write_wav(tmp_path / "a.wav", signal)

    loaded = load_audio(path)

    assert loaded.sample_rate == SR
    assert loaded.start == 0.0
    assert np.allclose(loaded.samples, signal, atol=1e-4)


def test_window_keeps_absolute_time(tmp_path: Path):
    """Отрезок помнит, откуда он взят: ноты из него не нужно сдвигать руками."""
    path = write_wav(tmp_path / "a.wav", tone(5.0))

    loaded = load_audio(path, start=1.5, max_seconds=2.0)

    assert loaded.start == 1.5
    assert loaded.duration == pytest.approx(2.0)
    assert loaded.end == pytest.approx(3.5)


def test_stereo_is_mixed_to_mono(tmp_path: Path):
    left = tone(1.0, 440)
    right = np.zeros_like(left)
    pcm = (np.stack([left, right], axis=1) * 32767).astype("<i2")
    path = tmp_path / "stereo.wav"
    with wave.open(str(path), "wb") as fh:
        fh.setnchannels(2)
        fh.setsampwidth(2)
        fh.setframerate(SR)
        fh.writeframes(pcm.tobytes())

    loaded = load_audio(path)

    assert loaded.samples.ndim == 1
    assert np.allclose(loaded.samples, left / 2, atol=1e-4)


def test_24_bit(tmp_path: Path):
    signal = tone(0.5)
    ints = np.round(signal * (2**23 - 1)).astype("<i4")
    raw = np.stack([(ints >> shift) & 0xFF for shift in (0, 8, 16)], axis=1).astype(np.uint8)
    path = tmp_path / "deep.wav"
    with wave.open(str(path), "wb") as fh:
        fh.setnchannels(1)
        fh.setsampwidth(3)
        fh.setframerate(SR)
        fh.writeframes(raw.tobytes())

    loaded = load_audio(path)

    assert np.allclose(loaded.samples, signal, atol=1e-5)


def test_missing_file(tmp_path: Path):
    with pytest.raises(TranscriptionError) as info:
        load_audio(tmp_path / "nope.wav")
    assert "не найден" in info.value.user_message


def test_start_beyond_end(tmp_path: Path):
    path = write_wav(tmp_path / "a.wav", tone(1.0))
    with pytest.raises(TranscriptionError, match="нет отсчётов"):
        load_audio(path, start=10.0)


def test_resample_keeps_start_and_duration():
    signal = AudioSignal(tone(1.0), SR, start=3.0)

    converted = signal.resampled(8000)

    assert converted.sample_rate == 8000
    assert converted.start == 3.0
    assert converted.duration == pytest.approx(1.0, abs=1e-3)


def test_resample_same_rate_is_noop():
    signal = AudioSignal(tone(0.5), SR)
    assert signal.resampled(SR) is signal
