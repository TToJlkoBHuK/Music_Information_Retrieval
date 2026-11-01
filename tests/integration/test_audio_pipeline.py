"""Этап 4 целиком: звуковой файл → ноты, атаки, темп.

Три уровня проверки, от быстрого к тяжёлому:

* без нейросети — атаки и темп на синтетике, секунды;
* с подставной нейросетью — проход конвейера, прогресс, отмена;
* с настоящей архитектурой ByteDance на случайных весах (`requires_torch`):
  потоковый проход по сегментам обязан давать в точности тот же ответ,
  что и исходная библиотека. Веса для этого не нужны — сравнивается
  механика, а не качество;
* с официальными весами (`requires_model`) — дымовой прогон, если веса
  уже скачаны.
"""

from __future__ import annotations

import contextlib
import importlib.util
import io
from pathlib import Path

import numpy as np
import pytest

from mir.audio import analyze_audio
from mir.audio.models import ModelRegistry
from mir.common.enums import Source, Stage
from mir.common.errors import CancelledError, TranscriptionError
from mir.config import MirConfig
from tests.fixtures.audio_synth import render_notes, scale_melody, write_wav

HAS_TORCH = all(
    importlib.util.find_spec(name) is not None
    for name in ("torch", "piano_transcription_inference", "torchlibrosa")
)
requires_torch = pytest.mark.skipif(
    not HAS_TORCH, reason="PyTorch и piano_transcription_inference не установлены"
)


@pytest.fixture(scope="module")
def melody_wav(tmp_path_factory: pytest.TempPathFactory) -> Path:
    notes = scale_melody(100, 8)
    path = tmp_path_factory.mktemp("audio") / "melody.wav"
    return write_wav(path, render_notes(notes))


class ConstantModel:
    """Подставная нейросеть: всегда «слышит» одну ноту."""

    sample_rate = 16000
    frames_per_second = 100
    segment_samples = 160000
    device = "fake"

    def forward(self, segments):
        shape = (len(segments), 1001, 88)
        return {"reg_onset_output": np.full(shape, 0.9, dtype=np.float32)}

    def decode(self, outputs, thresholds):
        return np.array([[1.0, 1.5, 60, 0.5]], dtype=np.float32), np.empty((0, 2), np.float32)


def test_tempo_and_onsets_without_network(melody_wav, config: MirConfig):
    result = analyze_audio(melody_wav, config, transcribe=False)

    assert result.notes == []
    assert result.device is None
    assert abs(result.tempo.bpm - 100) / 100 < 0.02
    assert len(result.onsets) >= 60
    assert 19.0 < result.duration < 21.0


def test_window_keeps_absolute_time(melody_wav, config: MirConfig):
    result = analyze_audio(melody_wav, config, transcribe=False, start_seconds=5.0, max_seconds=6.0)
    assert result.onsets.min() >= 5.0
    assert result.tempo.beats[0] >= 5.0


def test_progress_and_fake_network(melody_wav, config: MirConfig):
    stages: list[tuple[Stage, float]] = []

    result = analyze_audio(
        melody_wav, config, model=ConstantModel(), progress=lambda s, f: stages.append((s, f))
    )

    assert {stage for stage, _ in stages} == {Stage.AUDIO}
    fractions = [f for _, f in stages]
    assert fractions == sorted(fractions)
    assert fractions[-1] == 1.0
    assert result.device == "fake"
    assert result.notes[0].source is Source.AUDIO
    assert result.notes[0].confidence == pytest.approx(0.9)


def test_cancel(melody_wav, config: MirConfig):
    with pytest.raises(CancelledError):
        analyze_audio(melody_wav, config, model=ConstantModel(), cancel=lambda: True)


def test_manual_tempo_passed_through(melody_wav, config: MirConfig):
    result = analyze_audio(melody_wav, config, transcribe=False, force_bpm=75.0)
    assert result.tempo.is_manual
    assert result.tempo.bpm == pytest.approx(75.0, rel=1e-3)


@pytest.fixture(scope="session")
def random_checkpoint(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Файл весов со случайными значениями — той же структуры, что официальный."""
    import torch
    from piano_transcription_inference.models import Note_pedal

    torch.manual_seed(0)
    network = Note_pedal(frames_per_second=100, classes_num=88)
    path = tmp_path_factory.mktemp("weights") / "random.pth"
    state = {
        "note_model": network.note_model.state_dict(),
        "pedal_model": network.pedal_model.state_dict(),
    }
    torch.save({"model": state, "iteration": 0}, path)
    return path


@requires_torch
def test_streaming_matches_original_library(random_checkpoint, config: MirConfig):
    """Свой проход по сегментам и исходный дают одинаковую разметку до бита.

    Исходный путь держит все сегменты в памяти и печатает в консоль;
    свой — потоковый, с прогрессом и отменой. Разметка обязана совпасть,
    иначе качество разошлось бы с опубликованным.
    """
    from piano_transcription_inference import pytorch_utils
    from piano_transcription_inference.inference import PianoTranscription

    from mir.audio.bytedance import ByteDanceModel
    from mir.audio.transcriber import AudioTranscriber

    model = ByteDanceModel(random_checkpoint, "cpu", official=False)
    samples = render_notes(scale_melody(100, 5))
    ours = AudioTranscriber(model, config.audio).infer(samples)

    segment = model.segment_samples
    padded = np.zeros((1, int(np.ceil(len(samples) / segment)) * segment))
    padded[0, : len(samples)] = samples
    segments = PianoTranscription.enframe(None, padded, segment)
    with contextlib.redirect_stdout(io.StringIO()):
        reference = pytorch_utils.forward(model._network, segments, batch_size=1)

    for name, values in ours.items():
        expected = PianoTranscription.deframe(None, reference[name])[: len(values)]
        assert np.array_equal(values, expected), name


@requires_torch
def test_incomplete_weights_rejected(random_checkpoint, tmp_path: Path):
    """Исходная библиотека грузит с strict=False и молча работает на мусоре."""
    import torch

    from mir.audio.bytedance import ByteDanceModel

    data = torch.load(str(random_checkpoint), weights_only=True)
    first = next(k for k in data["model"]["note_model"] if "gru" in k)
    del data["model"]["note_model"][first]
    broken = tmp_path / "broken.pth"
    torch.save(data, broken)

    with pytest.raises(TranscriptionError, match="нет 1 параметров"):
        ByteDanceModel(broken, "cpu", official=False)


@requires_torch
def test_official_weights_smoke(melody_wav, config: MirConfig):
    """Если официальные веса уже скачаны — полный прогон на них."""
    found = ModelRegistry(config.ingest.cache_path, config.audio.model_path).locate()
    if found is None:
        pytest.skip("официальные веса не скачаны (run audio fetch-model)")

    result = analyze_audio(melody_wav, config, max_seconds=10.0)

    assert result.device in ("cpu", "cuda")
    assert all(21 <= n.pitch <= 108 for n in result.notes)
    assert all(n.offset > n.onset for n in result.notes)
