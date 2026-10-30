"""Рассинхрон на собранном ролике: картинка и звук из одних нот (#29).

Ролик рендерится генератором видео, звук — генератором звука, оба из
одного списка нот; звук сдвигается на известную величину, и всё
сводится FFmpeg в mp4 с AAC — как у настоящих роликов. Дальше работает
весь конвейер: детекция клавиатуры, подсветка, извлечение звука, атаки,
оценка сдвига. Нейросеть не нужна: сдвиг оценивается по атакам.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from mir.audio import analyze_audio
from mir.common.types import NoteEvent
from mir.config import MirConfig
from mir.fusion import estimate_av_offset_from_onsets
from mir.ingest.demuxer import Demuxer
from mir.vision.analyzer import analyze_file
from tests.fixtures.audio_synth import render_notes, write_wav
from tests.fixtures.synth import SynthConfig, render_visualizer_video

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="FFmpeg не установлен")

DELAY = 0.25
FPS = 30.0


def irregular_melody() -> list[NoteEvent]:
    """Неровный ритм: на ровном сдвиг по одним атакам неоднозначен."""
    rng = np.random.default_rng(11)
    notes, t = [], 1.0
    for _ in range(36):
        duration = float(rng.uniform(0.15, 0.4))
        pitch = int(rng.integers(48, 84))
        notes.append(NoteEvent(pitch, t, t + duration, 90))
        if rng.random() < 0.3:
            notes.append(NoteEvent(pitch - 12, t, t + duration, 80))
        t += float(rng.choice([0.2, 0.3, 0.45, 0.6]))
    return notes


@pytest.fixture(scope="module")
def delayed_clip(tmp_path_factory: pytest.TempPathFactory) -> Path:
    work = tmp_path_factory.mktemp("sync")
    notes = irregular_melody()
    video = render_visualizer_video(
        notes, work / "video.mp4", SynthConfig(width=854, height=480, fps=FPS)
    )
    samples = render_notes(notes)
    delayed = np.concatenate([np.zeros(int(DELAY * 16000), np.float32), samples])
    audio = write_wav(work / "audio.wav", delayed)
    clip = work / "clip.mp4"
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", str(video), "-i", str(audio),
         "-c:v", "copy", "-c:a", "aac", "-b:a", "128k", str(clip)],
        check=True,
        capture_output=True,
    )  # fmt: skip
    return clip


def test_delay_recovered_end_to_end(delayed_clip: Path, config: MirConfig, tmp_path: Path):
    video_notes = analyze_file(delayed_clip, config).notes
    wav = Demuxer(config.ingest).extract_audio(delayed_clip, tmp_path / "out.wav")
    audio = analyze_audio(wav, config, transcribe=False)

    estimate = estimate_av_offset_from_onsets(video_notes, audio.onsets, config.fusion)

    # В найденный сдвиг входят запаздывание подсветки (0..33 мс при 30 fps)
    # и задержка детектора атак (~10 мс) — отсюда допуск в кадр.
    assert estimate.offset == pytest.approx(DELAY, abs=1 / FPS)
    assert estimate.is_reliable(config.fusion.min_offset_confidence)
