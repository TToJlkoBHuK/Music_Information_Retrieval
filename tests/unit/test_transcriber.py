"""Проход нейросети по сегментам и перевод ответа в события (#26).

Модель здесь подставная: вместо разметки она возвращает номер кадра,
который видит. Тогда склейка проверяется точно — в общей разметке кадр
номер `g` обязан содержать значение `g`, без пропусков и наложений.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import numpy.typing as npt
import pytest

from mir.audio.signal import AudioSignal
from mir.audio.transcriber import AudioTranscriber, DecodeThresholds, SegmentPlan
from mir.common.enums import Source
from mir.common.errors import CancelledError
from mir.config import AudioConfig, load_config

SR = 1000
FPS = 100
SEGMENT = 2000  # 2 с: короткие сегменты, чтобы тест шёл быстро
FRAMES = SEGMENT * FPS // SR


class FrameIndexModel:
    """Возвращает в каждом кадре его номер в общей разметке."""

    sample_rate = SR
    frames_per_second = FPS
    segment_samples = SEGMENT
    device = "fake"

    def __init__(
        self,
        notes: npt.NDArray[np.float32] | None = None,
        pedals: npt.NDArray[np.float32] | None = None,
    ):
        self.calls = 0
        self.position = 0  # номер следующего сегмента
        self.notes = np.empty((0, 4), dtype=np.float32) if notes is None else notes
        self.pedals = np.empty((0, 2), dtype=np.float32) if pedals is None else pedals

    def forward(self, segments):
        self.calls += 1
        batch = []
        for _ in segments:
            start = self.position * FRAMES // 2
            frames = np.arange(start, start + FRAMES + 1, dtype=np.float32)
            batch.append(np.repeat(frames[:, None], 88, axis=1))
            self.position += 1
        stacked = np.stack(batch)
        return {"frame_output": stacked, "reg_onset_output": np.full_like(stacked, 0.25)}

    def decode(self, outputs, thresholds):
        self.decoded = outputs
        return self.notes, self.pedals


@pytest.fixture
def config() -> AudioConfig:
    return load_config(use_user_config=False, use_env=False).audio


def run(config, seconds: float, batch_size: int = 1):
    model = FrameIndexModel()
    transcriber = AudioTranscriber(model, replace(config, batch_size=batch_size))
    samples = np.zeros(int(seconds * SR), dtype=np.float32)
    return transcriber.infer(samples), model


class TestSegmentPlan:
    @pytest.mark.parametrize("samples", [1, 1999, 2000, 2001, 9000, 20000])
    def test_kept_frames_tile_without_gaps(self, samples):
        plan = SegmentPlan.build(samples, SEGMENT, FRAMES)
        covered = np.zeros(plan.total_frames, dtype=int)
        for index in range(plan.count):
            lo, hi, at = plan.kept_frames(index)
            covered[at : at + hi - lo] += 1
        assert np.all(covered == 1)

    def test_segment_count(self):
        # 4.5 с → дополнение до 6 с → 3 целых сегмента → 5 с перекрытием.
        assert SegmentPlan.build(4500, SEGMENT, FRAMES).count == 5

    def test_frames_must_divide_by_four(self):
        with pytest.raises(ValueError, match="кратно 4"):
            SegmentPlan.build(100, SEGMENT, 199)


class TestInfer:
    @pytest.mark.parametrize("seconds", [0.5, 2.0, 2.3, 7.7])
    def test_every_frame_from_its_place(self, config, seconds):
        outputs, _ = run(config, seconds)
        values = outputs["frame_output"][:, 0]
        assert np.array_equal(values, np.arange(len(values)))

    def test_length_matches_signal(self, config):
        outputs, _ = run(config, 7.7)
        assert len(outputs["frame_output"]) == int(np.ceil(7.7 * FPS)) + 1

    def test_batches_change_nothing(self, config):
        single, model_single = run(config, 9.1, batch_size=1)
        batched, model_batched = run(config, 9.1, batch_size=4)
        assert np.array_equal(single["frame_output"], batched["frame_output"])
        assert model_batched.calls < model_single.calls

    def test_progress_reaches_one(self, config):
        seen: list[float] = []
        AudioTranscriber(FrameIndexModel(), config).infer(
            np.zeros(9000, dtype=np.float32), progress=seen.append
        )
        assert seen == sorted(seen)
        assert seen[-1] == pytest.approx(1.0)

    def test_cancel_stops_before_next_segment(self, config):
        model = FrameIndexModel()
        calls = iter([False, False, True])
        with pytest.raises(CancelledError):
            AudioTranscriber(model, config).infer(
                np.zeros(20000, dtype=np.float32), cancel=lambda: next(calls)
            )
        assert model.calls == 2


class TestEvents:
    def transcribe(self, config, notes, pedals=None, start=0.0):
        model = FrameIndexModel(np.array(notes, dtype=np.float32), pedals)
        signal = AudioSignal(np.zeros(5 * SR, dtype=np.float32), SR, start=start)
        return AudioTranscriber(model, config).transcribe(signal)

    def test_converted_to_note_events(self, config):
        result = self.transcribe(config, [[1.0, 1.5, 60, 0.5]])
        note = result.notes[0]
        assert (note.pitch, note.onset, note.offset) == (60, 1.0, 1.5)
        assert note.velocity == 64
        assert note.source is Source.AUDIO
        assert note.confidence == pytest.approx(0.25)

    def test_velocity_clamped_to_midi(self, config):
        """У исходной библиотеки громкость 1.0 превращалась в 128."""
        result = self.transcribe(config, [[1.0, 1.5, 60, 1.0], [2.0, 2.5, 62, 0.0]])
        assert [n.velocity for n in result.notes] == [127, 1]

    def test_zero_length_note_gets_one_frame(self, config):
        result = self.transcribe(config, [[1.0, 1.0, 60, 0.5]])
        assert result.notes[0].duration == pytest.approx(1 / FPS)

    def test_out_of_range_pitch_skipped(self, config):
        result = self.transcribe(config, [[1.0, 1.5, 15, 0.5], [1.0, 1.5, 60, 0.5]])
        assert [n.pitch for n in result.notes] == [60]

    def test_window_start_applied(self, config):
        result = self.transcribe(config, [[1.0, 1.5, 60, 0.5]], [[0.5, 2.0]], start=30.0)
        assert result.notes[0].onset == pytest.approx(31.0)
        assert result.pedals[0].onset == pytest.approx(30.5)

    def test_sorted_by_time(self, config):
        result = self.transcribe(config, [[2.0, 2.5, 60, 0.5], [1.0, 1.5, 64, 0.5]])
        assert [n.onset for n in result.notes] == [1.0, 2.0]

    def test_thresholds_from_config(self, config):
        thresholds = AudioTranscriber(FrameIndexModel(), config).thresholds
        assert thresholds == DecodeThresholds(
            config.onset_threshold,
            config.offset_threshold,
            config.frame_threshold,
            config.pedal_offset_threshold,
        )

    def test_other_sample_rate_resampled(self, config):
        model = FrameIndexModel()
        signal = AudioSignal(np.zeros(4 * SR * 2, dtype=np.float32), SR * 2)
        result = AudioTranscriber(model, config).transcribe(signal)
        # 4 с при частоте модели — ровно два сегмента, без «лишнего» кадра.
        assert len(model.decoded["frame_output"]) == 4 * FPS
        assert result.segments == 3
