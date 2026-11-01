"""Сетка долей и педаль: модель данных этапа 4."""

from __future__ import annotations

import numpy as np
import pytest

from mir.common.types import PedalEvent, TempoMap


class TestConstantGrid:
    def test_bpm_is_exact(self):
        tempo = TempoMap.constant(120.0, duration=10.0)
        assert tempo.bpm == pytest.approx(120.0)

    def test_downbeats_every_bar(self):
        tempo = TempoMap.constant(60.0, duration=8.0, beats_per_bar=3)
        assert np.allclose(np.diff(tempo.downbeats), 3.0)

    def test_roundtrip(self):
        tempo = TempoMap.constant(90.0, duration=20.0, start=0.3)
        for t in (0.3, 1.7, 11.11, 19.9):
            assert tempo.beats_to_seconds(tempo.seconds_to_beats(t)) == pytest.approx(t)

    def test_rejects_non_positive_bpm(self):
        with pytest.raises(ValueError, match="bpm"):
            TempoMap.constant(0.0, duration=5.0)


class TestVariableTempo:
    """Интерполяция по долям, а не деление на средний темп."""

    @staticmethod
    def accelerando() -> TempoMap:
        # Доли сжимаются от 1.0 до 0.5 с: темп растёт с 60 до 120 BPM.
        intervals = np.linspace(1.0, 0.5, 40)
        return TempoMap(beats=np.concatenate([[0.0], np.cumsum(intervals)]))

    def test_beat_positions_follow_the_grid(self):
        tempo = self.accelerando()
        for index in (0, 10, 25, 40):
            assert tempo.seconds_to_beats(float(tempo.beats[index])) == pytest.approx(index)

    def test_linear_formula_would_drift(self):
        """Деление на средний темп к концу ошибается на несколько долей."""
        tempo = self.accelerando()
        end = float(tempo.beats[-1])
        linear = end * tempo.bpm / 60.0
        assert linear == pytest.approx(40.0)  # по краям совпадает...
        middle = float(tempo.beats[20])
        assert abs(middle * tempo.bpm / 60.0 - 20) > 2  # ...а в середине — нет
        assert tempo.seconds_to_beats(middle) == pytest.approx(20.0)

    def test_local_bpm(self):
        tempo = self.accelerando()
        assert tempo.local_bpm(0.1) == pytest.approx(60.0, rel=0.02)
        assert tempo.local_bpm(float(tempo.beats[-1]) - 0.1) == pytest.approx(120.0, rel=0.02)

    def test_extrapolation_outside_grid(self):
        tempo = TempoMap(beats=np.array([1.0, 2.0, 3.0]))
        assert tempo.seconds_to_beats(0.0) == pytest.approx(-1.0)
        assert tempo.seconds_to_beats(5.0) == pytest.approx(4.0)
        assert tempo.beats_to_seconds(-1.0) == pytest.approx(0.0)
        assert tempo.beats_to_seconds(4.0) == pytest.approx(5.0)


class TestGridOperations:
    def test_nearest_grid_point_sixteenths(self):
        tempo = TempoMap.constant(60.0, duration=10.0)
        assert tempo.nearest_grid_point(2.26, subdivision=4) == pytest.approx(2.25)
        assert tempo.nearest_grid_point(2.4, subdivision=2) == pytest.approx(2.5)

    def test_shifted(self):
        tempo = TempoMap.constant(120.0, duration=4.0)
        moved = tempo.shifted(0.25)
        assert np.allclose(moved.beats, tempo.beats + 0.25)
        assert np.allclose(moved.downbeats, tempo.downbeats + 0.25)
        assert moved.bpm == pytest.approx(tempo.bpm)

    def test_arrays_are_read_only(self):
        """frozen=True не защищает содержимое массива — защищает флаг."""
        tempo = TempoMap.constant(100.0, duration=4.0)
        with pytest.raises(ValueError):
            tempo.beats[0] = 5.0

    def test_source_array_not_shared(self):
        beats = np.array([0.0, 0.5, 1.0])
        tempo = TempoMap(beats=beats)
        beats[0] = -1.0
        assert tempo.beats[0] == 0.0


class TestValidation:
    def test_needs_two_beats(self):
        with pytest.raises(ValueError, match="двух долей"):
            TempoMap(beats=np.array([1.0]))

    def test_beats_must_increase(self):
        with pytest.raises(ValueError, match="возрастать"):
            TempoMap(beats=np.array([0.0, 1.0, 1.0]))

    def test_confidence_range(self):
        with pytest.raises(ValueError, match="confidence"):
            TempoMap(beats=np.array([0.0, 1.0]), confidence=2.0)


class TestPedalEvent:
    def test_duration(self):
        assert PedalEvent(1.0, 2.5).duration == pytest.approx(1.5)

    def test_offset_after_onset(self):
        with pytest.raises(ValueError, match="offset"):
            PedalEvent(2.0, 2.0)
