"""Модель данных и переводы времени."""

from __future__ import annotations

from fractions import Fraction

import pytest

from mir.common.enums import Source, Stage
from mir.common.timeutil import (
    DEFAULT_PPQ,
    beats_to_note_value,
    frame_to_seconds,
    seconds_to_frame,
    seconds_to_ticks,
    ticks_to_seconds,
)
from mir.common.types import (
    PITCH_MAX,
    PITCH_MIN,
    KeyboardLayout,
    KeySlot,
    NoteEvent,
    QualityReport,
)


class TestNoteEvent:
    def test_valid(self) -> None:
        note = NoteEvent(pitch=60, onset=1.0, offset=1.5, velocity=80)
        assert note.duration == pytest.approx(0.5)

    @pytest.mark.parametrize("pitch", [20, 109, 0, 127])
    def test_pitch_out_of_piano_range(self, pitch: int) -> None:
        with pytest.raises(ValueError, match="вне диапазона фортепиано"):
            NoteEvent(pitch=pitch, onset=0.0, offset=1.0, velocity=64)

    @pytest.mark.parametrize("pitch", [PITCH_MIN, 60, PITCH_MAX])
    def test_pitch_boundaries_ok(self, pitch: int) -> None:
        NoteEvent(pitch=pitch, onset=0.0, offset=1.0, velocity=64)

    def test_offset_must_follow_onset(self) -> None:
        with pytest.raises(ValueError, match="offset"):
            NoteEvent(pitch=60, onset=1.0, offset=1.0, velocity=64)

    def test_velocity_range(self) -> None:
        with pytest.raises(ValueError, match="velocity"):
            NoteEvent(pitch=60, onset=0.0, offset=1.0, velocity=200)

    def test_confidence_range(self) -> None:
        with pytest.raises(ValueError, match="confidence"):
            NoteEvent(pitch=60, onset=0.0, offset=1.0, velocity=64, confidence=1.5)

    def test_immutable(self) -> None:
        """Этапы создают новые списки, а не правят существующие."""
        note = NoteEvent(pitch=60, onset=0.0, offset=1.0, velocity=64)
        with pytest.raises(AttributeError):
            note.pitch = 61  # type: ignore[misc]


class TestSourceFlags:
    def test_combination_gives_both(self) -> None:
        assert Source.VIDEO | Source.AUDIO is Source.BOTH

    def test_membership(self) -> None:
        assert Source.VIDEO in Source.BOTH
        assert Source.AUDIO in Source.BOTH


class TestStage:
    def test_all_stages_have_russian_titles(self) -> None:
        for stage in Stage:
            assert stage.title_ru
            assert stage.title_ru != stage.value


class TestKeyboardLayout:
    @staticmethod
    def _layout() -> KeyboardLayout:
        keys = (
            KeySlot(pitch=60, x_min=0, x_max=20, is_black=False),
            KeySlot(pitch=61, x_min=15, x_max=25, is_black=True),
            KeySlot(pitch=62, x_min=20, x_max=40, is_black=False),
        )
        return KeyboardLayout(
            bbox=(0, 100, 40, 50),
            keys=keys,
            lowest_pitch=60,
            highest_pitch=62,
            is_cropped=True,
            confidence=0.9,
        )

    def test_black_key_wins_overlap(self) -> None:
        """Чёрные клавиши перекрывают белые, поэтому проверяются первыми."""
        assert self._layout().pitch_at(18) == 61

    def test_white_key(self) -> None:
        assert self._layout().pitch_at(5) == 60

    def test_outside(self) -> None:
        assert self._layout().pitch_at(500) is None

    def test_covers(self) -> None:
        layout = self._layout()
        assert layout.covers(61)
        assert not layout.covers(80)


class TestTimeUtil:
    def test_frame_roundtrip(self) -> None:
        fps = 30000 / 1001
        assert seconds_to_frame(frame_to_seconds(100, fps), fps) == 100

    def test_fractional_fps_not_rounded(self) -> None:
        """29.97 против 30: на часовом ролике разница около 3.6 секунды."""
        fps = 30000 / 1001
        drift = abs(frame_to_seconds(108000, fps) - frame_to_seconds(108000, 30.0))
        assert drift > 3.0

    def test_zero_fps_rejected(self) -> None:
        with pytest.raises(ValueError, match="fps"):
            frame_to_seconds(10, 0.0)

    def test_ticks_roundtrip(self) -> None:
        assert ticks_to_seconds(seconds_to_ticks(2.5, 120.0), 120.0) == pytest.approx(2.5)

    @pytest.mark.parametrize(
        ("beats", "expected"),
        [
            (4.0, Fraction(1, 1)),
            (2.0, Fraction(1, 2)),
            (1.0, Fraction(1, 4)),
            (0.5, Fraction(1, 8)),
        ],
    )
    def test_note_values(self, beats: float, expected: Fraction) -> None:
        assert beats_to_note_value(beats) == expected

    def test_triplets_sum_exactly(self) -> None:
        """Ключевая причина использовать Fraction вместо float."""
        triplet = beats_to_note_value(1 / 3)
        assert triplet * 3 == beats_to_note_value(1.0)

    def test_default_ppq_divisible(self) -> None:
        for divisor in (2, 3, 4, 5):
            assert DEFAULT_PPQ % divisor == 0


class TestQualityReport:
    def test_agreement_rate(self) -> None:
        report = QualityReport(
            notes_confirmed=80, notes_from_video_only=10, notes_from_audio_only=10
        )
        assert report.agreement_rate == pytest.approx(0.8)

    def test_empty_report_no_division_error(self) -> None:
        assert QualityReport().agreement_rate == 0.0
