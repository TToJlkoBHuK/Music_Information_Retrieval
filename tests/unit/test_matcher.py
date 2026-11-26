"""Сопоставление нот двух каналов."""

from __future__ import annotations

from mir.common.types import NoteEvent
from mir.fusion.matcher import match_notes

TOLERANCE = 0.08


def note(pitch: int, onset: float, duration: float = 0.3) -> NoteEvent:
    return NoteEvent(pitch=pitch, onset=onset, offset=onset + duration, velocity=64)


def test_chord_is_paired_note_by_note():
    """Аккорд из четырёх нот: каждая находит свою пару, а не одну общую."""
    video = [note(p, 1.0) for p in (48, 55, 60, 64)]
    audio = [note(p, 1.02) for p in (48, 55, 60, 64)]

    result = match_notes(video, audio, TOLERANCE)

    assert sorted((v.pitch, a.pitch) for v, a in result.pairs) == [(p, p) for p in (48, 55, 60, 64)]
    assert not result.video_only and not result.audio_only


def test_pitch_must_match():
    result = match_notes([note(60, 1.0)], [note(61, 1.0)], TOLERANCE)
    assert not result.pairs
    assert len(result.video_only) == len(result.audio_only) == 1


def test_tolerance_is_respected():
    result = match_notes([note(60, 1.0)], [note(60, 1.0 + TOLERANCE + 0.01)], TOLERANCE)
    assert not result.pairs


def test_one_audio_note_confirms_one_video_note():
    """Два повтора в видео и одна нота звука: подтверждён только ближайший."""
    video = [note(60, 1.00, 0.04), note(60, 1.06, 0.04)]
    audio = [note(60, 1.05)]

    result = match_notes(video, audio, TOLERANCE)

    assert [(v.onset, a.onset) for v, a in result.pairs] == [(1.06, 1.05)]
    assert [v.onset for v in result.video_only] == [1.00]


def test_missed_repeat_does_not_shift_pairs():
    """Звук пропустил первый из трёх быстрых повторов.

    Жадный проход отдал бы первую ноту видео первой ноте звука (она в
    допуске) и оставил бы без пары последнюю — настоящую. Лучшее
    сопоставление целиком оставляет без пары именно пропущенную.
    """
    video = [note(60, 1.00, 0.05), note(60, 1.07, 0.05), note(60, 1.14, 0.05)]
    audio = [note(60, 1.075), note(60, 1.145)]

    result = match_notes(video, audio, TOLERANCE)

    assert [(v.onset, a.onset) for v, a in result.pairs] == [(1.07, 1.075), (1.14, 1.145)]
    assert [v.onset for v in result.video_only] == [1.00]


def test_more_pairs_beat_closer_pairs():
    """Сначала число пар, потом близость: иначе одна точная пара вытеснила бы две."""
    video = [note(60, 1.00, 0.02), note(60, 1.05, 0.02)]
    audio = [note(60, 1.045), note(60, 1.12)]

    result = match_notes(video, audio, TOLERANCE)

    assert len(result.pairs) == 2
    assert [(v.onset, a.onset) for v, a in result.pairs] == [(1.00, 1.045), (1.05, 1.12)]


def test_empty_channels():
    assert match_notes([], [], TOLERANCE).pairs == []
    assert len(match_notes([note(60, 1.0)], [], TOLERANCE).video_only) == 1
    assert len(match_notes([], [note(60, 1.0)], TOLERANCE).audio_only) == 1


def test_results_are_sorted_by_time():
    video = [note(64, 3.0), note(60, 1.0), note(62, 2.0)]
    audio = [note(62, 2.0), note(64, 3.0), note(60, 1.0), note(70, 0.5)]

    result = match_notes(video, audio, TOLERANCE)

    assert [a.onset for _, a in result.pairs] == [1.0, 2.0, 3.0]
    assert [a.pitch for a in result.audio_only] == [70]
