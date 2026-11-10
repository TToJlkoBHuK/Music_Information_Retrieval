"""Правила для нот, найденных только звуком."""

from __future__ import annotations

from mir.common.enums import Source
from mir.common.types import NoteEvent
from mir.fusion.decisions import Reason
from mir.fusion.resolver import CandidatePool, VideoIndex, resolve_audio_only

TOLERANCE = 0.08
THRESHOLD = 0.8


def seen(pitch: int, onset: float, duration: float = 0.4) -> NoteEvent:
    return NoteEvent(pitch, onset, onset + duration, 64)


def heard(pitch: int, onset: float, confidence: float) -> NoteEvent:
    return NoteEvent(pitch, onset, onset + 0.5, 50, source=Source.AUDIO, confidence=confidence)


def resolve(
    note: NoteEvent,
    video: list[NoteEvent] | None = None,
    candidates: list[NoteEvent] | None = None,
    in_view: bool = True,
) -> Reason:
    pool = CandidatePool(candidates or [])
    return resolve_audio_only(note, VideoIndex(video or []), pool, THRESHOLD, TOLERANCE, in_view)[0]


def test_key_out_of_view_is_accepted():
    """Видео эту клавишу видеть не могло — звук единственный свидетель."""
    assert resolve(heard(22, 1.0, 0.4), in_view=False) is Reason.OUT_OF_VIEW


def test_candidate_block_confirms_unsure_note():
    reason = resolve(heard(60, 1.0, 0.4), candidates=[seen(60, 1.03)])
    assert reason is Reason.CANDIDATE


def test_candidate_must_match_pitch_and_time():
    assert resolve(heard(60, 1.0, 0.4), candidates=[seen(61, 1.0)]) is Reason.WEAK
    assert resolve(heard(60, 1.0, 0.4), candidates=[seen(60, 1.2)]) is Reason.WEAK


def test_candidate_confirms_only_one_note():
    pool = CandidatePool([seen(60, 1.0)])
    index = VideoIndex([])
    first = resolve_audio_only(heard(60, 1.0, 0.4), index, pool, THRESHOLD, TOLERANCE, True)
    second = resolve_audio_only(heard(60, 1.02, 0.4), index, pool, THRESHOLD, TOLERANCE, True)

    assert first[0] is Reason.CANDIDATE and first[1] is not None
    assert second[0] is Reason.WEAK


def test_note_hidden_behind_neighbour_of_other_colour():
    """Фа в видео звучит, соль-бемоль слышен только в звуке: блоки слились."""
    assert resolve(heard(66, 1.0, 0.7), video=[seen(65, 1.0, 1.5)]) is Reason.HIDDEN
    assert resolve(heard(66, 1.8, 0.7), video=[seen(65, 1.0, 1.5)]) is Reason.HIDDEN


def test_white_neighbours_do_not_hide_each_other():
    """Ми и фа — обе белые: их блоки датчик разделяет сам."""
    assert resolve(heard(65, 1.0, 0.7), video=[seen(64, 1.0, 1.5)]) is Reason.WEAK


def test_confident_note_is_accepted_without_video():
    assert resolve(heard(60, 1.0, 0.9)) is Reason.CONFIDENT


def test_confident_note_on_held_key_is_a_restrike():
    """Блоки повторов с зазором меньше кадра видео показывает одной нотой."""
    holder = seen(60, 1.0, 2.0)
    reason, related = resolve_audio_only(
        heard(60, 2.0, 0.9), VideoIndex([holder]), CandidatePool([]), THRESHOLD, TOLERANCE, True
    )
    assert reason is Reason.RESTRIKE
    assert related is holder


def test_unsure_note_on_held_key_is_rejected():
    assert resolve(heard(60, 2.0, 0.5), video=[seen(60, 1.0, 2.0)]) is Reason.KEY_HELD


def test_unsure_note_an_octave_from_video_note_is_harmonic():
    assert resolve(heard(72, 1.0, 0.5), video=[seen(60, 1.02)]) is Reason.HARMONIC
    assert resolve(heard(48, 1.0, 0.5), video=[seen(60, 1.02)]) is Reason.HARMONIC
    assert resolve(heard(79, 1.0, 0.5), video=[seen(60, 1.02)]) is Reason.HARMONIC


def test_unsure_repeat_right_after_video_note_is_echo():
    """Нота видео кончилась, а звук через 0.2 с слышит ту же высоту ещё раз."""
    assert resolve(heard(60, 1.2, 0.5), video=[seen(60, 1.0, 0.1)]) is Reason.ECHO


def test_unsure_note_without_any_trace_is_weak():
    assert resolve(heard(60, 5.0, 0.5), video=[seen(60, 1.0, 0.1)]) is Reason.WEAK


def test_reasons_know_whether_note_is_kept():
    kept = {r for r in Reason if r.accepted}
    assert Reason.CANDIDATE in kept and Reason.RESTRIKE in kept
    assert Reason.KEY_HELD not in kept and Reason.OUTSIDE not in kept
    assert all(reason.title_ru for reason in Reason)
