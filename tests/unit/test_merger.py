"""Слияние каналов: сборка итоговых нот и решений (#30, F-10)."""

from __future__ import annotations

from dataclasses import replace

import pytest

from mir.common.enums import Hand, Source
from mir.common.types import KeyboardLayout, NoteEvent
from mir.config import FusionConfig, load_config
from mir.fusion import Reason, fuse

OFFSET = 0.12
"""Звук отстаёт от видео: в шкале видео нота на столько раньше."""


@pytest.fixture(scope="module")
def config() -> FusionConfig:
    return load_config(use_user_config=False, use_env=False).fusion


def base(count: int = 40) -> list[NoteEvent]:
    """Мелодия без повторов высоты подряд: её хватает, чтобы совместить каналы."""
    pitches = [60, 64, 67, 72, 62, 65, 69, 74, 59, 63]
    return [
        NoteEvent(pitches[i % len(pitches)], 2.0 + i * 0.37, 2.0 + i * 0.37 + 0.25, 60 + i % 20)
        for i in range(count)
    ]


def as_video(notes: list[NoteEvent]) -> list[NoteEvent]:
    return [
        replace(n, onset=n.onset - OFFSET, offset=n.offset - OFFSET, velocity=64, hand=Hand.RIGHT)
        for n in notes
    ]


def as_audio(notes: list[NoteEvent], confidence: float | None = None) -> list[NoteEvent]:
    """Ноты звука: длиннее из-за педали, уверенность 0.86–0.92."""
    return [
        replace(
            n,
            offset=n.offset + 0.8,
            source=Source.AUDIO,
            confidence=confidence if confidence is not None else 0.86 + 0.02 * (i % 4),
        )
        for i, n in enumerate(notes)
    ]


def audio_note(pitch: int, onset: float, confidence: float, velocity: int = 50) -> NoteEvent:
    return NoteEvent(
        pitch, onset, onset + 0.6, velocity, source=Source.AUDIO, confidence=confidence
    )


def video_note(pitch: int, audio_onset: float, duration: float) -> NoteEvent:
    onset = audio_onset - OFFSET
    return NoteEvent(pitch, onset, onset + duration, 64, hand=Hand.LEFT, confidence=0.9)


def reasons(result, pitch: int) -> list[Reason]:
    """Причины решений по нотам этой высоты, по времени."""
    found = []
    for decision in result.decisions:
        note = decision.note or decision.audio or decision.video
        if note is not None and note.pitch == pitch:
            found.append(decision.reason)
    return found


def test_pair_takes_each_property_from_the_better_channel(config: FusionConfig):
    truth = base()
    result = fuse(as_video(truth), as_audio(truth), config)

    assert len(result.notes) == len(truth)
    note, original = result.notes[5], truth[5]
    assert note.source is Source.BOTH
    assert note.onset == pytest.approx(original.onset, abs=1e-6)  # время — из звука
    assert note.duration == pytest.approx(0.25, abs=1e-6)  # удержание клавиши — из видео
    assert note.velocity == original.velocity  # громкость — из звука
    assert note.hand is Hand.RIGHT  # рука — из видео
    assert note.confidence == pytest.approx(1 - (1 - 1.0) * (1 - 0.88), abs=0.05)
    assert result.lead is Source.VIDEO
    assert result.video_trust == result.audio_trust == 1.0
    assert result.curve.at(5.0) == pytest.approx(OFFSET, abs=0.005)


def test_every_source_note_gets_exactly_one_decision(config: FusionConfig):
    truth = base()
    video = [*as_video(truth), video_note(50, 6.0, 0.3)]
    audio = [*as_audio(truth), audio_note(80, 7.0, 0.4), audio_note(81, 99.0, 0.9)]

    result = fuse(video, audio, config)

    assert sum(d.video is not None for d in result.decisions) == len(video)
    assert sum(d.audio is not None for d in result.decisions) == len(audio)
    assert len(result.notes) == sum(d.note is not None for d in result.decisions)


def test_audio_after_last_block_is_dropped(config: FusionConfig):
    """После последнего блока — титры: уверенные ноты звука там не музыка пьесы."""
    truth = base()
    outro = [audio_note(52 + i, truth[-1].offset + 3 + i, 0.9) for i in range(5)]

    result = fuse(as_video(truth), [*as_audio(truth), *outro], config)

    assert result.counts[Reason.OUTSIDE] == 5
    assert len(result.notes) == len(truth)
    assert any("за пределами видеоряда" in w for w in result.warnings)


def test_video_only_note_is_kept_with_neighbours_loudness(config: FusionConfig):
    """Звук пропустил ноту: она остаётся, громкость — как у соседей, а не 64."""
    truth = [replace(n, velocity=90) for n in base()]
    video = [*as_video(truth), video_note(48, 9.0, 0.5)]

    result = fuse(video, as_audio(truth), config)

    note = next(n for n in result.notes if n.pitch == 48)
    assert note.source is Source.VIDEO
    assert note.velocity == 90
    assert note.onset == pytest.approx(9.0, abs=0.01)
    assert note.confidence < 0.95
    assert reasons(result, 48) == [Reason.VIDEO_ONLY]


def test_confident_restrike_splits_held_video_note(config: FusionConfig):
    """Видео слило два удара в одну ноту на 2 с; звук уверенно слышит оба."""
    truth = base()
    video = [*as_video(truth), video_note(40, 5.0, 2.0)]
    audio = [*as_audio(truth), audio_note(40, 5.0, 0.9), audio_note(40, 6.0, 0.9)]

    result = fuse(video, audio, config)

    low = [n for n in result.notes if n.pitch == 40]
    assert [round(n.onset, 2) for n in low] == [5.0, 6.0]
    assert low[0].offset == pytest.approx(6.0, abs=0.01)  # первая обрезана вторым ударом
    assert low[1].offset == pytest.approx(7.0, abs=0.01)  # вторая держится до конца блока
    assert reasons(result, 40) == [Reason.CONFIRMED, Reason.RESTRIKE]


def test_unsure_restrike_on_held_key_is_rejected(config: FusionConfig):
    truth = base()
    video = [*as_video(truth), video_note(40, 5.0, 2.0)]
    audio = [*as_audio(truth), audio_note(40, 5.0, 0.9), audio_note(40, 6.0, 0.5)]

    result = fuse(video, audio, config)

    low = [n for n in result.notes if n.pitch == 40]
    assert len(low) == 1
    assert low[0].duration == pytest.approx(2.0, abs=0.01)
    assert Reason.KEY_HELD in reasons(result, 40)


def test_split_block_is_joined_back(config: FusionConfig):
    """Датчик разорвал долгий блок надвое, звук слышит одну ноту (В-10)."""
    truth = base()
    video = [*as_video(truth), video_note(40, 5.0, 1.2), video_note(40, 6.21, 0.8)]
    long_note = replace(audio_note(40, 5.0, 0.9), offset=8.0)

    result = fuse(video, [*as_audio(truth), long_note], config)

    low = [n for n in result.notes if n.pitch == 40]
    assert len(low) == 1
    assert low[0].offset == pytest.approx(7.01, abs=0.02)
    assert Reason.SPLIT in reasons(result, 40)


def test_real_repeat_after_a_pause_is_not_joined(config: FusionConfig):
    """Вторая нота началась через полсекунды после первой — это не разрыв блока."""
    truth = base()
    video = [*as_video(truth), video_note(40, 5.0, 0.5), video_note(40, 6.0, 0.5)]
    long_note = replace(audio_note(40, 5.0, 0.9), offset=8.0)

    result = fuse(video, [*as_audio(truth), long_note], config)

    assert len([n for n in result.notes if n.pitch == 40]) == 2


def test_candidate_block_and_unsure_audio_make_a_note(config: FusionConfig):
    truth = base()
    candidate = replace(video_note(40, 5.0, 0.45), confidence=0.3)

    result = fuse(
        as_video(truth),
        [*as_audio(truth), audio_note(40, 5.02, 0.5)],
        config,
        candidates=[candidate],
    )

    note = next(n for n in result.notes if n.pitch == 40)
    assert note.source is Source.BOTH
    assert note.onset == pytest.approx(5.02)
    assert note.duration == pytest.approx(0.45, abs=0.01)  # длина блока — от кандидата
    assert reasons(result, 40) == [Reason.CANDIDATE]


def test_unsure_audio_without_block_is_rejected(config: FusionConfig):
    truth = base()
    # 5.15 с — между нотами мелодии: иначе нота сошла бы за обертон соседней.
    result = fuse(as_video(truth), [*as_audio(truth), audio_note(40, 5.15, 0.5)], config)

    assert not [n for n in result.notes if n.pitch == 40]
    assert reasons(result, 40) == [Reason.WEAK]
    assert len(result.rejected(Source.AUDIO)) == 1


def test_confident_audio_without_block_is_accepted(config: FusionConfig):
    truth = base()
    result = fuse(as_video(truth), [*as_audio(truth), audio_note(40, 5.0, 0.95)], config)

    note = next(n for n in result.notes if n.pitch == 40)
    assert note.source is Source.AUDIO
    assert note.hand is Hand.UNKNOWN
    assert reasons(result, 40) == [Reason.CONFIDENT]


def test_key_outside_frame_is_taken_from_audio(config: FusionConfig):
    """Обрезанная клавиатура: нота ниже видимого диапазона есть только в звуке."""
    truth = base()
    layout = KeyboardLayout(
        (0, 0, 100, 10), (), lowest_pitch=48, highest_pitch=84, is_cropped=True, confidence=1.0
    )

    result = fuse(
        as_video(truth), [*as_audio(truth), audio_note(36, 5.0, 0.5)], config, layout=layout
    )

    assert reasons(result, 36) == [Reason.OUT_OF_VIEW]
    assert any(n.pitch == 36 for n in result.notes)


def test_late_pair_takes_onset_from_audio(config: FusionConfig):
    """Дым на входе сдвинул начало в видео на 150 мс раньше: это та же нота."""
    truth = base()
    video = [*as_video(truth), video_note(40, 4.85, 1.0)]

    result = fuse(video, [*as_audio(truth), audio_note(40, 5.0, 0.9)], config)

    low = [n for n in result.notes if n.pitch == 40]
    assert len(low) == 1
    assert low[0].onset == pytest.approx(5.0)
    assert low[0].source is Source.BOTH
    assert reasons(result, 40) == [Reason.LATE_PAIR]


def test_octave_error_of_cropped_keyboard_is_corrected(config: FusionConfig):
    truth = base()
    video = [replace(n, pitch=n.pitch - 12) for n in as_video(truth)]

    result = fuse(video, as_audio(truth), config)

    assert result.pitch_shift == 12
    assert sorted(n.pitch for n in result.notes) == sorted(n.pitch for n in truth)
    assert all(n.source is Source.BOTH for n in result.notes)
    assert any("Октава" in w for w in result.warnings)


def test_full_keyboard_is_never_transposed(config: FusionConfig):
    """Все 88 клавиш в кадре: ошибиться в октаве видео не могло."""
    truth = base()
    video = [replace(n, pitch=n.pitch - 12) for n in as_video(truth)]
    layout = KeyboardLayout(
        (0, 0, 100, 10), (), lowest_pitch=21, highest_pitch=108, is_cropped=False, confidence=1.0
    )

    assert fuse(video, as_audio(truth), config, layout=layout).pitch_shift == 0


def test_unreliable_video_gives_way_to_audio(config: FusionConfig):
    """Видео нашло треть нот и вдвое больше мусора: итог берётся из звука.

    Так выглядит подсветка клавиш под руками исполнителя (В-05).
    """
    truth = base(60)
    junk = [video_note(30 + i % 50, 2.5 + i * 0.21, 0.2) for i in range(120)]
    video = [*as_video(truth)[::3], *junk]

    result = fuse(video, as_audio(truth), config)

    assert result.lead is Source.AUDIO
    assert sorted((n.pitch, round(n.onset, 2)) for n in result.notes) == sorted(
        (n.pitch, round(n.onset, 2)) for n in truth
    )
    assert result.counts[Reason.UNTRUSTED] > 100
    assert any("ненадёжен" in w for w in result.warnings)


def test_channels_that_cannot_be_aligned_fall_back_to_one(config: FusionConfig):
    truth = base()
    other = [audio_note(40 + i % 30, 1.0 + i * 0.53, 0.9) for i in range(40)]

    by_blocks = fuse(as_video(truth), other, config, video_reliable=True)
    by_keys = fuse(as_video(truth), other, config, video_reliable=False)

    assert by_blocks.lead is Source.VIDEO and len(by_blocks.notes) == len(truth)
    assert by_keys.lead is Source.AUDIO and len(by_keys.notes) == len(other)
    assert not by_blocks.curve.reliable
    assert any("не удалось совместить" in w for w in by_blocks.warnings)


def test_single_channel_input(config: FusionConfig):
    truth = base()
    only_video = fuse(as_video(truth), [], config)
    only_audio = fuse([], as_audio(truth), config)

    assert only_video.lead is Source.VIDEO and len(only_video.notes) == len(truth)
    assert only_audio.lead is Source.AUDIO and len(only_audio.notes) == len(truth)
    assert only_video.warnings and only_audio.warnings
    assert fuse([], [], config).notes == []


def test_audio_without_confidence_needs_video_support(config: FusionConfig):
    """В CSV нет столбца уверенности — у всех нот 1.0: по ней решать нельзя."""
    truth = base()
    audio = [*as_audio(truth, confidence=1.0), audio_note(40, 5.0, 1.0)]

    result = fuse(as_video(truth), audio, config)

    assert not [n for n in result.notes if n.pitch == 40]
    assert any("нет оценки уверенности" in w for w in result.warnings)


def test_quality_report_counts_sources(config: FusionConfig):
    truth = base()
    video = [*as_video(truth), video_note(48, 9.0, 0.5)]
    audio = [*as_audio(truth), audio_note(40, 5.0, 0.95)]

    report = fuse(video, audio, config).quality_report()

    assert report.notes_confirmed == len(truth)
    assert report.notes_from_video_only == 1
    assert report.notes_from_audio_only == 1
    assert report.av_offset_ms == pytest.approx(OFFSET * 1000, abs=5)
    assert report.agreement_rate == pytest.approx(len(truth) / (len(truth) + 2))


def test_result_is_sorted_and_valid(config: FusionConfig):
    truth = base()
    result = fuse(as_video(truth), as_audio(truth), config)

    onsets = [n.onset for n in result.notes]
    assert onsets == sorted(onsets)
    assert all(n.offset > n.onset for n in result.notes)
