"""Исполнительский MIDI: файл для проверки результата на слух."""

from __future__ import annotations

import struct
from pathlib import Path

from mir.common.enums import Hand
from mir.common.types import NoteEvent, PedalEvent
from mir.export import write_performance_midi

TICKS_PER_SECOND = 960  # 480 тиков на четверть при 120 ударах в минуту


def read_tracks(path: Path) -> tuple[int, int, list[list[tuple[int, bytes]]]]:
    """Разобрать файл: формат, тиков на четверть, события дорожек (тик, байты)."""
    data = path.read_bytes()
    assert data[:4] == b"MThd"
    _, kind, count, ppq = struct.unpack(">IHHH", data[4:14])
    position, tracks = 14, []
    for _ in range(count):
        assert data[position : position + 4] == b"MTrk"
        (length,) = struct.unpack(">I", data[position + 4 : position + 8])
        body = data[position + 8 : position + 8 + length]
        position += 8 + length
        events, cursor, tick = [], 0, 0
        while cursor < len(body):
            delta = 0
            while True:
                byte = body[cursor]
                cursor += 1
                delta = (delta << 7) | (byte & 0x7F)
                if not byte & 0x80:
                    break
            tick += delta
            if body[cursor] == 0xFF:
                size = body[cursor + 2]
                events.append((tick, body[cursor : cursor + 3 + size]))
                cursor += 3 + size
            else:
                events.append((tick, body[cursor : cursor + 3]))
                cursor += 3
        tracks.append(events)
    return kind, ppq, tracks


def note_ons(track: list[tuple[int, bytes]]) -> list[tuple[int, int, int]]:
    return [(tick, e[1], e[2]) for tick, e in track if e[0] & 0xF0 == 0x90]


def test_notes_keep_their_time_in_seconds(tmp_path: Path):
    notes = [
        NoteEvent(60, 0.5, 1.0, 80, Hand.RIGHT),
        NoteEvent(64, 2.25, 2.5, 50, Hand.RIGHT),
        NoteEvent(36, 0.5, 3.0, 70, Hand.LEFT),
    ]
    kind, ppq, tracks = read_tracks(write_performance_midi(notes, tmp_path / "out.mid"))

    assert (kind, ppq, len(tracks)) == (1, 480, 3)  # темп и по дорожке на руку
    assert tracks[0][0][1][:3] == b"\xff\x51\x03"  # первая запись — темп
    assert note_ons(tracks[1]) == [(480, 60, 80), (2160, 64, 50)]
    assert note_ons(tracks[2]) == [(480, 36, 70)]
    offs = [(tick, e[1]) for tick, e in tracks[2] if e[0] & 0xF0 == 0x80]
    assert offs == [(3 * TICKS_PER_SECOND, 36)]


def test_unknown_hand_gets_its_own_track(tmp_path: Path):
    notes = [NoteEvent(60, 0.0, 1.0, 80), NoteEvent(62, 1.0, 2.0, 80, Hand.RIGHT)]
    _, _, tracks = read_tracks(write_performance_midi(notes, tmp_path / "out.mid"))

    assert len(tracks) == 3
    assert [pitch for _, pitch, _ in note_ons(tracks[1])] == [62]
    assert [pitch for _, pitch, _ in note_ons(tracks[2])] == [60]


def test_repeat_starting_where_previous_note_ends(tmp_path: Path):
    """Отпускание раньше нажатия в тот же тик — иначе повтор заглушится."""
    notes = [NoteEvent(60, 0.0, 1.0, 80), NoteEvent(60, 1.0, 2.0, 80)]
    _, _, tracks = read_tracks(write_performance_midi(notes, tmp_path / "out.mid"))

    at_one_second = [e[0] & 0xF0 for tick, e in tracks[1] if tick == TICKS_PER_SECOND]
    assert at_one_second == [0x80, 0x90]


def test_pedal_is_written_as_sustain_controller(tmp_path: Path):
    notes = [NoteEvent(60, 0.0, 1.0, 80)]
    path = write_performance_midi(notes, tmp_path / "out.mid", pedals=[PedalEvent(0.5, 1.5)])
    _, _, tracks = read_tracks(path)

    pedal = [(tick, e[2]) for tick, e in tracks[1] if e[0] == 0xB0 and e[1] == 64]
    assert pedal == [(480, 127), (1440, 0)]


def test_long_piece_uses_multibyte_delta(tmp_path: Path):
    """Пауза в две минуты — разность времён не помещается в один байт."""
    notes = [NoteEvent(60, 0.0, 0.5, 80), NoteEvent(62, 120.0, 120.5, 80)]
    _, _, tracks = read_tracks(write_performance_midi(notes, tmp_path / "out.mid"))

    assert [tick for tick, _, _ in note_ons(tracks[1])] == [0, 120 * TICKS_PER_SECOND]


def test_empty_list_gives_valid_file(tmp_path: Path):
    kind, _, tracks = read_tracks(write_performance_midi([], tmp_path / "out.mid"))
    assert kind == 1 and len(tracks) == 1
