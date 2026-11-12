"""Запись MIDI.

Пока — только **исполнительский MIDI**: ноты в том времени, в каком они
сыграны, без тактов и квантования. Его назначение — проверка на слух:
открыть результат слияния каналов в любом плеере и сравнить с роликом.
Нотный MIDI с темпом, размером и тональностью (F-22) появится на этапе 6,
когда будет готова нотная запись.

Файл пишется без сторонних библиотек: формат прост, а лишняя зависимость
в дистрибутиве ни к чему (NF-14).
"""

from __future__ import annotations

import struct
from collections.abc import Sequence
from pathlib import Path

from mir.common.enums import Hand
from mir.common.timeutil import DEFAULT_PPQ
from mir.common.types import NoteEvent, PedalEvent

__all__ = ["write_performance_midi"]

_TEMPO_BPM = 120
"""Темп в заголовке файла. На звучание не влияет: время нот задано в секундах
и пересчитано в тики именно при этом темпе; он лишь делает пересчёт целым —
при 120 ударах в минуту четверть длится полсекунды."""

_SUSTAIN_CONTROLLER = 64
"""Номер контроллера правой педали в стандарте MIDI."""

_TRACK_NAMES = {Hand.RIGHT: "Right hand", Hand.LEFT: "Left hand", Hand.UNKNOWN: "Piano"}


def write_performance_midi(
    notes: Sequence[NoteEvent],
    path: Path,
    pedals: Sequence[PedalEvent] = (),
    ppq: int = DEFAULT_PPQ,
) -> Path:
    """Записать ноты как сыграны: время в секундах, без тактовой сетки.

    Файл многодорожечный (тип 1): дорожка темпа и по дорожке на каждую
    руку, встретившуюся в нотах. Ноты с неизвестной рукой идут отдельной
    дорожкой — так их видно в редакторе.

    Args:
        notes: Ноты.
        path: Куда писать.
        pedals: Нажатия правой педали; пишутся в первую дорожку с нотами.
        ppq: Тиков на четверть.

    Returns:
        Путь к записанному файлу.
    """
    ticks_per_second = ppq * _TEMPO_BPM / 60.0

    def tick(seconds: float) -> int:
        return max(round(seconds * ticks_per_second), 0)

    tempo = _meta(0x51, struct.pack(">I", round(60_000_000 / _TEMPO_BPM))[1:])
    tracks = [_track([(0, tempo)])]
    hands = [
        hand for hand in (Hand.RIGHT, Hand.LEFT, Hand.UNKNOWN) if any(n.hand is hand for n in notes)
    ]
    for channel, hand in enumerate(hands):
        events: list[tuple[int, bytes]] = [(0, _meta(0x03, _TRACK_NAMES[hand].encode("ascii")))]
        for note in notes:
            if note.hand is not hand:
                continue
            start = tick(note.onset)
            end = max(tick(note.offset), start + 1)
            events.append((start, bytes([0x90 | channel, note.pitch, max(note.velocity, 1)])))
            events.append((end, bytes([0x80 | channel, note.pitch, 0])))
        if channel == 0:
            for pedal in pedals:
                events.append((tick(pedal.onset), bytes([0xB0, _SUSTAIN_CONTROLLER, 127])))
                events.append((tick(pedal.offset), bytes([0xB0, _SUSTAIN_CONTROLLER, 0])))
        tracks.append(_track(events))

    header = b"MThd" + struct.pack(">IHHH", 6, 1, len(tracks), ppq)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(header + b"".join(tracks))
    return path


def _track(events: list[tuple[int, bytes]]) -> bytes:
    """Собрать дорожку: события по времени, между ними — разности в тиках."""
    # При равном времени отпускание идёт раньше нажатия: иначе повтор той же
    # клавиши, начавшийся в тик конца предыдущей ноты, тут же заглушился бы.
    ordered = sorted(events, key=lambda item: (item[0], item[1][0] & 0xF0 == 0x90))
    body = bytearray()
    last = 0
    for time, data in ordered:
        body += _variable_length(time - last) + data
        last = time
    body += _variable_length(0) + _meta(0x2F, b"")
    return b"MTrk" + struct.pack(">I", len(body)) + bytes(body)


def _meta(kind: int, data: bytes) -> bytes:
    """Мета-событие: темп, название дорожки, конец дорожки."""
    return bytes([0xFF, kind]) + _variable_length(len(data)) + data


def _variable_length(value: int) -> bytes:
    """Число переменной длины: по семь бит на байт, старший бит — «есть ещё»."""
    chunks = [value & 0x7F]
    value >>= 7
    while value:
        chunks.append((value & 0x7F) | 0x80)
        value >>= 7
    return bytes(reversed(chunks))
