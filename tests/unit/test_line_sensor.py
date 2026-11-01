"""Линия-датчик: ноты по падающим блокам.

Кадры рисуются прямо в памяти, без кодирования видео: так проверяется сама
логика датчика — скорость, линия касания, отсев дыма и неподвижного, —
и тест идёт доли секунды.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pytest

from mir.common.enums import Hand
from mir.common.types import NoteEvent
from mir.config import SensorConfig
from mir.vision.keyboard_geometry import KeyboardGeometry
from mir.vision.line_sensor import LineSensor, SensorBank, _hue_groups, _split_hands

WIDTH, HEIGHT = 640, 400
TOP = 300  # верх клавиатуры
CONTACT = 290  # линия касания на 10 пикселей выше — как планка у Rousseau
FPS = 30.0
SPEED = 200.0
BACKGROUND = (20, 18, 15)
BLOCK = (240, 240, 240)

GEOMETRY = KeyboardGeometry(x=0, width=WIDTH, lowest_pitch=48, highest_pitch=72)
LAYOUT = GEOMETRY.build_layout(y=TOP, height=HEIGHT - TOP)


def blank() -> npt.NDArray[np.uint8]:
    frame = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    frame[:] = BACKGROUND
    frame[TOP:] = (230, 230, 230)
    return frame


def draw_block(
    frame: npt.NDArray[np.uint8],
    pitch: int,
    onset: float,
    offset: float,
    t: float,
    colour: tuple[int, int, int] = BLOCK,
):
    bottom = CONTACT - (onset - t) * SPEED
    top = bottom - (offset - onset) * SPEED
    y0, y1 = int(max(top, 0)), int(min(bottom, CONTACT))
    if y1 > y0:
        left, right = GEOMETRY.bounds(pitch)
        frame[y0:y1, int(left) + 1 : int(right) - 1] = colour


def render(
    notes: list[tuple[int, float, float]],
    seconds: float,
    extra=None,
    start: float = 0.0,
    colour: tuple[int, int, int] = BLOCK,
):
    """Кадры ролика из блоков; `extra(frame, t)` (или их список) дорисовывает помехи."""
    times = np.arange(start, seconds, 1 / FPS)
    painters = extra if isinstance(extra, list) else [extra] if extra is not None else []
    frames = []
    for t in times:
        frame = blank()
        for pitch, onset, offset in notes:
            draw_block(frame, pitch, onset, offset, float(t), colour)
        for paint in painters:
            paint(frame, t)
        frames.append(frame)
    return frames, times


def run(
    notes: list[tuple[int, float, float]],
    seconds: float,
    extra=None,
    start: float = 0.0,
):
    """Прогнать датчик по ролику из блоков."""
    frames, times = render(notes, seconds, extra, start)
    sample = frames[:: max(len(frames) // 60, 1)]
    sensor = LineSensor(LAYOUT, sample, FPS, SensorConfig())
    for t, frame in zip(times, frames, strict=True):
        sensor.process_frame(frame, float(t))
    return sensor.finish(), sensor


def melody(jitter: float = 0.0) -> list[tuple[int, float, float]]:
    """13 нот; `jitter` сдвигает каждую следующую на долю кадра."""
    pitches = [60, 62, 64, 65, 67, 69, 71, 72, 48, 52, 55, 61, 63]
    return [
        (p, 1.5 + i * (0.4 + jitter), 1.5 + i * (0.4 + jitter) + 0.3) for i, p in enumerate(pitches)
    ]


@pytest.mark.parametrize("jitter", [0.0, 0.013])
def test_notes_found_with_subframe_timing(jitter):
    """Без сдвига все ноты в одной фазе кадра, со сдвигом — в разных."""
    truth = melody(jitter)
    notes, _ = run(truth, 9.0)

    assert [n.pitch for n in notes] == [p for p, _, _ in truth]
    errors = [abs(n.onset - onset) for n, (_, onset, _) in zip(notes, truth, strict=True)]
    assert max(errors) < 0.5 / FPS  # точнее полукадра
    durations = [abs(n.duration - (off - on)) for n, (_, on, off) in zip(notes, truth, strict=True)]
    assert max(durations) < 1 / FPS


def test_speed_and_contact_measured():
    _, sensor = run(melody(), 8.0)
    assert sensor.report.fall_speed == pytest.approx(SPEED, rel=0.03)
    assert sensor.report.contact_y == pytest.approx(CONTACT, abs=2)


def test_rising_smoke_is_rejected():
    """Дым поднимается от места касания: в нижней полосе раньше, чем в верхней."""

    def smoke(frame, t):
        if 2.0 <= t < 4.0:
            bottom = CONTACT - (t - 2.0) * 120  # поднимается
            left, right = GEOMETRY.bounds(71)  # клавиша, на которой нот нет
            frame[int(bottom) - 60 : int(bottom), int(left) : int(right)] = (200, 200, 210)

    truth = melody()[:6]  # настоящие ноты нужны, чтобы измерить скорость
    notes, sensor = run(truth, 6.0, extra=smoke)
    assert [n.pitch for n in notes] == [p for p, _, _ in truth]
    assert sensor.report.rejected >= 1


def test_static_overlay_is_rejected():
    """Неподвижная полоса стоит в обеих полосах сразу: покрытие есть, сдвига нет."""

    def beam(frame, t):
        if 3.0 <= t < 5.0:
            left, right = GEOMETRY.bounds(69)
            frame[40:CONTACT, int(left) : int(right)] = (235, 235, 235)

    notes, _ = run(melody()[:5], 6.0, extra=beam)
    assert 69 not in [n.pitch for n in notes]


def test_block_already_in_flight_at_start():
    """Разбор начался после заставки, а блок первой ноты уже падает."""
    notes, _ = run([(60, 0.3, 0.9), *melody()], 8.0, start=0.0)
    first = notes[0]
    assert first.pitch == 60
    assert first.onset == pytest.approx(0.3, abs=1 / FPS)


def test_disabled_without_blocks():
    """Визуализатор без блоков: датчик честно выключается, а не выдумывает ноты."""
    notes, sensor = run([], 4.0)
    assert notes == []
    assert not sensor.enabled
    assert "нет блоков" in sensor.report.reason


def test_hue_groups_split_two_colours_across_red():
    """Два цвета, один из них — красный по обе стороны нуля шкалы оттенков."""
    hues = np.array([178.0, 2.0, 1.0, 179.0, 60.0, 61.0, 59.0])
    groups = _hue_groups(hues)
    assert groups is not None
    assert list(groups) == [groups[0]] * 4 + [not groups[0]] * 3


def test_hue_groups_single_colour_with_outlier():
    """Один цвет и пара искр другого оттенка — руки не различаются."""
    assert (
        _hue_groups(np.array([90.0, 91.0, 92.0, 93.0, 94.0, 95.0, 97.0, 91.0, 92.0, 93.0])) is None
    )
    hues = np.array([90.0] * 20 + [30.0])  # одна нота из двадцати одной: выброс
    assert _hue_groups(hues) is None


def _found(hands_by_time: bool) -> list[tuple[NoteEvent, npt.NDArray[np.float64]]]:
    """Ноты двух цветов: вперемешку (руки) или первая половина одним цветом (смена палитры)."""
    green, blue = np.array([80.0, 200.0, 80.0]), np.array([220.0, 140.0, 60.0])
    found = []
    for i in range(40):
        low = i % 2 == 0
        colour = (green if i < 20 else blue) if hands_by_time else (blue if low else green)
        pitch = 45 + i % 5 if low else 70 + i % 5
        note = NoteEvent(pitch=pitch, onset=1.0 + i * 0.25, offset=1.2 + i * 0.25, velocity=64)
        found.append((note, colour))
    return found


def test_two_colours_played_together_are_hands():
    notes = _split_hands(_found(hands_by_time=False))
    assert all((n.hand is Hand.LEFT) == (n.pitch < 60) for n in notes)


def test_palette_change_is_not_hands():
    """Палитра сменилась посреди ролика (Rousseau, Clair de lune): цветов два, рук нет."""
    notes = _split_hands(_found(hands_by_time=True))
    assert all(n.hand is Hand.UNKNOWN for n in notes)


def test_bank_keeps_high_level_on_clean_blocks():
    """Чистые яркие блоки: более низкий порог ничего не добавляет, остаётся 0.6."""
    truth = melody()
    frames, times = render(truth, 9.0)
    bank = SensorBank(LAYOUT, frames[:: max(len(frames) // 60, 1)], FPS, SensorConfig())
    for t, frame in zip(times, frames, strict=True):
        bank.process_frame(frame, float(t))
    notes = bank.finish()

    assert [n.pitch for n in notes] == [p for p, _, _ in truth]
    assert bank.report.solid_level == pytest.approx(0.6)
    assert len(bank.report.candidates) == 3


def test_bank_lowers_level_when_blocks_are_dim():
    """Тусклые блоки при ярких вспышках: порог подбирается под блоки.

    Вспышки задают яркость «самых ярких блоков», и при пороге 0.6 тусклые
    блоки не проходят вовсе — датчик с этим порогом не может даже измерить
    скорость. При 0.5 и 0.4 ноты находятся все, а вспышки (обе полосы сразу)
    отсеиваются. На ролике Interstellar механизм тоньше — блок искажался,
    а не пропадал, — но итог тот же: прежний единый порог терял ноты.
    """
    truth = melody()[:8]
    dim = (110, 110, 110)  # отклонение от фона ~90 при ~220 у вспышки

    def flashes(frame, t):
        if int(t * 2) % 3 == 0:  # мигающая вспышка на свободной клавише, в обеих полосах
            left, right = GEOMETRY.bounds(50)
            frame[150:CONTACT, int(left) : int(right)] = (245, 245, 245)

    def brighter_on_top(frame, t):
        # Верхняя часть тусклого блока чуть ярче: так у полупрозрачных столбов
        # с градиентом — порог 0.6 задевает блок не от низа, а от середины.
        for pitch, onset, offset in truth:
            bottom = CONTACT - (onset - t) * SPEED
            top = bottom - (offset - onset) * SPEED
            mid = int((top + bottom) / 2)
            left, right = GEOMETRY.bounds(pitch)
            y0, y1 = int(max(top, 0)), int(min(mid, CONTACT))
            if y1 > y0:
                frame[y0:y1, int(left) + 1 : int(right) - 1] = (150, 150, 150)

    frames, times = render(truth, 7.0, extra=[flashes, brighter_on_top], colour=dim)
    bank = SensorBank(LAYOUT, frames[:: max(len(frames) // 60, 1)], FPS, SensorConfig())
    for t, frame in zip(times, frames, strict=True):
        bank.process_frame(frame, float(t))
    notes = bank.finish()

    assert bank.report.solid_level < 0.6
    assert sorted(n.pitch for n in notes) == sorted(p for p, _, _ in truth)
