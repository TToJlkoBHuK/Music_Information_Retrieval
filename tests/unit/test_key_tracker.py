"""Гистерезис, антидребезг и компенсация его задержки."""

from __future__ import annotations

import numpy as np
import pytest

from mir.config import TrackerConfig
from mir.vision.key_tracker import KeyTracker
from mir.vision.keyboard_geometry import KeyboardGeometry

FPS = 30.0
STEP = 1.0 / FPS

WIDTH = 420
HEIGHT = 200
KEYBOARD_TOP = 120
KEYBOARD_HEIGHT = 80


@pytest.fixture
def layout():
    geometry = KeyboardGeometry(x=0, width=WIDTH, lowest_pitch=60, highest_pitch=71)
    return geometry.build_layout(y=KEYBOARD_TOP, height=KEYBOARD_HEIGHT)


@pytest.fixture
def blank():
    """Кадр с «пустой» клавиатурой: белые клавиши без подсветки."""
    frame = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    frame[KEYBOARD_TOP:, :] = (245, 245, 245)
    return frame


def light_up(frame, layout, pitch, color=(60, 200, 60)):
    """Подсветить клавишу заданным цветом."""
    out = frame.copy()
    key = next(k for k in layout.keys if k.pitch == pitch)
    out[KEYBOARD_TOP + 45 : KEYBOARD_TOP + KEYBOARD_HEIGHT - 2, key.x_min + 3 : key.x_max - 3] = (
        color
    )
    return out


def run(tracker, frames):
    """Прогнать последовательность кадров, вернуть все события."""
    notes = []
    for index, frame in enumerate(frames):
        notes.extend(tracker.process_frame(frame, index * STEP))
    notes.extend(tracker.flush(len(frames) * STEP))
    return notes


def test_blank_frames_produce_nothing(layout, blank):
    tracker = KeyTracker(layout, blank)

    assert run(tracker, [blank] * 20) == []


def test_single_press_is_detected(layout, blank):
    tracker = KeyTracker(layout, blank)
    lit = light_up(blank, layout, 62)
    frames = [blank] * 5 + [lit] * 10 + [blank] * 5

    notes = run(tracker, frames)

    assert len(notes) == 1
    assert notes[0].pitch == 62


def test_onset_is_not_delayed_by_debounce(layout, blank):
    """Антидребезг требует двух кадров, но начало ноты — первый из них.

    Иначе все ноты уезжали бы на 33 мс вперёд при 30 fps — две трети
    стандартного допуска в 50 мс, отданные на ровном месте.
    """
    tracker = KeyTracker(layout, blank, TrackerConfig(min_frames_on=3))
    lit = light_up(blank, layout, 65)
    frames = [blank] * 4 + [lit] * 10 + [blank] * 3

    notes = run(tracker, frames)

    assert len(notes) == 1
    assert notes[0].onset == pytest.approx(4 * STEP, abs=1e-6)


def test_single_frame_flash_is_rejected(layout, blank):
    """Один кадр — это сбой сжатия, а не нота."""
    tracker = KeyTracker(layout, blank, TrackerConfig(min_frames_on=2))
    lit = light_up(blank, layout, 67)
    frames = [blank] * 5 + [lit] + [blank] * 5

    assert run(tracker, frames) == []


def test_hysteresis_keeps_note_whole(layout, blank):
    """Просадка яркости в середине ноты не должна её разрывать.

    С единым порогом такой кадр закрыл бы ноту и открыл новую.
    """
    tracker = KeyTracker(layout, blank, TrackerConfig(on_threshold=0.25, off_threshold=0.08))
    bright = light_up(blank, layout, 60, color=(60, 200, 60))
    dim = light_up(blank, layout, 60, color=(60, 150, 120))
    frames = [blank] * 3 + [bright] * 4 + [dim] * 2 + [bright] * 4 + [blank] * 3

    notes = run(tracker, frames)

    assert len(notes) == 1
    assert notes[0].duration == pytest.approx(10 * STEP, abs=STEP)


def test_note_still_lit_at_end_is_closed(layout, blank):
    """Ноту, звучащую на последнем кадре, нельзя терять."""
    tracker = KeyTracker(layout, blank)
    lit = light_up(blank, layout, 69)

    notes = run(tracker, [blank] * 3 + [lit] * 8)

    assert len(notes) == 1
    assert notes[0].offset == pytest.approx(11 * STEP, abs=1e-6)


def test_neighbouring_keys_do_not_bleed(layout, blank):
    """Подсветка одной клавиши не должна задевать соседей.

    Пробы берутся с отступом от краёв именно ради этого.
    """
    tracker = KeyTracker(layout, blank)
    lit = light_up(blank, layout, 64)

    notes = run(tracker, [blank] * 3 + [lit] * 8 + [blank] * 3)

    assert [n.pitch for n in notes] == [64]


def test_chord_notes_are_independent(layout, blank):
    tracker = KeyTracker(layout, blank)
    chord = blank
    for pitch in (60, 64, 67):
        chord = light_up(chord, layout, pitch)

    notes = run(tracker, [blank] * 3 + [chord] * 8 + [blank] * 3)

    assert sorted(n.pitch for n in notes) == [60, 64, 67]


def test_velocity_grows_with_deviation(layout, blank):
    """Сильнее отличается цвет — выше сила нажатия."""
    weak = KeyTracker(layout, blank)
    strong = KeyTracker(layout, blank)

    quiet = run(
        weak, [blank] * 3 + [light_up(blank, layout, 60, (100, 190, 200))] * 8 + [blank] * 3
    )
    loud = run(strong, [blank] * 3 + [light_up(blank, layout, 60, (60, 240, 60))] * 8 + [blank] * 3)

    assert quiet[0].velocity < loud[0].velocity
    assert 1 <= quiet[0].velocity <= 127
