"""Автокалибровка: цвета блоков, скорость падения, разделение рук."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from mir.common.enums import Hand
from mir.common.types import NoteEvent
from mir.vision.block_tracker import BlockTracker
from mir.vision.calibration import _block_colors, _hue_peaks, calibrate
from mir.vision.keyboard_geometry import KeyboardGeometry

WIDTH = 320
HEIGHT = 240
HIT_LINE = 180

BACKGROUND_BGR = (30, 25, 20)
GREEN_BGR = (80, 200, 80)
ORANGE_BGR = (220, 140, 60)


def to_hsv(frame_bgr):
    return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)


def scene(colors: list[tuple[int, int, int]]):
    """Кадр с фоном и вертикальными полосами заданных цветов."""
    frame = np.zeros((HIT_LINE, WIDTH, 3), dtype=np.uint8)
    frame[:, :] = BACKGROUND_BGR
    span = WIDTH // (len(colors) + 1)
    for index, colour in enumerate(colors):
        frame[20:160, index * span + 10 : (index + 1) * span] = colour
    return to_hsv(frame)


@pytest.fixture
def layout():
    return KeyboardGeometry(x=0, width=WIDTH, lowest_pitch=36, highest_pitch=96).build_layout(
        y=HIT_LINE, height=HEIGHT - HIT_LINE
    )


def test_two_hand_colours_are_separated():
    frames = [scene([GREEN_BGR, ORANGE_BGR])] * 3

    colors, share = _block_colors(frames, (96.0, 80.0, 24.0))

    assert len(colors) == 2
    hues = sorted(c[0] for c in colors)
    assert hues[0] == pytest.approx(60.0, abs=3)
    assert hues[1] == pytest.approx(105.0, abs=3)
    assert 0.3 < share < 0.7


def test_single_colour_is_not_split_in_two():
    """Одноцветную схему нельзя выдавать за двуручную.

    Делить руки наугад хуже, чем честно признать, что цвет
    их не различает.
    """
    frames = [scene([GREEN_BGR, GREEN_BGR])] * 3

    colors, _ = _block_colors(frames, (96.0, 80.0, 24.0))

    assert len(colors) == 1


def test_result_is_reproducible():
    """Ответ обязан быть одинаковым при каждом запуске.

    K-средних на одних и тех же кадрах то делил руки, то сливал их
    в усреднённый цвет.
    """
    frames = [scene([GREEN_BGR, ORANGE_BGR])] * 3

    runs = [_block_colors(frames, (96.0, 80.0, 24.0))[0] for _ in range(5)]

    assert all(run == runs[0] for run in runs)


def test_desaturated_blocks_give_one_colour():
    """Белые блоки: тона нет, разделять нечего."""
    frames = [scene([(240, 240, 240), (200, 200, 200)])] * 3

    colors, _ = _block_colors(frames, (96.0, 80.0, 24.0))

    assert len(colors) == 1


def test_hue_peaks_wrap_around_zero():
    """Красный цвет лежит по обе стороны нуля и не должен дробиться."""
    hues = np.array([178, 179, 0, 1, 2] * 40, dtype=np.int32)

    peaks = _hue_peaks(hues)

    assert len(peaks) == 1


def test_hue_peaks_ignore_empty_histogram():
    assert _hue_peaks(np.array([], dtype=np.int32)) == []


def test_calibration_reports_missing_hand_colours(layout):
    frames = [
        cv2.cvtColor(np.full((HEIGHT, WIDTH, 3), BACKGROUND_BGR, dtype=np.uint8), cv2.COLOR_HSV2BGR)
        for _ in range(4)
    ]

    profile = calibrate(frames, layout, fps=30.0)

    assert profile.has_hand_colors is False
    assert profile.hit_line_y == HIT_LINE


def note(pitch: int, onset: float, hand: Hand) -> NoteEvent:
    return NoteEvent(pitch=pitch, onset=onset, offset=onset + 0.4, velocity=64, hand=hand)


def test_lower_group_becomes_left_hand():
    """Руки решаются по регистру: ниже звучит левая.

    Порядок цветов в профиле произволен и опорой служить не может.
    """
    notes = [
        note(40, 0.0, Hand.RIGHT),
        note(43, 0.5, Hand.RIGHT),
        note(72, 0.0, Hand.LEFT),
        note(76, 0.5, Hand.LEFT),
    ]

    result = BlockTracker._assign_hands_by_register(notes)

    assert [n.hand for n in result if n.pitch < 60] == [Hand.LEFT, Hand.LEFT]
    assert [n.hand for n in result if n.pitch > 60] == [Hand.RIGHT, Hand.RIGHT]


def test_correct_assignment_is_left_alone():
    notes = [note(40, 0.0, Hand.LEFT), note(72, 0.0, Hand.RIGHT)]

    assert BlockTracker._assign_hands_by_register(notes) == notes


def test_single_group_is_not_reassigned():
    notes = [note(40, 0.0, Hand.UNKNOWN), note(72, 0.0, Hand.UNKNOWN)]

    assert BlockTracker._assign_hands_by_register(notes) == notes
