"""Геометрия клавиатуры — общая основа генератора и детектора."""

from __future__ import annotations

from itertools import pairwise

import pytest

from mir.common.types import PITCH_MAX, PITCH_MIN
from mir.vision.keyboard_geometry import (
    KeyboardGeometry,
    black_pattern,
    is_black_key,
    white_keys_in_range,
)


@pytest.mark.parametrize(
    ("pitch", "black"),
    [
        (60, False),  # до первой октавы
        (61, True),  # до-диез
        (64, False),  # ми
        (65, False),  # фа — сразу после ми, чёрной между ними нет
        (66, True),  # фа-диез
        (21, False),  # ля субконтроктавы, нижняя клавиша
        (108, False),  # до пятой октавы, верхняя клавиша
    ],
)
def test_black_key_classification(pitch: int, black: bool):
    assert is_black_key(pitch) is black


def test_full_keyboard_has_52_white_keys():
    """У фортепиано 88 клавиш: 52 белые и 36 чёрных."""
    white = white_keys_in_range(PITCH_MIN, PITCH_MAX)

    assert len(white) == 52
    assert PITCH_MAX - PITCH_MIN + 1 - len(white) == 36


def test_black_pattern_alternates_by_semitone():
    """В хроматическом ряду чёрные клавиши подряд не идут."""
    pattern = black_pattern(60, 71)

    assert pattern == [
        False,
        True,
        False,
        True,
        False,
        False,
        True,
        False,
        True,
        False,
        True,
        False,
    ]
    assert not any(a and b for a, b in pairwise(pattern))


def test_black_keys_form_groups_of_two_and_three():
    """Узор 2-3 — то, по чему клавиатура привязывается к абсолютным нотам.

    Группы видны не в хроматическом ряду, а в ответе на вопрос «есть ли
    чёрная клавиша справа от этой белой»: до, ре — да; ми — нет; фа, соль,
    ля — да; си — нет.
    """
    octave = list(range(60, 72))
    has_black_after = [
        is_black_key(pitch + 1)
        for pitch in octave
        if not is_black_key(pitch) and pitch + 1 <= octave[-1] + 1
    ]

    groups = []
    run = 0
    for black in has_black_after:
        if black:
            run += 1
        elif run:
            groups.append(run)
            run = 0
    if run:
        groups.append(run)

    assert groups == [2, 3]


def test_octave_boundaries_must_be_white():
    with pytest.raises(ValueError, match="белые клавиши"):
        KeyboardGeometry(x=0, width=100, lowest_pitch=61, highest_pitch=96)


def test_lowest_must_not_exceed_highest():
    with pytest.raises(ValueError, match="не больше"):
        KeyboardGeometry(x=0, width=100, lowest_pitch=96, highest_pitch=36)


def test_white_keys_tile_the_area_without_gaps():
    geometry = KeyboardGeometry(x=100, width=520, lowest_pitch=36, highest_pitch=96)

    whites = geometry.white_pitches
    first_left, _ = geometry.bounds(whites[0])
    _, last_right = geometry.bounds(whites[-1])

    assert first_left == pytest.approx(100.0)
    assert last_right == pytest.approx(620.0)

    for left_pitch, right_pitch in pairwise(whites):
        assert geometry.bounds(left_pitch)[1] == pytest.approx(geometry.bounds(right_pitch)[0])


def test_black_key_sits_on_white_boundary_and_is_narrower():
    geometry = KeyboardGeometry(x=0, width=520, lowest_pitch=36, highest_pitch=96)

    left, right = geometry.bounds(61)  # до-диез
    white_right = geometry.bounds(60)[1]

    assert (left + right) / 2 == pytest.approx(white_right)
    assert right - left < geometry.white_width


def test_layout_marks_cropped_range():
    full = KeyboardGeometry(x=0, width=520).build_layout(y=10, height=100)
    partial = KeyboardGeometry(x=0, width=520, lowest_pitch=36, highest_pitch=96).build_layout(
        y=10, height=100
    )

    assert full.is_cropped is False
    assert partial.is_cropped is True
    assert len(partial.keys) == 96 - 36 + 1
