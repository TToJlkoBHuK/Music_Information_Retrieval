"""Детекция клавиатуры и привязка к абсолютным нотам (F-07)."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from mir.common.errors import KeyboardNotFoundError
from mir.common.types import Frame
from mir.vision.keyboard_detector import (
    KeyboardDetector,
    build_median_frame,
    match_black_pattern,
)
from mir.vision.keyboard_geometry import is_black_key


def _has_black_after(lowest: int, count: int) -> list[bool]:
    """Узор «есть ли чёрная справа» для count белых клавиш подряд."""
    out: list[bool] = []
    pitch = lowest
    while len(out) < count:
        if not is_black_key(pitch):
            out.append(is_black_key(pitch + 1))
        pitch += 1
    return out


def test_pattern_match_finds_c_at_start():
    """Ряд, начинающийся с «до», должен дать нулевое смещение."""
    offset, score = match_black_pattern(_has_black_after(60, 21))

    assert offset == 0
    assert score == pytest.approx(1.0)


@pytest.mark.parametrize("start_white", [60, 62, 64, 65, 67, 69, 71])
def test_pattern_match_recovers_any_starting_key(start_white: int):
    """Клавиатура может начинаться с любой белой клавиши, не только с «до».

    Возвращается индекс той белой клавиши в ряду, которая является «до»:
    ряд, начатый с «ре», даёт 6 — следующее «до» стоит шестым.
    """
    shift = sum(1 for p in range(60, start_white) if not is_black_key(p)) % 7
    expected = (7 - shift) % 7

    offset, score = match_black_pattern(_has_black_after(start_white, 21))

    assert offset == expected
    assert score == pytest.approx(1.0)


def test_pattern_match_survives_one_error():
    """Один сбой сегментации не должен ломать привязку целиком."""
    pattern = _has_black_after(60, 21)
    pattern[5] = not pattern[5]

    offset, score = match_black_pattern(pattern)

    assert offset == 0
    assert 0.9 < score < 1.0


def test_pattern_match_rejects_degenerate_input():
    """Вырожденный узор не должен приниматься за клавиатуру.

    Ряд из одних «есть чёрная справа» совпадает с эталоном на 5/7
    при любом сдвиге и без проверки отрыва прошёл бы порог 0.7.
    """
    _, score = match_black_pattern([True] * 21)

    assert score == pytest.approx(0.0)


def test_pattern_match_rejects_alternating_noise():
    _, score = match_black_pattern([True, False] * 10)

    assert score < 0.7


def test_pattern_match_handles_empty_input():
    assert match_black_pattern([]) == (0, 0.0)


def test_median_removes_transient_highlight():
    """Подсветка появляется в меньшинстве кадров и обязана исчезнуть."""
    empty = np.full((30, 30, 3), 200, dtype=np.uint8)
    lit = empty.copy()
    lit[10:20, 10:20] = 40

    median = build_median_frame([empty, empty.copy(), lit])

    assert np.all(median == 200)


def test_median_rejects_empty_input():
    with pytest.raises(ValueError):
        build_median_frame([])


def test_detector_rejects_frame_without_keyboard():
    """Ролик без клавиатуры должен давать понятную ошибку, а не мусор."""
    noise = [np.full((240, 320, 3), 120, dtype=np.uint8) for _ in range(5)]

    with pytest.raises(KeyboardNotFoundError):
        KeyboardDetector().detect(noise)


def _keyboard_strip(width: int, height: int) -> Frame:
    """Полоса с регулярным чёрно-белым узором — как у клавиатуры."""
    strip = np.full((height, width, 3), 240, dtype=np.uint8)
    for x in range(0, width, 12):
        strip[:, x : x + 1] = 40
    return strip


def test_band_found_when_keyboard_is_not_at_the_bottom():
    """Клавиатура не обязана упираться в низ кадра.

    У реальных роликов там оказываются чёрные поля, тень от клавиш
    или полоса педали. Прежний вариант в таком кадре не находил ничего.
    """
    frame = np.full((400, 480, 3), 25, dtype=np.uint8)
    frame[240:340] = _keyboard_strip(480, 100)

    band = KeyboardDetector()._find_band(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))

    assert 235 <= band.top <= 245
    assert 335 <= band.bottom <= 345


def test_lower_band_wins_over_upper_one():
    """Сверху похожий узор дают титры и ряды блоков; клавиатура — снизу."""
    frame = np.full((400, 480, 3), 25, dtype=np.uint8)
    frame[20:160] = _keyboard_strip(480, 140)
    frame[250:350] = _keyboard_strip(480, 100)

    band = KeyboardDetector()._find_band(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))

    assert band.top > 200
