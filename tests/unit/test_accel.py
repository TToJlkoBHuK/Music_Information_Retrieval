"""Нативное ядро и запасная реализация обязаны совпадать.

Расхождение здесь означало бы, что результат распознавания зависит
от наличия компилятора у пользователя.
"""

from __future__ import annotations

import numpy as np
import pytest

from mir.vision import accel


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(20240516)


@pytest.fixture
def scene(rng: np.random.Generator):
    hsv = rng.integers(0, 255, size=(240, 352, 3), dtype=np.uint8)
    regions = np.array(
        [[i * 8, (i + 1) * 8, 100, 200] for i in range(44)],
        dtype=np.int32,
    )
    references = (rng.random((44, 3)) * 200.0).astype(np.float32)
    return hsv, regions, references


def test_backend_name_reports_reality():
    assert accel.backend_name() == ("mir_core (C++)" if accel.HAS_NATIVE else "numpy")


def test_sample_regions_uniform_patch():
    hsv = np.zeros((40, 40, 3), dtype=np.uint8)
    hsv[:, :] = (60, 200, 180)
    regions = np.array([[0, 20, 0, 10]], dtype=np.int32)

    colors = accel.sample_regions(hsv, regions)

    assert colors.shape == (1, 3)
    np.testing.assert_allclose(colors[0], (60.0, 200.0, 180.0), atol=0.01)


def test_sample_regions_outside_frame_is_zero():
    hsv = np.full((20, 20, 3), 100, dtype=np.uint8)
    regions = np.array([[100, 140, 50, 60]], dtype=np.int32)

    colors = accel.sample_regions(hsv, regions)

    np.testing.assert_array_equal(colors[0], (0.0, 0.0, 0.0))


def test_identical_colors_give_zero_deviation():
    colors = np.array([[30.0, 100.0, 120.0]], dtype=np.float32)

    assert accel.deviation_from_colors(colors, colors)[0] == pytest.approx(0.0)


def test_hue_wraps_around():
    """Тон 179 и тон 0 — соседние оттенки, а не противоположные.

    Цвета взяты насыщенными: у бесцветных тон в расчёт не идёт.
    """
    near_end = np.array([[179.0, 255.0, 255.0]], dtype=np.float32)
    zero = np.array([[0.0, 255.0, 255.0]], dtype=np.float32)

    assert accel.deviation_from_colors(near_end, zero)[0] == pytest.approx(
        (1.0 / 90.0) * 0.5, abs=1e-5
    )


def test_hue_outweighs_brightness():
    """Блик меняет яркость, подсветка — тон. Путать их нельзя."""
    reference = np.array([[30.0, 200.0, 120.0]], dtype=np.float32)
    brighter = np.array([[30.0, 200.0, 240.0]], dtype=np.float32)
    other_hue = np.array([[120.0, 200.0, 120.0]], dtype=np.float32)

    dev_bright = accel.deviation_from_colors(brighter, reference)[0]
    dev_hue = accel.deviation_from_colors(other_hue, reference)[0]

    assert dev_hue > dev_bright


def test_hue_ignored_for_colourless_pair():
    """Тон белого пикселя не определён.

    Белая клавиша (245, 245, 245) при малейшем шуме даёт произвольный
    оттенок; без этой поправки шум сжатия на 480p давал ложные нажатия
    с отклонением до 0.34 при пороге 0.25.
    """
    almost_white = np.array([[170.0, 3.0, 245.0]], dtype=np.float32)
    also_white = np.array([[20.0, 2.0, 245.0]], dtype=np.float32)

    assert accel.deviation_from_colors(almost_white, also_white)[0] < 0.01


def test_white_key_against_colour_still_responds():
    """Поправка на бесцветность не должна гасить полезный сигнал.

    Вес тона переходит насыщенности, которая здесь и меняется.
    """
    white = np.array([[0.0, 2.0, 245.0]], dtype=np.float32)
    green = np.array([[60.0, 200.0, 200.0]], dtype=np.float32)

    assert accel.deviation_from_colors(green, white)[0] > 0.4


def test_deviation_never_exceeds_one():
    far = np.array([[90.0, 255.0, 255.0]], dtype=np.float32)
    zero = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)

    assert accel.deviation_from_colors(far, zero)[0] <= 1.0


def test_median_frame_drops_outlier():
    dark = np.full((8, 8, 3), 10, dtype=np.uint8)
    bright = np.full((8, 8, 3), 250, dtype=np.uint8)

    result = accel.median_frame([dark, dark.copy(), bright])

    assert np.all(result == 10)


def test_median_frame_rejects_empty():
    with pytest.raises(ValueError, match="хотя бы один кадр"):
        accel.median_frame([])


@pytest.mark.skipif(not accel.HAS_NATIVE, reason="нативное ядро не собрано")
def test_native_matches_numpy_on_sampling(scene):
    hsv, regions, _ = scene

    native = accel.sample_regions(hsv, regions)
    fallback = accel._sample_regions_numpy(hsv, regions, accel.SAMPLE_INSET)

    np.testing.assert_allclose(native, fallback, atol=1e-3)


@pytest.mark.skipif(not accel.HAS_NATIVE, reason="нативное ядро не собрано")
def test_native_matches_numpy_on_deviations(scene):
    hsv, regions, references = scene

    native = accel.key_deviations(hsv, regions, references)
    fallback = accel.deviation_from_colors(
        accel._sample_regions_numpy(hsv, regions, accel.SAMPLE_INSET), references
    )

    np.testing.assert_allclose(native, fallback, atol=1e-4)


@pytest.mark.skipif(not accel.HAS_NATIVE, reason="нативное ядро не собрано")
def test_native_matches_numpy_on_median(rng: np.random.Generator):
    frames = [rng.integers(0, 255, size=(24, 32, 3), dtype=np.uint8) for _ in range(7)]

    native = accel.median_frame(frames)
    fallback = np.median(np.stack(frames), axis=0).astype(np.uint8)

    np.testing.assert_array_equal(native, fallback)


@pytest.mark.skipif(not accel.HAS_NATIVE, reason="нативное ядро не собрано")
def test_native_rejects_mismatched_shapes():
    hsv = np.zeros((20, 20, 3), dtype=np.uint8)
    regions = np.array([[0, 10, 0, 10], [10, 20, 0, 10]], dtype=np.int32)
    references = np.zeros((1, 3), dtype=np.float32)

    with pytest.raises(ValueError):
        accel.key_deviations(hsv, regions, references)
