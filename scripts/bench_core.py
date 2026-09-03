"""Сравнение нативного ядра с запасной реализацией на numpy.

Замеряет обе ветки [`mir.vision.accel`][] на данных реального размера
и печатает таблицу ускорения. Числа из этой таблицы идут в пояснительную
записку, поэтому важно, чтобы обе ветки считали одно и то же — это
проверяется здесь же.

Запуск:

```
python scripts/bench_core.py
python scripts/bench_core.py --frames 200 --keys 88
```
"""

from __future__ import annotations

import argparse
import sys
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np
import numpy.typing as npt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mir.common.types import Frame
from mir.vision import accel


def _build_scene(
    width: int, height: int, keys: int, rng: np.random.Generator
) -> tuple[Frame, npt.NDArray[np.int32], npt.NDArray[np.float32]]:
    """Кадр и разметка клавиш, приближённые к реальному ролику."""
    hsv = rng.integers(0, 255, size=(height, width, 3), dtype=np.uint8)
    key_width = width // keys
    regions = np.array(
        [[i * key_width, (i + 1) * key_width, height - 160, height - 20] for i in range(keys)],
        dtype=np.int32,
    )
    references = rng.random((keys, 3), dtype=np.float32) * 200.0
    return hsv, regions, references


def _time(fn: Callable[[], object], repeats: int) -> float:
    """Среднее время одного вызова в миллисекундах."""
    fn()  # прогрев: первый вызов включает выделение буферов
    started = time.perf_counter()
    for _ in range(repeats):
        fn()
    return (time.perf_counter() - started) / repeats * 1000.0


def main() -> int:
    """Точка входа."""
    parser = argparse.ArgumentParser(description="Замер нативного ядра mir_core")
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--keys", type=int, default=88)
    parser.add_argument("--repeats", type=int, default=50, help="повторов на замер")
    parser.add_argument("--frames", type=int, default=25, help="кадров для медианы")
    parser.add_argument("--video-minutes", type=float, default=4.0)
    parser.add_argument("--fps", type=float, default=30.0)
    args = parser.parse_args()

    if not accel.HAS_NATIVE:
        print("Нативное ядро не собрано — сравнивать не с чем.")
        print("Соберите его: cmake -B core/build -S core && cmake --build core/build")
        return 1

    rng = np.random.default_rng(20240516)
    hsv, regions, references = _build_scene(args.width, args.height, args.keys, rng)
    frames = [
        rng.integers(0, 255, size=(args.height // 4, args.width // 4, 3), dtype=np.uint8)
        for _ in range(args.frames)
    ]

    native = accel.key_deviations(hsv, regions, references)
    fallback = accel.deviation_from_colors(
        accel._sample_regions_numpy(hsv, regions, accel.SAMPLE_INSET), references
    )
    max_diff = float(np.max(np.abs(native - fallback)))

    print(f"кадр            : {args.width}×{args.height}, клавиш {args.keys}")
    print(f"расхождение веток: {max_diff:.2e}  (допуск 1e-4)")
    if max_diff > 1e-4:
        print("ВНИМАНИЕ: реализации расходятся, замер недействителен")
        return 1

    rows: list[tuple[str, float, float]] = []

    rows.append(
        (
            "key_deviations, 1 кадр",
            _time(lambda: accel.key_deviations(hsv, regions, references), args.repeats),
            _time(
                lambda: accel.deviation_from_colors(
                    accel._sample_regions_numpy(hsv, regions, accel.SAMPLE_INSET), references
                ),
                args.repeats,
            ),
        )
    )
    rows.append(
        (
            f"median_frame, {args.frames} кадров",
            _time(lambda: accel.median_frame(frames), max(args.repeats // 10, 1)),
            _time(
                lambda: np.median(np.stack(frames), axis=0).astype(np.uint8),
                max(args.repeats // 10, 1),
            ),
        )
    )

    width = max(len(name) for name, _, _ in rows)
    print()
    print(f"{'операция'.ljust(width)}  {'C++':>10}  {'numpy':>10}  {'ускорение':>10}")
    print("-" * (width + 36))
    for name, native_ms, numpy_ms in rows:
        speedup = numpy_ms / native_ms if native_ms > 0 else float("inf")
        print(f"{name.ljust(width)}  {native_ms:>8.3f} мс  {numpy_ms:>8.3f} мс  {speedup:>9.1f}×")

    total_frames = args.video_minutes * 60.0 * args.fps
    per_frame_native = rows[0][1] / 1000.0
    per_frame_numpy = rows[0][2] / 1000.0
    print()
    print(f"ролик {args.video_minutes:.0f} мин при {args.fps:.0f} fps — {total_frames:.0f} кадров:")
    print(f"  C++   : {total_frames * per_frame_native:>7.1f} с")
    print(f"  numpy : {total_frames * per_frame_numpy:>7.1f} с")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
