"""Отладочный разбор видеоряда: разметка, ноты, картинки, метрики.

Инструмент для проверки этапа `vision` глазами. Он умеет три вещи:

* разобрать любой ролик и показать, что нашлось;
* нарисовать поверх кадра найденную клавиатуру — так сразу видно,
  промахнулась ли детекция;
* собрать синтетический ролик с известным ответом и посчитать метрики.

Примеры:

```
python scripts/vision_cli.py demo
python scripts/vision_cli.py demo --scheme dark --noise 6 --report demo.png
python scripts/vision_cli.py analyze video.mp4 --overlay layout.png --csv notes.csv
python scripts/vision_cli.py analyze video.mp4 --truth notes.csv
```
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
from pathlib import Path

try:
    import cv2
    import numpy as np
except ImportError as exc:  # pragma: no cover - зависит от окружения
    # Частый случай на Windows: несколько установок Python, и `python`
    # в PATH указывает не на ту, куда pip ставил пакеты. Traceback здесь
    # ничего не объясняет, поэтому сообщение прямое.
    raise SystemExit(
        f"Не найдена библиотека {exc.name}.\n\n"
        f"Текущий интерпретатор: {sys.executable}\n\n"
        "Установите зависимости именно в него:\n"
        '    python -m pip install -e ".[dev]"\n\n'
        "Либо запускайте через run.cmd — он сам выберет подходящий Python:\n"
        "    run demo"
    ) from exc

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mir.common.enums import Hand
from mir.common.errors import KeyboardNotFoundError
from mir.common.logging import setup_logging
from mir.common.types import Frame, KeyboardLayout, NoteEvent
from mir.eval import evaluate
from mir.vision import accel
from mir.vision.analyzer import VisionResult, analyze_file

_HAND_BGR = {
    Hand.LEFT: (60, 140, 220),
    Hand.RIGHT: (80, 200, 80),
    Hand.UNKNOWN: (180, 180, 180),
}


def draw_layout(frame: Frame, layout: KeyboardLayout) -> Frame:
    """Нарисовать найденную клавиатуру поверх кадра.

    Белые клавиши обводятся зелёным, чёрные заливаются полупрозрачным
    синим, каждое «до» подписывается. Съехавшая на клавишу разметка
    видна на такой картинке мгновенно.
    """
    canvas = frame.copy()
    x, y, width, height = layout.bbox
    cv2.rectangle(canvas, (x, y), (x + width, y + height), (0, 255, 255), 2)

    overlay = canvas.copy()
    for key in layout.keys:
        if key.is_black:
            cv2.rectangle(
                overlay, (key.x_min, y), (key.x_max, y + int(height * 0.6)), (255, 120, 0), -1
            )
        else:
            cv2.rectangle(canvas, (key.x_min, y), (key.x_max, y + height), (0, 200, 0), 1)
    cv2.addWeighted(overlay, 0.45, canvas, 0.55, 0, canvas)

    for key in layout.keys:
        if key.pitch % 12 == 0:
            cv2.putText(
                canvas,
                f"C{key.pitch // 12 - 1}",
                (key.x_min, y + height - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                (0, 0, 255),
                1,
                cv2.LINE_AA,
            )

    caption = (
        f"{layout.lowest_pitch}..{layout.highest_pitch}  "
        f"keys={len(layout.keys)}  conf={layout.confidence:.2f}"
    )
    cv2.putText(
        canvas, caption, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA
    )
    return canvas


def draw_piano_roll(
    notes: list[NoteEvent], width: int = 1200, height: int = 420, title: str = ""
) -> Frame:
    """Нарисовать найденные ноты в виде фортепианного валика.

    Глазами такая картинка читается быстрее списка: пропуски, дубли
    и съехавшие октавы видно сразу.
    """
    canvas = np.full((height, width, 3), 24, dtype=np.uint8)
    if not notes:
        cv2.putText(
            canvas, "нот не найдено", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2
        )
        return canvas

    margin = 40
    span = max(n.offset for n in notes)
    low = min(n.pitch for n in notes) - 2
    high = max(n.pitch for n in notes) + 2
    scale_x = (width - 2 * margin) / max(span, 1e-6)
    scale_y = (height - 2 * margin) / max(high - low, 1)

    for pitch in range(low, high + 1):
        y = int(height - margin - (pitch - low) * scale_y)
        if pitch % 12 == 0:
            cv2.line(canvas, (margin, y), (width - margin, y), (70, 70, 70), 1)
            cv2.putText(
                canvas,
                f"C{pitch // 12 - 1}",
                (4, y + 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                (140, 140, 140),
                1,
                cv2.LINE_AA,
            )

    for second in range(int(span) + 1):
        x = int(margin + second * scale_x)
        cv2.line(canvas, (x, margin), (x, height - margin), (55, 55, 55), 1)
        cv2.putText(
            canvas,
            f"{second}s",
            (x - 8, height - 14),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            (140, 140, 140),
            1,
            cv2.LINE_AA,
        )

    for note in notes:
        x0 = int(margin + note.onset * scale_x)
        x1 = max(int(margin + note.offset * scale_x), x0 + 2)
        y = int(height - margin - (note.pitch - low) * scale_y)
        cv2.rectangle(canvas, (x0, y - 4), (x1, y + 4), _HAND_BGR[note.hand], -1)

    if title:
        cv2.putText(
            canvas,
            title,
            (margin, 26),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (220, 220, 220),
            1,
            cv2.LINE_AA,
        )
    return canvas


def print_result(result: VisionResult) -> None:
    """Вывести сводку разбора."""
    layout, profile = result.layout, result.profile
    speed = result.frames_processed / result.elapsed_sec if result.elapsed_sec else 0.0

    print(f"ядро            : {accel.backend_name()}")
    print(
        f"кадров          : {result.frames_processed} за {result.elapsed_sec:.1f} с "
        f"({speed:.0f} кадров/с)"
    )
    print(
        f"клавиатура      : {layout.lowest_pitch}..{layout.highest_pitch}, "
        f"клавиш {len(layout.keys)}, уверенность {layout.confidence:.2f}"
        f"{', обрезана' if layout.is_cropped else ''}"
    )
    print(f"линия касания   : y={profile.hit_line_y}")
    print(f"скорость падения: {profile.fall_speed:.0f} px/с")
    print(
        f"цветов блоков   : {len(profile.block_colors_hsv)}"
        f"{' (руки различимы)' if profile.has_hand_colors else ' (руки неразличимы)'}"
    )
    print(
        f"нот найдено     : {len(result.notes)} "
        f"(подсветка {result.notes_from_keys}, блоки {result.notes_from_blocks})"
    )
    for warning in result.warnings:
        print(f"  предупреждение: {warning}")


def write_csv(notes: list[NoteEvent], path: Path) -> None:
    """Сохранить ноты в CSV — тот же формат читается ключом --truth."""
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["pitch", "onset", "offset", "velocity", "hand"])
        for note in notes:
            writer.writerow(
                [
                    note.pitch,
                    f"{note.onset:.4f}",
                    f"{note.offset:.4f}",
                    note.velocity,
                    note.hand.name,
                ]
            )


def read_csv(path: Path) -> list[NoteEvent]:
    """Прочитать эталонные ноты."""
    with path.open(encoding="utf-8", newline="") as fh:
        return [
            NoteEvent(
                pitch=int(row["pitch"]),
                onset=float(row["onset"]),
                offset=float(row["offset"]),
                velocity=int(row.get("velocity") or 64),
                hand=Hand[row.get("hand") or "UNKNOWN"],
            )
            for row in csv.DictReader(fh)
        ]


def _save(image: Frame, path: Path, what: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), image)
    print(f"{what}: {path}")


def _first_frame(video: Path) -> Frame | None:
    capture = cv2.VideoCapture(str(video))
    capture.set(cv2.CAP_PROP_POS_FRAMES, int(capture.get(cv2.CAP_PROP_FRAME_COUNT) // 2))
    ok, frame = capture.read()
    capture.release()
    return frame if ok else None


VIDEO_SUFFIXES = frozenset({".mp4", ".mkv", ".webm", ".avi", ".mov", ".m4v", ".flv"})


def _dump_median(video: Path, path: Path) -> bool:
    """Сохранить медианный кадр — то, на чём работает детектор.

    Первое, что нужно увидеть при неудачной детекции: заставка,
    неверные пропорции и посторонние наложения видны сразу.
    """
    import numpy as np_local

    capture = cv2.VideoCapture(str(video))
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        capture.release()
        return False

    frames = []
    for index in np_local.linspace(total * 0.15, total * 0.92, 40, dtype=int):
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(index))
        ok, frame = capture.read()
        if ok:
            frames.append(frame)
    capture.release()

    if not frames:
        return False
    _save(accel.median_frame(frames), path, "медианный кадр")
    return True


def command_analyze(args: argparse.Namespace) -> int:
    """Разобрать готовый ролик."""
    video = Path(args.video)
    if not video.exists():
        print(f"файл не найден: {video}")
        return 1

    if video.suffix.lower() not in VIDEO_SUFFIXES:
        print(f"{video.name} — это не видеофайл, а {video.suffix or 'файл без расширения'}.")
        print()
        print("Этап загрузки кладёт рядом аудиодорожку, а само видео — в кэш.")
        print("Путь к видео печатает команда fetch строкой «Видео:», или посмотрите кэш:")
        print(f"    dir {Path('~/.mir_cache').expanduser()}")
        return 1

    try:
        result = analyze_file(video, max_seconds=args.seconds, start_seconds=args.start)
    except KeyboardNotFoundError as exc:
        print(f"Клавиатура не найдена: {exc.user_message}")
        print(f"  причина: {exc}")
        dump = Path(args.overlay or "build/median.png").with_name("median.png")
        if _dump_median(video, dump):
            print()
            print(f"Медианный кадр сохранён: {dump}")
            print("По нему видно, что именно разбирал детектор. Если там заставка")
            print("или логотип вместо клавиатуры — начните разбор позже: --start 60")
        return 1

    print_result(result)

    if args.csv:
        write_csv(result.notes, Path(args.csv))
        print(f"ноты сохранены: {args.csv}")

    if args.overlay:
        frame = _first_frame(video)
        if frame is None:
            print("не удалось прочитать кадр для картинки")
        else:
            _save(draw_layout(frame, result.layout), Path(args.overlay), "разметка клавиатуры")

    if args.roll:
        _save(draw_piano_roll(result.notes, title=video.name), Path(args.roll), "валик")

    if args.truth:
        score = evaluate(read_csv(Path(args.truth)), result.notes)
        print()
        print(score.format_report(f"сравнение с {Path(args.truth).name}"))
        return 0 if score.f1 >= args.min_f1 else 1

    return 0


def command_demo(args: argparse.Namespace) -> int:
    """Собрать синтетический ролик и проверить на нём себя.

    Эталон известен точно, потому что ролик рисуется из него же.
    """
    from tests.fixtures.synth import (
        ColorScheme,
        SynthConfig,
        render_visualizer_video,
        simple_melody,
    )

    schemes = {"light": ColorScheme(), "dark": ColorScheme.dark(), "mono": ColorScheme.monochrome()}
    config = SynthConfig(
        width=args.width,
        height=args.height,
        noise_sigma=args.noise,
        glow=args.glow,
        colors=schemes[args.scheme],
        watermark=args.watermark,
    )

    truth = simple_melody()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    video = render_visualizer_video(truth, out_dir / "demo.mp4", config)
    print(f"ролик собран    : {video}")

    result = analyze_file(video)
    print_result(result)

    score = evaluate(truth, result.notes)
    print()
    print(score.format_report(f"схема {args.scheme}, шум {args.noise}"))

    frame = _first_frame(video)
    if frame is not None:
        _save(draw_layout(frame, result.layout), out_dir / "layout.png", "разметка клавиатуры")
    _save(
        np.vstack(
            [
                draw_piano_roll(truth, title="эталон"),
                draw_piano_roll(result.notes, title=f"найдено, F1={score.f1:.3f}"),
            ]
        ),
        out_dir / "roll.png",
        "эталон и результат",
    )

    if args.report:
        _save(draw_piano_roll(result.notes, title=f"F1={score.f1:.3f}"), Path(args.report), "отчёт")

    return 0 if score.f1 >= args.min_f1 else 1


def build_parser() -> argparse.ArgumentParser:
    """Собрать разбор аргументов."""
    parser = argparse.ArgumentParser(description="Отладка разбора видеоряда")
    parser.add_argument("--verbose", action="store_true", help="подробный журнал")
    parser.add_argument(
        "--min-f1", type=float, default=0.9, help="код возврата 1, если F1 ниже (для CI)"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    analyze = sub.add_parser("analyze", help="разобрать готовый ролик")
    analyze.add_argument("video")
    analyze.add_argument("--csv", help="куда сохранить найденные ноты")
    analyze.add_argument("--overlay", help="картинка с разметкой клавиатуры")
    analyze.add_argument("--roll", help="картинка с фортепианным валиком")
    analyze.add_argument("--truth", help="CSV с эталонными нотами для сравнения")
    analyze.add_argument(
        "--seconds",
        type=float,
        help="разобрать только N секунд — быстрая проверка разметки",
    )
    analyze.add_argument(
        "--start",
        type=float,
        default=0.0,
        help="с какой секунды начинать: ролики часто открываются заставкой",
    )
    analyze.set_defaults(func=command_analyze)

    demo = sub.add_parser("demo", help="синтетический ролик с известным ответом")
    demo.add_argument("--scheme", choices=["light", "dark", "mono"], default="light")
    demo.add_argument("--width", type=int, default=1280)
    demo.add_argument("--height", type=int, default=720)
    demo.add_argument("--noise", type=float, default=0.0, help="сигма шума сжатия")
    demo.add_argument("--glow", action="store_true", help="свечение вокруг блоков")
    demo.add_argument("--watermark", help="надпись поверх кадра")
    demo.add_argument("--out-dir", default="build/vision_demo")
    demo.add_argument("--report", help="дополнительная картинка с результатом")
    demo.set_defaults(func=command_demo)

    return parser


def main(argv: list[str] | None = None) -> int:
    """Точка входа."""
    args = build_parser().parse_args(argv)
    setup_logging(level=logging.DEBUG if args.verbose else logging.WARNING)
    result: int = args.func(args)
    return result


if __name__ == "__main__":
    raise SystemExit(main())
