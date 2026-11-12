"""Слияние каналов: из нот видео и нот звука — один список (этап 5, #30).

Каналы берутся из готовых CSV или разбираются на месте:

```
python scripts/fusion_cli.py fuse --video-csv video.csv --audio-csv audio.csv --midi fused.mid
python scripts/fusion_cli.py fuse video.mp4 --video-csv video.csv --audio-csv audio.csv
python scripts/fusion_cli.py fuse "https://youtu.be/..." --csv fused.csv --roll fused.png
```

Файл канала, который уже есть, читается; которого нет — считается из ролика
и сохраняется туда же. Поэтому вторая команда после первого запуска
отрабатывает за секунду, а не за семь минут.

Что смотреть:

* сводку — сколько нот подтверждено обоими каналами и почему приняты или
  отброшены остальные;
* `--midi` — итог можно послушать в любом плеере и сравнить с роликом;
* `--roll` — валик: зелёным подтверждённое, синим только видео, оранжевым
  принятое из звука, серым отброшенное;
* `--decisions` — таблица решений по каждой ноте, для разбора ошибок.
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
import tempfile
from pathlib import Path

try:
    import numpy as np
except ImportError as exc:  # pragma: no cover - зависит от окружения
    raise SystemExit(
        f"Не найдена библиотека {exc.name}.\n\n"
        f"Текущий интерпретатор: {sys.executable}\n\n"
        "Установите зависимости именно в него:\n"
        '    python -m pip install -e ".[dev]"\n\n'
        "Либо запускайте через run.cmd — он сам выберет подходящий Python:\n"
        "    run fuse --help"
    ) from exc

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mir.common.enums import Source
from mir.common.errors import MirError
from mir.common.logging import setup_logging
from mir.common.types import KeyboardLayout, NoteEvent
from mir.config import MirConfig, load_config
from mir.eval import candidates_path, read_notes_csv, write_notes_csv
from mir.export import write_performance_midi
from mir.fusion import FusionResult, Reason, fuse

_NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def note_name(pitch: int) -> str:
    """MIDI-номер → научное обозначение: 60 → C4."""
    return f"{_NOTE_NAMES[pitch % 12]}{pitch // 12 - 1}"


class Channels:
    """Ноты двух каналов и то, что о них известно."""

    def __init__(self) -> None:
        self.video: list[NoteEvent] = []
        self.candidates: list[NoteEvent] = []
        self.audio: list[NoteEvent] = []
        self.layout: KeyboardLayout | None = None
        self.video_reliable = True


def load_channels(args: argparse.Namespace, config: MirConfig, work_dir: Path) -> Channels:
    """Собрать ноты каналов: из CSV, а чего нет — разбором ролика.

    Raises:
        MirError: Канал не из чего взять или разбор не удался.
    """
    channels = Channels()
    video_csv = Path(args.video_csv) if args.video_csv else None
    audio_csv = Path(args.audio_csv) if args.audio_csv else None
    need_video = video_csv is None or not video_csv.exists()
    need_audio = audio_csv is None or not audio_csv.exists()
    if (need_video or need_audio) and not args.source:
        missing = "видео" if need_video else "звука"
        raise MirError(
            f"нет нот {missing}",
            f"Нет файла с нотами {missing}. Укажите ролик — канал будет разобран, "
            "или путь к готовому CSV",
        )

    video_path = audio_path = None
    if args.source and (need_video or need_audio):
        from audio_cli import resolve_media

        video_path, audio_path = resolve_media(args.source, config, work_dir)

    if not need_video and video_csv is not None:
        channels.video = read_notes_csv(video_csv)
        unproven = candidates_path(video_csv)
        if unproven.exists():
            channels.candidates = read_notes_csv(unproven)
        print(f"ноты видео      : {len(channels.video)} из {video_csv}")
    else:
        if video_path is None:
            raise MirError("в источнике нет видеоряда", "Для нот видео нужен ролик, а не звук")
        _analyze_video(channels, video_path, video_csv, config)

    if not need_audio and audio_csv is not None:
        channels.audio = read_notes_csv(audio_csv)
        print(f"ноты звука      : {len(channels.audio)} из {audio_csv}")
    else:
        assert audio_path is not None
        _analyze_audio(channels, audio_path, audio_csv, config)
    return channels


def _analyze_video(
    channels: Channels, video: Path, save_to: Path | None, config: MirConfig
) -> None:
    from audio_cli import ProgressLine

    from mir.vision.analyzer import analyze_file

    progress = ProgressLine()
    result = analyze_file(video, config, progress=progress)
    progress.finish()
    channels.video, channels.candidates = result.notes, result.unconfirmed
    channels.layout = result.layout
    channels.video_reliable = result.source == "blocks"
    origin = "блоки над клавиатурой" if result.source == "blocks" else "подсветка клавиш"
    print(f"ноты видео      : {len(result.notes)}, источник — {origin}")
    for warning in result.warnings:
        print(f"  предупреждение: {warning}")
    if save_to is not None:
        write_notes_csv(result.notes, save_to)
        write_notes_csv(result.unconfirmed, candidates_path(save_to))
        print(f"  сохранены: {save_to}")


def _analyze_audio(
    channels: Channels, audio: Path, save_to: Path | None, config: MirConfig
) -> None:
    from audio_cli import ProgressLine

    from mir.audio import analyze_audio

    progress = ProgressLine()
    result = analyze_audio(audio, config, progress=progress)
    progress.finish()
    channels.audio = result.notes
    print(f"ноты звука      : {len(result.notes)}")
    for warning in result.warnings:
        print(f"  предупреждение: {warning}")
    if not result.notes:
        raise MirError(
            "транскрипция звука не дала нот",
            "Нот звука нет. Если нейросеть не установлена: run audio-setup",
        )
    if save_to is not None:
        write_notes_csv(result.notes, save_to)
        print(f"  сохранены: {save_to}")


def print_report(result: FusionResult, channels: Channels) -> None:
    """Сводка слияния: согласие каналов и решения по причинам."""
    curve = result.curve
    if curve.reliable:
        low, high = float(curve.offsets.min()) * 1000, float(curve.offsets.max()) * 1000
        print(
            f"рассинхрон      : {curve.overall.offset_ms:+.0f} мс в целом; по ходу ролика "
            f"от {low:+.0f} до {high:+.0f} мс"
        )
        print(
            f"согласие каналов: звук подтверждает {result.video_trust:.0%} нот видео, "
            f"видео — {result.audio_trust:.0%} нот звука"
        )
    lead = "видео" if result.lead is Source.VIDEO else "звук"
    print(f"ведущий канал   : {lead}")
    if np.isfinite(result.confidence_threshold):
        print(
            f"порог для звука : нота только из звука принимается при уверенности "
            f"от {result.confidence_threshold:.2f}"
        )
    if result.pitch_shift:
        print(f"октава          : ноты видео сдвинуты на {result.pitch_shift:+d} полутонов")

    by_source = {source: sum(n.source is source for n in result.notes) for source in Source}
    print(
        f"итог            : {len(result.notes)} нот — оба канала {by_source[Source.BOTH]}, "
        f"только видео {by_source[Source.VIDEO]}, только звук {by_source[Source.AUDIO]}"
    )
    if channels.candidates:
        print(f"  блоков-кандидатов видео было {len(channels.candidates)}")
    counts = result.counts
    for title, accepted in (("принято", True), ("отброшено", False)):
        rows = [(counts[r], r) for r in Reason if counts[r] and r.accepted is accepted]
        if rows:
            print(f"{title}:")
            for count, reason in rows:
                print(f"  {count:6d}  {reason.title_ru}")
    for warning in result.warnings:
        print(f"предупреждение  : {warning}")


def write_decisions(result: FusionResult, path: Path) -> None:
    """Таблица решений: по строке на каждую исходную ноту или пару."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            ["onset", "pitch", "name", "verdict", "reason", "explanation", "video", "audio",
             "audio_confidence", "audio_velocity"]
        )  # fmt: skip
        for decision in result.decisions:
            note = decision.note or decision.audio or decision.video
            assert note is not None
            audio = decision.audio
            writer.writerow(
                [
                    f"{note.onset:.3f}",
                    note.pitch,
                    note_name(note.pitch),
                    "+" if decision.note is not None else "-",
                    decision.reason.value,
                    decision.reason.title_ru,
                    int(decision.video is not None),
                    int(audio is not None),
                    f"{audio.confidence:.3f}" if audio else "",
                    audio.velocity if audio else "",
                ]
            )
    print(f"решения         : {path}")


def draw_roll(result: FusionResult, start: float, end: float, out: Path) -> None:
    """Валик итога: чем подтверждена каждая нота и что отброшено."""
    import cv2

    width, height, left, bottom = 1600, 720, 60, 30
    canvas = np.full((height, width, 3), 255, dtype=np.uint8)

    def y(pitch: float) -> int:
        return int(height - bottom - (pitch - 21) / 87 * (height - bottom - 30))

    def x(t: float) -> int:
        return int(left + (t - start) / (end - start) * (width - left - 20))

    for pitch in range(24, 109, 12):
        cv2.line(canvas, (left, y(pitch)), (width - 20, y(pitch)), (225, 225, 225), 1)
        cv2.putText(canvas, note_name(pitch), (8, y(pitch) + 4), 0, 0.4, (0, 0, 0), 1)
    span = max(int((end - start) / 10), 1)
    for second in range(int(start), int(end) + 1, span):
        cv2.line(canvas, (x(second), 25), (x(second), height - bottom), (235, 235, 235), 1)
        cv2.putText(canvas, f"{second}s", (x(second) - 10, height - 10), 0, 0.4, (0, 0, 0), 1)

    colours = {
        Source.BOTH: (40, 160, 40),
        Source.VIDEO: (200, 90, 30),
        Source.AUDIO: (20, 140, 240),
    }
    dropped = [d.audio or d.video for d in result.decisions if d.note is None]
    layers = [(note, (170, 170, 170), 3) for note in dropped if note is not None]
    layers += [(note, colours[note.source], 0) for note in result.notes]
    for note, colour, lift in layers:
        if note.offset < start or note.onset > end:
            continue
        top = y(note.pitch) + lift
        left_x, right_x = x(max(note.onset, start)), x(min(note.offset, end))
        cv2.rectangle(canvas, (left_x, top - 2), (max(right_x, left_x + 1), top + 1), colour, -1)
    legend = "green: both channels   blue: video only   orange: audio only   grey: rejected"
    cv2.putText(canvas, legend, (left + 10, 18), 0, 0.5, (0, 0, 0), 1)
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), canvas)
    print(f"валик           : {out}")


def command_fuse(args: argparse.Namespace) -> int:
    """Слить каналы и сохранить итог."""
    config = load_config()
    try:
        with tempfile.TemporaryDirectory(prefix="mir_fuse_") as tmp:
            channels = load_channels(args, config, Path(tmp))
    except MirError as error:
        print(f"ошибка: {error.user_message}")
        return 1

    result = fuse(
        channels.video,
        channels.audio,
        config.fusion,
        candidates=channels.candidates,
        layout=channels.layout,
        video_reliable=channels.video_reliable,
    )
    print_report(result, channels)

    if args.csv:
        write_notes_csv(result.notes, Path(args.csv))
        print(f"ноты сохранены  : {args.csv}")
    if args.midi:
        write_performance_midi(result.notes, Path(args.midi))
        print(f"MIDI для слуха  : {args.midi}")
    if args.decisions:
        write_decisions(result, Path(args.decisions))
    if args.roll and result.notes:
        end = args.end or max(note.offset for note in result.notes)
        draw_roll(result, args.start, end, Path(args.roll))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Слияние нот видео и звука")
    parser.add_argument("--verbose", action="store_true", help="подробный журнал")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("fuse", help="слить каналы в один список нот")
    run.add_argument("source", nargs="?", help="ролик или ссылка — если каналы ещё не разобраны")
    run.add_argument("--video-csv", help="ноты видео: есть — читаются, нет — считаются сюда")
    run.add_argument("--audio-csv", help="ноты звука: есть — читаются, нет — считаются сюда")
    run.add_argument("--csv", help="куда сохранить итоговые ноты")
    run.add_argument("--midi", help="MIDI итога — послушать и сравнить с роликом")
    run.add_argument("--decisions", help="таблица решений по каждой ноте, CSV")
    run.add_argument("--roll", help="картинка: валик итога")
    run.add_argument("--start", type=float, default=0.0, help="начало окна картинки, с")
    run.add_argument("--end", type=float, help="конец окна картинки, с")
    run.set_defaults(handler=command_fuse)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    setup_logging(logging.INFO if args.verbose else logging.WARNING)
    handler = args.handler
    return int(handler(args))


if __name__ == "__main__":
    sys.exit(main())
