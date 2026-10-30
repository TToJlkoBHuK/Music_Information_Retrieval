"""Отладочный разбор звука: ноты, атаки, темп, рассинхрон.

Проверка этапа 4 на слух и в цифрах:

* разобрать звук ролика и сохранить ноты в CSV;
* наложить щелчки на найденные доли или атаки — темп проверяется ухом
  за десять секунд, без всякой разметки;
* оценить рассинхрон звука и картинки;
* собрать синтетику с известным ответом и посчитать метрики.

Примеры:

```
python scripts/audio_cli.py demo
python scripts/audio_cli.py analyze video.mp4 --clicks beats.wav
python scripts/audio_cli.py analyze video.mp4 --no-transcribe --seconds 30
python scripts/audio_cli.py analyze audio.wav --csv notes_audio.csv --truth notes.csv
python scripts/audio_cli.py sync video.mp4 --video-csv notes_video.csv
python scripts/audio_cli.py fetch-model
python scripts/audio_cli.py doctor
```
"""

from __future__ import annotations

import argparse
import logging
import sys
import tempfile
import time
from dataclasses import replace
from pathlib import Path

try:
    import numpy as np
    import numpy.typing as npt
except ImportError as exc:  # pragma: no cover - зависит от окружения
    raise SystemExit(
        f"Не найдена библиотека {exc.name}.\n\n"
        f"Текущий интерпретатор: {sys.executable}\n\n"
        "Установите зависимости именно в него:\n"
        '    python -m pip install -e ".[dev]"\n\n'
        "Либо запускайте через run.cmd — он сам выберет подходящий Python:\n"
        "    run audio demo"
    ) from exc

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mir.audio import AudioResult, analyze_audio, load_audio
from mir.audio.models import (
    MODEL_URL,
    ModelRegistry,
    backend_status,
    select_device,
)
from mir.audio.signal import save_wav
from mir.common.enums import Stage
from mir.common.errors import MirError
from mir.common.logging import setup_logging
from mir.common.types import NoteEvent
from mir.config import MirConfig, load_config
from mir.eval import evaluate, read_notes_csv, write_notes_csv
from mir.fusion import (
    OffsetEstimate,
    apply_offset,
    estimate_av_offset,
    estimate_av_offset_from_onsets,
)

VIDEO_SUFFIXES = {".mp4", ".mkv", ".webm", ".mov", ".avi", ".flv"}

_NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def note_name(pitch: int) -> str:
    """MIDI-номер → научное обозначение: 60 → C4."""
    return f"{_NOTE_NAMES[pitch % 12]}{pitch // 12 - 1}"


def _config(args: argparse.Namespace) -> MirConfig:
    overrides: dict[str, dict[str, object]] = {}
    if getattr(args, "device", None):
        overrides["audio"] = {"device": args.device}
    return load_config(overrides=overrides or None)


def prepare_audio(path: Path, config: MirConfig, work_dir: Path) -> Path:
    """Видео → WAV через FFmpeg; звуковой файл возвращается как есть."""
    if path.suffix.lower() not in VIDEO_SUFFIXES:
        return path
    from mir.ingest.demuxer import Demuxer

    out = work_dir / f"{path.stem}.wav"
    print(f"извлечение звука из {path.name}…")
    return Demuxer(config.ingest).extract_audio(path, out)


def _click(sample_rate: int, accent: bool) -> npt.NDArray[np.float64]:
    t = np.arange(int(0.025 * sample_rate)) / sample_rate
    freq = 2000.0 if accent else 1200.0
    click: npt.NDArray[np.float64] = np.sin(2 * np.pi * freq * t) * np.exp(-t / 0.005)
    return click * (0.9 if accent else 0.6)


def mix_clicks(
    audio_path: Path,
    times: npt.NDArray[np.float64],
    accents: npt.NDArray[np.float64],
    out: Path,
    start: float,
    seconds: float | None,
) -> None:
    """Наложить щелчки на звук: высокий — на сильную долю, обычный — на прочие.

    Это самая быстрая проверка темпа: если щелчки идут вместе с музыкой,
    сетка верна; если «плывут» или попадают между долями — нет.
    """
    signal = load_audio(audio_path, start, seconds)
    mixed = signal.samples.astype(np.float64) * 0.7
    accent_set = {round(float(t), 4) for t in accents}
    for moment in times:
        click = _click(signal.sample_rate, round(float(moment), 4) in accent_set)
        at = int(np.rint((moment - signal.start) * signal.sample_rate))
        if 0 <= at < len(mixed):
            end = min(at + len(click), len(mixed))
            mixed[at:end] += click[: end - at]
    save_wav(out, mixed, signal.sample_rate)
    print(f"щелчки наложены: {out}")


def print_audio_result(result: AudioResult) -> None:
    """Сводка разбора звука."""
    details = result.tempo_details
    speed = result.elapsed_sec / result.duration if result.duration else 0.0
    print(
        f"звук            : {result.duration:.1f} с, разобран за {result.elapsed_sec:.1f} с "
        f"({speed:.2f}× реального времени)"
    )
    if result.device is None:
        print("транскрипция    : отключена")
    else:
        print(
            f"транскрипция    : {len(result.notes)} нот, педаль {len(result.pedals)} раз "
            f"({result.device})"
        )
        if result.notes:
            pitches = [n.pitch for n in result.notes]
            print(f"  диапазон      : {note_name(min(pitches))}..{note_name(max(pitches))}")
    print(f"атак            : {len(result.onsets)}")
    tempo = result.tempo
    print(
        f"темп            : {tempo.bpm:.1f} BPM"
        f"{' (задан вручную)' if tempo.is_manual else ''}, "
        f"уверенность {tempo.confidence:.2f}, разброс долей {details.variation:.1%}"
    )
    if details.correction != 1.0:
        print(f"  ошибка вдвое  : исправлена, {details.raw_bpm:.1f} → {tempo.bpm:.1f} BPM")
    if details.median_ioi_beats is not None:
        print(f"  типичная нота : {details.median_ioi_beats:.2f} доли")
    print(
        f"долей           : {len(tempo.beats)}, сильных {len(tempo.downbeats)} "
        f"(в такте {tempo.beats_per_bar}, предварительно)"
    )
    for warning in result.warnings:
        print(f"  предупреждение: {warning}")


class ProgressLine:
    """Строка прогресса с оценкой оставшегося времени.

    На полном ролике нейросеть на процессоре работает минутами, и без
    индикатора консоль выглядит зависшей. Строка перерисовывается на месте
    и обновляется не чаще раза в секунду — печать не должна тормозить счёт.
    """

    def __init__(self) -> None:
        self.started = time.monotonic()
        self.shown = 0.0
        self.active = False

    def __call__(self, stage: Stage, fraction: float) -> None:
        """Обновить строку: этап и доля выполнения 0..1."""
        now = time.monotonic()
        if fraction < 1.0 and now - self.shown < 1.0:
            return
        self.shown = now
        elapsed = now - self.started
        remaining = elapsed / fraction * (1 - fraction) if fraction > 0.02 else None
        tail = f", осталось ~{_clock(remaining)}" if remaining is not None else ""
        print(
            f"\r{stage.title_ru}: {fraction:4.0%} (прошло {_clock(elapsed)}{tail})   ",
            end="",
            flush=True,
        )
        self.active = True

    def finish(self) -> None:
        """Перевести строку, чтобы следующий вывод не налез на прогресс."""
        if self.active:
            print()
        self.active = False
        self.started = time.monotonic()


def _clock(seconds: float) -> str:
    minutes, rest = divmod(int(seconds), 60)
    return f"{minutes}:{rest:02d}"


def command_analyze(args: argparse.Namespace) -> int:
    """Разобрать звук ролика или звуковой файл."""
    source = Path(args.file)
    if not source.exists():
        print(f"файл не найден: {source}")
        return 1
    config = _config(args)

    if not args.no_transcribe and (problem := backend_status()) is not None:
        print(problem)
        print()
        print("Без нейросети можно посчитать атаки и темп: добавьте --no-transcribe")
        return 1

    with tempfile.TemporaryDirectory() as tmp:
        audio_path = prepare_audio(source, config, Path(tmp))
        progress = ProgressLine()
        result = analyze_audio(
            audio_path,
            config,
            transcribe=not args.no_transcribe,
            force_bpm=args.bpm,
            start_seconds=args.start,
            max_seconds=args.seconds,
            progress=progress,
        )
        progress.finish()
        print_audio_result(result)

        if args.clicks:
            mix_clicks(
                audio_path,
                result.tempo.beats,
                result.tempo.downbeats,
                Path(args.clicks),
                args.start,
                args.seconds,
            )
        if args.onset_clicks:
            # Все щелчки атак одинаковые: акцент здесь не несёт смысла.
            mix_clicks(
                audio_path,
                result.onsets,
                np.empty(0),
                Path(args.onset_clicks),
                args.start,
                args.seconds,
            )

    if args.csv:
        write_notes_csv(result.notes, Path(args.csv))
        print(f"ноты сохранены  : {args.csv}")

    if args.truth:
        if not result.notes:
            print("сравнивать не с чем: транскрипция отключена")
            return 1
        score = evaluate(read_notes_csv(Path(args.truth)), result.notes)
        print()
        print(score.format_report(f"сравнение с {Path(args.truth).name}"))
        return 0 if score.f1 >= args.min_f1 else 1
    return 0


def _print_offset(estimate: OffsetEstimate, threshold: float) -> None:
    method = "по нотам с учётом высоты" if estimate.method == "notes" else "по атакам звука"
    verdict = "применяется" if estimate.is_reliable(threshold) else "НЕ применяется — неуверенно"
    print(
        f"рассинхрон {method}: {estimate.offset_ms:+.0f} мс "
        f"(уверенность {estimate.confidence:.2f}, пар {estimate.pairs}) — {verdict}"
    )


def command_sync(args: argparse.Namespace) -> int:
    """Оценить сдвиг звука относительно картинки."""
    video = Path(args.video)
    if not video.exists():
        print(f"файл не найден: {video}")
        return 1
    config = _config(args)

    if args.video_csv:
        video_notes = read_notes_csv(Path(args.video_csv))
        print(f"ноты видео      : {len(video_notes)} из {args.video_csv}")
    else:
        from mir.vision.analyzer import analyze_file

        progress = ProgressLine()
        video_notes = analyze_file(video, config, progress, args.seconds, args.start).notes
        progress.finish()
        print(f"ноты видео      : {len(video_notes)}")
    if args.start or args.seconds:
        end = args.start + args.seconds if args.seconds else float("inf")
        video_notes = [n for n in video_notes if args.start <= n.onset < end]

    transcribe = not args.no_transcribe and backend_status() is None
    with tempfile.TemporaryDirectory() as tmp:
        audio_path = prepare_audio(video, config, Path(tmp))
        progress = ProgressLine()
        audio = analyze_audio(
            audio_path,
            config,
            transcribe=transcribe,
            start_seconds=args.start,
            max_seconds=args.seconds,
            progress=progress,
        )
        progress.finish()

    fusion = config.fusion
    threshold = fusion.min_offset_confidence
    estimates: list[OffsetEstimate] = []
    if audio.notes:
        estimates.append(estimate_av_offset(video_notes, audio.notes, fusion))
    estimates.append(estimate_av_offset_from_onsets(video_notes, audio.onsets, fusion))
    print()
    for estimate in estimates:
        _print_offset(estimate, threshold)
    if not transcribe:
        print("  (без транскрипции: оценка по атакам слабее на ровном ритме)")

    best = estimates[0]
    if args.aligned_csv and not audio.notes:
        print("ноты звука не сохранены: без транскрипции их нет")
    if args.aligned_csv and audio.notes:
        shift = -best.offset if best.is_reliable(threshold) else 0.0
        write_notes_csv(apply_offset(audio.notes, shift), Path(args.aligned_csv))
        print(f"ноты звука на шкале видео: {args.aligned_csv}")
    return 0


def _onset_report(truth: list[NoteEvent], onsets: npt.NDArray[np.float64]) -> str:
    targets = np.unique(np.round([n.onset for n in truth], 4))
    used: set[int] = set()
    errors: list[float] = []
    for t in targets:
        distance = np.abs(onsets - t)
        index = int(np.argmin(distance))
        if distance[index] <= 0.05 and index not in used:
            used.add(index)
            errors.append(float(onsets[index] - t))
    recall = len(errors) / len(targets)
    precision = len(errors) / max(len(onsets), 1)
    err = np.array(errors)
    return (
        f"полнота {recall:.3f}, точность {precision:.3f}, задержка {err.mean() * 1000:+.1f} мс, "
        f"разброс {err.std() * 1000:.1f} мс"
    )


def command_demo(args: argparse.Namespace) -> int:
    """Синтетика с известным ответом: атаки, темп, рассинхрон.

    Нейросеть здесь не проверяется: синтезированный тон — не рояль,
    и её качество на нём ничего не скажет. Для неё — `analyze --truth`
    на живой записи.
    """
    from tests.fixtures.audio_synth import render_notes, scale_melody

    config = load_config()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    truth = scale_melody(args.bpm, args.bars, subdivision=args.subdivision)
    samples = render_notes(truth, noise_db=args.noise)
    wav = save_wav(out_dir / "demo.wav", samples, 16000)
    print(f"синтетика       : {wav} ({args.bpm} BPM, по {args.subdivision} ноты на долю)")

    result = analyze_audio(wav, config, transcribe=False)
    print_audio_result(result)

    errors = [abs(result.tempo.bpm - args.bpm * k) / (args.bpm * k) for k in (0.5, 1.0, 2.0)]
    level = ["вдвое медленнее", "тот же", "вдвое быстрее"][int(np.argmin(errors))]
    tempo_ok = min(errors) <= 0.02
    print()
    print(
        f"темп            : ошибка {min(errors):.2%} (метрический уровень: {level}) — "
        f"{'в пределах NF-08' if tempo_ok else 'ХУЖЕ NF-08 (2 %)'}"
    )
    print(f"атаки           : {_onset_report(truth, result.onsets)}")

    delay = args.av_delay
    shifted = np.concatenate([np.zeros(int(delay * 16000), np.float32), samples])
    shifted_wav = save_wav(out_dir / "demo_delayed.wav", shifted, 16000)
    delayed = analyze_audio(shifted_wav, config, transcribe=False)
    fps = 30.0
    video_view = [
        replace(n, onset=float(np.ceil(n.onset * fps) / fps)) for n in truth
    ]  # подсветка — с первого кадра после атаки
    # Транскрипция имитируется эталоном, сдвинутым на ту же задержку:
    # проверяется оценка сдвига, а не нейросеть.
    audio_notes = [replace(n, onset=n.onset + delay, offset=n.offset + delay) for n in truth]
    lag = float(np.mean([v.onset - t.onset for v, t in zip(video_view, truth, strict=True)]))
    print(
        f"рассинхрон      : задан {delay * 1000:+.0f} мс, ожидается {(delay - lag) * 1000:+.0f} мс "
        f"(подсветка видна с первого кадра после атаки — здесь в среднем на "
        f"{lag * 1000:.0f} мс позже)"
    )
    threshold = config.fusion.min_offset_confidence
    _print_offset(estimate_av_offset(video_view, audio_notes, config.fusion), threshold)
    _print_offset(
        estimate_av_offset_from_onsets(video_view, delayed.onsets, config.fusion), threshold
    )
    if args.subdivision > 1:
        print("  (по атакам на ровном ритме сдвиг неоднозначен: соседняя доля совпадает")
        print("   почти так же хорошо — поэтому он и не применяется)")

    mix_clicks(wav, result.tempo.beats, result.tempo.downbeats, out_dir / "beats.wav", 0.0, None)
    return 0 if tempo_ok else 1


def command_fetch_model(args: argparse.Namespace) -> int:
    """Скачать веса модели заранее."""
    config = load_config()
    registry = ModelRegistry(config.ingest.cache_path, config.audio.model_path)
    found = registry.locate()
    if found is not None:
        print(f"веса уже на месте: {found[0]} ({'официальные' if found[1] else 'свои'})")
        return 0

    last = -1

    def show(done: int, total: int) -> None:
        nonlocal last
        percent = done * 100 // max(total, 1)
        if percent != last:
            last = percent
            print(f"\rзагрузка: {percent:3d}% ({done / 2**20:.0f} МБ)", end="", flush=True)

    try:
        path, _ = registry.ensure(show)
    except MirError as exc:
        print()
        print(exc.user_message)
        return 1
    print(f"\nвеса сохранены и проверены: {path}")
    return 0


def command_doctor(args: argparse.Namespace) -> int:
    """Проверить окружение этапа 4."""
    import importlib.metadata

    config = load_config()
    print(f"Python          : {sys.version.split()[0]} ({sys.executable})")
    for name in ("numpy", "librosa", "soundfile", "torch", "piano_transcription_inference"):
        try:
            print(f"{name:16s}: {importlib.metadata.version(name)}")
        except importlib.metadata.PackageNotFoundError:
            print(f"{name:16s}: не установлен")

    problem = backend_status()
    if problem is None:
        device, warning = select_device(config.audio.device)
        print(f"устройство      : {device}{' — ' + warning if warning else ''}")
    else:
        print()
        print(problem)

    found = ModelRegistry(config.ingest.cache_path, config.audio.model_path).locate()
    if found is None:
        print(f"веса модели     : не скачаны — run audio fetch-model ({MODEL_URL})")
    else:
        print(f"веса модели     : {found[0]} ({'официальные' if found[1] else 'свои'})")
    return 0 if problem is None and found is not None else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--verbose", action="store_true", help="подробный журнал")
    sub = parser.add_subparsers(dest="command", required=True)

    analyze = sub.add_parser("analyze", help="разобрать звук ролика или звуковой файл")
    analyze.add_argument("file", help="видео (звук извлекается FFmpeg) или WAV/FLAC/MP3")
    analyze.add_argument("--no-transcribe", action="store_true", help="только атаки и темп")
    analyze.add_argument("--bpm", type=float, help="задать темп вручную")
    analyze.add_argument("--start", type=float, default=0.0, help="с какой секунды")
    analyze.add_argument("--seconds", type=float, help="сколько секунд разобрать")
    analyze.add_argument("--device", choices=["auto", "cpu", "cuda"], help="где считать")
    analyze.add_argument("--csv", help="куда сохранить найденные ноты")
    analyze.add_argument("--truth", help="CSV с эталонными нотами для сравнения")
    analyze.add_argument("--min-f1", type=float, default=0.0, help="порог F1 для кода выхода")
    analyze.add_argument("--clicks", help="WAV со щелчками на долях — проверка темпа на слух")
    analyze.add_argument("--onset-clicks", help="WAV со щелчками на атаках")
    analyze.set_defaults(func=command_analyze)

    sync = sub.add_parser("sync", help="рассинхрон звука и картинки")
    sync.add_argument("video")
    sync.add_argument("--video-csv", help="ноты видео из vision_cli analyze --csv")
    sync.add_argument("--no-transcribe", action="store_true", help="оценка только по атакам")
    sync.add_argument("--start", type=float, default=0.0)
    sync.add_argument("--seconds", type=float)
    sync.add_argument("--device", choices=["auto", "cpu", "cuda"])
    sync.add_argument("--aligned-csv", help="сохранить ноты звука, перенесённые на шкалу видео")
    sync.set_defaults(func=command_sync)

    demo = sub.add_parser("demo", help="синтетика с известным ответом")
    demo.add_argument("--bpm", type=float, default=100.0)
    demo.add_argument("--bars", type=int, default=12)
    demo.add_argument("--subdivision", type=int, default=2, choices=[1, 2, 3, 4])
    demo.add_argument("--noise", type=float, help="шум, дБ относительно пика (например -25)")
    demo.add_argument("--av-delay", type=float, default=0.18, help="искусственный сдвиг звука, с")
    demo.add_argument("--out-dir", default="build/audio_demo")
    demo.set_defaults(func=command_demo)

    fetch = sub.add_parser("fetch-model", help="скачать веса нейросети (172 МБ)")
    fetch.set_defaults(func=command_fetch_model)

    doctor = sub.add_parser("doctor", help="проверить библиотеки, устройство и веса")
    doctor.set_defaults(func=command_doctor)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Точка входа."""
    args = build_parser().parse_args(argv)
    setup_logging(level=logging.INFO if args.verbose else logging.WARNING)
    try:
        result: int = args.func(args)
    except MirError as exc:
        print(f"ошибка: {exc.user_message}")
        print(f"  причина: {exc}")
        return 1
    return result


if __name__ == "__main__":
    raise SystemExit(main())
