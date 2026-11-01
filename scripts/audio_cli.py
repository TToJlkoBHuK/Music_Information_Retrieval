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
python scripts/audio_cli.py analyze "https://youtu.be/..." --clicks beats.wav
python scripts/audio_cli.py analyze video.mp4 --clicks beats.wav
python scripts/audio_cli.py analyze video.mp4 --no-transcribe --seconds 30
python scripts/audio_cli.py analyze audio.wav --csv notes_audio.csv --truth notes.csv
python scripts/audio_cli.py sync video.mp4 --video-csv notes_video.csv --audio-csv notes_audio.csv
python scripts/audio_cli.py compare --video-csv v.csv --audio-csv a.csv --roll roll.png --end 40
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
from mir.common.errors import MirError, UnsupportedSourceError
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


def resolve_media(source: str, config: MirConfig, work_dir: Path) -> tuple[Path | None, Path]:
    """Ссылка, видео или звуковой файл → (видео, звук).

    Ссылка проходит этап загрузки целиком: ролик скачивается в кэш,
    поэтому второй запуск с той же ссылкой не качает заново — только
    извлекает звук, это секунды.

    Returns:
        Путь к видео (`None` для звукового файла) и путь к WAV или
        исходному звуковому файлу.

    Raises:
        MirError: Ссылка не распознана, скачать не удалось, файла нет.
    """
    from mir.ingest import ingest
    from mir.ingest.demuxer import Demuxer
    from mir.ingest.sources import is_url

    if is_url(source):
        progress = ProgressLine()
        bundle = ingest(source, config, work_dir=work_dir, progress=progress)
        progress.finish()
        print(f"ролик           : {bundle.title or '—'}, {bundle.duration:.0f} с")
        print(f"видео в кэше    : {bundle.video_path}")
        return bundle.video_path, bundle.audio_path

    path = Path(source)
    if not path.exists():
        raise UnsupportedSourceError(f"{path} не найден", f"Файл не найден: {path}")
    if path.suffix.lower() not in VIDEO_SUFFIXES:
        return None, path
    out = work_dir / f"{path.stem}.wav"
    print(f"извлечение звука из {path.name}…")
    return path, Demuxer(config.ingest).extract_audio(path, out)


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
        self.stage: Stage | None = None

    def __call__(self, stage: Stage, fraction: float) -> None:
        """Обновить строку: этап и доля выполнения 0..1."""
        if stage is not self.stage:
            # Новый этап — новая строка и свой отсчёт времени: иначе оценка
            # остатка для распаковки считалась бы по скорости загрузки.
            self.finish()
            self.stage = stage
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
    """Разобрать звук ролика по ссылке, видеофайла или звукового файла."""
    config = _config(args)

    if not args.no_transcribe and (problem := backend_status()) is not None:
        print(problem)
        print()
        print("Без нейросети можно посчитать атаки и темп: добавьте --no-transcribe")
        return 1

    with tempfile.TemporaryDirectory() as tmp:
        _, audio_path = resolve_media(args.file, config, Path(tmp))
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
    config = _config(args)
    with tempfile.TemporaryDirectory() as tmp:
        return _sync(args, config, Path(tmp))


def _sync(args: argparse.Namespace, config: MirConfig, work_dir: Path) -> int:
    video, audio_path = resolve_media(args.video, config, work_dir)
    if video is None:
        print("для оценки рассинхрона нужно видео или ссылка, а не звуковой файл")
        return 1

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

    # Готовые ноты звука из `analyze --csv` избавляют от второго прогона
    # нейросети: на полном ролике это минуты. Атаки всё равно считаются
    # заново — это секунды.
    audio_notes: list[NoteEvent] = []
    if args.audio_csv:
        audio_notes = read_notes_csv(Path(args.audio_csv))
        print(f"ноты звука      : {len(audio_notes)} из {args.audio_csv}")
    transcribe = not args.audio_csv and not args.no_transcribe and backend_status() is None
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
    if transcribe:
        audio_notes = audio.notes
    if args.start or args.seconds:
        end = args.start + args.seconds if args.seconds else float("inf")
        audio_notes = [n for n in audio_notes if args.start <= n.onset < end]

    fusion = config.fusion
    threshold = fusion.min_offset_confidence
    estimates: list[OffsetEstimate] = []
    if audio_notes:
        estimates.append(estimate_av_offset(video_notes, audio_notes, fusion))
    estimates.append(estimate_av_offset_from_onsets(video_notes, audio.onsets, fusion))
    print()
    for estimate in estimates:
        _print_offset(estimate, threshold)
    if not audio_notes:
        print("  (без транскрипции: оценка по атакам слабее на ровном ритме)")

    best = estimates[0]
    if args.aligned_csv and not audio_notes:
        print("ноты звука не сохранены: без транскрипции их нет")
    if args.aligned_csv and audio_notes:
        shift = -best.offset if best.is_reliable(threshold) else 0.0
        write_notes_csv(apply_offset(audio_notes, shift), Path(args.aligned_csv))
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


def draw_two_rolls(
    video: list[NoteEvent], audio: list[NoteEvent], start: float, end: float, out: Path
) -> None:
    """Два валика на одной картинке: звук зелёным, видео красным.

    Совпавшие ноты ложатся друг на друга. Если красные полосы стабильно
    смещены по высоте относительно зелёных — видео ошибается в нотах;
    если по времени — это рассинхрон.
    """
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
    for notes, colour, lift in ((audio, (40, 160, 40), -2), (video, (40, 40, 220), 2)):
        for note in notes:
            if note.offset < start or note.onset > end:
                continue
            top = y(note.pitch) + lift
            cv2.rectangle(
                canvas,
                (x(max(note.onset, start)), top - 2),
                (x(min(note.offset, end)), top + 1),
                colour,
                -1,
            )
    cv2.putText(canvas, "green: audio   red: video", (left + 10, 18), 0, 0.5, (0, 0, 0), 1)
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), canvas)
    print(f"валики          : {out}")


def command_compare(args: argparse.Namespace) -> int:
    """Сверить ноты видео и звука без эталона."""
    from mir.eval import compare_channels

    video_all = read_notes_csv(Path(args.video_csv))
    audio_all = read_notes_csv(Path(args.audio_csv))
    if not video_all or not audio_all:
        print("сравнивать нечего: один из файлов пуст")
        return 1
    # Сравнивается только общий отрезок: видео часто разобрано с --seconds,
    # а звук целиком, и лишние ноты одного канала занизили бы совпадение.
    start = max(args.start, min(n.onset for n in video_all), min(n.onset for n in audio_all))
    end = min(max(n.offset for n in video_all), max(n.offset for n in audio_all))
    if args.end is not None:
        end = min(end, args.end)
    video = [n for n in video_all if start <= n.onset < end]
    audio = [n for n in audio_all if start <= n.onset < end]
    print(f"окно сравнения  : {start:.1f}–{end:.1f} с (общий отрезок обоих каналов)")
    print(f"нот             : видео {len(video)}, звук {len(audio)}")
    if not video or not audio:
        print("в общем отрезке нот нет")
        return 1

    result = compare_channels(video, audio)
    if abs(result.time_shift) >= 0.95:
        where = "сдвиг по времени не найден — упёрся в предел поиска ±1 с"
    else:
        where = f"сдвиг звука относительно видео {result.time_shift * 1000:+.0f} мс"
    print(f"совпадение      : {result.similarity:.2f} ({where})")
    top = sorted(result.pitch_shifts.items(), key=lambda item: -item[1])[:3]
    print("лучшие сдвиги высоты видео: " + ", ".join(f"{k:+d} → {v:.2f}" for k, v in top))
    if result.pitch_consistent:
        print("  высоты каналов согласованы")
    else:
        print(
            f"  совпадение лучше при сдвиге {result.best_pitch_shift:+d} полутонов: "
            "один из каналов ошибается в высоте нот"
        )
    print("  (ориентир: 1.0 — валики совпадают полностью; у верных каналов")
    print("   длительности расходятся из-за педали, поэтому заметно меньше 1)")

    # По началам нот: педаль на них не влияет, поэтому оценка строже валиков.
    # Звук здесь вместо эталона — его ошибки тоже войдут в счёт.
    fusion = load_config().fusion
    estimate = estimate_av_offset(video, audio, fusion)
    shifted = [
        replace(n, onset=n.onset + estimate.offset, offset=n.offset + estimate.offset)
        for n in video
    ]
    score = evaluate(audio, shifted)
    print(
        f"по началам нот  : совпало {score.matched} (видео {len(video)}, звук {len(audio)}), "
        f"P {score.precision:.3f} R {score.recall:.3f} F1 {score.f1:.3f}"
    )
    print(
        f"  после сдвига видео на {estimate.offset_ms:+.0f} мс "
        f"(уверенность {estimate.confidence:.2f}); "
        f"средняя ошибка начала {score.onset_error_avg * 1000:.1f} мс"
    )
    if estimate.confidence < fusion.min_offset_confidence:
        # Так бывает, когда файлы от разных роликов: окно и числа при этом
        # печатаются как обычно, и ошибку легко не заметить.
        print(
            "  ВНИМАНИЕ: сдвиг не найден уверенно, каналы почти не совпадают.\n"
            "  Проверьте, что оба файла от одного ролика; если да — один из\n"
            "  каналов ошибается (так выглядел разбор с подсветкой клавиш при руках, В-05)."
        )

    if args.roll:
        draw_two_rolls(video, audio, start, end, Path(args.roll))
    return 0


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
    analyze.add_argument("file", help="ссылка на ролик, видеофайл или WAV/FLAC/MP3")
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
    sync.add_argument("video", help="ссылка на ролик или видеофайл")
    sync.add_argument("--video-csv", help="ноты видео из vision_cli analyze --csv")
    sync.add_argument("--audio-csv", help="ноты звука из audio_cli analyze --csv")
    sync.add_argument("--no-transcribe", action="store_true", help="оценка только по атакам")
    sync.add_argument("--start", type=float, default=0.0)
    sync.add_argument("--seconds", type=float)
    sync.add_argument("--device", choices=["auto", "cpu", "cuda"])
    sync.add_argument("--aligned-csv", help="сохранить ноты звука, перенесённые на шкалу видео")
    sync.set_defaults(func=command_sync)

    compare = sub.add_parser("compare", help="сверить ноты видео и звука без эталона")
    compare.add_argument("--video-csv", required=True, help="ноты из run analyze --csv")
    compare.add_argument("--audio-csv", required=True, help="ноты из run audio analyze --csv")
    compare.add_argument("--roll", help="картинка: оба валика на одной оси")
    compare.add_argument("--start", type=float, default=0.0, help="начало окна сравнения, с")
    compare.add_argument("--end", type=float, help="конец окна сравнения, с")
    compare.set_defaults(func=command_compare)

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
