"""Синтез звука с заранее известным ответом.

Тот же приём, что и в `synth.py` для видео: звук рендерится из списка нот,
поэтому эталон точен и бесплатен. Ручная разметка аудио стоит ещё дороже
видео — на слух моменты атак с точностью 10 мс не отметить.

Это не имитация настоящего рояля. Тон собран из гармоник с затуханием
и коротким ударом молоточка — ровно столько, чтобы у звука была резкая
атака, спадающая огибающая и высота. Для проверки детектора атак, темпа
и рассинхрона этого достаточно; для оценки нейросетевой транскрипции —
нет, там нужен живой рояль (см. `mir/audio/README.md`).
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import numpy.typing as npt

from mir.audio.signal import save_wav
from mir.common.types import NoteEvent

__all__ = [
    "SAMPLE_RATE",
    "metronome",
    "render_notes",
    "scale_melody",
    "tempo_ramp_clicks",
    "write_wav",
]

SAMPLE_RATE = 16000
"""Частота дискретизации по умолчанию — та же, что готовит `ingest`."""

_RELEASE_SEC = 0.06
_ATTACK_SEC = 0.003
_HAMMER_SEC = 0.006


def _note_frequency(pitch: int) -> float:
    return float(440.0 * 2.0 ** ((pitch - 69) / 12))


def _render_tone(
    pitch: int, duration: float, velocity: int, sr: int, rng: np.random.Generator
) -> npt.NDArray[np.float64]:
    """Один звук: гармоники с затуханием и удар молоточка."""
    length = int((duration + _RELEASE_SEC) * sr)
    t = np.arange(length) / sr
    f0 = _note_frequency(pitch)
    loudness = (velocity / 127.0) ** 2

    # Низкие ноты звучат дольше высоких — как у настоящих струн.
    tau = 1.6 * (f0 / 261.6) ** -0.6
    brightness = 2.2 - 0.9 * velocity / 127.0

    tone = np.zeros(length)
    for k in range(1, 13):
        fk = k * f0 * np.sqrt(1.0 + 1e-4 * k * k)
        if fk >= 0.45 * sr:
            break
        decay = np.exp(-t * (1.0 + 0.35 * (k - 1)) / tau)
        tone += np.sin(2 * np.pi * fk * t + rng.uniform(0, 2 * np.pi)) * decay / k**brightness

    envelope = np.minimum(t / _ATTACK_SEC, 1.0)
    release_from = int(duration * sr)
    tail = np.arange(length - release_from) / sr
    envelope[release_from:] *= np.exp(-tail / (_RELEASE_SEC / 4))
    tone *= envelope

    hammer = int(_HAMMER_SEC * sr)
    tone[:hammer] += rng.normal(0, 0.25, hammer) * np.linspace(1, 0, hammer)

    peak = float(np.max(np.abs(tone))) or 1.0
    result: npt.NDArray[np.float64] = tone / peak * loudness
    return result


def render_notes(
    notes: Sequence[NoteEvent],
    sr: int = SAMPLE_RATE,
    duration: float | None = None,
    noise_db: float | None = None,
    seed: int = 0,
) -> npt.NDArray[np.float32]:
    """Отрендерить ноты в моно-сигнал.

    Args:
        notes: Ноты эталона.
        sr: Частота дискретизации.
        duration: Длительность результата. По умолчанию — до конца последней
            ноты с запасом на затухание.
        noise_db: Белый шум относительно пика сигнала, дБ. `None` — без шума.
        seed: Зерно генератора: фазы гармоник и шум воспроизводимы.

    Returns:
        Сигнал `float32` в диапазоне примерно -1..1.
    """
    rng = np.random.default_rng(seed)
    if duration is None:
        duration = max((n.offset for n in notes), default=1.0) + 0.5
    out: npt.NDArray[np.float64] = np.zeros(int(duration * sr))

    for note in notes:
        start = round(note.onset * sr)
        if start >= len(out):
            continue
        tone = _render_tone(note.pitch, note.duration, note.velocity, sr, rng)
        end = min(start + len(tone), len(out))
        out[start:end] += tone[: end - start]

    peak = float(np.max(np.abs(out))) or 1.0
    out = out / peak * 0.8
    if noise_db is not None:
        out = out + rng.normal(0, 0.8 * 10 ** (noise_db / 20), len(out))
    return out.astype(np.float32)


def _click(sr: int, accent: bool) -> npt.NDArray[np.float64]:
    length = int(0.03 * sr)
    t = np.arange(length) / sr
    freq = 1500.0 if accent else 1000.0
    click: npt.NDArray[np.float64] = np.sin(2 * np.pi * freq * t) * np.exp(-t / 0.006)
    return click * (1.0 if accent else 0.6)


def metronome(
    bpm: float,
    seconds: float,
    sr: int = SAMPLE_RATE,
    beats_per_bar: int = 4,
    start: float = 0.5,
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float64]]:
    """Метроном с акцентом на сильную долю.

    Returns:
        Сигнал и точные моменты щелчков в секундах.
    """
    period = 60.0 / bpm
    times = np.arange(start, seconds - 0.05, period, dtype=np.float64)
    return _clicks_at(times, seconds, sr, beats_per_bar), times


def tempo_ramp_clicks(
    bpm_from: float,
    bpm_to: float,
    seconds: float,
    sr: int = SAMPLE_RATE,
    start: float = 0.5,
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float64]]:
    """Щелчки с плавно меняющимся темпом — для проверки переменного темпа."""
    times = [start]
    while True:
        progress = times[-1] / seconds
        bpm = bpm_from + (bpm_to - bpm_from) * progress
        nxt = times[-1] + 60.0 / bpm
        if nxt >= seconds - 0.05:
            break
        times.append(nxt)
    clicks = np.asarray(times)
    return _clicks_at(clicks, seconds, sr, 4), clicks


def _clicks_at(
    times: npt.NDArray[np.float64], seconds: float, sr: int, beats_per_bar: int
) -> npt.NDArray[np.float32]:
    out: npt.NDArray[np.float64] = np.zeros(int(seconds * sr))
    for index, time in enumerate(times):
        click = _click(sr, accent=index % beats_per_bar == 0)
        start = int(np.rint(time * sr))
        end = min(start + len(click), len(out))
        out[start:end] += click[: end - start]
    return out.astype(np.float32)


def scale_melody(
    bpm: float = 100.0,
    bars: int = 8,
    start: float = 0.5,
    subdivision: int = 2,
) -> list[NoteEvent]:
    """Мелодия по гамме с аккомпанементом по долям — проверка темпа и атак.

    Левая рука отмечает каждую долю, сильная доля такта громче. Без этого
    доля в звуке ничем не выделена: у ровных шестнадцатых любой кратный
    им период одинаково правдоподобен, и тест проверял бы не трекер,
    а удачу. В живой фортепианной музыке доли отмечены — басом, сменой
    гармонии, акцентом.

    Args:
        bpm: Темп.
        bars: Сколько тактов 4/4.
        start: Начало первой ноты.
        subdivision: Нот мелодии на долю: 2 — восьмые, 4 — шестнадцатые.

    Returns:
        Ноты, отсортированные по времени начала.
    """
    beat = 60.0 / bpm
    step = beat / subdivision
    scale = [60, 62, 64, 65, 67, 69, 71, 72, 71, 69, 67, 65, 64, 62]
    bass = [36, 43, 40, 43]
    notes: list[NoteEvent] = []
    for index in range(bars * 4 * subdivision):
        onset = start + index * step
        on_beat = index % subdivision == 0
        notes.append(
            NoteEvent(scale[index % len(scale)], onset, onset + step * 0.9, 80 if on_beat else 58)
        )
    for index in range(bars * 4):
        onset = start + index * beat
        velocity = 104 if index % 4 == 0 else 78
        notes.append(NoteEvent(bass[index % 4], onset, onset + beat * 0.95, velocity))
    return sorted(notes, key=lambda n: (n.onset, n.pitch))


def write_wav(path: Path, samples: npt.NDArray[np.floating], sr: int = SAMPLE_RATE) -> Path:
    """Записать моно-сигнал в 16-битный WAV — тот же формат, что даёт `ingest`."""
    return save_wav(path, samples, sr)
