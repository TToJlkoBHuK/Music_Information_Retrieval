"""Звуковой сигнал в памяти и его чтение с диска.

Время отсчитывается от начала ролика, даже если загружен только отрезок:
поле `start` хранит, с какой секунды отрезок взят. Иначе ноты из отрезка
пришлось бы сдвигать вручную на каждом этапе, и рано или поздно где-то
сдвиг забыли бы.
"""

from __future__ import annotations

import wave
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt

from mir.common.errors import TranscriptionError
from mir.common.logging import get_logger

__all__ = ["AudioSignal", "load_audio", "save_wav"]

_log = get_logger(__name__)

_PCM_SCALE = {1: 128.0, 2: 32768.0, 4: 2147483648.0}
"""Делитель для перевода целых PCM-отсчётов в диапазон -1..1."""


@dataclass(frozen=True, eq=False)
class AudioSignal:
    """Моно-сигнал с привязкой ко времени ролика.

    Attributes:
        samples: Отсчёты `float32` в диапазоне примерно -1..1.
        sample_rate: Частота дискретизации, Гц.
        start: Момент ролика, которому соответствует первый отсчёт, секунды.
    """

    samples: npt.NDArray[np.float32]
    sample_rate: int
    start: float = 0.0

    def __post_init__(self) -> None:
        if self.sample_rate <= 0:
            raise ValueError(f"sample_rate={self.sample_rate} должен быть положительным")
        if self.samples.ndim != 1:
            raise ValueError("ожидается моно-сигнал: одномерный массив отсчётов")

    @property
    def duration(self) -> float:
        """Длительность сигнала, секунды."""
        return len(self.samples) / self.sample_rate

    @property
    def end(self) -> float:
        """Момент ролика, на котором сигнал заканчивается."""
        return self.start + self.duration

    def resampled(self, sample_rate: int) -> AudioSignal:
        """Пересчитать на другую частоту дискретизации.

        Модель транскрипции обучена на 16 кГц. `ingest` готовит именно
        такой файл, но частоту можно поменять в настройках — тогда
        пересчёт делается здесь, а не падением с невнятной ошибкой.
        """
        if sample_rate == self.sample_rate:
            return self
        import librosa

        _log.info("пересчёт звука %d → %d Гц", self.sample_rate, sample_rate)
        converted = librosa.resample(
            self.samples, orig_sr=self.sample_rate, target_sr=sample_rate, res_type="soxr_hq"
        )
        return AudioSignal(converted.astype(np.float32), sample_rate, self.start)


def _read_wav(
    path: Path, start: float, max_seconds: float | None
) -> tuple[npt.NDArray[np.float32], int]:
    with wave.open(str(path), "rb") as fh:
        sample_rate = fh.getframerate()
        channels = fh.getnchannels()
        width = fh.getsampwidth()
        first = int(start * sample_rate)
        fh.setpos(min(first, fh.getnframes()))
        count = fh.getnframes() - fh.tell()
        if max_seconds is not None:
            count = min(count, int(max_seconds * sample_rate))
        raw = fh.readframes(count)

    if width == 3:
        # 24 бита numpy напрямую не читает: дополняем каждый отсчёт до 32.
        bytes_ = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3)
        padded = np.zeros((len(bytes_), 4), dtype=np.uint8)
        padded[:, 1:] = bytes_
        data = padded.view("<i4").reshape(-1).astype(np.float32) / _PCM_SCALE[4]
    elif width == 1:
        data = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / _PCM_SCALE[1]
    elif width in (2, 4):
        dtype = "<i2" if width == 2 else "<i4"
        data = np.frombuffer(raw, dtype=dtype).astype(np.float32) / _PCM_SCALE[width]
    else:  # pragma: no cover - экзотический формат
        raise TranscriptionError(f"{path.name}: неподдерживаемая разрядность {8 * width} бит")

    if channels > 1:
        data = data.reshape(-1, channels).mean(axis=1)
    return data.astype(np.float32), sample_rate


def _read_other(
    path: Path, start: float, max_seconds: float | None
) -> tuple[npt.NDArray[np.float32], int]:
    import soundfile

    info = soundfile.info(str(path))
    first = int(start * info.samplerate)
    stop = None if max_seconds is None else first + int(max_seconds * info.samplerate)
    data, sample_rate = soundfile.read(
        str(path), start=first, stop=stop, dtype="float32", always_2d=True
    )
    mono: npt.NDArray[np.float32] = data.mean(axis=1).astype(np.float32)
    return mono, int(sample_rate)


def save_wav(path: Path, samples: npt.NDArray[np.floating], sample_rate: int) -> Path:
    """Записать моно-сигнал в 16-битный WAV.

    Значения за пределами -1..1 обрезаются, а не заворачиваются: иначе
    громкий щелчок поверх музыки превратился бы в треск.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    pcm = (np.clip(samples, -1.0, 1.0) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as fh:
        fh.setnchannels(1)
        fh.setsampwidth(2)
        fh.setframerate(sample_rate)
        fh.writeframes(pcm.tobytes())
    return path


def load_audio(path: Path, start: float = 0.0, max_seconds: float | None = None) -> AudioSignal:
    """Прочитать звук с диска.

    WAV читается стандартной библиотекой: это формат, который готовит
    `ingest`, и для него не нужны внешние зависимости. Прочие форматы
    (FLAC, OGG, MP3) — через `soundfile`.

    Args:
        path: Звуковой файл.
        start: С какой секунды читать.
        max_seconds: Сколько секунд прочитать. `None` — до конца.

    Raises:
        TranscriptionError: Файл не найден, пуст или не читается.
    """
    if not path.exists():
        raise TranscriptionError(f"{path} не найден", "Звуковой файл не найден")
    if start < 0:
        raise ValueError(f"start={start} не может быть отрицательным")

    try:
        if path.suffix.lower() == ".wav":
            samples, sample_rate = _read_wav(path, start, max_seconds)
        else:
            samples, sample_rate = _read_other(path, start, max_seconds)
    except (wave.Error, EOFError, RuntimeError) as exc:
        raise TranscriptionError(
            f"не удалось прочитать {path}: {exc}",
            "Звуковой файл повреждён или записан в неподдерживаемом формате",
        ) from exc

    if len(samples) == 0:
        raise TranscriptionError(
            f"{path}: нет отсчётов начиная с {start:.1f} с",
            "Звуковая дорожка пуста",
        )
    return AudioSignal(samples, sample_rate, start)
