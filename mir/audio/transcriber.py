"""Нейросетевая транскрипция звука (#26, F-09, F-13).

Сама нейросеть — готовая модель ByteDance (Kong et al., 2021): на входе
звук 16 кГц, на выходе покадровые вероятности начала, конца и звучания
каждой из 88 клавиш, плюс педаль. Здесь — всё, что вокруг неё:

* потоковый проход по 10-секундным сегментам с перекрытием в половину,
  ровно как обучена модель, но с прогрессом и отменой после каждого
  сегмента. Исходная библиотека держит все сегменты ролика в памяти
  сразу, печатает ход работы в консоль и не даёт себя прервать;
* перевод ответа в [`NoteEvent`][mir.common.types.NoteEvent] с проверкой
  диапазонов: у исходной библиотеки громкость может выйти 128 при
  допустимых 0..127;
* уверенность ноты — вероятность начала из самой модели.

Модель отделена протоколом [`SegmentModel`][mir.audio.transcriber.SegmentModel]:
логика сегментации проверяется тестами без PyTorch и без весов, на
подставной модели с заранее известным ответом.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
import numpy.typing as npt

from mir.audio.signal import AudioSignal
from mir.common.enums import Hand, Source
from mir.common.errors import CancelledError
from mir.common.logging import get_logger
from mir.common.types import PITCH_MAX, PITCH_MIN, CancelCheck, NoteEvent, PedalEvent
from mir.config import AudioConfig

__all__ = [
    "AudioTranscriber",
    "DecodeThresholds",
    "SegmentModel",
    "SegmentPlan",
    "TranscriptionResult",
]

_log = get_logger(__name__)

Outputs = dict[str, npt.NDArray[np.float32]]
"""Покадровые выходы модели: имя → массив (кадры, классы)."""

_VELOCITY_SCALE = 128
"""Модель предсказывает громкость, делённую на 128, — так она обучена."""


@dataclass(frozen=True)
class DecodeThresholds:
    """Пороги перевода вероятностей в события. Значения — из `[audio]`."""

    onset: float
    offset: float
    frame: float
    pedal_offset: float


class SegmentModel(Protocol):
    """Нейросеть, размечающая звук покадрово по сегментам фиксированной длины."""

    sample_rate: int
    """Частота дискретизации, на которой обучена модель."""

    frames_per_second: int
    """Кадров разметки в секунду."""

    segment_samples: int
    """Длина сегмента в отсчётах."""

    device: str
    """Устройство вычислений — для отчёта."""

    def forward(self, segments: npt.NDArray[np.float32]) -> Outputs:
        """Разметить пачку сегментов.

        Args:
            segments: Массив (пачка, `segment_samples`).

        Returns:
            Выходы формы (пачка, кадры сегмента + 1, классы). Лишний кадр —
            следствие центрированного спектрального анализа, он отбрасывается.
        """
        ...

    def decode(
        self, outputs: Outputs, thresholds: DecodeThresholds
    ) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
        """Перевести покадровые выходы всего ролика в события.

        Returns:
            Ноты (n, 4): начало, конец в секундах, MIDI-номер, громкость 0..1;
            педаль (m, 2): начало и конец в секундах.
        """
        ...


@dataclass(frozen=True)
class SegmentPlan:
    """Раскладка сигнала на сегменты с перекрытием в половину.

    Сигнал дополняется нулями до целого числа сегментов `k`, затем
    нарезается с шагом в полсегмента — выходит `2k - 1` сегментов. Из
    каждого берётся средняя половина кадров, из крайних — ещё и внешняя
    четверть. Края сегмента модель видит без контекста и размечает хуже
    середины, поэтому ответ берётся там, где у неё есть контекст
    с обеих сторон.

    Схема та же, что у авторов модели, — иначе качество разметки
    разошлось бы с опубликованным.

    Attributes:
        segment_samples: Длина сегмента в отсчётах.
        segment_frames: Кадров разметки на сегмент (без лишнего).
        padded_samples: Длина сигнала после дополнения нулями.
    """

    segment_samples: int
    segment_frames: int
    padded_samples: int

    @classmethod
    def build(cls, n_samples: int, segment_samples: int, segment_frames: int) -> SegmentPlan:
        """Составить раскладку для сигнала длиной `n_samples`."""
        if segment_frames % 4:
            raise ValueError(f"кадров в сегменте ({segment_frames}) должно быть кратно 4")
        full = max(int(np.ceil(n_samples / segment_samples)), 1)
        return cls(segment_samples, segment_frames, full * segment_samples)

    @property
    def count(self) -> int:
        """Число сегментов."""
        return 2 * (self.padded_samples // self.segment_samples) - 1

    @property
    def total_frames(self) -> int:
        """Кадров разметки на весь дополненный сигнал."""
        return self.padded_samples // self.segment_samples * self.segment_frames

    def sample_range(self, index: int) -> tuple[int, int]:
        """Отсчёты сигнала, попадающие в сегмент `index`."""
        start = index * self.segment_samples // 2
        return start, start + self.segment_samples

    def kept_frames(self, index: int) -> tuple[int, int, int]:
        """Какие кадры сегмента идут в результат.

        Returns:
            Первый и последний (исключая) кадр внутри сегмента и позиция
            первого из них в общей разметке.
        """
        frames = self.segment_frames
        quarter = frames // 4
        first = 0 if index == 0 else quarter
        last = frames if index == self.count - 1 else frames - quarter
        return first, last, index * frames // 2 + first


@dataclass
class TranscriptionResult:
    """Ноты и педаль, распознанные по звуку.

    Attributes:
        notes: События с `source=AUDIO`, рука не определена.
        pedals: Нажатия правой педали.
        device: На чём считалось.
        segments: Сколько сегментов обработано.
        elapsed_sec: Время работы — для таблиц производительности (NF-01, NF-02).
        warnings: Предупреждения для пользователя.
    """

    notes: list[NoteEvent]
    pedals: list[PedalEvent]
    device: str
    segments: int
    elapsed_sec: float
    warnings: list[str] = field(default_factory=list)


class AudioTranscriber:
    """Прогон модели по всему сигналу и перевод ответа в события.

    Args:
        model: Нейросеть — обычно
            [`ByteDanceModel`][mir.audio.bytedance.ByteDanceModel].
        config: Параметры из секции `[audio]`.
    """

    def __init__(self, model: SegmentModel, config: AudioConfig) -> None:
        self.model = model
        self.config = config

    @property
    def thresholds(self) -> DecodeThresholds:
        """Пороги постобработки из настроек."""
        return DecodeThresholds(
            onset=self.config.onset_threshold,
            offset=self.config.offset_threshold,
            frame=self.config.frame_threshold,
            pedal_offset=self.config.pedal_offset_threshold,
        )

    def plan(self, n_samples: int) -> SegmentPlan:
        """Раскладка сигнала длиной `n_samples` на сегменты модели."""
        model = self.model
        segment_frames = model.segment_samples * model.frames_per_second // model.sample_rate
        return SegmentPlan.build(n_samples, model.segment_samples, segment_frames)

    def infer(
        self,
        samples: npt.NDArray[np.float32],
        progress: Callable[[float], None] | None = None,
        cancel: CancelCheck | None = None,
    ) -> Outputs:
        """Покадровая разметка всего сигнала.

        Память занимает только результат — около 25 МБ на минуту звука,
        а не все сегменты ролика сразу.

        Raises:
            CancelledError: Пользователь отменил обработку.
        """
        model = self.model
        plan = self.plan(len(samples))
        padded = np.zeros(plan.padded_samples, dtype=np.float32)
        padded[: len(samples)] = samples

        outputs: Outputs = {}
        batch = max(self.config.batch_size, 1)
        for first in range(0, plan.count, batch):
            if cancel is not None and cancel():
                raise CancelledError("распознавание звука отменено")
            indices = range(first, min(first + batch, plan.count))
            segments = np.stack([padded[slice(*plan.sample_range(i))] for i in indices])
            result = model.forward(segments)

            for name, values in result.items():
                if name not in outputs:
                    outputs[name] = np.zeros(
                        (plan.total_frames, values.shape[-1]), dtype=np.float32
                    )
                for offset, index in enumerate(indices):
                    lo, hi, at = plan.kept_frames(index)
                    outputs[name][at : at + hi - lo] = values[offset, lo:hi]

            if progress is not None:
                progress(min(first + batch, plan.count) / plan.count)

        frames = int(np.ceil(len(samples) / model.sample_rate * model.frames_per_second)) + 1
        return {name: values[: min(frames, plan.total_frames)] for name, values in outputs.items()}

    def transcribe(
        self,
        signal: AudioSignal,
        progress: Callable[[float], None] | None = None,
        cancel: CancelCheck | None = None,
    ) -> TranscriptionResult:
        """Распознать ноты и педаль.

        Args:
            signal: Звук; при другой частоте дискретизации пересчитывается.
            progress: Доля выполнения 0..1.
            cancel: Опрос отмены.

        Raises:
            CancelledError: Пользователь отменил обработку.
        """
        started = time.perf_counter()
        prepared = signal.resampled(self.model.sample_rate)
        outputs = self.infer(prepared.samples, progress, cancel)
        note_array, pedal_array = self.model.decode(dict(outputs), self.thresholds)

        notes = self._to_notes(note_array, outputs, prepared.start)
        pedals = self._to_pedals(pedal_array, prepared.start)
        elapsed = time.perf_counter() - started
        _log.info(
            "звук: %d нот, %d нажатий педали за %.1f с (%s, %.1f× реального времени)",
            len(notes),
            len(pedals),
            elapsed,
            self.model.device,
            elapsed / max(signal.duration, 1e-9),
        )
        return TranscriptionResult(
            notes=notes,
            pedals=pedals,
            device=self.model.device,
            segments=self.plan(len(prepared.samples)).count,
            elapsed_sec=elapsed,
        )

    def _to_notes(
        self, array: npt.NDArray[np.float32], outputs: Outputs, start: float
    ) -> list[NoteEvent]:
        """Ноты модели → события конвейера.

        Две поправки к ответу исходной библиотеки. Громкость там считается
        как `int(v * 128)` и при `v = 1` выходит 128 — вне шкалы MIDI;
        здесь она ограничивается диапазоном 1..127 (ноль в MIDI означает
        отпускание клавиши, а не тихую ноту).
        И нота может получить конец, совпадающий с началом, если модель
        сочла её длиной меньше кадра; такой ноте отводится один кадр.
        """
        fps = self.model.frames_per_second
        min_duration = 1.0 / fps
        onset_prob = outputs.get("reg_onset_output")
        notes: list[NoteEvent] = []
        for onset, offset, pitch, velocity in np.asarray(array, dtype=np.float64).reshape(-1, 4):
            midi = int(np.rint(pitch))
            if not PITCH_MIN <= midi <= PITCH_MAX:
                continue
            confidence = 1.0
            if onset_prob is not None and len(onset_prob):
                frame = int(np.clip(round(onset * fps), 0, len(onset_prob) - 1))
                confidence = float(np.clip(onset_prob[frame, midi - PITCH_MIN], 0.0, 1.0))
            notes.append(
                NoteEvent(
                    pitch=midi,
                    onset=start + float(onset),
                    offset=start + max(float(offset), float(onset) + min_duration),
                    velocity=int(np.clip(int(velocity * _VELOCITY_SCALE), 1, 127)),
                    hand=Hand.UNKNOWN,
                    source=Source.AUDIO,
                    confidence=confidence,
                )
            )
        notes.sort(key=lambda n: (n.onset, n.pitch))
        return notes

    def _to_pedals(self, array: npt.NDArray[np.float32], start: float) -> list[PedalEvent]:
        min_duration = 1.0 / self.model.frames_per_second
        pedals = [
            PedalEvent(start + float(on), start + max(float(off), float(on) + min_duration))
            for on, off in np.asarray(array, dtype=np.float64).reshape(-1, 2)
        ]
        pedals.sort(key=lambda p: p.onset)
        return pedals
