"""Отслеживание подсветки клавиш (F-08).

Клавиша меняет цвет на время звучания ноты. Сравнивая каждый кадр
с «пустой» клавиатурой, получаем моменты нажатия и отпускания.

Три решения определяют качество результата:

* **гистерезис** — порог включения выше порога выключения. При едином
  пороге шум сжатия на границе даёт дребезг, и одна нота рассыпается
  на десяток коротких;
* **работа в HSV** — канал тона устойчив к изменению яркости, поэтому
  свечение и блики не считаются нажатием;
* **пакетный расчёт отклонений** — области всех клавиш собраны в один
  массив и уходят в ядро [`accel`][mir.vision.accel] одним вызовом.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
import numpy.typing as npt

from mir.common.enums import Hand, Source
from mir.common.logging import get_logger
from mir.common.types import Frame, KeyboardLayout, NoteEvent
from mir.config import TrackerConfig
from mir.vision import accel

__all__ = ["KeyTracker", "sample_key_color"]

_log = get_logger(__name__)

_BLACK_KEY_SAMPLE_DEPTH = 0.4
"""Доля высоты клавиатуры, на которой пробуется чёрная клавиша.

Ниже этой глубины чёрной клавиши уже нет — там белая соседка.
"""


def sample_key_color(
    hsv: Frame,
    x_min: int,
    x_max: int,
    y_min: int,
    y_max: int,
) -> npt.NDArray[np.float32]:
    """Средний цвет прямоугольной пробы внутри клавиши."""
    region = np.array([[x_min, x_max, y_min, y_max]], dtype=np.int32)
    colors: npt.NDArray[np.float32] = accel.sample_regions(hsv, region)[0]
    return colors


@dataclass
class _KeyState:
    """Состояние одной клавиши между кадрами."""

    pressed: bool = False
    frames_on: int = 0
    onset: float = 0.0
    candidate_onset: float = 0.0
    peak_deviation: float = 0.0
    hand: Hand = Hand.UNKNOWN


class KeyTracker:
    """Покадровое отслеживание нажатий.

    Args:
        layout: Разметка клавиатуры.
        reference_frame: «Пустой» кадр, обычно медианный.
        config: Пороги гистерезиса и антидребезга.
    """

    def __init__(
        self,
        layout: KeyboardLayout,
        reference_frame: Frame,
        config: TrackerConfig | None = None,
    ) -> None:
        self.layout = layout
        self.config = config or TrackerConfig()
        self._pitches: list[int] = [key.pitch for key in layout.keys]
        self._states: list[_KeyState] = [_KeyState() for _ in layout.keys]
        self._regions = self._build_regions()
        self._references = accel.sample_regions(
            cv2.cvtColor(reference_frame, cv2.COLOR_BGR2HSV), self._regions
        )

    def _build_regions(self) -> npt.NDArray[np.int32]:
        """Собрать области проб для всех клавиш в один массив.

        Чёрные клавиши пробуются выше белых: ниже своей длины чёрной
        клавиши уже нет, там видна соседняя белая.
        """
        _, top, _, height = self.layout.bbox
        black_bottom = top + int(height * _BLACK_KEY_SAMPLE_DEPTH)
        white_top = black_bottom + 1

        rows = []
        for key in self.layout.keys:
            if key.is_black:
                y_min, y_max = top + 2, black_bottom
            else:
                y_min, y_max = white_top, top + height - 2
            rows.append((key.x_min, key.x_max, y_min, max(y_max, y_min + 1)))
        return np.array(rows, dtype=np.int32)

    def process_frame(self, frame: Frame, timestamp: float) -> list[NoteEvent]:
        """Обработать кадр и вернуть завершившиеся на нём ноты.

        Args:
            frame: Кадр в BGR.
            timestamp: Время кадра в секундах от начала ролика.

        Returns:
            Ноты, отпущенные на этом кадре.
        """
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        deviations = accel.key_deviations(hsv, self._regions, self._references)
        finished: list[NoteEvent] = []

        for index, state in enumerate(self._states):
            deviation = float(deviations[index])

            if state.pressed:
                if deviation < self.config.off_threshold:
                    note = self._release(self._pitches[index], state, timestamp)
                    if note is not None:
                        finished.append(note)
                else:
                    state.peak_deviation = max(state.peak_deviation, deviation)
            elif deviation > self.config.on_threshold:
                if state.frames_on == 0:
                    # Время первого превышения порога, а не момента
                    # подтверждения: иначе антидребезг сдвинул бы все ноты
                    # на min_frames_on кадров вперёд — при 30 fps это 33 мс
                    # систематической ошибки при допуске MIR в 50 мс
                    state.candidate_onset = timestamp
                state.frames_on += 1
                if state.frames_on >= self.config.min_frames_on:
                    state.pressed = True
                    state.onset = state.candidate_onset
                    state.peak_deviation = deviation
            else:
                state.frames_on = 0

        return finished

    def flush(self, end_timestamp: float) -> list[NoteEvent]:
        """Закрыть ноты, звучащие в конце ролика."""
        finished: list[NoteEvent] = []
        for pitch, state in zip(self._pitches, self._states, strict=True):
            if state.pressed:
                note = self._release(pitch, state, end_timestamp)
                if note is not None:
                    finished.append(note)
        return finished

    @staticmethod
    def _deviation(current: npt.NDArray[np.float32], reference: npt.NDArray[np.float32]) -> float:
        """Насколько цвет отличается от эталонного, 0..1."""
        pair = accel.deviation_from_colors(
            current.reshape(1, 3).astype(np.float32),
            reference.reshape(1, 3).astype(np.float32),
        )
        return float(pair[0])

    def _release(self, pitch: int, state: _KeyState, timestamp: float) -> NoteEvent | None:
        """Сформировать событие ноты и сбросить состояние клавиши."""
        onset, peak = state.onset, state.peak_deviation
        state.pressed = False
        state.frames_on = 0
        state.peak_deviation = 0.0

        if timestamp <= onset:
            return None

        return NoteEvent(
            pitch=pitch,
            onset=onset,
            offset=timestamp,
            velocity=self._to_velocity(peak),
            hand=state.hand,
            source=Source.VIDEO,
            confidence=min(1.0, peak * 1.5),
        )

    def _to_velocity(self, deviation: float) -> int:
        """Перевести отклонение цвета в силу нажатия.

        Связь нелинейна: восприятие громкости логарифмическое, а яркость
        подсветки в визуализаторах обычно линейна по velocity.
        """
        scaled = deviation**self.config.velocity_gamma
        return int(np.clip(round(scaled * 127), 1, 127))
