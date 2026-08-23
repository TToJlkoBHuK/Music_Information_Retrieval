"""Отслеживание падающих блоков (F-08) и разделение рук по цвету.

Блоки несут больше сведений, чем подсветка клавиш:

* длина блока делённая на скорость падения сразу даёт длительность ноты —
  точнее, чем подсчёт кадров подсветки;
* нота видна за секунды до нажатия;
* цвет блока обычно кодирует руку.

Главная тонкость — момент касания клавиатуры. При 30 fps шаг между кадрами
33 мс, что грубо при допуске MIR в 50 мс. Поэтому момент вычисляется
линейной интерполяцией траектории между соседними кадрами, а не
приравнивается к времени кадра.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import cv2
import numpy as np

from mir.common.enums import Hand, Source
from mir.common.logging import get_logger
from mir.common.types import Frame, KeyboardLayout, NoteEvent
from mir.config import TrackerConfig
from mir.vision.calibration import VisualizerProfile

__all__ = ["Block", "BlockTracker"]

_log = get_logger(__name__)

_MIN_BLOCK_AREA = 20
"""Меньшие пятна — частицы, блики и артефакты сжатия, а не ноты."""

_DUPLICATE_ONSET_GAP = 0.06
"""Порог слияния дублей одной ноты, секунды.

Блок иногда распадается на части: сглаживание краёв, наложенный эффект,
артефакт сжатия. Части ведутся как отдельные треки и дают по событию
на ту же ноту с разницей в кадр. Настоящие повторы ноты быстрее 60 мс
на фортепиано не встречаются, поэтому такие пары безопасно склеивать.
"""

_MIN_BLOCK_WIDTH_KEYS = 0.35
"""Доля ширины клавиши, ниже которой пятно считается краевым артефактом.

Блок рисуется во всю ширину клавиши. Узкие полоски по бокам возникают
из-за сглаживания и попадают центроидом на соседнюю клавишу, порождая
ноты, которых не было.
"""

_MATCH_TOLERANCE_PX = 12.0
"""Допуск сопоставления блока между кадрами по предсказанной позиции."""

_SEPARATOR_MARGIN_PX = 4
"""Отступ от линии касания: там проходит разделительная полоса."""

_NEW_TRACK_CLEARANCE_PX = 6.0
"""Насколько выше линии касания должен быть блок, чтобы завести новый трек.

Блок, уже пересёкший линию, — это хвост приземлившейся ноты, а не новая.
Без этого условия остаток блока порождал бы новый трек на каждом кадре,
и одна нота попадала бы в результат до десяти раз.
"""

_MAX_BLOCK_WIDTH_KEYS = 1.6
"""Предел ширины блока в ширинах белой клавиши.

Блок соответствует одной клавише. Всё, что заметно шире, — разделительная
линия, полоса педали, водяной знак или наложенный эффект.
"""


@dataclass
class Block:
    """Отслеживаемый блок.

    Attributes:
        pitch: Нота, определённая по горизонтальной позиции.
        hand: Партия, определённая по цвету.
        top: Верхняя граница в пикселях.
        bottom: Нижняя граница.
        hue: Тон цвета блока.
        last_seen: Время последнего наблюдения.
        landed: Блок уже коснулся клавиатуры.
    """

    pitch: int
    hand: Hand
    top: float
    bottom: float
    hue: float
    last_seen: float
    landed: bool = False
    onset: float | None = None

    @property
    def height(self) -> float:
        """Длина блока в пикселях."""
        return self.bottom - self.top


@dataclass
class _Detection:
    """Блок, найденный в одном кадре."""

    pitch: int
    top: float
    bottom: float
    hue: float
    saturation: float


class BlockTracker:
    """Трекинг блоков с субкадровой точностью момента касания.

    Args:
        layout: Разметка клавиатуры — по ней блок сопоставляется с нотой.
        profile: Профиль визуализатора: цвет фона, палитра, скорость падения.
        fps: Частота кадров.
    """

    def __init__(
        self,
        layout: KeyboardLayout,
        profile: VisualizerProfile,
        fps: float,
        config: TrackerConfig | None = None,
    ) -> None:
        self.layout = layout
        self.profile = profile
        self.fps = fps
        self.config = config or TrackerConfig()
        self._tracked: list[Block] = []
        self._completed: list[NoteEvent] = []
        self._hand_hues = self._resolve_hand_hues()

    def _resolve_hand_hues(self) -> dict[Hand, float]:
        """Разделить цвета блоков на две группы.

        Привязка к конкретной руке здесь произвольна: по цвету понять,
        где левая, нельзя. Группы получают имена рук только для удобства,
        а окончательное решение принимает `_assign_hands_by_register`
        по высоте нот, когда весь ролик уже разобран.
        """
        colors = self.profile.block_colors_hsv
        if len(colors) < 2:
            return {}
        return {Hand.RIGHT: colors[0][0], Hand.LEFT: colors[1][0]}

    def process_frame(self, frame: Frame, timestamp: float) -> list[NoteEvent]:
        """Обработать кадр.

        Args:
            frame: Кадр в BGR.
            timestamp: Время кадра в секундах.

        Returns:
            Ноты, чьи блоки коснулись клавиатуры на этом кадре.
        """
        detections = self._detect(frame)
        landed = self._update_tracks(detections, timestamp)
        self._drop_stale(timestamp)
        return landed

    def flush(self) -> list[NoteEvent]:
        """Вернуть ноты за весь ролик.

        Дубли одного блока сливаются, слишком короткие события
        отбрасываются: это обрывки соседних блоков, попавшие центроидом
        на чужую клавишу.
        """
        merged = self._deduplicate(self._completed)
        kept = [n for n in merged if n.duration >= self.config.min_note_duration]
        return self._assign_hands_by_register(kept)

    @staticmethod
    def _assign_hands_by_register(notes: list[NoteEvent]) -> list[NoteEvent]:
        """Решить, какой цвет какой руке принадлежит.

        Цвет сам по себе о руке не говорит: порядок цветов в профиле
        задаётся занимаемой площадью и меняется от ролика к ролику.
        Зато партия левой руки звучит ниже — это и есть признак.
        Сравниваются медианы высот двух групп: медиана, в отличие
        от среднего, не съезжает от нескольких перекрёстных нот.
        """
        left = sorted(n.pitch for n in notes if n.hand is Hand.LEFT)
        right = sorted(n.pitch for n in notes if n.hand is Hand.RIGHT)
        if not left or not right or left[len(left) // 2] <= right[len(right) // 2]:
            return notes

        swap = {Hand.LEFT: Hand.RIGHT, Hand.RIGHT: Hand.LEFT}
        return [
            replace(note, hand=swap[note.hand]) if note.hand in swap else note for note in notes
        ]

    @staticmethod
    def _deduplicate(notes: list[NoteEvent]) -> list[NoteEvent]:
        """Склеить события, относящиеся к одной ноте.

        Из пары остаётся более длинное событие: короткое — это обрывок
        блока, потерявший верхнюю или нижнюю часть.
        """
        best: dict[tuple[int, int], NoteEvent] = {}
        for note in sorted(notes, key=lambda n: n.onset):
            bucket = (note.pitch, int(note.onset / _DUPLICATE_ONSET_GAP))
            neighbours = [
                key
                for key in (bucket, (note.pitch, bucket[1] - 1), (note.pitch, bucket[1] + 1))
                if key in best and abs(best[key].onset - note.onset) < _DUPLICATE_ONSET_GAP
            ]
            if not neighbours:
                best[bucket] = note
                continue
            key = neighbours[0]
            if note.duration > best[key].duration:
                best[key] = note
        return sorted(best.values(), key=lambda n: (n.onset, n.pitch))

    def _detect(self, frame: Frame) -> list[_Detection]:
        """Найти блоки в зоне выше клавиатуры.

        Верхняя граница клавиатуры обычно подчёркнута разделительной линией.
        Она контрастна фону и тянется через весь кадр, поэтому без отступа
        попадает в маску одной огромной компонентой и порождает ложные ноты.
        """
        hit_line = self.profile.hit_line_y - _SEPARATOR_MARGIN_PX
        if hit_line <= 2:
            return []

        region = frame[:hit_line]
        hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
        background = np.array(self.profile.background_hsv, dtype=np.float32)
        distance = np.abs(hsv.astype(np.float32) - background).sum(axis=2)
        mask = (distance > 60).astype(np.uint8)

        count, _labels, stats, centroids = cv2.connectedComponentsWithStats(
            np.ascontiguousarray(mask, dtype=np.uint8), connectivity=8
        )

        key_width = self._white_key_width()
        max_width = key_width * _MAX_BLOCK_WIDTH_KEYS
        min_width = key_width * _MIN_BLOCK_WIDTH_KEYS
        min_height = max(2.0, self.profile.fall_speed * self.config.min_note_duration)

        detections: list[_Detection] = []
        for index in range(1, count):
            x, y, w, h, area = stats[index]
            if area < _MIN_BLOCK_AREA:
                continue
            if w < min_width or w > max_width or h < min_height:
                continue

            pitch = self.layout.pitch_at(int(centroids[index][0]))
            if pitch is None:
                continue

            patch = hsv[y : y + h, x : x + w].reshape(-1, 3)
            detections.append(
                _Detection(
                    pitch=pitch,
                    top=float(y),
                    bottom=float(y + h),
                    hue=float(np.median(patch[:, 0].astype(np.float32))),
                    saturation=float(np.median(patch[:, 1].astype(np.float32))),
                )
            )
        return detections

    def _white_key_width(self) -> float:
        """Ширина белой клавиши в пикселях — мера допустимой ширины блока."""
        whites = [k for k in self.layout.keys if not k.is_black]
        if not whites:
            return float(self.layout.bbox[2])
        return float(np.median([k.x_max - k.x_min for k in whites]))

    def _update_tracks(self, detections: list[_Detection], timestamp: float) -> list[NoteEvent]:
        """Сопоставить обнаружения с отслеживаемыми блоками."""
        landed: list[NoteEvent] = []
        step = self.profile.fall_speed / self.fps if self.fps else 0.0
        unmatched = list(detections)

        for block in self._tracked:
            predicted_bottom = block.bottom + step
            tolerance = (
                max(_MATCH_TOLERANCE_PX, block.height) if block.landed else _MATCH_TOLERANCE_PX
            )
            best, best_gap = None, tolerance
            for detection in unmatched:
                if detection.pitch != block.pitch:
                    continue
                gap = abs(detection.bottom - predicted_bottom)
                if gap < best_gap:
                    best, best_gap = detection, gap

            if best is None:
                # Блок мог скрыться за эффектом или водяным знаком: ведём
                # его по предсказанной траектории, а не бросаем
                note = self._check_landing(block, predicted_bottom, timestamp)
                if note is not None:
                    landed.append(note)
                block.bottom = predicted_bottom
                block.top += step
                continue

            unmatched.remove(best)
            note = self._check_landing(block, best.bottom, timestamp)
            if note is not None:
                landed.append(note)
            block.top, block.bottom, block.last_seen = best.top, best.bottom, timestamp

        hit_line = self.profile.hit_line_y - _SEPARATOR_MARGIN_PX
        for detection in unmatched:
            if detection.bottom > hit_line - _NEW_TRACK_CLEARANCE_PX:
                continue
            self._tracked.append(
                Block(
                    pitch=detection.pitch,
                    hand=self._hand_for(detection.hue),
                    top=detection.top,
                    bottom=detection.bottom,
                    hue=detection.hue,
                    last_seen=timestamp,
                )
            )
        return landed

    def _check_landing(self, block: Block, new_bottom: float, timestamp: float) -> NoteEvent | None:
        """Проверить касание клавиатуры и вычислить точный момент.

        Момент касания находится линейной интерполяцией между положением
        блока на предыдущем и текущем кадре — это поднимает точность
        с 33 мс (шаг кадра при 30 fps) до единиц миллисекунд.
        """
        hit_line = self.profile.hit_line_y - _SEPARATOR_MARGIN_PX
        if block.landed or new_bottom < hit_line:
            return None

        travelled = new_bottom - block.bottom
        frame_step = 1.0 / self.fps if self.fps else 0.0
        if travelled > 1e-6:
            fraction = (hit_line - block.bottom) / travelled
            onset = timestamp - frame_step * (1.0 - float(np.clip(fraction, 0.0, 1.0)))
        else:
            onset = timestamp

        block.landed = True
        block.onset = onset

        duration = (
            block.height / self.profile.fall_speed if self.profile.fall_speed > 0 else frame_step
        )
        if duration <= 0:
            return None

        note = NoteEvent(
            pitch=block.pitch,
            onset=onset,
            offset=onset + duration,
            velocity=self._velocity_from(block),
            hand=block.hand,
            source=Source.VIDEO,
            confidence=0.9,
        )
        self._completed.append(note)
        return note

    def _velocity_from(self, block: Block) -> int:
        """Оценить силу нажатия.

        Цвет блока о громкости не говорит: у большинства визуализаторов он
        одинаков для всех нот партии. Возвращается среднее значение —
        реальную динамику даст аудиоканал на этапе слияния.
        """
        del block
        return 64

    def _hand_for(self, hue: float) -> Hand:
        """Определить руку по тону блока."""
        if not self._hand_hues:
            return Hand.UNKNOWN

        best_hand, best_gap = Hand.UNKNOWN, 90.0
        for hand, reference in self._hand_hues.items():
            gap = abs(hue - reference)
            gap = min(gap, 180.0 - gap)
            if gap < best_gap:
                best_hand, best_gap = hand, gap
        return best_hand if best_gap < 25.0 else Hand.UNKNOWN

    def _drop_stale(self, timestamp: float, max_age: float = 0.5) -> None:
        """Забыть блоки, пропавшие из виду."""
        self._tracked = [
            b for b in self._tracked if timestamp - b.last_seen <= max_age and not b.landed
        ]
