"""Генератор синтетических piano-visualizer-роликов.

Ручная разметка нот стоит часов на каждый ролик. Здесь обратный ход:
видео рендерится из известного списка нот, поэтому эталон точен, бесплатен
и доступен в любом количестве. Варьируя параметры рендера, можно получить
и «идеальный» ролик, и низкобитрейтный с обрезанной клавиатурой.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from mir.common.enums import Hand
from mir.common.types import Frame, NoteEvent
from mir.vision.keyboard_geometry import (
    BLACK_KEY_HEIGHT_RATIO,
    KeyboardGeometry,
    is_black_key,
)

__all__ = ["ColorScheme", "SynthConfig", "render_visualizer_video", "simple_melody"]


@dataclass(frozen=True)
class ColorScheme:
    """Палитра визуализатора в BGR — порядок, принятый в OpenCV.

    Attributes:
        background: Фон зоны падающих блоков.
        white_key: Ненажатая белая клавиша.
        black_key: Ненажатая чёрная клавиша.
        right_hand: Блоки и подсветка правой руки.
        left_hand: Блоки и подсветка левой руки.
        separator: Линия между зонами.
    """

    background: tuple[int, int, int] = (30, 25, 20)
    white_key: tuple[int, int, int] = (245, 245, 245)
    black_key: tuple[int, int, int] = (25, 25, 25)
    right_hand: tuple[int, int, int] = (80, 200, 80)
    left_hand: tuple[int, int, int] = (220, 140, 60)
    separator: tuple[int, int, int] = (200, 200, 200)

    @staticmethod
    def dark() -> ColorScheme:
        """Тёмная тема с фиолетово-синими блоками."""
        return ColorScheme(
            background=(12, 10, 10),
            white_key=(220, 220, 220),
            black_key=(15, 15, 15),
            right_hand=(200, 90, 180),
            left_hand=(200, 170, 60),
        )

    @staticmethod
    def monochrome() -> ColorScheme:
        """Один цвет на обе руки — разделение по цвету невозможно."""
        return ColorScheme(right_hand=(90, 190, 240), left_hand=(90, 190, 240))


@dataclass
class SynthConfig:
    """Параметры рендера.

    Attributes:
        width: Ширина кадра.
        height: Высота кадра.
        fps: Частота кадров.
        keyboard_height_ratio: Доля высоты кадра под клавиатуру.
        fall_speed: Скорость падения блоков, пикселей в секунду.
        lowest_pitch: Нижняя видимая нота (для имитации обрезанной клавиатуры).
        highest_pitch: Верхняя видимая нота.
        colors: Палитра.
        glow: Рисовать свечение вокруг блоков.
        noise_sigma: Гауссов шум, имитация артефактов сжатия.
        watermark: Текст поверх кадра.
    """

    width: int = 1280
    height: int = 720
    fps: float = 30.0
    keyboard_height_ratio: float = 0.25
    fall_speed: float = 320.0
    lowest_pitch: int = 36
    highest_pitch: int = 96
    colors: ColorScheme = field(default_factory=ColorScheme)
    glow: bool = False
    noise_sigma: float = 0.0
    watermark: str | None = None
    intro_seconds: float = 0.0
    outro_seconds: float = 0.0

    @property
    def keyboard_top(self) -> int:
        """Y-координата линии касания: верхняя граница клавиатуры."""
        return int(self.height * (1 - self.keyboard_height_ratio))

    @property
    def keyboard_height(self) -> int:
        """Высота зоны клавиатуры."""
        return self.height - self.keyboard_top

    def geometry(self) -> KeyboardGeometry:
        """Раскладка клавиш на всю ширину кадра."""
        return KeyboardGeometry(
            x=0,
            width=self.width,
            lowest_pitch=self.lowest_pitch,
            highest_pitch=self.highest_pitch,
        )


def _hand_color(hand: Hand, colors: ColorScheme) -> tuple[int, int, int]:
    return colors.left_hand if hand is Hand.LEFT else colors.right_hand


def _draw_keyboard(frame: Frame, config: SynthConfig, active: dict[int, Hand]) -> None:
    """Нарисовать клавиатуру, подсветив звучащие клавиши.

    Белые рисуются первыми, чёрные поверх — иначе узкие чёрные клавиши
    будут затёрты соседними белыми.
    """
    geom = config.geometry()
    top, height = config.keyboard_top, config.keyboard_height
    black_height = int(height * BLACK_KEY_HEIGHT_RATIO)

    for pitch in geom.white_pitches:
        left, right = geom.bounds(pitch)
        colour = (
            _hand_color(active[pitch], config.colors)
            if pitch in active
            else config.colors.white_key
        )
        cv2.rectangle(frame, (int(left), top), (int(right) - 1, config.height), colour, -1)
        cv2.rectangle(frame, (int(left), top), (int(right) - 1, config.height), (60, 60, 60), 1)

    for pitch in range(config.lowest_pitch, config.highest_pitch + 1):
        if not is_black_key(pitch):
            continue
        left, right = geom.bounds(pitch)
        colour = (
            _hand_color(active[pitch], config.colors)
            if pitch in active
            else config.colors.black_key
        )
        cv2.rectangle(frame, (int(left), top), (int(right), top + black_height), colour, -1)


def _draw_blocks(frame: Frame, config: SynthConfig, notes: list[NoteEvent], t: float) -> None:
    """Нарисовать падающие блоки на момент времени t.

    Блок касается линии клавиатуры ровно в onset ноты, а его длина
    пропорциональна длительности — так же, как в настоящих визуализаторах.
    """
    geom = config.geometry()
    hit_line = config.keyboard_top

    for note in notes:
        bottom = hit_line - (note.onset - t) * config.fall_speed
        top = bottom - note.duration * config.fall_speed
        if bottom < 0 or top > hit_line:
            continue

        left, right = geom.bounds(note.pitch)
        y0, y1 = int(max(top, 0)), int(min(bottom, hit_line))
        if y1 <= y0:
            continue

        colour = _hand_color(note.hand, config.colors)
        if config.glow:
            cv2.rectangle(
                frame,
                (int(left) - 3, y0 - 3),
                (int(right) + 3, y1 + 3),
                tuple(int(c * 0.4) for c in colour),
                -1,
            )
        cv2.rectangle(frame, (int(left), y0), (int(right) - 1, y1), colour, -1)


def _draw_title_card(frame: Frame, config: SynthConfig) -> None:
    """Нарисовать заставку: обложка и название вместо клавиатуры.

    Так открывается почти каждый ролик на YouTube, и клавиатуры в этих
    кадрах нет вовсе.
    """
    cv2.rectangle(
        frame,
        (config.width // 4, config.height // 4),
        (config.width * 3 // 4, config.height * 3 // 4),
        (60, 50, 45),
        -1,
    )
    cv2.putText(
        frame,
        "PIANO",
        (config.width // 3, config.height // 2),
        cv2.FONT_HERSHEY_SIMPLEX,
        2.0,
        (220, 220, 220),
        3,
    )


def render_visualizer_video(
    notes: list[NoteEvent],
    out_path: Path,
    config: SynthConfig | None = None,
    duration: float | None = None,
) -> Path:
    """Отрендерить ролик из списка нот.

    Args:
        notes: Эталонные ноты.
        out_path: Куда писать mp4.
        config: Параметры рендера.
        duration: Длительность. По умолчанию — до конца последней ноты
            плюс время падения блоков.

    Returns:
        Путь к готовому файлу.

    Example:
        >>> notes = simple_melody()
        >>> render_visualizer_video(notes, Path("out.mp4"))  # doctest: +SKIP
    """
    config = config or SynthConfig()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if duration is None:
        last = max((n.offset for n in notes), default=1.0)
        duration = last + config.keyboard_top / config.fall_speed + 0.5
    duration += config.intro_seconds + config.outro_seconds

    writer = cv2.VideoWriter(
        str(out_path),
        cv2.VideoWriter.fourcc(*"mp4v"),
        config.fps,
        (config.width, config.height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"не удалось открыть {out_path} для записи")

    rng = np.random.default_rng(0)
    total = int(duration * config.fps)

    try:
        intro_frames = int(config.intro_seconds * config.fps)
        outro_frames = int(config.outro_seconds * config.fps)

        for index in range(total):
            t = (index - intro_frames) / config.fps
            # Тип указан явно: без него он выводится из np.full, и добавление
            # шума ниже перестаёт ему соответствовать — причём по-разному
            # в стабах numpy для разных версий Python.
            frame: Frame = np.full(
                (config.height, config.width, 3), config.colors.background, dtype=np.uint8
            )

            if index < intro_frames or index >= total - outro_frames:
                _draw_title_card(frame, config)
                writer.write(frame)
                continue

            _draw_blocks(frame, config, notes, t)

            active = {n.pitch: n.hand for n in notes if n.onset <= t < n.offset}
            _draw_keyboard(frame, config, active)

            cv2.line(
                frame,
                (0, config.keyboard_top),
                (config.width, config.keyboard_top),
                config.colors.separator,
                2,
            )

            if config.watermark:
                cv2.putText(
                    frame,
                    config.watermark,
                    (config.width // 3, config.height // 3),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.2,
                    (255, 255, 255),
                    2,
                )

            if config.noise_sigma > 0:
                noise = rng.normal(0, config.noise_sigma, frame.shape)
                frame = np.clip(frame.astype(np.float32) + noise, 0, 255).astype(np.uint8)

            writer.write(frame)
    finally:
        writer.release()

    return out_path


def simple_melody(
    start: float = 0.5, note_duration: float = 0.4, gap: float = 0.1
) -> list[NoteEvent]:
    """Гамма до мажор в две руки — базовый набор для проверки детектора.

    Правая рука играет от до первой октавы вверх, левая — те же ступени
    двумя октавами ниже, поэтому разделение по цвету имеет смысл.
    """
    notes: list[NoteEvent] = []
    scale = [60, 62, 64, 65, 67, 69, 71, 72]
    t = start
    for pitch in scale:
        notes.append(
            NoteEvent(
                pitch=pitch,
                onset=t,
                offset=t + note_duration,
                velocity=90,
                hand=Hand.RIGHT,
            )
        )
        notes.append(
            NoteEvent(
                pitch=pitch - 24,
                onset=t,
                offset=t + note_duration,
                velocity=70,
                hand=Hand.LEFT,
            )
        )
        t += note_duration + gap
    return notes
