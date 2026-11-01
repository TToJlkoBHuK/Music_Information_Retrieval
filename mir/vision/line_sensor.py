"""Линия-датчик: ноты по падающим блокам, без участия клавиатуры.

Нота начинается, когда нижний край блока касается клавиатуры, и кончается,
когда через линию касания проходит его верхний край. Значит, достаточно
следить за узкой полосой над клавиатурой: пока в столбце клавиши стоит
блок, нота звучит. Руки исполнителя здесь не мешают — они лежат на
клавиатуре, ниже полосы. Именно руки ломают канал подсветки на роликах
Rousseau и подобных (`docs/known-issues.md`, В-05).

Полос две, одна выше другой. Это решает две задачи сразу:

* **скорость падения** — по тому, за какое время блок проходит от верхней
  полосы к нижней. Она одинакова для всех блоков ролика. Фазовая
  корреляция, которой скорость мерилась раньше, на светящихся блоках
  с частицами давала ноль (В-01);
* **отсев дыма и искр.** После касания визуализатор пускает над клавишей
  дым и искры — яркие, плотные, на высоте полосы. Но они поднимаются, а блок
  падает. Нота засчитывается, только если блок из верхней полосы через
  нужное время прошёл и нижнюю.

Блок отличается от дыма и тем, что сплошной: в его столбце яркой оказывается
каждая строка полосы, а у дыма — лишь часть. Поэтому столбец засчитывается
по минимуму по строкам, а не по среднему.

**Время точнее кадра.** Когда блок впервые замечен в полосе, его нижний край
уже ушёл ниже — на сколько, видно по столбцу клавиши. Расстояние, делённое
на скорость, — это время, прошедшее с пересечения середины полосы. То же
с верхним краем в конце ноты. Так время находится с точностью до пикселя,
а не до кадра.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field, replace

import cv2
import numpy as np
import numpy.typing as npt

from mir.common.enums import Hand, Source
from mir.common.logging import get_logger
from mir.common.types import Frame, KeyboardLayout, NoteEvent
from mir.config import SensorConfig
from mir.vision.block_tracker import BlockTracker

__all__ = ["LineSensor", "SensorBank", "SensorReport"]

_log = get_logger(__name__)

_BLOCK_LEVEL_PERCENTILE = 99.5
"""Яркость блока — верхний перцентиль отклонения от фона в полосе.

Блоки занимают в полосе несколько процентов площади, фон — остальное;
99.5-й перцентиль по кадрам выборки приходится на тело блока.
"""

_MIN_DELAY_PAIRS = 5
"""Меньше пар «верх → низ» — скорость не определить."""

_MATCH_FRAMES = 2.0
"""Допуск сопоставления событий двух полос, в кадрах."""

_UNKNOWN_VELOCITY = 64
"""Громкость ноты по блоку не узнать: блоки её не кодируют.

64 — середина шкалы MIDI («mf»). При слиянии каналов громкость
берётся из звука.
"""

_CENTER_HALF_WIDTH = 2
"""Полуширина столбца в пикселях, по которому прослеживаются края блока."""

_MAX_TRACES = 3000
"""Сколько раз проследить блок до линии касания — с запасом для моды."""

_MIN_HUE_SATURATION = 40.0
"""Ниже этой насыщенности цвет блока не говорит о руке."""

_MIN_HAND_HUE_GAP = 20.0
"""Оттенки рук должны различаться хотя бы на столько (шкала OpenCV 0..180, то есть 40°).

У Rousseau все блоки одного оттенка (90–98), у синтетики руки зелёная и синяя
(60 и 105). Разброс оттенка у одного цвета — единицы.
"""

_MIN_HAND_SHARE = 0.1
"""Группа меньше этой доли нот — выбросы (искры другого цвета), а не рука."""

_MIN_HAND_MIXING = 0.1
"""Насколько цвета должны перемешиваться во времени, чтобы быть руками.

Руки играют одновременно, и по ходу ролика цвета нот чередуются постоянно.
Число смен цвета в последовательности нот сравнивается с ожидаемым при
случайном порядке (критерий серий Вальда — Вольфовица). Замеры: у Rousseau,
Clair de lune, палитра сменилась с синей на фиолетовую на 151-й секунде —
отношение 0.003; у синтетики, где руки играют вместе, — больше 1. Порог 0.1
пропускает и пьесы, где руки подолгу играют по очереди (до 20 нот одной
рукой подряд). Ролика, где руки действительно окрашены по-разному, в
проверке пока не было — сверить на #13.
"""

_BACKGROUND_ROWS = 32
"""По сколько строк считать фон: около 50 МБ памяти на часть при 1080p."""

_WIDE_BLOCK = 1.6
"""Сплошной участок шире стольких белых клавиш — это несколько слившихся блоков."""


@dataclass
class _Event:
    """Блок, прошедший через полосу.

    Время — кадр, на котором край замечен, плюс расстояние, на которое край
    к этому кадру ушёл за середину полосы. В секунды расстояние переводится,
    когда известна скорость.

    Расстояние ограничено тем, что физически возможно: на прошлом кадре
    края в этом месте ещё не было, значит, за кадр он ушёл не дальше
    `скорость × шаг`. Без ограничения неподвижный объект, возникший сразу
    во весь столбец (луч, полоса заставки), выглядел бы блоком, давно
    пересёкшим полосу, — и его время в двух полосах сходилось бы, как у
    настоящего блока.
    """

    key: int
    start_time: float
    start_shift: float
    end_time: float
    end_shift: float
    """`nan`, если верхний край не найден."""
    colour: npt.NDArray[np.float64]
    clipped: bool
    """Блок уже стоял в полосе на первом кадре — начало неизвестно."""
    half: float
    """Половина высоты полосы."""
    step: float
    """Шаг между кадрами, секунды."""

    def start(self, speed: float) -> float:
        # Блок засчитывается, когда закрыл всю полосу: низ уже на `half` ниже середины.
        shift = float(np.clip(self.start_shift, self.half, self.half + speed * self.step))
        return self.start_time - shift / speed

    def measured(self, speed: float) -> bool:
        """Низ блока прослежен правдоподобно, а не утёк в свечение под ним.

        Под блоком у нижней полосы часто светятся дым и вспышка касания:
        прослеживание уходит в них, и расстояние превышает возможное за кадр.
        Такое начало обрезалось бы до границы и смещалось на долю кадра,
        поэтому его лучше не учитывать вовсе. Запас в пиксель — на размытие края.
        """
        return self.start_shift <= self.half + speed * self.step + 1.0

    def end(self, speed: float) -> float:
        # Блок пропал, когда верх вошёл в полосу: не выше `half` над серединой.
        upper = -self.half + speed * self.step
        raw = upper if np.isnan(self.end_shift) else self.end_shift
        return self.end_time - float(np.clip(raw, -self.half, upper)) / speed


@dataclass
class _Active:
    """Блок, который сейчас в полосе."""

    start_time: float
    start_shift: float
    colour: npt.NDArray[np.float64]
    clipped: bool
    pending: tuple[float, float] | None = None
    """Кадр, на котором блок пропал, и положение его верхнего края."""


@dataclass
class _Band:
    """Горизонтальная полоса-датчик."""

    top: int
    bottom: int
    background: npt.NDArray[np.int16]
    """Фон от верха полосы до клавиатуры: по нему прослеживаются края блока."""
    level: float
    active: dict[int, _Active] = field(default_factory=dict)
    events: list[_Event] = field(default_factory=list)

    @property
    def middle(self) -> float:
        return (self.top + self.bottom) / 2


@dataclass
class SensorReport:
    """Подробности работы датчика — для отчёта и отладки.

    Attributes:
        enabled: Блоки найдены, датчик работал.
        fall_speed: Скорость падения, пикселей в секунду.
        contact_y: Найденная линия касания.
        confirmed: Нот подтверждено нижней полосой.
        rejected: Событий верхней полосы отброшено как дым, искры и прочее.
        reason: Почему датчик не работал.
        solid_level: Порог «сплошного» столбца, с которым получены ноты.
        candidates: Испробованные пороги и доля отсеянного при каждом.
    """

    enabled: bool
    fall_speed: float = 0.0
    contact_y: float = 0.0
    confirmed: int = 0
    rejected: int = 0
    reason: str = ""
    solid_level: float = 0.0
    candidates: tuple[tuple[float, float], ...] = ()

    @property
    def rejected_share(self) -> float:
        """Доля отсеянного среди всех событий верхней полосы."""
        total = self.confirmed + self.rejected
        return self.rejected / total if total else 1.0


def _background(sample: list[Frame], top: int, bottom: int) -> npt.NDArray[np.int16]:
    """Фон строк `top:bottom` — каким столбец бывает без блока.

    Медиана по кадрам выборки годится, пока клавиша свободна больше половины
    времени. Но басовая нота, которую держат почти весь ролик (соль-диез
    в «Лунной сонате»), делает медиану своего столбца полусветлой, и блок
    на этом фоне перестаёт выделяться — нота терялась целиком. Поэтому
    в каждом столбце берутся только те кадры, где он похож на фон строки
    в целом (медиану по всем столбцам: занятых клавиш всегда мало), —
    четверть самых похожих, и из них медиана. Неподвижное — разделительные
    линии, виньетка — остаётся своим: такой столбец всегда одинаков, и
    похожие на себя кадры у него те же самые. Строки обрабатываются частями,
    чтобы не держать в памяти всю выборку целиком.
    """
    keep = max(len(sample) // 4, 1)
    parts: list[npt.NDArray[np.int16]] = []
    for start in range(top, bottom, _BACKGROUND_ROWS):
        end = min(start + _BACKGROUND_ROWS, bottom)
        stack = np.stack([frame[start:end] for frame in sample]).astype(np.int16)
        reference = np.median(np.median(stack, axis=0), axis=1, keepdims=True)
        distance = np.abs(stack - reference).max(axis=3)
        nearest = np.argpartition(distance, keep - 1, axis=0)[:keep]
        chosen = np.take_along_axis(stack, nearest[..., None], axis=0)
        parts.append(np.median(chosen, axis=0).astype(np.int16))
    return np.concatenate(parts)


def _zone_end(
    background: npt.NDArray[np.int16],
    low_rows: tuple[int, int],
    x_range: tuple[int, int],
    contrast: float,
) -> int:
    """Где кончается зона падения блоков — строка от начала `background`.

    У многих визуализаторов над клавишами проходит планка (у Rousseau —
    серая, 8 пикселей). Когда блок касается её, она вспыхивает, и без
    ограничения низ блока прослеживался бы сквозь вспышку до самых клавиш —
    линия касания уезжала на 13 пикселей (54 мс). Планка видна в фоне как
    ровная полоса во всю ширину клавиатуры, заметно светлее зоны над ней;
    зона кончается на её верхнем крае — по половине перепада. Порог
    «заметно» — `contrast`: та же яркость, ниже которой не считаются блоки.
    Планки нет — зона доходит до клавиатуры.
    """
    profile = np.median(background.max(axis=2)[:, x_range[0] : x_range[1]], axis=1)
    zone = float(np.median(profile[low_rows[0] : low_rows[1]]))
    below = np.abs(profile[low_rows[1] :] - zone)
    if len(below) == 0 or below.max() < contrast:
        return len(profile)
    return low_rows[1] + int(np.argmax(below >= below.max() / 2))


def _runs(mask: npt.NDArray[np.bool_]) -> list[tuple[int, int]]:
    """Непрерывные участки `True`: пары (начало, конец исключая)."""
    edges = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1), strict=True))


class LineSensor:
    """Ноты по блокам, проходящим через две полосы над клавиатурой.

    Args:
        layout: Разметка клавиатуры.
        sample: Кадры выборки по всему ролику — для фона полос и яркости блоков.
        fps: Частота кадров.
        config: Параметры из `[vision.sensor]`.
        solid_level: Порог «сплошного» столбца; по умолчанию первый из
            `config.solid_levels`.
        background: Готовый фон от другого датчика того же ролика — чтобы не
            считать его заново (`SensorBank`).
    """

    def __init__(
        self,
        layout: KeyboardLayout,
        sample: list[Frame],
        fps: float,
        config: SensorConfig,
        solid_level: float | None = None,
        background: npt.NDArray[np.int16] | None = None,
    ) -> None:
        self.layout = layout
        self.fps = fps
        self.config = config
        self.solid_level = solid_level if solid_level is not None else config.solid_levels[0]
        self.background: npt.NDArray[np.int16] | None = None
        self.report = SensorReport(enabled=False, solid_level=self.solid_level)
        self._bands: list[_Band] = []
        self._bottoms: list[int] = []
        self._initial_bottom: dict[int, int] = {}
        self._frames = 0
        self._first_time: float | None = None
        self._last_time = 0.0

        keyboard_top = layout.bbox[1]
        if not sample:
            self.report.reason = "нет кадров для настройки"
            return
        height = max(round(sample[0].shape[0] * config.band_height), 4)
        tops = [
            round(keyboard_top * (1.0 - offset))
            for offset in (config.high_offset, config.low_offset)
        ]
        if min(tops) < 0 or max(tops) + height >= keyboard_top:
            self.report.reason = "полоса датчика не помещается над клавиатурой"
            return
        first = min(tops)
        if background is None:
            background = _background(sample, first, keyboard_top)
        self.background = background
        x_range = (int(layout.bbox[0]), int(layout.bbox[0] + layout.bbox[2]))
        low_rows = (max(tops) - first, max(tops) - first + height)
        self._zone_end = first + _zone_end(background, low_rows, x_range, config.min_block_contrast)
        for top in tops:
            rows = background[top - first : self._zone_end - first]
            self._bands.append(self._make_band(top, top + height, rows, sample))

        self._centers = np.array([(k.x_min + k.x_max) / 2 for k in layout.keys])
        self._pitches = [k.pitch for k in layout.keys]
        whites = [k.x_max - k.x_min for k in layout.keys if not k.is_black]
        self._white_width = float(np.median(whites)) if whites else 1.0
        offsets = np.arange(-_CENTER_HALF_WIDTH, _CENTER_HALF_WIDTH + 1)
        self._columns = np.clip(
            self._centers.astype(int)[:, None] + offsets[None, :], 0, sample[0].shape[1] - 1
        )

        level = min(band.level for band in self._bands) / self.solid_level
        if level < config.min_block_contrast:
            self.report.reason = (
                f"над клавиатурой нет блоков: яркость {level:.0f} при пороге "
                f"{config.min_block_contrast:.0f}"
            )
            _log.info("датчик блоков отключён: %s", self.report.reason)
            return
        self.report.enabled = True

    def _make_band(
        self, top: int, bottom: int, background: npt.NDArray[np.int16], sample: list[Frame]
    ) -> _Band:
        rows = bottom - top
        stack = np.stack([frame[top:bottom] for frame in sample]).astype(np.int16)
        deviation = np.abs(stack - background[:rows]).max(axis=3).mean(axis=1)
        block_level = float(np.percentile(deviation, _BLOCK_LEVEL_PERCENTILE))
        return _Band(top, bottom, background, self.solid_level * block_level)

    @property
    def enabled(self) -> bool:
        """Блоки найдены, и датчик ведёт наблюдение."""
        return self.report.enabled

    def process_frame(self, frame: Frame, timestamp: float) -> None:
        """Обработать кадр в BGR."""
        if not self.enabled:
            return
        if self._first_time is None:
            self._first_time = timestamp
        for band in self._bands:
            self._observe(band, frame, timestamp)
        low = self._bands[1]
        for key in list(low.active):
            bottom = self._bottom_of(frame, low, key, low.bottom)
            if bottom is None:
                continue
            if self._frames == 0:
                self._initial_bottom[key] = bottom
            if len(self._bottoms) < _MAX_TRACES:
                self._bottoms.append(bottom)
        self._frames += 1
        self._last_time = timestamp

    def _bright_column(self, frame: Frame, band: _Band, key: int) -> npt.NDArray[np.bool_]:
        """Яркие строки столбца клавиши от верха полосы до конца зоны падения."""
        columns = self._columns[key]
        keyboard_top = band.top + band.background.shape[0]
        column = frame[band.top : keyboard_top, columns].astype(np.int16)
        deviation = np.abs(column - band.background[:, columns]).max(axis=2).min(axis=1)
        bright: npt.NDArray[np.bool_] = deviation >= band.level
        return bright

    def _bottom_of(self, frame: Frame, band: _Band, key: int, start: int) -> int | None:
        """Нижний край блока: первая тёмная строка ниже `start`."""
        bright = self._bright_column(frame, band, key)
        offset = start - band.top
        dark = np.flatnonzero(~bright[offset:])
        if len(dark) == 0:
            return band.top + len(bright)
        return start + int(dark[0])

    def _top_of(self, frame: Frame, band: _Band, key: int) -> int | None:
        """Верхний край блока: первая яркая строка от верха полосы вниз."""
        bright = self._bright_column(frame, band, key)
        lit = np.flatnonzero(bright)
        return band.top + int(lit[0]) if len(lit) else None

    def _observe(self, band: _Band, frame: Frame, timestamp: float) -> None:
        """Обновить состояние клавиш в одной полосе."""
        region = frame[band.top : band.bottom].astype(np.int16)
        deviation = np.abs(region - band.background[: band.bottom - band.top]).max(axis=2)
        solid = self._solid_keys(deviation, band.level)
        first = self._frames == 0

        for key in np.flatnonzero(solid):
            index = int(key)
            active = band.active.get(index)
            if active is not None:
                active.pending = None  # блок не пропадал: был кадр с искрой
                continue
            bottom = self._bottom_of(frame, band, index, band.bottom)
            shift = (bottom if bottom is not None else band.bottom) - band.middle
            colour = region[:, self._columns[index]].reshape(-1, 3).mean(axis=0)
            band.active[index] = _Active(timestamp, shift, colour, clipped=first)

        for key in [k for k in band.active if not solid[k]]:
            active = band.active[key]
            if active.pending is None:
                # Первый кадр без блока: ждём второго, чтобы одна искра
                # или кадр сжатия не разрезали ноту надвое.
                top = self._top_of(frame, band, key)
                shift = (top - band.middle) if top is not None else float("nan")
                active.pending = (timestamp, shift)
                continue
            end_time, end_shift = active.pending
            del band.active[key]
            self._close(band, key, active, end_time, end_shift)

    def _close(
        self, band: _Band, key: int, active: _Active, end_time: float, end_shift: float
    ) -> None:
        """Записать блок, покинувший полосу."""
        half = (band.bottom - band.top) / 2
        band.events.append(
            _Event(
                key,
                active.start_time,
                active.start_shift,
                end_time,
                end_shift,
                active.colour,
                active.clipped,
                half,
                1.0 / self.fps,
            )
        )

    def _solid_keys(self, deviation: npt.NDArray[np.int16], level: float) -> npt.NDArray[np.bool_]:
        """Клавиши, над которыми в полосе стоит сплошной блок.

        Столбец сплошной, если ярка каждая его строка — минимум по строкам.
        Сплошные участки шире порога сопоставляются клавише с ближайшим
        центром; слишком широкие (соседние блоки слились) делятся на части
        по ширине белой клавиши.
        """
        solid = np.zeros(len(self._pitches), dtype=bool)
        unit = self._white_width
        for start, end in _runs(deviation.min(axis=0) >= level):
            width = end - start
            if width < self.config.min_block_width * unit:
                continue
            parts = max(round(width / unit), 1) if width > _WIDE_BLOCK * unit else 1
            for part in range(parts):
                center = start + (part + 0.5) * width / parts
                solid[int(np.argmin(np.abs(self._centers - center)))] = True
        return solid

    def finish(self) -> list[NoteEvent]:
        """Закрыть незавершённые события и вернуть подтверждённые ноты."""
        if not self.enabled:
            return []
        for band in self._bands:
            for key, active in band.active.items():
                end_time, end_shift = active.pending or (self._last_time, 0.0)
                self._close(band, key, active, end_time, end_shift)
            band.active.clear()

        high, low = self._bands
        speed = self._speed(high, low)
        if speed <= 0:
            self.report.enabled = False
            self.report.reason = "не удалось сопоставить полосы: блоки не падают сверху вниз"
            _log.warning("датчик блоков: %s", self.report.reason)
            return []

        contact = self._contact_line(low)
        found, rejected = self._confirm(high, low, contact, speed)
        notes = _split_hands(found)
        notes.sort(key=lambda n: (n.onset, n.pitch))

        self.report = SensorReport(
            enabled=True,
            fall_speed=speed,
            contact_y=contact,
            confirmed=len(notes),
            rejected=rejected,
            solid_level=self.solid_level,
        )
        _log.info(
            "датчик блоков: %d нот, отброшено %d, скорость %.0f px/с, касание y=%.0f",
            len(notes),
            rejected,
            speed,
            contact,
        )
        return notes

    def _speed(self, high: _Band, low: _Band) -> float:
        """Скорость падения по парам «верхняя полоса → нижняя» одной клавиши.

        Сначала пары отбираются по моде задержки в кадрах: у всех блоков
        ролика скорость одна, и верные пары собираются в один столбец
        гистограммы, а случайные — дым, искры — размазаны. Затем по каждой
        верной паре скорость считается точно, с учётом того, насколько край
        блока ушёл за середину каждой полосы к кадру, где его заметили:
        `v = (Δy + сдвиг_внизу − сдвиг_вверху) / (t_внизу − t_вверху)`.
        Медиана таких оценок не зависит от фазы кадров — по одной лишь
        задержке в кадрах ошибка доходила до 5 %.
        """
        by_key: dict[int, list[_Event]] = {}
        for event in low.events:
            if not event.clipped:
                by_key.setdefault(event.key, []).append(event)
        pairs = [
            (event, match)
            for event in high.events
            if not event.clipped
            for match in by_key.get(event.key, [])
            if 0 < match.start_time - event.start_time < self.config.max_delay
        ]
        if len(pairs) < _MIN_DELAY_PAIRS:
            return 0.0
        frame_time = 1.0 / self.fps
        delays = np.array([m.start_time - e.start_time for e, m in pairs])
        bins = np.arange(0.0, self.config.max_delay + frame_time, frame_time)
        counts, edges = np.histogram(delays, bins=bins)
        peak = int(np.argmax(counts))
        near = (delays >= edges[peak] - frame_time) & (delays <= edges[peak + 1] + frame_time)
        distance = low.middle - high.middle
        chosen = [pair for pair, keep in zip(pairs, near, strict=True) if keep]
        # Сдвиги ограничены через скорость (`_Event.start`), поэтому сначала
        # грубая оценка по задержке, затем два уточнения: хватает с запасом.
        speed = distance / float(np.median([m.start_time - e.start_time for e, m in chosen]))
        for _ in range(2):
            delays = np.array(
                [
                    m.start(speed) - e.start(speed)
                    for e, m in chosen
                    if e.measured(speed) and m.measured(speed)
                ]
            )
            delays = delays[delays > 0]
            if len(delays) == 0:
                return 0.0
            speed = float(np.median(distance / delays))
        return speed

    def _confirm(
        self, high: _Band, low: _Band, contact: float, speed: float
    ) -> tuple[list[tuple[NoteEvent, npt.NDArray[np.float64]]], int]:
        """Оставить блоки, дошедшие до нижней полосы, и найти их время.

        Возвращает ноты вместе с цветом блока — по нему потом делятся руки.

        Начало и конец ноты оцениваются по обеим полосам и усредняются: две
        независимые оценки уменьшают разброс. Если в верхней полосе начало
        обрезано первым кадром, остаётся оценка по нижней.
        """
        delay = (low.middle - high.middle) / speed
        tolerance = _MATCH_FRAMES / self.fps
        lead_high = (contact - high.middle) / speed
        lead_low = (contact - low.middle) / speed
        covered = self._coverage_index(low, speed)
        low_by_key: dict[int, list[_Event]] = {}
        for event in low.events:
            low_by_key.setdefault(event.key, []).append(event)

        notes: list[tuple[NoteEvent, npt.NDArray[np.float64]]] = []
        used: set[int] = set()
        rejected = 0
        for event in high.events:
            start, end = event.start(speed), event.end(speed)
            candidates = low_by_key.get(event.key, [])
            share = covered(event.key, start + delay, end + delay)
            # Дым от прошлой ноты нередко уже стоит в нижней полосе, когда туда
            # входит блок: они сливаются, и начало в нижней полосе выходит
            # раньше ожидаемого. На «Лунной сонате» так терялись настоящие ноты
            # с опережением в 2–3 кадра. Поэтому раньше срока допускается до
            # половины задержки: неподвижное (обе полосы сразу) и поднимающийся
            # дым (нижняя раньше верхней) по-прежнему не проходят.
            early = max(tolerance, delay / 2)
            match = self._match(event, candidates, delay, (early, tolerance), speed)
            # Блок обязан дойти до нижней полосы (доля покрытия) и пройти её
            # со сдвигом на задержку — войти или выйти. Второе условие
            # отсекает неподвижное: световой луч или полосу заставки, которые
            # стоят в обеих полосах сразу, — покрытие у них полное, а сдвига
            # нет. Достаточно одного из краёв. Конец в нижней полосе может
            # запоздать: над долгой нотой поднимается её же дым. У блока,
            # стоявшего в полосе с первого кадра, начала нет — нужен конец:
            # иначе луч, светивший с начала разбора, проходил как нота.
            ended = any(-tolerance <= c.end(speed) - (end + delay) <= early for c in candidates)
            shifted = ended if event.clipped else match is not None or ended
            # Нижняя полоса бывает залита дымом прошлых ударов по той же
            # клавише, и у быстрых повторов (остинато Interstellar, 0.27 с)
            # в ней нет ни входа, ни выхода блока. Тогда довольно того, что
            # блок вошёл в верхнюю полосу сверху: низ прослежен на расстоянии,
            # возможном за кадр. Луч или вспышка загораются во весь столбец
            # сразу, дым приходит снизу — у них такого входа нет.
            entered = not event.clipped and event.measured(speed)
            if share < self.config.confirm_coverage or not (shifted or entered):
                rejected += 1
                continue
            onsets = [] if event.clipped else [start + lead_high]
            offsets = [end + lead_high]
            if match is not None:
                used.add(id(match))
                # Во время берутся только края нижней полосы, согласные с
                # верхней: слившийся с блоком дым сдвигает их — начало раньше,
                # конец позже (над длинной нотой поднимается её же дым).
                on_time = abs(match.start(speed) - (start + delay)) <= tolerance
                if not match.clipped and match.measured(speed) and (on_time or event.clipped):
                    onsets.append(match.start(speed) + lead_low)
                if abs(match.end(speed) - (end + delay)) <= tolerance:
                    offsets.append(match.end(speed) + lead_low)
            if not onsets:
                initial = self._initial_onset(event.key, contact, speed)
                if initial is None:
                    continue
                onsets.append(initial)
            notes.append((self._note(event.key, onsets, offsets, share), event.colour))

        # Блок, к началу наблюдения уже миновавший верхнюю полосу, виден
        # только нижней. Дыма в первые секунды ещё нет, и такой блок — нота.
        # Если же клавиша светилась и в верхней полосе с первого кадра, это
        # не тот случай: падающий блок там разобран выше, а неподвижный луч
        # стоит в обеих полосах сразу и нотой не считается.
        first = self._first_time if self._first_time is not None else 0.0
        lit_on_top = {event.key for event in high.events if event.clipped}
        for event in low.events:
            early = event.start_time < first + delay + tolerance
            if not early or id(event) in used or event.key in lit_on_top:
                continue
            onset = (
                event.start(speed) + lead_low
                if not event.clipped
                else self._initial_onset(event.key, contact, speed)
            )
            if onset is not None:
                offsets = [event.end(speed) + lead_low]
                notes.append((self._note(event.key, [onset], offsets, 0.5), event.colour))
        return notes, rejected

    def _initial_onset(self, key: int, contact: float, speed: float) -> float | None:
        """Начало ноты, чей блок к первому кадру уже прошёл обе полосы.

        Так бывает в самом начале разбора: ролик начинается не с первой ноты,
        а с отрезка после заставки, и блоки первых нот уже в полёте. Низ блока
        на первом кадре известен — его проследили до линии касания, — и время
        до касания считается по скорости. Блок, уже стоящий на линии, звучит
        с первого кадра.
        """
        bottom = self._initial_bottom.get(key)
        if bottom is None or self._first_time is None:
            return None
        return self._first_time + max(contact - bottom, 0.0) / speed

    @staticmethod
    def _match(
        event: _Event,
        candidates: list[_Event],
        delay: float,
        window: tuple[float, float],
        speed: float,
    ) -> _Event | None:
        """Событие нижней полосы, порождённое тем же блоком.

        `window` — насколько раньше и позже ожидаемого может начаться событие.
        """
        if event.clipped:
            start, end = event.start(speed) + delay, event.end(speed) + delay
            overlapping = [c for c in candidates if c.start(speed) < end and c.end(speed) > start]
            return overlapping[0] if overlapping else None
        expected = event.start(speed) + delay
        early, late = window
        near = [c for c in candidates if -early <= c.start(speed) - expected <= late]
        return min(near, key=lambda c: abs(c.start(speed) - expected)) if near else None

    def _note(self, key: int, onsets: list[float], offsets: list[float], share: float) -> NoteEvent:
        onset = float(np.mean(onsets))
        offset = max(float(np.mean(offsets)), onset + 1.0 / self.fps)
        return NoteEvent(
            pitch=self._pitches[key],
            onset=onset,
            offset=offset,
            velocity=_UNKNOWN_VELOCITY,
            hand=Hand.UNKNOWN,
            source=Source.VIDEO,
            confidence=float(np.clip(share, 0.0, 1.0)),
        )

    def _contact_line(self, low: _Band) -> float:
        """Линия касания: самое частое положение низа блока под нижней полосой.

        Блок, дошедший до клавиатуры, обрывается на линии касания и стоит на
        ней, пока через неё проходит его верх. Поэтому самое частое положение
        низа блока и есть эта линия, а положения в полёте размазаны по многим
        значениям. У Rousseau она на 20 пикселей выше клавиш — там проходит
        планка, и считать касанием верх клавиатуры значило бы опаздывать
        на 80 мс. Если проследить не удалось, линией считается верх клавиатуры.
        """
        if len(self._bottoms) < _MIN_DELAY_PAIRS:
            return float(self._zone_end)
        counts = np.bincount(np.asarray(self._bottoms) - low.bottom)
        contact = float(low.bottom + int(np.argmax(counts)))
        return float(np.clip(contact, low.bottom, self._zone_end))

    @staticmethod
    def _coverage_index(low: _Band, speed: float) -> Callable[[int, float, float], float]:
        """Функция «какую долю отрезка клавиша была занята в нижней полосе»."""
        by_key: dict[int, list[tuple[float, float]]] = {}
        for event in low.events:
            by_key.setdefault(event.key, []).append((event.start(speed), event.end(speed)))

        def covered(key: int, start: float, end: float) -> float:
            span = end - start
            if span <= 0:
                return 0.0
            total = sum(max(0.0, min(end, e) - max(start, s)) for s, e in by_key.get(key, []))
            return total / span

        return covered


class SensorBank:
    """Несколько датчиков с разными порогами — остаётся самый согласный.

    Порог «сплошного» столбца — доля яркости самых ярких блоков ролика. У
    Rousseau блоки ровные и яркие, а дым и свечение вокруг — светлые: порог
    нужен высокий. У визуализатора Interstellar (другой автор) долгие ноты
    рисуются тусклыми полупрозрачными столбами с градиентом: при высоком
    пороге блок замечался в верхней полосе на 0.4 с позже, задержка между
    полосами не сходилась, и настоящая нота отбрасывалась — полнота 0.548.

    Подсказка, какой порог верен, есть в самом ролике: доля отсеянных событий.
    Слишком высокий порог искажает тусклые блоки, слишком низкий пропускает
    дым — и в обоих случаях проверка двумя полосами отсеивает больше. Замеры
    (порог → отсеяно → F1 против звука): Interstellar 0.6 → 22 %, 0.5 → 9.5 %,
    0.4 → 18 %, лучший F1 при 0.45–0.5; Clair de lune 0.6 → 1.7 %, 0.5 → 8 %,
    лучший F1 при 0.6; «Лунная» 0.6 → 15 %, 0.5 → 32 %, лучший при 0.6. На
    всех трёх минимум отсева совпал с лучшим F1. Ролик проходится один раз:
    все датчики получают одни и те же кадры, фон считается однажды.

    Датчик, который при своём пороге не смог даже измерить скорость, из
    выбора выпадает. Ограничение: если при высоком пороге одни блоки видны
    чисто, а другие, тусклые, не видны вовсе (не искажены, а пропали), отсев
    этого не покажет — останется высокий порог. Таких роликов в проверке
    не было.
    """

    def __init__(
        self, layout: KeyboardLayout, sample: list[Frame], fps: float, config: SensorConfig
    ) -> None:
        levels = config.solid_levels
        first = LineSensor(layout, sample, fps, config, solid_level=levels[0])
        self.sensors = [first]
        if first.background is not None:
            self.sensors += [
                LineSensor(
                    layout, sample, fps, config, solid_level=level, background=first.background
                )
                for level in levels[1:]
            ]
        self.report = first.report

    @property
    def enabled(self) -> bool:
        """Блоки найдены, и датчики ведут наблюдение."""
        return any(sensor.enabled for sensor in self.sensors)

    def process_frame(self, frame: Frame, timestamp: float) -> None:
        """Обработать кадр в BGR всеми датчиками."""
        for sensor in self.sensors:
            sensor.process_frame(frame, timestamp)

    def finish(self) -> list[NoteEvent]:
        """Ноты датчика с наименьшей долей отсеянного."""
        results = [(sensor, sensor.finish()) for sensor in self.sensors]
        working = [(sensor, notes) for sensor, notes in results if sensor.report.enabled]
        if not working:
            self.report = self.sensors[0].report
            return []
        # При равной доле остаётся более высокий порог — он первый в списке.
        chosen, notes = min(working, key=lambda item: item[0].report.rejected_share)
        candidates = tuple(
            (sensor.solid_level, sensor.report.rejected_share) for sensor, _ in working
        )
        self.report = replace(chosen.report, candidates=candidates)
        _log.info(
            "порог блоков %.2f: отсеяно %s",
            chosen.solid_level,
            ", ".join(f"{level:.2f} → {share:.1%}" for level, share in candidates),
        )
        return notes


def _split_hands(found: list[tuple[NoteEvent, npt.NDArray[np.float64]]]) -> list[NoteEvent]:
    """Разделить руки по цвету блоков — если у ролика руки различаются цветом.

    Цвета берутся из самих подтверждённых блоков, а не из профиля калибровки:
    профиль собирает самые частые цвета зоны блоков, и на тёмных роликах
    туда попадает фон и дым. Если оттенки не делятся на две заметно разные
    группы, рука остаётся неизвестной — её определит этап нотации по
    голосоведению. Две группы, сменяющие друг друга во времени, — это смена
    палитры, а не руки (`_MIN_HAND_MIXING`). Какая группа левая, решает
    регистр: партия левой руки ниже.
    """
    notes = [note for note, _ in found]
    if not found:
        return notes
    pixels = np.clip([colour for _, colour in found], 0, 255).astype(np.uint8)
    hsv = cv2.cvtColor(pixels.reshape(-1, 1, 3), cv2.COLOR_BGR2HSV).reshape(-1, 3).astype(float)
    coloured = hsv[:, 1] >= _MIN_HUE_SATURATION
    if coloured.mean() < 0.5:
        return notes  # блоки почти белые: оттенок не измерить
    groups = _hue_groups(hsv[coloured, 0])
    if groups is None:
        return notes
    onsets = np.array([note.onset for note in notes])[coloured]
    if _mixing(groups[np.argsort(onsets, kind="stable")]) < _MIN_HAND_MIXING:
        # Цвета сменяют друг друга во времени, а не звучат вместе: это смена
        # палитры по ходу ролика, а не две руки.
        _log.info("два цвета блоков сменяют друг друга во времени — это не руки")
        return notes
    hands = iter([Hand.RIGHT if group else Hand.LEFT for group in groups])
    labelled = [
        replace(note, hand=next(hands)) if colour_known else note
        for note, colour_known in zip(notes, coloured, strict=True)
    ]
    return BlockTracker._assign_hands_by_register(labelled)


def _mixing(sequence: npt.NDArray[np.bool_]) -> float:
    """Число серий одного цвета подряд относительно ожидаемого при случайном порядке."""
    first = int(sequence.sum())
    second = len(sequence) - first
    if first == 0 or second == 0:
        return 0.0
    runs = 1 + int(np.count_nonzero(sequence[1:] != sequence[:-1]))
    expected = 1 + 2 * first * second / (first + second)
    return runs / expected


def _hue_groups(hues: npt.NDArray[np.float64]) -> npt.NDArray[np.bool_] | None:
    """Две группы оттенков на круге (метки `True`/`False`) или `None`, если цвет один.

    Оттенки сортируются по кругу; две группы разделены двумя самыми большими
    промежутками. Если меньший из них не дотягивает до порога или одна группа
    слишком мала, цвет считается одним.
    """
    if len(hues) < 2:
        return None
    order = np.sort(hues)
    gaps = np.diff(np.concatenate((order, [order[0] + 180.0])))
    first, second = sorted(int(i) for i in np.argsort(gaps)[-2:])
    if min(gaps[first], gaps[second]) < _MIN_HAND_HUE_GAP:
        return None
    # Промежутки first < second режут круг на дугу order[first+1..second] и остаток.
    groups: npt.NDArray[np.bool_] = (hues >= order[first + 1]) & (hues <= order[second])
    if min(groups.mean(), 1.0 - groups.mean()) < _MIN_HAND_SHARE:
        return None
    return groups
