# mir/common — модель данных и общие утилиты

Фундамент проекта. Здесь описано, чем модули обмениваются между собой. Читать этот файл нужно первым: если понятны типы, понятен и весь конвейер.

Правило: **`common` ни от кого не зависит**. Никаких импортов из `vision`, `audio`, `notation`. Только стандартная библиотека, numpy и dataclasses.

## Файлы

```
common/
├── types.py       модели данных (главный файл)
├── enums.py       перечисления
├── errors.py      иерархия исключений
├── timeutil.py    перевод времени: кадры ↔ секунды ↔ доли ↔ тики MIDI
└── logging.py     настройка логирования
```

## `enums.py`

```python
class Hand(IntEnum):
    LEFT = 0
    RIGHT = 1
    UNKNOWN = 2

class Source(IntFlag):
    """Откуда пришло событие. Флаги, чтобы VIDEO | AUDIO давало BOTH."""
    VIDEO = 1
    AUDIO = 2
    BOTH = VIDEO | AUDIO

class Clef(IntEnum):
    TREBLE = 0   # скрипичный
    BASS = 1     # басовый
```

## `types.py` — главные структуры

### `NoteEvent` — атом всей системы

Через него проходит вся информация от распознавания до экспорта.

```python
@dataclass(frozen=True, slots=True)
class NoteEvent:
    pitch: int            # MIDI-номер: 21 (ля субконтроктавы) .. 108 (до 5-й октавы)
    onset: float          # начало звучания, секунды от старта ролика
    offset: float         # конец звучания, секунды
    velocity: int         # сила нажатия, 0..127 (шкала MIDI)
    hand: Hand = Hand.UNKNOWN
    source: Source = Source.VIDEO
    confidence: float = 1.0   # 0..1, насколько уверены в событии

    @property
    def duration(self) -> float:
        return self.offset - self.onset

    def __post_init__(self) -> None:
        if not 21 <= self.pitch <= 108:
            raise ValueError(f"pitch {self.pitch} вне диапазона фортепиано")
        if self.offset <= self.onset:
            raise ValueError("offset должен быть строго больше onset")
```

`frozen=True` осознанно: события не мутируются на месте. Каждый этап конвейера создаёт новый список — это сильно упрощает отладку (всегда можно сравнить «до» и «после») и делает невозможными скрытые побочные эффекты.

`slots=True` — событий бывают тысячи, экономия памяти заметна.

Обратите внимание: `velocity` уже в шкале MIDI 0..127, а не 0..1. Конвертация из нормализованной оценки C++-модуля происходит на границе, в `vision`.

### `KeyboardLayout` — разметка клавиатуры

Python-зеркало структуры из `core/include/mir_core/types.hpp`.

```python
@dataclass(frozen=True)
class KeySlot:
    pitch: int
    x_min: int
    x_max: int
    is_black: bool

@dataclass(frozen=True)
class KeyboardLayout:
    bbox: tuple[int, int, int, int]   # x, y, width, height области клавиатуры
    keys: tuple[KeySlot, ...]
    lowest_pitch: int
    highest_pitch: int
    is_cropped: bool                  # видны не все 88 клавиш
    confidence: float

    def pitch_at(self, x: int) -> int | None:
        """По горизонтальной координате вернуть номер ноты. Бинарный поиск."""

    @property
    def visible_range(self) -> tuple[int, int]:
        return self.lowest_pitch, self.highest_pitch
```

`is_cropped` важен дальше по конвейеру: если клавиатура обрезана, `fusion` должен доверять аудиоканалу для нот за пределами видимого диапазона, а не отбрасывать их как ложные.

### `TempoMap` — временна́я сетка

```python
@dataclass(frozen=True)
class TempoMap:
    bpm: float                  # средний темп
    beats: npt.NDArray[np.float64]      # моменты долей, секунды
    downbeats: npt.NDArray[np.float64]  # моменты сильных долей (начала тактов)
    confidence: float

    def seconds_to_beats(self, t: float) -> float:
        """Секунды → позиция в долях. Интерполяция по сетке beats,
        поэтому корректно работает при переменном темпе."""

    def beats_to_seconds(self, b: float) -> float: ...

    def nearest_grid_point(self, t: float, subdivision: int = 4) -> float:
        """Ближайший узел сетки. subdivision=4 → шестнадцатые."""
```

Метод `seconds_to_beats` через интерполяцию, а не через деление на BPM: живое исполнение с rubato имеет плавающий темп, и линейная формула даёт накопительную ошибку к концу произведения.

### `Transcription` — результат распознавания целиком

```python
@dataclass
class Transcription:
    notes: list[NoteEvent]
    tempo: TempoMap
    key_signature: str | None = None       # "C major", "A minor", ...
    time_signature: tuple[int, int] | None = None   # (4, 4)
    measures: list[Measure] = field(default_factory=list)
    title: str | None = None
    source_url: str | None = None

@dataclass
class Measure:
    """Один такт. Заполняется в notation."""
    index: int
    start_beat: float
    end_beat: float
    treble: list[NoteEvent | Rest]   # верхний стан
    bass: list[NoteEvent | Rest]     # нижний стан

@dataclass(frozen=True)
class Rest:
    """Пауза. В MIDI её нет, но в нотном тексте нужна явно."""
    onset: float
    duration: float
    clef: Clef
```

`Transcription` — это то, что кэшируется между `run` и `rebuild`. Она содержит всё нужное, чтобы пересобрать ноты без повторного разбора видео.

### `MediaBundle` — результат загрузки

```python
@dataclass(frozen=True)
class MediaBundle:
    video_path: Path
    audio_path: Path
    fps: float
    duration: float
    width: int
    height: int
    title: str | None
    source_url: str | None
```

### `QualityReport` — что показать пользователю

```python
@dataclass
class QualityReport:
    keyboard_confidence: float
    keyboard_cropped: bool
    notes_from_video_only: int    # не подтверждены аудио
    notes_from_audio_only: int    # не подтверждены видео
    notes_confirmed: int          # подтверждены обоими каналами
    detected_bpm: float
    detected_key: str
    detected_time_signature: str
    av_offset_ms: float           # найденный рассинхрон
    warnings: list[str]           # человекочитаемые предупреждения
```

Отчёт нужен и интерфейсу (показать, чему можно верить), и экспериментальной главе диплома — из него собираются таблицы.

## `timeutil.py`

Четыре системы отсчёта времени, между которыми постоянно нужны переводы. Собраны в одном месте, чтобы не плодить магические формулы по всему коду.

```python
def frame_to_seconds(frame_idx: int, fps: float) -> float: ...
def seconds_to_frame(t: float, fps: float) -> int: ...
def seconds_to_ticks(t: float, bpm: float, ppq: int = 480) -> int:
    """Секунды → тики MIDI. ppq — тиков на четверть."""
def ticks_to_seconds(ticks: int, bpm: float, ppq: int = 480) -> float: ...
def beats_to_note_value(beats: float, beat_unit: int = 4) -> Fraction:
    """Длительность в долях → нотная длительность (1 = целая, 1/4 = четвертная)."""
```

## `errors.py`

```python
class MirError(Exception):
    user_message: str = "Произошла ошибка"

    def __init__(self, technical: str, user_message: str | None = None):
        super().__init__(technical)
        if user_message:
            self.user_message = user_message
```

Разделение технического текста (в лог) и пользовательского (в диалог Qt) — чтобы интерфейс не показывал музыканту traceback про numpy.

## Соглашение о времени

Единица времени в проекте — **секунда от начала ролика, float**. Кадры и тики MIDI используются только внутри модулей, наружу не протекают. Все `onset` и `offset` в `NoteEvent` — секунды. Это сознательное решение: иначе на границах модулей постоянно путаница, что тут за единицы.

## При изменении типов

`types.py` и `core/include/mir_core/types.hpp` описывают одни и те же структуры. Менять нужно синхронно, иначе биндинги сломаются молча — pybind11 отдаст объект с недостающими полями. Стоит завести тест `tests/unit/test_type_parity.py`, который сверяет набор полей.
