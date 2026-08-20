# mir/vision — анализ видеоряда

Тонкий Python-слой над C++ модулем `mir_core`. Сама тяжёлая работа делается в `core/`, здесь — вызов, конвертация типов и то, что на C++ писать неудобно.

## Зачем нужен слой, если есть C++

Три причины:

1. **Граница типов.** C++ отдаёт свои структуры, дальше по конвейеру ходят Python-датаклассы из `common`. Конвертация должна быть в одном месте, а не размазана по коду.
2. **Изоляция от нативного модуля.** Если `mir_core` не собран (типичная ситуация при первом запуске на новой машине), нужно выдать понятную ошибку, а не `ImportError: DLL load failed`.
3. **Отладочная визуализация.** Рисовать разметку поверх кадров удобнее в Python — это не в горячем цикле, и matplotlib под рукой.

## Файлы

```
vision/
├── analyzer.py     основной вызов C++ и конвертация результата
├── adapters.py     mir_core.* → mir.common.*
├── debug.py        отрисовка разметки для отладки и скриншотов в диплом
└── fallback.py     чистый Python-путь на случай отсутствия mir_core
```

## `analyzer.py`

```python
class VisionAnalyzer:
    def __init__(self, use_native: bool = True):
        """use_native=False — принудительно медленный Python-путь.
        Нужно для сравнения реализаций в экспериментальной главе."""

    def analyze(self, media: MediaBundle, progress: ProgressCallback | None = None) -> VisionResult:
        """
        Вход:  MediaBundle (нужен video_path и fps)
        Выход: VisionResult
        Бросает: KeyboardNotFoundError — ролик не является piano visualizer
        """


@dataclass
class VisionResult:
    notes: list[NoteEvent]  # source=Source.VIDEO
    layout: KeyboardLayout
    profile: VisualizerProfile
    frames_processed: int
    elapsed_sec: float  # для таблиц производительности в дипломе
```

Внутри `analyze` — по сути три строчки:

```python
raw = mir_core.VideoAnalyzer().analyze(str(media.video_path), progress)
notes = [adapt_note(n) for n in raw.notes]
layout = adapt_layout(raw.layout)
```

Вся сложность — в C++ (см. `core/README.md`).

## `adapters.py`

```python
def adapt_note(raw: "mir_core.NoteEventRaw") -> NoteEvent:
    """Ключевой момент: velocity из C++ приходит нормализованным (0..1),
    оценённым по яркости подсветки. Здесь масштабируется в MIDI-шкалу 0..127.

    Формула нелинейная — восприятие громкости логарифмическое,
    а яркость подсветки в визуализаторах обычно линейна по velocity:
        midi_velocity = round(127 * raw.velocity ** 0.7)
    Показатель 0.7 подобран эмпирически, вынести в конфиг."""


def adapt_layout(raw: "mir_core.KeyboardLayout") -> KeyboardLayout: ...
def adapt_profile(raw: "mir_core.VisualizerProfile") -> VisualizerProfile: ...
```

Адаптеры — единственное место, где встречаются нативные и Python-типы. Всё остальное работает только с `common`.

## `debug.py`

Не участвует в основном сценарии, но окупается многократно при отладке и даёт готовые иллюстрации для пояснительной записки.

```python
def draw_layout(frame: np.ndarray, layout: KeyboardLayout) -> np.ndarray:
    """Нарисовать поверх кадра границы клавиш и подписи нот.
    Сразу видно, если разметка съехала на октаву."""


def draw_blocks(frame: np.ndarray, blocks: list[Block]) -> np.ndarray:
    """Рамки вокруг найденных блоков с id и определённой нотой."""


def export_piano_roll(notes: list[NoteEvent], path: Path) -> None:
    """Piano-roll найденных нот в PNG. Удобно класть рядом
    с эталоном и сравнивать глазами."""


def dump_frames(
    media: MediaBundle, timestamps: list[float], out_dir: Path, layout: KeyboardLayout | None = None
) -> None:
    """Сохранить кадры в конкретные моменты с наложенной разметкой.
    Отсюда берутся картинки для диплома."""
```

## `fallback.py`

Чистая Python-реализация того же алгоритма на OpenCV-Python. Медленнее в разы, но:

- позволяет запустить проект без компиляции C++ (важно для проверяющего),
- служит эталоном при отладке C++: если результаты расходятся, ошибка в оптимизациях,
- даёт материал для таблицы «C++ против Python» в экспериментальной главе — это прямое обоснование выбора C++ в работе.

```python
class FallbackAnalyzer:
    """Тот же интерфейс, что у VisionAnalyzer.
    Векторизация через numpy там, где возможно, но покадровый
    цикл всё равно остаётся узким местом."""

    def analyze(
        self, media: MediaBundle, progress: ProgressCallback | None = None
    ) -> VisionResult: ...
```

## Контракт этапа

```python
def analyze_video(media: MediaBundle, progress=None) -> VisionResult
```

Выход этого модуля — события с `source=Source.VIDEO`. Все они помечены `confidence` из C++ (качество детекции подсветки). Дальше `fusion` будет пересматривать эти оценки с учётом аудиоканала.

## Известные ограничения

Стоит зафиксировать честно, для главы про результаты:

- ролики, где клавиатура снята под углом (не фронтально), не поддерживаются — потребовалась бы гомография;
- если поверх блоков идут крупные эффекты (снег, вспышки на весь экран), трекинг деградирует;
- при полностью статичной картинке (клавиатура нарисована, но не подсвечивается) работает только трекинг блоков.
