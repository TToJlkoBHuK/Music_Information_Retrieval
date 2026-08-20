# mir — Python-пакет с логикой конвейера

Основной пакет проекта. Название — от Music Information Retrieval, области, к которой относится задача.

## Принцип организации

Пакет разбит по **этапам конвейера**, а не по типам файлов. Каждый подмодуль — самостоятельный шаг: принимает данные предыдущего, отдаёт данные следующему, ничего не знает о том, кто его вызывает.

```
mir/
├── common/      модель данных и утилиты, общие для всех  ← начинать чтение отсюда
├── ingest/      ссылка/файл → видео + аудио
├── vision/      видео → события нот (обёртка над C++ core)
├── audio/       аудио → события нот + темп
├── fusion/      два потока событий → один выверенный
├── notation/    события → музыкально осмысленная партитура
├── export/      партитура → MIDI / MusicXML / PDF
├── pipeline.py  сборка всех этапов в один вызов
└── config.py    загрузка настроек и пресетов
```

Зависимости направлены строго вниз по списку. `notation` может импортировать `common`, но не `vision`. Проверять это стоит линтером (`import-linter`) — на защите легко получить вопрос про связность модулей.

## Точка входа

```python
from mir.pipeline import Pipeline, PipelineConfig

pipeline = Pipeline(PipelineConfig())
result = pipeline.run(
    source="https://www.youtube.com/watch?v=...",
    output_dir="./output",
    progress=lambda stage, pct: print(f"{stage}: {pct:.0%}"),
)
# result.pdf_path, result.midi_path, result.musicxml_path
```

### `pipeline.py`

```python
@dataclass
class PipelineConfig:
    proxy: str | None = None  # для yt-dlp, актуально из-за VPN
    quantize_strength: float = 0.7  # 0 = не квантовать, 1 = жёстко в сетку
    force_bpm: float | None = None  # ручное переопределение темпа
    force_key: str | None = None  # например "D major"
    force_time_signature: str | None = None
    use_gpu: bool = True
    keep_intermediate: bool = False  # оставлять video.mp4 / audio.wav


@dataclass
class PipelineResult:
    transcription: Transcription
    midi_path: Path
    musicxml_path: Path
    pdf_path: Path
    report: QualityReport  # что определилось автоматически и с какой уверенностью


class Pipeline:
    def __init__(self, config: PipelineConfig): ...

    def run(
        self, source: str | Path, output_dir: Path, progress: ProgressCallback | None = None
    ) -> PipelineResult:
        """Полный цикл. source — URL или путь к локальному файлу."""

    def rebuild(
        self, transcription: Transcription, config: PipelineConfig, output_dir: Path
    ) -> PipelineResult:
        """Пересборка нот из уже готовой транскрипции с новыми настройками.
        Нужна для интерфейса: пользователь двигает ползунок квантизации
        или правит тональность — видео заново не разбирается."""
```

Разделение `run` / `rebuild` — не мелочь, а требование интерфейса. Полный разбор ролика идёт минуты, а подбор параметров должен быть мгновенным. Поэтому результат `fusion` кэшируется, и `rebuild` начинает с этапа `notation`.

## Этапы и их контракты

| Этап | Вход | Выход |
|---|---|---|
| `ingest` | URL или путь | `MediaBundle` (пути к видео и аудио, fps, длительность) |
| `vision` | `MediaBundle` | `list[NoteEvent]` (source=VIDEO), `KeyboardLayout` |
| `audio` | `MediaBundle` | `list[NoteEvent]` (source=AUDIO), `TempoMap` |
| `fusion` | два списка + `TempoMap` | `list[NoteEvent]` (выверенный, source=BOTH/VIDEO/AUDIO) |
| `notation` | события + `TempoMap` | `Transcription` (с тональностью, размером, тактами, паузами, руками) |
| `export` | `Transcription` | файлы MIDI / MusicXML / PDF |

Все типы описаны в `common/README.md`.

## Обработка ошибок

Своя иерархия исключений в `common/errors.py`, чтобы интерфейс мог показать человеческое сообщение вместо стектрейса:

```python
class MirError(Exception):
    """Базовое. Поле .user_message — текст для показа пользователю (по-русски)."""


class DownloadError(MirError): ...  # видео недоступно, нет сети, нужен VPN


class KeyboardNotFoundError(MirError): ...  # не нашли клавиатуру — не тот формат ролика


class TranscriptionError(MirError): ...


class ExportError(MirError): ...  # не найден MuseScore / LilyPond
```

## Стиль и инструменты

- Python 3.10+ (нужен `match` и `X | None` в аннотациях)
- Типизация обязательна, проверка через `mypy --strict`
- Форматирование `ruff format`, линт `ruff`
- Датаклассы для моделей данных, `frozen=True` где возможно — события нот не должны меняться на месте
- Логирование через `logging`, не `print`: интерфейс перехватывает лог и показывает в панели прогресса
