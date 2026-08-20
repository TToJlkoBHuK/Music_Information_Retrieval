# mir/export — выгрузка результата

Последний этап конвейера. Превращает `Transcription` в файлы, которые пользователь получает на руки.

## Три формата — три сценария

| Формат | Кому и зачем |
|---|---|
| **PDF** | Основной результат: распечатать и поставить на пюпитр |
| **MusicXML** | Открыть в MuseScore/Sibelius и доправить руками — автоматика не идеальна |
| **MIDI** | Загрузить в DAW, прослушать, разучивать по частям |

MusicXML принципиально важен: он честно признаёт, что распознавание может ошибаться, и даёт пользователю возможность исправить. Закрытые сервисы часто отдают только PDF, и поправить в нём ничего нельзя.

## Файлы

```
export/
├── midi_writer.py       Transcription → .mid
├── musicxml_writer.py   Transcription → .musicxml
├── pdf_renderer.py      .musicxml → .pdf
├── engines.py           поиск MuseScore / LilyPond в системе
└── metadata.py          заголовок, автор, источник
```

## `midi_writer.py`

```python
def write_midi(transcription: Transcription, path: Path, ppq: int = 480) -> Path:
    """
    Вход:  Transcription
    Выход: путь к .mid

    Структура файла (Type 1, многодорожечный):
      Трек 0: мета-события — темп, размер, тональность, название
      Трек 1: правая рука (канал 0)
      Трек 2: левая рука (канал 1)
      Педаль — CC64 в треке 1, если распознана

    ppq=480 — общепринятое значение, при нём триоли и шестнадцатые
    выражаются целыми числами тиков без округления
    (480 / 3 = 160, 480 / 4 = 120).
    """
```

Разделение рук по трекам, а не по каналам — так файл корректно открывается и в DAW, и в нотных редакторах.

## `musicxml_writer.py`

```python
def write_musicxml(transcription: Transcription, path: Path, compressed: bool = False) -> Path:
    """
    Вход:  Transcription
    Выход: путь к .musicxml (или .mxl при compressed=True)

    Построение через music21:
      Score
      ├── Metadata (название, источник, дата распознавания)
      └── PartStaff x2, объединённые в StaffGroup (фигурная скобка)
          ├── Part «Правая рука»: TrebleClef
          └── Part «Левая рука»: BassClef
              каждый: KeySignature, TimeSignature,
                      Measure[] с Note / Rest / Chord

    Ключевые моменты:
      - одновременные ноты одной руки собираются в Chord,
        иначе редактор рисует их как отдельные голоса;
      - знаки альтерации не проставляются вручную — music21
        выводит их из KeySignature, это и даёт чистый текст;
      - ноты, разрезанные тактовой чертой, связываются Tie;
      - педаль экспортируется как PedalMark (Ped. / *).
    """


def _validate(score: "music21.stream.Score") -> list[str]:
    """Проверка перед записью. Возвращает список проблем:
      - такты с неверной суммой длительностей
      - ноты вне диапазона фортепиано
      - пересекающиеся ноты одной высоты на одном стане
    Такие ошибки MuseScore проглотит молча и нарисует мусор,
    поэтому ловим их у себя."""
```

## `pdf_renderer.py`

```python
@dataclass
class RenderOptions:
    page_size: str = "A4"
    staff_size_mm: float = 7.0  # высота нотоносца
    measures_per_line: int | None = None  # None = автоматика движка
    show_measure_numbers: bool = True
    title: str | None = None
    subtitle: str | None = None  # сюда — ссылку на исходный ролик


def render_pdf(
    musicxml_path: Path, out_path: Path, options: RenderOptions, engine: Engine | None = None
) -> Path:
    """
    Вход:  MusicXML
    Выход: PDF

    Вёрстка не пишется своими руками — это отдельная большая задача
    (расстановка нот по ширине, переносы строк, разрешение коллизий
    штилей и лиг). Используется готовый движок.

    MuseScore CLI:
        mscore -o out.pdf in.musicxml
    LilyPond (через промежуточную конвертацию):
        musicxml2ly in.musicxml -o out.ly && lilypond out.ly

    MuseScore — основной вариант: проще, быстрее, качество вёрстки
    достаточное. LilyPond — запасной: вёрстка красивее, но цепочка
    длиннее и настройка тоньше.
    """
```

## `engines.py`

```python
class Engine(StrEnum):
    MUSESCORE = "musescore"
    LILYPOND = "lilypond"


def find_engine(preferred: Engine | None = None) -> tuple[Engine, Path]:
    """
    Ищет движок вёрстки:
      1. Путь из настроек приложения
      2. Переменная окружения MIR_MUSESCORE_PATH
      3. Стандартные места установки:
         Windows: C:\\Program Files\\MuseScore 4\\bin\\MuseScore4.exe
         Linux:   /usr/bin/mscore, flatpak
         macOS:   /Applications/MuseScore 4.app/Contents/MacOS/mscore
      4. PATH

    Бросает ExportError с инструкцией по установке,
    если ничего не найдено.
    """
```

**Вопрос распространения.** Тянуть MuseScore в установщик — это +400 МБ и вопросы с лицензией (GPL). Решение: приложение работает без него, но кнопка «PDF» неактивна с подсказкой «Установите MuseScore, чтобы получать готовые ноты в PDF». MIDI и MusicXML при этом доступны всегда. Уточнить у научрука, приемлемо ли это для защиты.

## `metadata.py`

```python
def build_metadata(transcription: Transcription, source_url: str | None) -> Metadata:
    """Заголовок = название ролика (из yt-dlp), очищенное от мусора
    вроде «(Piano Tutorial)», «[Synthesia]», «| Easy».

    В подзаголовок — ссылка на источник и пометка
    «Автоматическая транскрипция», чтобы не выдавать
    машинный разбор за авторскую редакцию нот."""
```

Пометка об автоматическом происхождении — вопрос честности и заодно снимает часть претензий по авторским правам.

## Контракт этапа

```python
def export_all(transcription: Transcription, out_dir: Path,
               options: RenderOptions) -> ExportResult

@dataclass
class ExportResult:
    midi_path: Path
    musicxml_path: Path
    pdf_path: Path | None      # None, если движок вёрстки не найден
    warnings: list[str]
```

## Что тестировать

- **Круговой тест**: `Transcription → MIDI → чтение обратно` должно дать те же ноты. Ловит ошибки в тиках и округлении.
- Сгенерированный MusicXML открывается в MuseScore без предупреждений — проверяется запуском `mscore -o /dev/null` и разбором stderr.
- `_validate` ловит подсунутый такт с неверной суммой длительностей.
- Отсутствие движка не роняет пайплайн: MIDI и MusicXML сохраняются, PDF отсутствует, в `warnings` понятное сообщение.
- Очистка названия: «Chopin Nocturne op.9 no.2 (Piano Tutorial) [Synthesia]» → «Chopin Nocturne op.9 no.2».
