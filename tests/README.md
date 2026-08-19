# tests — тестирование

Тесты здесь нужны не для галочки: половина модулей проекта содержит алгоритмы, которые ломаются молча. Неверная тональность или съехавшая на октаву разметка не вызовут исключения — они просто дадут неправильные ноты.

## Структура

```
tests/
├── unit/          отдельные функции, быстро, без файлов на диске
├── integration/   связки модулей и полный конвейер
├── fixtures/      генераторы синтетических данных
└── conftest.py    общие фикстуры pytest
```

## Главный приём: синтетические данные

Ручная разметка нот для эталона — это часы работы на каждый ролик. Вместо неё — обратный ход: **берём известный MIDI и генерируем из него и видео, и аудио**.

```
эталонный MIDI ──┬──► рендер видео (свой генератор Synthesia-подобной картинки)
                 └──► синтез аудио (FluidSynth + SoundFont рояля)
```

Тогда ground truth известен точно, бесплатно и в любом количестве. Можно варьировать что угодно: темп, тональность, плотность фактуры, разрешение, fps, качество сжатия.

### `fixtures/synth.py`

```python
def render_visualizer_video(
    midi_path: Path,
    out_path: Path,
    fps: int = 30,
    resolution: tuple[int, int] = (1920, 1080),
    color_scheme: ColorScheme = ColorScheme.DEFAULT,
    crop_keyboard: tuple[int, int] | None = None,
    bitrate_kbps: int | None = None,
) -> Path:
    """Сгенерировать piano-visualizer-видео из MIDI.

    crop_keyboard=(48, 84) — сымитировать обрезанную клавиатуру.
    bitrate_kbps=500 — сымитировать плохое качество.
    Оба параметра нужны для стресс-тестов."""


def synthesize_audio(
    midi_path: Path, out_path: Path, soundfont: Path, reverb: float = 0.0, noise_db: float = -60.0
) -> Path:
    """Синтез аудио из того же MIDI. Реверберация и шум —
    чтобы проверить устойчивость аудиоканала."""


def midi_to_note_events(midi_path: Path) -> list[NoteEvent]:
    """Эталон в формате проекта — с ним сравниваются результаты."""
```

Генератор видео стоит написать одним из первых: он окупится на всех этапах отладки и даст воспроизводимые цифры для экспериментальной главы.

## Что покрывать в `unit/`

| Файл | Что проверяет |
|---|---|
| `test_types.py` | валидация `NoteEvent`, границы pitch, offset > onset |
| `test_timeutil.py` | переводы кадры ↔ секунды ↔ тики, отсутствие накопления ошибки |
| `test_type_parity.py` | набор полей в Python-типах и C++-структурах совпадает |
| `test_sources.py` | `normalize_url`, `detect_platform` |
| `test_quantizer.py` | известный ритм → ожидаемые длительности; триоли не схлопываются |
| `test_key_detector.py` | фрагменты в заведомых тональностях |
| `test_meter.py` | размер и затакт |
| `test_rests.py` | **инвариант: сумма длительностей в такте равна размеру** |
| `test_staves.py` | перекрещивание рук |
| `test_matcher.py` | аккорд из 4 нот не схлопывается в одну |
| `test_aligner.py` | искусственный сдвиг находится с точностью до 10 мс |
| `test_resolver.py` | обертон на октаву отсеивается; нота вне обрезанного кадра сохраняется |

## Что покрывать в `integration/`

| Файл | Что проверяет |
|---|---|
| `test_pipeline_synthetic.py` | полный прогон на синтетическом ролике, F1 выше порога |
| `test_fusion_gain.py` | **слияние даёт F1 выше, чем каждый канал по отдельности** |
| `test_export_roundtrip.py` | `Transcription → MIDI → чтение` даёт те же ноты |
| `test_musicxml_valid.py` | сгенерированный MusicXML открывается MuseScore без предупреждений |
| `test_no_engine.py` | без MuseScore пайплайн не падает, MIDI и MusicXML сохраняются |
| `test_rebuild.py` | `rebuild` с другими параметрами не трогает видео и укладывается в секунды |

`test_fusion_gain.py` — самый ценный тест в проекте. Он проверяет ровно то, что заявлено новизной работы, и его результат идёт прямо в диплом графиком.

## Метрики (`scripts/evaluate.py`)

```python
@dataclass
class TranscriptionMetrics:
    note_precision: float
    note_recall: float
    note_f1: float
    onset_mae_ms: float  # средняя ошибка по времени атаки
    duration_mae_beats: float
    key_correct: bool
    time_signature_correct: bool
    tempo_error_percent: float
    hand_accuracy: float  # доля нот в правильной руке


def evaluate(
    predicted: list[NoteEvent], reference: list[NoteEvent], onset_tolerance: float = 0.05
) -> TranscriptionMetrics:
    """Допуск 50 мс — стандарт в MIR (используется в MIREX
    и в mir_eval), берём его, чтобы цифры были сопоставимы
    с опубликованными результатами других работ."""
```

Считать через `mir_eval.transcription`, а не своими руками: так результаты сравнимы с литературой, и на защите не будет вопросов к методике.

## Запуск

```bash
pytest tests/unit -q                    # быстрые, при каждом изменении
pytest tests/integration -q             # медленные, перед коммитом
pytest --cov=mir --cov-report=html      # покрытие
```

Тесты, требующие GPU или установленного MuseScore, помечаются:

```python
@pytest.mark.requires_gpu
@pytest.mark.requires_musescore
```

и пропускаются, если условие не выполняется. Иначе проверяющий не сможет запустить набор у себя.

## C++ тесты

Живут отдельно, в `core/tests/`, запускаются через CTest — см. `core/README.md`.
