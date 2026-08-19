# mir/audio — анализ аудиодорожки

Второй, независимый от видео источник истины. Даёт то, чего видео дать не может: точное время атак, ноты за пределами кадра и темповую сетку.

## Роль в проекте

Аналоги делятся на две группы: видео-парсеры и аудио-транскрипторы. Ни те, ни другие не используют оба канала. Здесь аудио — не замена видео, а **перекрёстная проверка**:

| Что даёт видео | Что даёт аудио |
|---|---|
| Точная высота (позиция клавиши однозначна) | Точное время (разрешение ~5–10 мс против 33 мс) |
| Разделение рук по цвету | Ноты, не попавшие в кадр |
| Работает при плохом звуке | Работает при плохом видео |
| Длительность из длины блока | Темп и сетка долей |

## Файлы

```
audio/
├── transcriber.py   нейросетевая транскрипция (ByteDance)
├── tempo.py         beat-tracking, определение BPM и сетки долей
├── onsets.py        детекция атак звука
└── models.py        загрузка и кэширование весов модели
```

## `transcriber.py`

```python
@dataclass
class TranscriberConfig:
    device: str = "cuda"  # "cuda" | "cpu", автоопределение при старте
    chunk_seconds: float = 60.0  # длинные ролики обрабатываются кусками
    overlap_seconds: float = 2.0  # перекрытие, чтобы не резать ноты на стыке
    onset_threshold: float = 0.3
    offset_threshold: float = 0.3


class AudioTranscriber:
    def __init__(self, config: TranscriberConfig): ...

    def transcribe(self, audio_path: Path, progress: ProgressCallback | None = None) -> AudioResult:
        """
        Вход:  WAV 16 кГц моно (подготовлен в ingest)
        Выход: AudioResult
        Бросает: TranscriptionError при отсутствии весов модели
        """


@dataclass
class AudioResult:
    notes: list[NoteEvent]  # source=Source.AUDIO
    pedal_events: list[PedalEvent]  # использование правой педали
    elapsed_sec: float


@dataclass(frozen=True)
class PedalEvent:
    onset: float
    offset: float
```

**Почему кусками.** Модель ByteDance держит в памяти всю спектрограмму. Ролик на 10 минут при 16 кГц — это около 10 млн отсчётов, и на слабой машине без GPU процесс упирается в память. Разбиение на минутные куски с перекрытием 2 секунды решает проблему; ноты в зоне перекрытия дедуплицируются по (pitch, onset) с допуском.

**Педаль.** Модель ByteDance умеет её распознавать, и это редкая возможность — большинство транскрипторов педаль игнорируют. В нотном тексте она превращается в обозначения Ped./*, что заметно повышает достоверность записи для фортепианной музыки.

## `tempo.py`

```python
class TempoAnalyzer:
    def analyze(self, audio_path: Path, force_bpm: float | None = None) -> TempoMap:
        """
        Вход:  аудиофайл
        Выход: TempoMap (bpm, моменты долей, моменты сильных долей)

        Основа — librosa.beat.beat_track с onset_envelope.
        Сильные доли (downbeats) определяются отдельно: доли группируются
        по 2/3/4 и выбирается группировка, при которой на первую долю
        приходится максимум энергии атак.
        """

    def estimate_confidence(self, tempo_map: TempoMap, onset_env: np.ndarray) -> float:
        """Насколько сетка совпадает с реальными атаками.
        Низкое значение = rubato или сложный ритм, интерфейсу
        стоит предупредить пользователя и предложить задать BPM руками."""
```

**Типичная ловушка.** Beat-tracker регулярно ошибается вдвое: определяет 140 BPM вместо 70 или наоборот. Проверка — по средней длительности нот: если при найденном темпе большинство нот оказывается тридцать вторыми, темп завышен вдвое. Такая эвристика ловит большинство случаев, остальное правится вручную в интерфейсе.

## `onsets.py`

```python
def detect_onsets(audio_path: Path, hop_length: int = 256) -> npt.NDArray[np.float64]:
    """
    Вход:  аудиофайл
    Выход: массив моментов атак в секундах

    hop_length=256 при 16 кГц даёт шаг 16 мс — заметно точнее
    видеокадра при 30 fps. Значение по умолчанию в librosa (512)
    для нашей задачи грубовато.
    """


def refine_note_onsets(
    notes: list[NoteEvent], onsets: npt.NDArray[np.float64], max_shift: float = 0.05
) -> list[NoteEvent]:
    """Подтянуть время нот к ближайшей обнаруженной атаке,
    если та не дальше max_shift. Это чистит дрожание,
    накопившееся при округлении к кадрам."""
```

## `models.py`

```python
MODEL_URL = "https://zenodo.org/record/4034264/files/CRNN_note_F1=0.9677_pedal_F1=0.9186.pth"


class ModelRegistry:
    def ensure_downloaded(self, progress=None) -> Path:
        """Скачать веса при первом запуске (~170 МБ), проверить sha256,
        положить в ~/.mir_cache/models/.

        Осознанное решение: не класть веса в дистрибутив.
        Иначе установщик распухает, а большинству пользователей
        хватит одной загрузки."""

    def select_device(self) -> str:
        """cuda если доступна и хватает памяти, иначе cpu.
        На cpu транскрипция минутного фрагмента идёт примерно
        в 3–5 раз дольше — предупредить пользователя в интерфейсе."""
```

## Контракт этапа

```python
def analyze_audio(media: MediaBundle, config: TranscriberConfig,
                  progress=None) -> tuple[AudioResult, TempoMap]
```

## Что тестировать

- транскрипция синтезированного из MIDI аудио — сравнение с исходным MIDI даёт готовый ground truth без ручной разметки
- склейка кусков: нота, попавшая на границу chunk, не должна раздваиваться
- `TempoAnalyzer` на метрономе с известным BPM
- проверка на ошибку вдвое: подать материал в 60 и 120 BPM
- `refine_note_onsets` не двигает ноты дальше `max_shift`
