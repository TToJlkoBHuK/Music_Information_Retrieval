# config — конфигурация и пресеты

Все настраиваемые числа проекта в одном месте. Правило простое: **никаких магических констант в коде** — если значение подобрано эмпирически, ему место здесь.

Это не педантизм. В экспериментальной главе диплома придётся объяснять, откуда взялся каждый порог, и удобно, когда все они собраны и подписаны, а не разбросаны по десяти файлам.

## Структура

```
config/
├── default.toml       значения по умолчанию
├── presets/           профили под конкретные визуализаторы
│   ├── synthesia.toml
│   ├── pianoroll_dark.toml
│   └── generic.toml
└── schema.py          загрузка и валидация
```

Формат TOML: читается человеком, поддерживается стандартной библиотекой Python 3.11 (`tomllib`), в отличие от YAML не требует зависимостей и не преподносит сюрпризов с типами.

## `default.toml`

```toml
[ingest]
max_height = 1080
prefer_fps = 60
timeout_sec = 300
cache_dir = "~/.mir_cache"

[vision.keyboard]
sample_frames = 90          # кадров для медианы при детекции клавиатуры
min_key_width_px = 4.0
black_key_height_ratio = 0.6

[vision.tracker]
on_threshold = 0.25         # порог включения подсветки
off_threshold = 0.15        # порог выключения (ниже — это гистерезис)
min_frames_on = 2           # антидребезг
velocity_gamma = 0.7        # яркость → velocity, нелинейно

[audio]
sample_rate = 16000         # требование модели ByteDance
chunk_seconds = 60.0
overlap_seconds = 2.0
onset_threshold = 0.3
offset_threshold = 0.3
hop_length = 256            # 16 мс при 16 кГц

[fusion]
onset_tolerance = 0.08      # допуск сопоставления, секунды
max_av_offset = 1.0         # предел поиска рассинхрона
min_offset_confidence = 0.3 # ниже — сдвиг не применяем
duration_weight = 0.3
octave_artifact_ratio = 0.8 # порог отсева обертонов

[notation]
quantize_strength = 0.7
max_subdivision = 16
allow_triplets = true
min_duration_beats = 0.0625
hand_split_pitch = 60       # до первой октавы — граница по умолчанию
dynamics_window_beats = 4.0

[export]
ppq = 480
page_size = "A4"
staff_size_mm = 7.0
show_measure_numbers = true
```

## Пороги, требующие обоснования

Часть значений подобрана эмпирически. Для каждого стоит держать наготове ответ:

| Параметр | Почему такое значение |
|---|---|
| `on_threshold` / `off_threshold` | Разные пороги дают гистерезис. При равных на границе возникает дребезг: одна нота распадается на серию коротких. Разрыв 0.10 подобран так, чтобы шум сжатия на 480p не пробивал его. |
| `velocity_gamma = 0.7` | Восприятие громкости логарифмическое, яркость подсветки в визуализаторах обычно линейна по velocity. Степень 0.7 — компромисс, подобранный сравнением с эталонными MIDI. |
| `onset_tolerance = 0.08` | Чуть больше двух кадров при 30 fps (66 мс) плюс запас. Меньше — теряются совпадения из-за дискретизации по кадрам, больше — начинают ошибочно сопоставляться соседние ноты в быстрых пассажах. |
| `quantize_strength = 0.7` | Найдено перебором на тестовом наборе: максимум читаемости при сохранении живости. Вынесено в интерфейс ползунком, так как оптимум зависит от стиля произведения. |
| `hop_length = 256` | Даёт 16 мс — вдвое точнее кадра при 30 fps. Значение по умолчанию в librosa (512) для задачи грубовато. |
| `chunk_seconds = 60` | Компромисс между памятью и накладными расходами на склейку. При 60 с модель ByteDance укладывается в 4 ГБ на CPU. |
| `octave_artifact_ratio = 0.8` | Обертон обычно короче породившей его ноты. Порог отсекает кандидатов длиннее 80 % от основной. |

Эта таблица — заготовка раздела «Обоснование параметров» в пояснительной записке. Заполнять по мере подбора, с указанием, на каких данных значение проверялось.

## Пресеты (`presets/`)

Автокалибровка (`core/calibration.hpp`) определяет параметры визуализатора сама, но на сложных роликах может ошибиться. Пресет — способ подсказать ей стартовые значения:

```toml
# presets/synthesia.toml
name = "Synthesia (стандартная тема)"

[colors]
background_hsv = [0, 0, 20]
right_hand_hsv = [120, 200, 200]   # зелёный
left_hand_hsv = [210, 200, 200]    # синий
tolerance = 25

[geometry]
keyboard_position = "bottom"
fall_direction = "down"
```

В интерфейсе — выпадающий список «Тип видео» со значением «Определить автоматически» по умолчанию.

## `schema.py`

```python
@dataclass
class MirConfig:
    ingest: IngestConfig
    vision: VisionConfig
    audio: AudioConfig
    fusion: FusionConfig
    notation: NotationConfig
    export: ExportConfig

def load_config(path: Path | None = None,
                overrides: dict | None = None) -> MirConfig:
    """
    Приоритет источников, от низшего к высшему:
      1. config/default.toml
      2. пользовательский файл (~/.mir/config.toml)
      3. переменные окружения MIR_*
      4. overrides — то, что пришло из интерфейса

    Валидация при загрузке: проверка диапазонов и понятная
    ошибка вместо падения где-то в глубине конвейера.
    """

def validate(config: MirConfig) -> list[str]:
    """Проверки, которые нельзя выразить типами:
      - off_threshold < on_threshold (иначе гистерезис не работает)
      - overlap_seconds < chunk_seconds
      - 0 <= quantize_strength <= 1
      - hand_split_pitch в диапазоне 21..108
    """
```
