# Как смотреть тесты

Тесты — не только проверка кода, но и материал для экспериментальной главы: они показывают, что именно проверено и какие инварианты гарантированы.

## Подробный вывод (по умолчанию)

```bash
docker compose run --rm mir pytest tests
```

Каждый тест виден отдельной строкой:

```
tests/unit/test_sources.py::TestNormalizeUrl::test_strips_tracking_and_timestamp PASSED [ 60%]
tests/unit/test_sources.py::TestNormalizeUrl::test_same_video_same_key        PASSED [ 61%]
tests/unit/test_types.py::TestTimeUtil::test_fractional_fps_not_rounded       PASSED [ 78%]
tests/unit/test_types.py::TestTimeUtil::test_triplets_sum_exactly             PASSED [ 79%]
```

Имена читаются как утверждения: `test_fractional_fps_not_rounded` — «дробный fps не округляется».

## Краткий вывод

Если нужен только итог:

```bash
docker compose run --rm mir pytest tests -q
```

## HTML-отчёт

Отчёт, который можно открыть в браузере, показать научруку или приложить к записке:

```bash
docker compose run --rm mir pytest tests --html=data/output/report.html --self-contained-html
```

Файл появится в `data/output/report.html`. Внутри — таблица со всеми тестами, временем выполнения, фильтрами по результату и раскрывающимися подробностями упавших проверок.

## Только один модуль

```bash
docker compose run --rm mir pytest tests/unit/test_sources.py
docker compose run --rm mir pytest tests/unit -k "cache"
docker compose run --rm mir pytest tests -k "fps or triplet"
```

## Покрытие

```bash
docker compose run --rm mir pytest tests --cov=mir --cov-report=term-missing
```

Колонка `Missing` показывает непокрытые строки. HTML-версия:

```bash
docker compose run --rm mir pytest tests --cov=mir --cov-report=html:data/output/coverage
```

## Что сейчас проверяется

210 тестов на Python (188 модульных, 22 интеграционных) и 16 проверок в C++.

### Этап 2 — загрузка

| Файл | Тестов | Что проверяет |
|---|---|---|
| `test_sources.py` | 39 | Распознавание площадок, нормализация ссылок, отклонение неподдерживаемых источников |
| `test_types.py` | 31 | Модель данных, границы диапазонов, переводы времени |
| `test_config.py` | 18 | Загрузка настроек, приоритет источников, валидация инвариантов |
| `test_cache.py` | 10 | Кэш: попадания, вытеснение по объёму, устойчивость к повреждённому индексу |
| `test_proxy_modes.py` | 10 | Режимы `auto`, `none`, `manual`; прямое соединение прежде поиска прокси |
| `test_proxycheck.py` | 5 | Диагностика сети и обнаружение запущенного VPN-клиента |
| `test_ingest.py` | 11 | Полный этап загрузки на реальных файлах через FFmpeg |

### Этап 3 — разбор видеоряда

| Файл | Тестов | Что проверяет |
|---|---|---|
| `test_accel.py` | 15 | Совпадение нативного ядра и запасной реализации, поведение метрики цвета |
| `test_keyboard_detector.py` | 15 | Привязка к абсолютным нотам, устойчивость к сбоям сегментации, отказ на роликах без клавиатуры |
| `test_keyboard_geometry.py` | 15 | Раскладка клавиш, положение чёрных, обрезанные диапазоны |
| `test_metrics.py` | 11 | Правила MIREX: допуск 50 мс, дубли как ложные срабатывания, отдельная проверка длительности |
| `test_calibration.py` | 10 | Разделение цветов рук, воспроизводимость, назначение руки по регистру |
| `test_key_tracker.py` | 9 | Гистерезис, антидребезг без задержки, независимость соседних клавиш |
| `test_vision_pipeline.py` | 11 | Полный разбор синтетических роликов в разных условиях |
| `core/tests` (C++) | 16 | Усреднение по области, границы кадра, замыкание тона, медиана |

Интеграционные тесты этапа 3 печатают отчёт с метриками прямо в вывод pytest:

```
=== синтетический ролик, чистые условия ===
эталон / найдено : 16 / 16
совпало          : 16
precision        : 1.000
recall           : 1.000
F1               : 1.000
onset            : средняя 0.0 мс, макс 0.0 мс
рука             : 16/16 (1.000)
```

Пороги в этих тестах — зафиксированные требования к качеству: правка алгоритма, ухудшающая результат, уронит сборку.

### Проверки, за которыми стоят конкретные ошибки

Эти тесты стоит упомянуть на защите — каждый защищает от реальной проблемы:

| Тест | От чего защищает |
|---|---|
| `test_fractional_fps_not_rounded` | Округление 29.97 → 30 даёт на часовом ролике расхождение около 3.6 секунды |
| `test_triplets_sum_exactly` | `0.333 × 3 = 0.999` ломает проверку заполненности такта; отсюда `Fraction` вместо `float` |
| `test_same_video_same_key` | Без нормализации ссылки кэш скачает один ролик дважды |
| `test_hysteresis_enforced` | `off_threshold ≥ on_threshold` — подсветка клавиш дребезжит, и одна нота рассыпается на десяток |
| `test_ppq_divisible_by_12` | PPQ, не кратный 12, не выражает триоли целым числом тиков |
| `test_black_key_wins_overlap` | Чёрные клавиши перекрывают белые: без приоритета нота определится неверно |
| `test_sample_rate_matches_model_requirement` | Модель ByteDance принимает только 16 кГц моно |
| `test_video_not_reencoded` | Перекодирование испортило бы качество, критичное для детекции |
| `test_onset_is_not_delayed_by_debounce` | Антидребезг сдвигал все ноты на 33 мс вперёд — две трети допуска MIREX |
| `test_hue_ignored_for_colourless_pair` | Тон белой клавиши не определён: на 480p с шумом давал ложные нажатия с отклонением 0.34 при пороге 0.25 |
| `test_result_is_reproducible` | K-средних на одних и тех же кадрах то разделял руки, то сливал их |
| `test_pattern_match_rejects_degenerate_input` | Вырожденный узор совпадает с эталоном на 5/7 при любом сдвиге и прошёл бы порог |
| `test_native_matches_numpy_on_deviations` | Результат распознавания не должен зависеть от наличия компилятора |
| `test_cropped_keyboard_keeps_absolute_pitch` | Обрезанная клавиатура: аналоги здесь требуют ручной калибровки |

## Медленные тесты

В конце вывода pytest показывает пять самых долгих:

```
============================= slowest 5 durations ==============================
0.42s call     tests/integration/test_ingest.py::TestIngestPipeline::test_local_file
0.31s setup    tests/integration/test_ingest.py::TestProbe::test_reads_parameters
```

Долгие — интеграционные: они реально запускают FFmpeg и генерируют тестовое видео.

## Тесты, требующие внешних программ

Помечены и пропускаются, если программы нет:

```python
@pytest.mark.requires_ffmpeg
@pytest.mark.requires_gpu
@pytest.mark.requires_musescore
```

Пропуск виден как `SKIPPED` с причиной — набор не «падает» на чужой машине.
