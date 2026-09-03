# Запуск и проверка

Проект собирается по этапам, поэтому запускать можно то, что уже готово. Сейчас это **этап 2 — модуль загрузки**: ссылка или файл превращаются в подготовленный видеоряд и аудиодорожку.

!!! warning "Windows: команды пишутся в одну строку"
    Примеры ниже даны в одну строку намеренно. В Windows CMD обратный слеш `\` **не переносит команду** на следующую строку — он будет воспринят как аргумент, и команда отработает неправильно, не сообщив об ошибке.

    Перенос строки: в CMD — `^`, в PowerShell — обратная кавычка `` ` ``. Проще писать одной строкой.

    Пути и ссылки **всегда в кавычках**: без них имя файла с пробелами разобьётся на несколько аргументов.

## Вариант 1: Docker (ничего не нужно устанавливать)

Единственная зависимость — Docker. FFmpeg, Python и библиотеки уже внутри образа.

```bash
git clone https://github.com/TToJlkoBHuK/Music_Information_Retrieval.git
cd Music_Information_Retrieval

docker compose build
docker compose run --rm mir pytest tests
```

Подробнее о просмотре результатов — в разделе [Как смотреть тесты](testing.md).

### Разбор ссылки без обращения к сети

```bash
docker compose run --rm mir python -m scripts.ingest_cli url "https://www.youtube.com/watch?v=dQw4w9WgXcQ&list=PLabc&t=42"
```

```
Площадка:    youtube
Нормализация:https://www.youtube.com/watch?v=dQw4w9WgXcQ
Ключ кэша:   0424974c68530290
```

Трекинг-параметры и метка времени отброшены — иначе кэш скачал бы один ролик дважды.

### Подготовка локального файла

Положите видео в папку `data/videos/`, она примонтирована в контейнер. **Имя файла с пробелами обязательно в кавычках:**

```bash
docker compose run --rm mir python -m scripts.ingest_cli fetch "data/videos/Geometry Dash 2025.07.30.mp4" -o data/output
```

```
Подготовка данных      [##############################] 100.0%

Видео:       data/videos/Geometry Dash 2025.07.30.mp4
Аудио:       data/output/Geometry Dash 2025.07.30.wav
Длительность:  184.32 с
Кадры:       1920x1080 @ 29.970 fps
Всего кадров:    5524
```

Чтобы не мучиться с пробелами, файл проще переименовать: `sample.mp4`.

### Скачивание по ссылке

YouTube из России требует прокси. Ссылка должна быть настоящей — `VIDEO_ID` в примере надо заменить:

```bash
docker compose run --rm mir python -m scripts.ingest_cli fetch "https://youtu.be/dQw4w9WgXcQ" -o data/output --proxy socks5://host.docker.internal:1080
```

Флаг `--proxy` надёжнее переменной окружения: в Windows CMD синтаксис `VAR=value команда` не работает, переменную пришлось бы задавать отдельной командой `set`.

Для VK Видео и Rutube прокси не нужен:

```bash
docker compose run --rm mir python -m scripts.ingest_cli fetch "https://rutube.ru/video/VIDEO_ID/" -o data/output
```

## VPN

**В готовом приложении настраивать нечего.** Оно запускается обычным процессом Windows, поэтому режим TUN у VPN-клиента (перехват всего системного трафика) покрывает его автоматически: включили VPN — программа скачивает.

Если клиент работает в режиме Proxy, а не TUN, программа сама переберёт типичные порты локальных клиентов и повторит попытку. Это поведение по умолчанию — `proxy_mode = "auto"`.

### Особый случай: Docker

Docker Desktop на Windows работает поверх WSL2, у которого **собственный сетевой стек**. Туннель, поднятый на сетевом интерфейсе Windows, до контейнера не достаёт — контейнер ходит в интернет напрямую, мимо VPN. Это ограничение среды разработки, к готовому приложению отношения не имеющее.

Поэтому при работе через Docker нужен режим **Proxy** у VPN-клиента (в Happ — кнопка рядом с TUN). Дальше автоподбор сработает сам:

```bash
docker compose run --rm mir python -m scripts.ingest_cli fetch "https://youtu.be/VIDEO_ID" -o data/output
```

В логе с `-v` будет видно, что произошло:

```
прямое соединение не удалось, ищу локальный прокси
найден локальный прокси: socks5://host.docker.internal:10808 (Happ, v2rayN, Xray — SOCKS5)
повторная попытка через socks5://host.docker.internal:10808
```

### Если автоподбор не сработал

Посмотреть, что доступно:

```bash
docker compose run --rm mir python -m scripts.ingest_cli check-proxy
```

Проверяются порты: 10808/10809 (Happ, v2rayN, Xray), 2080/2081 (Nekoray), 7890/7891 (Clash), 12334/12335 (Hiddify), 1080, 9050 (Tor).

Нестандартный порт задаётся явно:

```bash
docker compose run --rm mir python -m scripts.ingest_cli fetch "https://youtu.be/ID" -o data/output --proxy socks5://host.docker.internal:PORT
```

### Самый простой путь для Docker

Скачать ролик любым способом, положить в `data/videos/` и обработать как локальный файл — сеть не нужна вовсе:

```bash
docker compose run --rm mir python -m scripts.ingest_cli fetch "data/videos/ролик.mp4" -o data/output
```

## Если команда «зависла»

При скачивании без работающего прокси yt-dlp будет молча ждать таймаута (по умолчанию 300 секунд). Прервать — `Ctrl+C`, программа корректно завершится с сообщением «Прервано пользователем».

Чтобы увидеть, что происходит, добавьте `-v`:

```bash
docker compose run --rm mir python -m scripts.ingest_cli fetch "https://youtu.be/ID" -o data/output -v
```

## Вариант 2: локальная установка

Нужны Python 3.10+ и FFmpeg в `PATH`.

=== "Windows"

    ```powershell
    winget install Gyan.FFmpeg
    python -m venv .venv
    .venv\Scripts\activate
    pip install -e ".[dev,docs]"
    ```

=== "Linux"

    ```bash
    sudo apt install ffmpeg python3-venv
    python3 -m venv .venv
    source .venv/bin/activate
    pip install -e ".[dev,docs]"
    ```

Проверка:

```bash
pytest tests -q
python -m scripts.ingest_cli url "https://youtu.be/dQw4w9WgXcQ?t=42"
python -m scripts.ingest_cli fetch ./video.mp4 -o ./output
```

## Команды CLI

| Команда | Что делает |
|---|---|
| `url <ссылка>` | Разбор ссылки: площадка, нормализация, ключ кэша. Без сети |
| `probe <источник>` | Название, длительность, разрешение. Без скачивания |
| `fetch <источник>` | Скачать и подготовить видео с аудиодорожкой |
| `cache` | Показать состояние кэша |
| `cache --clear` | Очистить кэш |

Общие флаги: `-v` — подробный лог, `-q` — без прогресса, `--proxy` — прокси, `--no-cache` — игнорировать кэш.

## Настройка

Три способа, по возрастанию приоритета:

```bash
# 1. Файл ~/.mir/config.toml
[ingest]
proxy = "socks5://127.0.0.1:1080"
max_height = 720

# 2. Переменные окружения
export MIR_INGEST_PROXY="socks5://127.0.0.1:1080"

# 3. Аргументы командной строки
python -m scripts.ingest_cli fetch ... --proxy socks5://127.0.0.1:1080
```

Все параметры и обоснование значений — в разделе [Конфигурация](modules/config.md).

## Этап 3: разбор видеоряда

### Самопроверка без единого файла

```bash
python scripts/vision_cli.py demo
```

Собирает синтетический ролик из заранее известного списка нот, разбирает его и печатает метрики. Эталон точен по построению — ролик рисуется из него же.

В `build/vision_demo/` появятся две картинки: найденная клавиатура поверх кадра и сравнение эталона с результатом.

Условия варьируются: `--scheme dark`, `--scheme mono`, `--noise 6 --glow`, `--width 854 --height 480`, `--watermark "PIANO TUTORIAL"`.

### Разбор своего ролика

```bash
python scripts/vision_cli.py analyze video.mp4 --overlay layout.png --roll roll.png
python scripts/vision_cli.py analyze video.mp4 --csv notes.csv
```

Если разметка на `layout.png` съехала, дальше смотреть незачем — проблема в детекции клавиатуры, а не в трекинге.

### Нативное ядро (необязательно)

Проект работает и без него, только медленнее: `mir.vision.accel` переключается на numpy. Сборка ускоряет покадровый разбор в 27 раз.

```bash
run build
```

Команда сама найдёт нужный интерпретатор, поставит pybind11 и соберёт модуль. Вручную придётся указывать интерпретатор явно ключом `-DPython3_EXECUTABLE`, иначе CMake возьмёт первый Python из PATH и соберёт модуль, который не загрузится.

Ожидаемый ответ — `mir_core (C++)`. Если `numpy`, ядро не собралось или не найдено.

```bash
./core/build/mir_core_tests      # 16 проверок ядра
python scripts/bench_core.py     # таблица сравнения с numpy
```

## Проверка качества кода

```bash
ruff check mir tests scripts     # линтер
ruff format mir tests scripts    # форматирование
mypy mir scripts tests           # типизация в строгом режиме (NF-16)
pytest --cov=mir                 # покрытие тестами (NF-17)
```

Эти же проверки выполняет GitHub Actions при каждом коммите.

## Сборка документации

```bash
mkdocs serve     # локальный просмотр на http://127.0.0.1:8000
mkdocs build     # статический сайт в site/
```

Справочник API генерируется из docstrings при каждой сборке, поэтому не может разойтись с кодом.

## Что проверять при приёмке этапа 2

| Требование | Как проверить |
|---|---|
| F-01 загрузка по ссылке YouTube | `fetch "https://youtu.be/ID" --proxy ...` |
| F-02 локальный файл | `fetch ./video.mp4` |
| F-03 прокси | `--proxy socks5://...` или `MIR_INGEST_PROXY` |
| F-04 VK Видео и Rutube | `fetch "https://rutube.ru/video/ID/"` |
| F-05 кэш | Повторный `fetch` той же ссылки идёт мгновенно |
| F-35 понятные ошибки | `fetch ./missing.mp4` → «Файл не найден», без traceback |
| NF-16 типизация | `mypy mir` → «Success» |
| NF-18 без магических чисел | Все пороги в `config/default.toml` |
| NF-19 логирование | `-v` показывает лог, `print` в модулях не используется |

## Что проверять при приёмке этапа 3

| Требование | Как проверить |
|---|---|
| F-06 автодетекция клавиатуры | `vision_cli.py demo` → диапазон 36..96 без настройки |
| F-07 обрезанная клавиатура | `test_cropped_keyboard_keeps_absolute_pitch`: диапазон 53..72 восстановлен точно |
| F-08 трекинг нот | `vision_cli.py demo` → F1 = 1.000 |
| F-09 разделение рук | тот же вывод, строка «рука 16/16» |
| F-12 автокалибровка | `demo --scheme dark` и `--scheme mono` работают без правки настроек |
| NF-01 скорость | `bench_core.py`: 0.8 с против 21.2 с на четырёхминутном ролике |
| NF-16 типизация | `mypy mir scripts tests` → «Success» |
| NF-18 без магических чисел | Пороги трекинга в `config/default.toml`, пояснения — рядом |

## Известные ограничения

- Прямые трансляции не поддерживаются — нужна опубликованная запись.
- Видео без звуковой дорожки отклоняется: аудиоканал обязателен для сверки.
- Плейлисты не разворачиваются: нужна ссылка на конкретный ролик.
- Съёмка клавиатуры под углом не компенсируется: детектор откажется работать, а не выдаст мусор.
- При сильно обрезанной клавиатуре октава определяется неоднозначно — разрешится на этапе слияния с аудио.
- Одноцветная схема не позволяет разделить руки; ноты при этом находятся полностью.
