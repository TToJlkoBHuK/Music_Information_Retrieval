# core — C++ ядро компьютерного зрения

Самая нагруженная часть проекта: покадровый разбор видеоряда. Собирается в нативный Python-модуль `mir_core` через pybind11.

## Зачем C++

Ролик 4 минуты при 30 fps — 7200 кадров. На каждом нужно:

- проверить состояние 88 клавиш (сравнение цвета с эталоном),
- отсегментировать и сопоставить с предыдущим кадром десятки падающих блоков.

Это порядка 10⁶ операций над областями изображения. Python с построчными циклами тут проседает, а векторизовать логику трекинга через numpy неудобно. C++ с OpenCV решает задачу за секунды.

## Структура

```
core/
├── include/mir_core/     публичные заголовки (то, что видит внешний мир)
│   ├── types.hpp             структуры данных
│   ├── keyboard_detector.hpp детекция и разметка клавиатуры
│   ├── key_tracker.hpp       отслеживание подсветки клавиш
│   ├── block_tracker.hpp     трекинг падающих блоков
│   ├── calibration.hpp       автокалибровка под визуализатор
│   └── video_analyzer.hpp    фасад: единая точка входа
├── src/                  реализация (.cpp)
├── bindings/             pybind11-обёртка
│   └── module.cpp
├── tests/                C++ юнит-тесты (Catch2 или GoogleTest)
└── CMakeLists.txt
```

Правило: заголовки в `include/mir_core/` — это публичный контракт. Всё, что не должно быть видно снаружи (вспомогательные функции, детали реализации), живёт только в `src/`.

## Типы данных (`types.hpp`)

Зеркалят Python-модель из `mir/common/`. При изменении одного нужно менять оба.

```cpp
namespace mir {

enum class Hand : uint8_t { Left = 0, Right = 1, Unknown = 2 };

// Одна клавиша на изображении
struct KeySlot {
    int  pitch;        // MIDI-номер: 21 (A0) .. 108 (C8)
    int  x_min;        // левая граница в пикселях
    int  x_max;        // правая граница
    bool is_black;     // чёрная клавиша (уже, выше по кадру)
};

// Результат детекции клавиатуры
struct KeyboardLayout {
    cv::Rect             bbox;           // область клавиатуры в кадре
    std::vector<KeySlot> keys;           // слева направо
    int                  lowest_pitch;   // реально видимый диапазон
    int                  highest_pitch;
    bool                 is_cropped;     // видны не все 88 клавиш
    float                confidence;     // 0..1, качество детекции
};

// Событие нажатия/отпускания, найденное по видео
struct NoteEventRaw {
    int   pitch;
    float onset;        // секунды от начала ролика
    float offset;
    float velocity;     // 0..1, оценка по яркости; в MIDI масштабируется до 0..127
    Hand  hand;
    float confidence;
};

// Параметры визуализатора, полученные автокалибровкой
struct VisualizerProfile {
    cv::Scalar background_hsv;
    std::vector<cv::Scalar> block_colors;  // 1..N кластеров цветов блоков
    float fall_speed_px_per_sec;           // скорость падения блоков
    int   hit_line_y;                      // y-координата линии касания клавиатуры
};

} // namespace mir
```

## Классы

### `KeyboardDetector` (`keyboard_detector.hpp`)

Находит клавиатуру и строит карту клавиш. Работает один раз в начале обработки.

```cpp
class KeyboardDetector {
public:
    struct Params {
        int   sample_frames    = 90;    // сколько кадров усреднять
        float min_key_width_px = 4.0f;  // отсев мусора
        float black_key_ratio  = 0.6f;  // ожидаемая высота чёрной клавиши
    };

    explicit KeyboardDetector(Params p = {});

    // Вход:  кадры из разных мест ролика (для медианы)
    // Выход: разметка клавиатуры; бросает DetectionError, если не нашёл
    KeyboardLayout detect(const std::vector<cv::Mat>& frames);

private:
    cv::Mat buildMedianFrame(const std::vector<cv::Mat>& frames);
    cv::Rect findKeyboardBand(const cv::Mat& median);
    std::vector<int> findKeyBoundaries(const cv::Mat& band);
    // Ключевой шаг: привязка к абсолютным высотам по шаблону
    // чередования чёрных клавиш (группы по 2 и 3)
    std::vector<KeySlot> assignPitches(const std::vector<int>& boundaries,
                                       const std::vector<bool>& is_black);
};
```

**Как работает.** Берём кадры равномерно по ролику и считаем попиксельную медиану — подсветка и падающие блоки исчезают, остаётся «пустая» клавиатура. Ищем горизонтальную полосу с регулярным чёрно-белым узором (Canny + проекция градиентов на ось X: у клавиатуры характерный периодический профиль). Внутри полосы находим границы клавиш и определяем, какие из них чёрные (темнее, выше по кадру). Дальше главное: сопоставляем найденный узор чёрных клавиш с эталонным шаблоном октавы (2-3-2-3…). Это даёт однозначную привязку к абсолютным нотам даже при обрезанном кадре — именно здесь аналоги требуют ручной калибровки.

### `KeyTracker` (`key_tracker.hpp`)

Отслеживает подсветку клавиш покадрово.

```cpp
class KeyTracker {
public:
    struct Params {
        float on_threshold  = 0.25f;  // порог включения (доля отличия от эталона)
        float off_threshold = 0.15f;  // порог выключения — ниже, это гистерезис
        int   min_frames_on = 2;      // антидребезг
    };

    KeyTracker(const KeyboardLayout& layout, const cv::Mat& reference_frame, Params p = {});

    // Вход:  очередной кадр и его временная метка
    // Выход: события, завершившиеся на этом кадре
    std::vector<NoteEventRaw> processFrame(const cv::Mat& frame, float timestamp);

    // Закрыть все ещё «нажатые» ноты в конце ролика
    std::vector<NoteEventRaw> flush(float end_timestamp);

private:
    KeyboardLayout layout_;
    std::vector<cv::Scalar> reference_hsv_;     // эталонный цвет каждой клавиши
    std::vector<bool>       is_pressed_;
    std::vector<float>      press_start_;
};
```

**Почему гистерезис.** Два разных порога на включение и выключение: иначе на границе шум сжатия даёт дребезг — одна нота распадается на десяток коротких. Работа в HSV, а не в RGB: канал Hue устойчив к изменению яркости, важно при свечении и частицах.

### `BlockTracker` (`block_tracker.hpp`)

Отслеживает падающие блоки. Даёт более точные длительности, чем подсветка клавиш, и «видит» ноту заранее.

```cpp
struct Block {
    int      id;
    cv::Rect bbox;
    cv::Scalar color_hsv;
    int      pitch;          // по x-координате центра через KeyboardLayout
    Hand     hand;           // по цвету
    float    first_seen;
    bool     has_landed;
};

class BlockTracker {
public:
    BlockTracker(const KeyboardLayout& layout, const VisualizerProfile& profile);

    // Вход:  кадр + метка времени
    // Выход: ноты, чьи блоки коснулись клавиатуры на этом кадре
    std::vector<NoteEventRaw> processFrame(const cv::Mat& frame, float timestamp);

private:
    std::vector<Block> segmentBlocks(const cv::Mat& frame);
    void matchWithPrevious(std::vector<Block>& current, float dt);
    // Момент касания считается интерполяцией траектории,
    // а не номером кадра: при 30 fps шаг 33 мс — для трелей слишком грубо
    float interpolateHitTime(const Block& b, float timestamp, float dt);
};
```

**Ключевая деталь.** Длина блока в пикселях делится на скорость падения (`fall_speed_px_per_sec` из профиля) и даёт длительность ноты в секундах напрямую — точнее, чем измерение по кадрам подсветки. Момент касания линии клавиатуры вычисляется линейной интерполяцией между двумя соседними кадрами, что поднимает временное разрешение с 33 мс до единиц миллисекунд.

### `Calibrator` (`calibration.hpp`)

Определяет параметры конкретного визуализатора по первым секундам ролика.

```cpp
class Calibrator {
public:
    // Вход:  первые N секунд кадров + разметка клавиатуры
    // Выход: профиль визуализатора
    static VisualizerProfile calibrate(const std::vector<cv::Mat>& frames,
                                       const KeyboardLayout& layout);
};
```

Что определяет: цвет фона (мода по верхней зоне), цвета блоков (k-means по HSV пикселей, не относящихся к фону), скорость падения (кросс-корреляция вертикальных срезов соседних кадров), y-координату линии касания (верхняя граница `layout.bbox`).

### `VideoAnalyzer` (`video_analyzer.hpp`)

Фасад — то, что видит Python. Скрывает порядок вызовов остальных классов.

```cpp
class VideoAnalyzer {
public:
    struct Result {
        std::vector<NoteEventRaw> notes;
        KeyboardLayout            layout;
        VisualizerProfile         profile;
        float                     fps;
        float                     duration;
    };

    // Вход:  путь к видеофайлу, колбэк прогресса (0..1)
    // Выход: все события + метаданные разбора
    Result analyze(const std::string& video_path,
                   const std::function<void(float)>& progress = nullptr);
};
```

## Биндинги (`bindings/module.cpp`)

```cpp
PYBIND11_MODULE(mir_core, m) {
    py::class_<KeySlot>(m, "KeySlot")
        .def_readonly("pitch", &KeySlot::pitch)
        .def_readonly("x_min", &KeySlot::x_min)
        .def_readonly("x_max", &KeySlot::x_max)
        .def_readonly("is_black", &KeySlot::is_black);
    // ... остальные структуры
    py::class_<VideoAnalyzer>(m, "VideoAnalyzer")
        .def(py::init<>())
        .def("analyze", &VideoAnalyzer::analyze,
             py::arg("video_path"), py::arg("progress") = nullptr,
             py::call_guard<py::gil_scoped_release>());  // отпускаем GIL на время работы
}
```

`gil_scoped_release` обязателен: без него на время анализа видео зависает весь Python, включая обновление интерфейса.

## Сборка

```bash
cmake -B build -S . -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
```

Зависимости: OpenCV 4.x, pybind11, Python 3.10+. Готовый модуль (`mir_core.pyd` на Windows, `mir_core.so` на Linux) кладётся туда, где его найдёт Python.

## Что тестировать (`core/tests/`)

- `assignPitches` на синтетических шаблонах клавиатуры, в том числе обрезанных с обеих сторон
- гистерезис `KeyTracker` на последовательности с искусственным шумом
- `interpolateHitTime` — сравнение с аналитически известным ответом
- полный `VideoAnalyzer` на коротком синтетическом ролике, сгенерированном из известного MIDI
