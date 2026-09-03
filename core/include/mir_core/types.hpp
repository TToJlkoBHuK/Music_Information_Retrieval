// Общие типы ядра. Сознательно без OpenCV: ядро работает над сырыми
// буферами numpy, декодирование видео остаётся на стороне Python.
// Это убирает из сборки самую тяжёлую зависимость — C++-часть OpenCV,
// которую под Windows пришлось бы собирать отдельно.
#pragma once

#include <cstdint>
#include <cstddef>

namespace mir {

/// Прямоугольная проба внутри одной клавиши.
struct KeyRegion {
    int x_min;
    int x_max;
    int y_min;
    int y_max;
};

/// Цвет в HSV, каналы в шкале OpenCV: H 0..179, S и V 0..255.
struct ColorHsv {
    float h;
    float s;
    float v;
};

/// Веса каналов при сравнении цвета с эталоном.
///
/// Тон весит больше яркости: подсветка меняет именно цвет, тогда как
/// блики и затемнение трогают только яркость.
struct DeviationWeights {
    float hue = 0.5f;
    float saturation = 0.35f;
    float value = 0.15f;
};

/// Отступ от краёв клавиши при взятии пробы, доля ширины.
/// Края смазаны сжатием и задевают соседей, поэтому берётся середина.
inline constexpr float kSampleInset = 0.25f;

}  // namespace mir
