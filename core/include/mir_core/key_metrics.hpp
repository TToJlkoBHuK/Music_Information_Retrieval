// Горячий цикл покадрового разбора: усреднение цвета по областям клавиш
// и сравнение с эталоном. Вызывается для каждого кадра ролика, поэтому
// вынесен в C++.
#pragma once

#include <cstdint>
#include <cstddef>

#include "mir_core/types.hpp"

namespace mir {

/// Средний цвет одной пробы. Кадр — плотный HSV-буфер height×width×3.
ColorHsv sampleRegion(const std::uint8_t* frame, int width, int height,
                      const KeyRegion& region, float inset = kSampleInset);

/// Средние цвета всех клавиш за один проход.
///
/// @param out Массив длины count, заполняется вызовом.
void sampleRegions(const std::uint8_t* frame, int width, int height,
                   const KeyRegion* regions, std::size_t count, ColorHsv* out,
                   float inset = kSampleInset);

/// Расстояние между цветами в 0..1.
///
/// Тон замыкается по кругу: 179 и 0 — соседние оттенки, а не
/// противоположные, иначе красная подсветка давала бы ложное срабатывание
/// на каждом кадре.
float deviation(const ColorHsv& current, const ColorHsv& reference,
                const DeviationWeights& weights = {});

/// Отклонения всех клавиш от эталона за один проход по кадру.
///
/// Основная функция ядра: заменяет 88 отдельных срезов numpy на один
/// вызов и один проход по памяти кадра.
void keyDeviations(const std::uint8_t* frame, int width, int height,
                   const KeyRegion* regions, const ColorHsv* references,
                   std::size_t count, float* out,
                   const DeviationWeights& weights = {},
                   float inset = kSampleInset);

/// Попиксельная медиана стопки кадров.
///
/// Медиана убирает подсветку и падающие блоки: они не задерживаются
/// на месте, а клавиатура неподвижна. Среднее для этого не годится —
/// яркие блоки сдвинули бы его.
///
/// @param frames Указатели на count кадров одинакового размера.
/// @param out Буфер размера width*height*channels.
void medianFrame(const std::uint8_t* const* frames, std::size_t count, int width,
                 int height, int channels, std::uint8_t* out);

}  // namespace mir
