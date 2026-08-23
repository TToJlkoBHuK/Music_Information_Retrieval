#include "mir_core/key_metrics.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <vector>

namespace mir {
namespace {

/// Ограничить значение отрезком — на случай неверной разметки клавиатуры.
inline int clampInt(int value, int low, int high) {
    return value < low ? low : (value > high ? high : value);
}

}  // namespace

ColorHsv sampleRegion(const std::uint8_t* frame, int width, int height,
                      const KeyRegion& region, float inset) {
    const int span = region.x_max - region.x_min;
    const int pad = static_cast<int>(static_cast<float>(span) * inset);

    const int x0 = clampInt(region.x_min + pad, 0, width);
    const int x1 = clampInt(std::max(region.x_max - pad, x0 + 1), 0, width);
    const int y0 = clampInt(region.y_min, 0, height);
    const int y1 = clampInt(std::max(region.y_max, y0 + 1), 0, height);

    if (x1 <= x0 || y1 <= y0) {
        return ColorHsv{0.0f, 0.0f, 0.0f};
    }

    // Накопление в 64-битном целом: для области 100×100 сумма канала
    // не превышает 2.5e6, но привычка держать счётчик шире произведения
    // размеров на максимум канала избавляет от целого класса ошибок.
    std::uint64_t sum_h = 0;
    std::uint64_t sum_s = 0;
    std::uint64_t sum_v = 0;

    for (int y = y0; y < y1; ++y) {
        const std::uint8_t* row = frame + (static_cast<std::size_t>(y) * width + x0) * 3;
        for (int x = x0; x < x1; ++x, row += 3) {
            sum_h += row[0];
            sum_s += row[1];
            sum_v += row[2];
        }
    }

    const auto pixels = static_cast<float>((x1 - x0) * (y1 - y0));
    return ColorHsv{static_cast<float>(sum_h) / pixels, static_cast<float>(sum_s) / pixels,
                    static_cast<float>(sum_v) / pixels};
}

void sampleRegions(const std::uint8_t* frame, int width, int height,
                   const KeyRegion* regions, std::size_t count, ColorHsv* out, float inset) {
    for (std::size_t i = 0; i < count; ++i) {
        out[i] = sampleRegion(frame, width, height, regions[i], inset);
    }
}

float deviation(const ColorHsv& current, const ColorHsv& reference,
                const DeviationWeights& weights) {
    float hue = std::fabs(current.h - reference.h);
    hue = std::min(hue, 180.0f - hue) / 90.0f;
    const float sat = std::fabs(current.s - reference.s) / 255.0f;
    const float val = std::fabs(current.v - reference.v) / 255.0f;

    // Тон у бесцветного пикселя не определён: белая клавиша (245, 245, 245)
    // при малейшем шуме даёт произвольный оттенок, и усреднение по области
    // от этого не спасает. Поэтому вес тона умножается на насыщенность
    // менее насыщенного из двух цветов, а высвободившийся вес уходит
    // насыщенности — она в паре «белая клавиша против цветной подсветки»
    // и несёт весь сигнал.
    const float hue_confidence = std::min(current.s, reference.s) / 255.0f;
    const float sat_weight = weights.saturation + weights.hue * (1.0f - hue_confidence);

    const float score = hue * weights.hue * hue_confidence + sat * sat_weight +
                        val * weights.value;
    return std::min(1.0f, score);
}

void keyDeviations(const std::uint8_t* frame, int width, int height, const KeyRegion* regions,
                   const ColorHsv* references, std::size_t count, float* out,
                   const DeviationWeights& weights, float inset) {
    for (std::size_t i = 0; i < count; ++i) {
        const ColorHsv current = sampleRegion(frame, width, height, regions[i], inset);
        out[i] = deviation(current, references[i], weights);
    }
}

void medianFrame(const std::uint8_t* const* frames, std::size_t count, int width, int height,
                 int channels, std::uint8_t* out) {
    if (count == 0) {
        return;
    }

    const std::size_t total = static_cast<std::size_t>(width) * height * channels;
    std::vector<std::uint8_t> bucket(count);
    const std::size_t middle = count / 2;

    for (std::size_t offset = 0; offset < total; ++offset) {
        for (std::size_t f = 0; f < count; ++f) {
            bucket[f] = frames[f][offset];
        }
        // nth_element вместо сортировки: нужен только средний элемент,
        // O(n) против O(n log n) на каждый пиксель кадра.
        std::nth_element(bucket.begin(), bucket.begin() + static_cast<std::ptrdiff_t>(middle),
                         bucket.end());
        out[offset] = bucket[middle];
    }
}

}  // namespace mir
