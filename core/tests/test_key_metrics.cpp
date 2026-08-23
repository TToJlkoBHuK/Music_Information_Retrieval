// Тесты ядра. Своя мини-проверка вместо Catch2 или GoogleTest:
// эти рамки тянут загрузку зависимостей при сборке, а проверок здесь
// два десятка — цена не оправдана.
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "mir_core/key_metrics.hpp"

namespace {

int g_failures = 0;
int g_checks = 0;

void check(bool condition, const std::string& name) {
    ++g_checks;
    if (condition) {
        std::printf("  [ ok ] %s\n", name.c_str());
    } else {
        ++g_failures;
        std::printf("  [FAIL] %s\n", name.c_str());
    }
}

void checkNear(float actual, float expected, float tolerance, const std::string& name) {
    const bool ok = std::fabs(actual - expected) <= tolerance;
    ++g_checks;
    if (ok) {
        std::printf("  [ ok ] %s (%.4f)\n", name.c_str(), actual);
    } else {
        ++g_failures;
        std::printf("  [FAIL] %s: получено %.4f, ожидалось %.4f\n", name.c_str(), actual,
                    expected);
    }
}

/// Одноцветный кадр заданного размера.
std::vector<std::uint8_t> solidFrame(int width, int height, std::uint8_t h, std::uint8_t s,
                                     std::uint8_t v) {
    std::vector<std::uint8_t> frame(static_cast<std::size_t>(width) * height * 3);
    for (std::size_t i = 0; i < frame.size(); i += 3) {
        frame[i] = h;
        frame[i + 1] = s;
        frame[i + 2] = v;
    }
    return frame;
}

void testSampleRegion() {
    std::printf("sampleRegion\n");
    const auto frame = solidFrame(40, 20, 60, 200, 180);
    const mir::KeyRegion region{0, 20, 0, 10};

    const auto color = mir::sampleRegion(frame.data(), 40, 20, region);
    checkNear(color.h, 60.0f, 0.01f, "тон одноцветной области");
    checkNear(color.s, 200.0f, 0.01f, "насыщенность одноцветной области");
    checkNear(color.v, 180.0f, 0.01f, "яркость одноцветной области");

    // Область за границей кадра не должна приводить к чтению чужой памяти.
    const mir::KeyRegion outside{100, 140, 50, 60};
    const auto empty = mir::sampleRegion(frame.data(), 40, 20, outside);
    check(empty.h == 0.0f && empty.s == 0.0f && empty.v == 0.0f,
          "область вне кадра даёт нулевой цвет");

    // Отступ обязан отсечь края: половина области закрашена иначе.
    auto split = solidFrame(40, 20, 10, 10, 10);
    for (int y = 0; y < 20; ++y) {
        for (int x = 0; x < 4; ++x) {
            const std::size_t idx = (static_cast<std::size_t>(y) * 40 + x) * 3;
            split[idx] = 170;
        }
    }
    const auto inset_color = mir::sampleRegion(split.data(), 40, 20, mir::KeyRegion{0, 16, 0, 20});
    checkNear(inset_color.h, 10.0f, 0.01f, "отступ отсекает край области");
}

void testDeviation() {
    std::printf("deviation\n");
    const mir::ColorHsv reference{30.0f, 100.0f, 120.0f};
    checkNear(mir::deviation(reference, reference), 0.0f, 1e-6f, "одинаковые цвета дают ноль");

    // Тон замкнут: 179 и 0 отличаются на один шаг, а не на 179.
    // Цвета взяты насыщенными: у бесцветных тон не учитывается вовсе.
    const float wrapped = mir::deviation(mir::ColorHsv{179.0f, 255.0f, 255.0f},
                                         mir::ColorHsv{0.0f, 255.0f, 255.0f});
    checkNear(wrapped, (1.0f / 90.0f) * 0.5f, 1e-5f, "тон замкнут по кругу");

    // Тон бесцветного пикселя не определён: белая клавиша при шуме даёт
    // произвольный оттенок, и это не должно считаться нажатием.
    const float grey = mir::deviation(mir::ColorHsv{170.0f, 3.0f, 245.0f},
                                      mir::ColorHsv{20.0f, 2.0f, 245.0f});
    check(grey < 0.01f, "разный тон у двух почти белых цветов почти не влияет");

    // При этом переход «белая клавиша → цветная подсветка» обязан
    // оставаться сильным: сигнал несёт насыщенность.
    const float lit = mir::deviation(mir::ColorHsv{60.0f, 200.0f, 200.0f},
                                     mir::ColorHsv{0.0f, 2.0f, 245.0f});
    check(lit > 0.4f, "белая клавиша против цветной подсветки даёт сильный отклик");

    const float far = mir::deviation(mir::ColorHsv{90.0f, 255.0f, 255.0f},
                                     mir::ColorHsv{0.0f, 0.0f, 0.0f});
    check(far <= 1.0f, "отклонение не превышает единицы");

    // Изменение только яркости весит меньше изменения тона — иначе блики
    // считались бы нажатием.
    const float value_only = mir::deviation(mir::ColorHsv{30.0f, 100.0f, 240.0f}, reference);
    const float hue_only = mir::deviation(mir::ColorHsv{120.0f, 100.0f, 120.0f}, reference);
    check(hue_only > value_only, "тон весит больше яркости");
}

void testKeyDeviations() {
    std::printf("keyDeviations\n");
    const int width = 88;
    const int height = 10;
    auto frame = solidFrame(width, height, 20, 30, 40);

    // Вторая клавиша подсвечена: меняется и тон, и насыщенность —
    // так выглядит настоящая подсветка.
    for (int y = 0; y < height; ++y) {
        for (int x = 8; x < 16; ++x) {
            const std::size_t idx = (static_cast<std::size_t>(y) * width + x) * 3;
            frame[idx] = 120;
            frame[idx + 1] = 210;
        }
    }

    std::vector<mir::KeyRegion> regions;
    std::vector<mir::ColorHsv> references;
    for (int i = 0; i < 11; ++i) {
        regions.push_back(mir::KeyRegion{i * 8, i * 8 + 8, 0, height});
        references.push_back(mir::ColorHsv{20.0f, 30.0f, 40.0f});
    }

    std::vector<float> out(regions.size());
    mir::keyDeviations(frame.data(), width, height, regions.data(), references.data(),
                       regions.size(), out.data());

    checkNear(out[0], 0.0f, 1e-6f, "неподсвеченная клавиша даёт ноль");
    check(out[1] > 0.4f, "подсвеченная клавиша превышает порог");
    checkNear(out[2], 0.0f, 1e-6f, "соседняя клавиша не задета");
}

void testMedianFrame() {
    std::printf("medianFrame\n");
    const int width = 4;
    const int height = 4;

    // Три кадра: в двух пиксель тёмный, в одном яркий. Медиана обязана
    // выбрать тёмный — так с кадра уходит пролетающий блок.
    auto a = solidFrame(width, height, 10, 10, 10);
    auto b = solidFrame(width, height, 10, 10, 10);
    auto c = solidFrame(width, height, 250, 250, 250);

    const std::uint8_t* frames[] = {a.data(), b.data(), c.data()};
    std::vector<std::uint8_t> out(static_cast<std::size_t>(width) * height * 3);
    mir::medianFrame(frames, 3, width, height, 3, out.data());

    bool all_dark = true;
    for (const auto value : out) {
        all_dark = all_dark && value == 10;
    }
    check(all_dark, "медиана убирает выброс из одного кадра");

    // Одиночный кадр должен возвращаться без изменений.
    const std::uint8_t* single[] = {c.data()};
    mir::medianFrame(single, 1, width, height, 3, out.data());
    check(out[0] == 250, "медиана одного кадра равна ему самому");
}

}  // namespace

int main() {
    std::printf("=== mir_core: тесты ядра ===\n");
    testSampleRegion();
    testDeviation();
    testKeyDeviations();
    testMedianFrame();

    std::printf("\nпроверок: %d, провалено: %d\n", g_checks, g_failures);
    return g_failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
