// Обёртка pybind11. Границы клавиш и эталонные цвета приходят из Python
// массивами numpy, наружу возвращаются тоже массивы: копирование
// поштучно съело бы весь выигрыш от переноса цикла в C++.
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <stdexcept>
#include <vector>

#include "mir_core/key_metrics.hpp"

namespace py = pybind11;

namespace {

using FrameArray = py::array_t<std::uint8_t, py::array::c_style | py::array::forcecast>;
using RegionArray = py::array_t<std::int32_t, py::array::c_style | py::array::forcecast>;
using ColorArray = py::array_t<float, py::array::c_style | py::array::forcecast>;

/// Проверить, что кадр — плотный трёхканальный HSV-буфер.
void checkFrame(const FrameArray& frame) {
    if (frame.ndim() != 3 || frame.shape(2) != 3) {
        throw std::invalid_argument("ожидается кадр формы (height, width, 3)");
    }
}

/// Преобразовать массив (N, 4) в области клавиш.
std::vector<mir::KeyRegion> toRegions(const RegionArray& regions) {
    if (regions.ndim() != 2 || regions.shape(1) != 4) {
        throw std::invalid_argument("ожидается массив областей формы (N, 4)");
    }
    const auto view = regions.unchecked<2>();
    std::vector<mir::KeyRegion> out(static_cast<std::size_t>(view.shape(0)));
    for (py::ssize_t i = 0; i < view.shape(0); ++i) {
        out[static_cast<std::size_t>(i)] = mir::KeyRegion{view(i, 0), view(i, 1), view(i, 2),
                                                          view(i, 3)};
    }
    return out;
}

std::vector<mir::ColorHsv> toColors(const ColorArray& colors, std::size_t expected) {
    if (colors.ndim() != 2 || colors.shape(1) != 3) {
        throw std::invalid_argument("ожидается массив цветов формы (N, 3)");
    }
    if (static_cast<std::size_t>(colors.shape(0)) != expected) {
        throw std::invalid_argument("число цветов не совпадает с числом областей");
    }
    const auto view = colors.unchecked<2>();
    std::vector<mir::ColorHsv> out(expected);
    for (py::ssize_t i = 0; i < view.shape(0); ++i) {
        out[static_cast<std::size_t>(i)] = mir::ColorHsv{view(i, 0), view(i, 1), view(i, 2)};
    }
    return out;
}

ColorArray sampleRegionsPy(const FrameArray& frame, const RegionArray& regions, float inset) {
    checkFrame(frame);
    const auto boxes = toRegions(regions);
    const auto count = boxes.size();

    ColorArray out({static_cast<py::ssize_t>(count), py::ssize_t{3}});
    auto* out_ptr = reinterpret_cast<mir::ColorHsv*>(out.mutable_data());
    const auto* data = frame.data();
    const int height = static_cast<int>(frame.shape(0));
    const int width = static_cast<int>(frame.shape(1));

    {
        // GIL отпускается на время счёта: без этого интерфейс замирал бы
        // на всё время обработки ролика.
        py::gil_scoped_release release;
        mir::sampleRegions(data, width, height, boxes.data(), count, out_ptr, inset);
    }
    return out;
}

py::array_t<float> keyDeviationsPy(const FrameArray& frame, const RegionArray& regions,
                                   const ColorArray& references, float hue_weight,
                                   float sat_weight, float val_weight, float inset) {
    checkFrame(frame);
    const auto boxes = toRegions(regions);
    const auto count = boxes.size();
    const auto refs = toColors(references, count);

    py::array_t<float> out(static_cast<py::ssize_t>(count));
    auto* out_ptr = out.mutable_data();
    const auto* data = frame.data();
    const int height = static_cast<int>(frame.shape(0));
    const int width = static_cast<int>(frame.shape(1));
    const mir::DeviationWeights weights{hue_weight, sat_weight, val_weight};

    {
        py::gil_scoped_release release;
        mir::keyDeviations(data, width, height, boxes.data(), refs.data(), count, out_ptr, weights,
                           inset);
    }
    return out;
}

FrameArray medianFramePy(const py::list& frames) {
    if (frames.empty()) {
        throw std::invalid_argument("нужен хотя бы один кадр");
    }

    std::vector<FrameArray> holders;
    std::vector<const std::uint8_t*> pointers;
    holders.reserve(frames.size());
    pointers.reserve(frames.size());

    int height = 0;
    int width = 0;
    int channels = 0;

    for (const auto& item : frames) {
        auto array = item.cast<FrameArray>();
        if (array.ndim() != 3) {
            throw std::invalid_argument("ожидаются кадры формы (height, width, channels)");
        }
        if (holders.empty()) {
            height = static_cast<int>(array.shape(0));
            width = static_cast<int>(array.shape(1));
            channels = static_cast<int>(array.shape(2));
        } else if (array.shape(0) != height || array.shape(1) != width ||
                   array.shape(2) != channels) {
            throw std::invalid_argument("размеры кадров различаются");
        }
        pointers.push_back(array.data());
        holders.push_back(std::move(array));
    }

    FrameArray out({height, width, channels});
    auto* out_ptr = out.mutable_data();
    const auto count = pointers.size();
    const auto* const* sources = pointers.data();

    {
        py::gil_scoped_release release;
        mir::medianFrame(sources, count, width, height, channels, out_ptr);
    }
    return out;
}

}  // namespace

PYBIND11_MODULE(mir_core, m) {
    m.doc() = "Нативное ядро покадрового разбора видеоряда (проект MIR)";
    m.attr("__version__") = "0.3.0";

    m.def("sample_regions", &sampleRegionsPy, py::arg("frame"), py::arg("regions"),
          py::arg("inset") = mir::kSampleInset,
          "Средние цвета HSV по областям клавиш, массив (N, 3) float32.");

    m.def("key_deviations", &keyDeviationsPy, py::arg("frame"), py::arg("regions"),
          py::arg("references"), py::arg("hue_weight") = 0.5f, py::arg("sat_weight") = 0.35f,
          py::arg("val_weight") = 0.15f, py::arg("inset") = mir::kSampleInset,
          "Отклонения цвета клавиш от эталона, массив (N,) float32 в диапазоне 0..1.");

    m.def("median_frame", &medianFramePy, py::arg("frames"),
          "Попиксельная медиана стопки кадров одинакового размера.");
}
