cmake_minimum_required(VERSION 3.18)
project(mir_core LANGUAGES CXX VERSION 0.3.0)

# C++17 достаточно: ядро обходится стандартной библиотекой, а более
# свежий стандарт сузил бы список пригодных компиляторов без выигрыша.
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

option(MIR_BUILD_TESTS "Собирать C++-тесты" ON)

# pybind11 ищется через Python: так берётся та же версия, что стоит
# в окружении проекта, и заголовки заведомо совпадают с интерпретатором.
#
# Какой именно интерпретатор — важно: модуль расширения совместим только
# с той версией Python, под которую собран. CMake без подсказки берёт
# первый из PATH, а он часто не тот, в котором стоят зависимости проекта.
# Поэтому проще собирать через `run build`: там нужный путь подставляется
# сам. Вручную — ключом -DPython3_EXECUTABLE=путь\к\python.exe
find_package(Python3 COMPONENTS Interpreter Development.Module REQUIRED)

execute_process(
    COMMAND "${Python3_EXECUTABLE}" -m pybind11 --cmakedir
    OUTPUT_VARIABLE PYBIND11_CMAKE_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE PYBIND11_LOOKUP
)
if(NOT PYBIND11_LOOKUP EQUAL 0)
    message(FATAL_ERROR
        "pybind11 не найден в ${Python3_EXECUTABLE}.\n"
        "Установите его в этот же интерпретатор:\n"
        "  \"${Python3_EXECUTABLE}\" -m pip install pybind11\n"
        "Либо соберите ядро командой run build — она всё подставит сама.")
endif()
find_package(pybind11 REQUIRED PATHS "${PYBIND11_CMAKE_DIR}" NO_DEFAULT_PATH)

add_library(mir_core_impl STATIC src/key_metrics.cpp)
target_include_directories(mir_core_impl PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

if(MSVC)
    target_compile_options(mir_core_impl PRIVATE /W4 /O2 /utf-8)
else()
    target_compile_options(mir_core_impl PRIVATE -Wall -Wextra -O3)
endif()

pybind11_add_module(mir_core bindings/module.cpp)
target_link_libraries(mir_core PRIVATE mir_core_impl)
if(MSVC)
    target_compile_options(mir_core PRIVATE /utf-8)
endif()

# Готовый модуль кладётся в mir/vision/: там его найдёт обычный import,
# и не нужно ни менять PYTHONPATH, ни устанавливать пакет заново.
set_target_properties(mir_core PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/../mir/vision"
    LIBRARY_OUTPUT_DIRECTORY_RELEASE "${CMAKE_CURRENT_SOURCE_DIR}/../mir/vision"
    LIBRARY_OUTPUT_DIRECTORY_DEBUG "${CMAKE_CURRENT_SOURCE_DIR}/../mir/vision"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/../mir/vision"
    RUNTIME_OUTPUT_DIRECTORY_RELEASE "${CMAKE_CURRENT_SOURCE_DIR}/../mir/vision"
    RUNTIME_OUTPUT_DIRECTORY_DEBUG "${CMAKE_CURRENT_SOURCE_DIR}/../mir/vision"
)

if(MIR_BUILD_TESTS)
    enable_testing()
    add_executable(mir_core_tests tests/test_key_metrics.cpp)
    target_link_libraries(mir_core_tests PRIVATE mir_core_impl)
    if(MSVC)
        target_compile_options(mir_core_tests PRIVATE /utf-8)
    endif()
    add_test(NAME mir_core_tests COMMAND mir_core_tests)
endif()
