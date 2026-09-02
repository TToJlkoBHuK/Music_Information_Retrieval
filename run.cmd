@echo off
rem Кодировка вывода: без этого кириллица в CMD показывается кракозябрами,
rem потому что файл в UTF-8, а консоль по умолчанию в cp866.
chcp 65001 >nul
rem Запуск проекта тем интерпретатором, где есть зависимости.
rem
rem На машине обычно несколько установок Python, и `python` в PATH
rem указывает не на ту, куда pip ставил пакеты. Разбираться в этом
rem пользователь не обязан, поэтому подходящий интерпретатор ищется сам:
rem берётся первый, в котором импортируется OpenCV.
rem
rem   run demo                     синтетический ролик с известным ответом
rem   run analyze video.mp4        разбор своего ролика
rem   run fetch "https://..."      скачать ролик
rem   run build                    собрать нативное ядро C++
rem   run test                     прогнать тесты
rem   run doctor                   диагностика сети и FFmpeg
rem   run python -c "..."          произвольная команда нужным Python

setlocal enabledelayedexpansion
cd /d "%~dp0"

set "PY="
for %%C in ("py -3.12" "py -3.11" "py -3.10" "python" "py -3") do (
    if not defined PY (
        %%~C -c "import cv2, numpy" >nul 2>&1
        if !errorlevel! equ 0 set "PY=%%~C"
    )
)

if not defined PY (
    echo.
    echo Не найден Python с установленным OpenCV.
    echo.
    echo Создайте окружение и поставьте зависимости:
    echo     py -3.11 -m venv .venv
    echo     .venv\Scripts\activate
    echo     python -m pip install -e ".[dev]"
    echo.
    exit /b 1
)

if "%~1"=="" goto :usage
set "CMD=%~1"
shift

if /i "%CMD%"=="demo"     ( %PY% scripts\vision_cli.py demo %1 %2 %3 %4 %5 %6 %7 %8 %9 & exit /b !errorlevel! )
if /i "%CMD%"=="diagnose" ( %PY% scripts\vision_cli.py diagnose %1 %2 %3 %4 %5 %6 %7 %8 %9 & exit /b !errorlevel! )
if /i "%CMD%"=="analyze"  ( %PY% scripts\vision_cli.py analyze %1 %2 %3 %4 %5 %6 %7 %8 %9 & exit /b !errorlevel! )
if /i "%CMD%"=="fetch"    ( %PY% scripts\ingest_cli.py fetch %1 %2 %3 %4 %5 %6 %7 %8 %9 & exit /b !errorlevel! )
if /i "%CMD%"=="info"     ( %PY% scripts\ingest_cli.py probe %1 %2 %3 %4 %5 %6 %7 %8 %9 & exit /b !errorlevel! )
if /i "%CMD%"=="doctor"   ( %PY% scripts\ingest_cli.py doctor %1 %2 %3 %4 %5 %6 %7 %8 %9 & exit /b !errorlevel! )
if /i "%CMD%"=="bench"    ( %PY% scripts\bench_core.py %1 %2 %3 %4 %5 %6 %7 %8 %9 & exit /b !errorlevel! )
if /i "%CMD%"=="test"     ( %PY% -m pytest tests %1 %2 %3 %4 %5 %6 %7 %8 %9 & exit /b !errorlevel! )
if /i "%CMD%"=="python"   ( %PY% %1 %2 %3 %4 %5 %6 %7 %8 %9 & exit /b !errorlevel! )
if /i "%CMD%"=="build"    goto :build
if /i "%CMD%"=="which"    ( echo %PY% & %PY% -c "import sys; print(sys.executable, sys.version)" & exit /b 0 )

:build
rem Ядро собирается для того же интерпретатора, которым запускается
rem проект. Иначе CMake берёт первый Python из PATH, собирает модуль
rem под него, и готовый .pyd не загружается: у каждой версии Python
rem свой двоичный интерфейс.
for /f "delims=" %%i in ('%PY% -c "import sys; print(sys.executable)"') do set "PYEXE=%%i"
echo Интерпретатор: !PYEXE!

rem Кэш CMake привязан к абсолютным путям. Каталог сборки, созданный
rem в другом месте — например, перенесённый вместе с проектом, — приводит
rem к отказу конфигурации, поэтому чужой кэш просто удаляется.
if exist core\build\CMakeCache.txt (
    findstr /c:"%CD:\=/%" core\build\CMakeCache.txt >nul 2>&1 || (
        echo Кэш сборки от другого каталога, пересоздаю.
        rmdir /s /q core\build
    )
)
echo.
"!PYEXE!" -m pip install --quiet pybind11 cmake || exit /b 1
"!PYEXE!" -m cmake -B core\build -S core -DCMAKE_BUILD_TYPE=Release -DPython3_EXECUTABLE="!PYEXE!" || exit /b 1
"!PYEXE!" -m cmake --build core\build --config Release || exit /b 1
echo.
"!PYEXE!" -c "from mir.vision import accel; print('ядро:', accel.backend_name())"
exit /b 0

:usage
echo Использование: run ^<команда^> [аргументы]
echo.
echo   demo      синтетический ролик с известным ответом
echo   analyze   разбор своего ролика
echo   diagnose  почему детекция не сработала
echo   fetch     скачать ролик по ссылке
echo   info      сведения о ролике без скачивания
echo   doctor    диагностика сети и FFmpeg
echo   bench     сравнение C++ ядра с numpy
echo   build     собрать нативное ядро C++
echo   test      прогнать тесты
echo   which     показать выбранный интерпретатор
exit /b 1
