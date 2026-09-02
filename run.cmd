@echo off
rem Launcher. Runs the project with the interpreter that actually has
rem the dependencies installed.
rem
rem A machine usually carries several Python installations, and `python`
rem in PATH points at the wrong one. The user should not have to care,
rem so the first interpreter that can import OpenCV is picked here.
rem
rem This file is deliberately ASCII-only: CMD reads a batch file byte by
rem byte in the current code page, and non-ASCII text inside derails the
rem parser in ways that surface as random "not recognized" errors.
rem The Russian documentation lives in README.md next to this file.
rem
rem   run demo                     synthetic clip with a known answer
rem   run analyze video.mp4        analyse a real clip
rem   run diagnose video.mp4       why keyboard detection failed
rem   run fetch "https://..."      download a clip
rem   run build                    build the native C++ core
rem   run test                     run the test suite
rem   run which                    show the chosen interpreter

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
    echo No Python with OpenCV installed was found.
    echo.
    echo Create an environment and install the dependencies:
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
goto :usage

:build
rem The extension module only works with the Python version it was built
rem for, so CMake is told explicitly which interpreter to target. Left
rem alone it picks the first one in PATH, rarely the one holding the
rem project dependencies.
for /f "delims=" %%i in ('%PY% -c "import sys; print(sys.executable)"') do set "PYEXE=%%i"
echo Interpreter: !PYEXE!

rem A CMake cache is tied to absolute paths, so a build directory created
rem elsewhere refuses to configure. Drop a foreign one instead of failing.
if exist core\build\CMakeCache.txt (
    findstr /c:"%CD:\=/%" core\build\CMakeCache.txt >nul 2>&1 || (
        echo Build cache belongs to another directory, recreating.
        rmdir /s /q core\build
    )
)

"!PYEXE!" -m pip install --quiet pybind11 cmake || exit /b 1
"!PYEXE!" -m cmake -B core\build -S core -DCMAKE_BUILD_TYPE=Release -DPython3_EXECUTABLE="!PYEXE!" || exit /b 1
"!PYEXE!" -m cmake --build core\build --config Release || exit /b 1
echo.
"!PYEXE!" -c "from mir.vision import accel; print('backend:', accel.backend_name())"
exit /b 0

:usage
echo Usage: run ^<command^> [arguments]
echo.
echo   demo      synthetic clip with a known answer
echo   analyze   analyse a real clip
echo   diagnose  why keyboard detection failed
echo   fetch     download a clip by URL
echo   info      clip metadata without downloading
echo   doctor    network and FFmpeg diagnostics
echo   bench     compare the C++ core against numpy
echo   build     build the native C++ core
echo   test      run the test suite
echo   which     show the chosen interpreter
exit /b 1
