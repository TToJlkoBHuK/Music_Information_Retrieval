"""Общие фикстуры pytest."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from mir.config import MirConfig, load_config

pytest_plugins: list[str] = []


def _ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None


requires_ffmpeg = pytest.mark.skipif(not _ffmpeg_available(), reason="FFmpeg не установлен")


@pytest.fixture(scope="session")
def config() -> MirConfig:
    """Конфигурация по умолчанию без пользовательских переопределений."""
    return load_config(use_user_config=False, use_env=False)


@pytest.fixture(scope="session")
def sample_video(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Короткий тестовый ролик со звуком.

    Частота кадров задана как 30000/1001 (29.97) специально: на ней
    проверяется, что дробный fps не округляется.
    """
    if not _ffmpeg_available():
        pytest.skip("FFmpeg не установлен")

    out = tmp_path_factory.mktemp("media") / "sample.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc=size=640x360:rate=30000/1001:duration=2",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:duration=2",
            "-c:v",
            "libx264",
            "-c:a",
            "aac",
            "-shortest",
            str(out),
        ],
        check=True,
        capture_output=True,
    )
    return out


@pytest.fixture(scope="session")
def silent_video(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Ролик без звуковой дорожки."""
    if not _ffmpeg_available():
        pytest.skip("FFmpeg не установлен")

    out = tmp_path_factory.mktemp("media") / "silent.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc=size=320x240:rate=25:duration=1",
            "-c:v",
            "libx264",
            str(out),
        ],
        check=True,
        capture_output=True,
    )
    return out
