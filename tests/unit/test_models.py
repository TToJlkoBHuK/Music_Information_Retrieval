"""Веса модели: поиск, загрузка, проверка целостности.

Загрузка проверяется на локальном файле через `file://` — без сети
и без 172 МБ. Размер и сумма официального файла подменяются на
значения маленького тестового.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from mir.audio import models
from mir.audio.models import ModelRegistry, backend_status, select_device
from mir.common.errors import CancelledError, TranscriptionError

PAYLOAD = b"weights" * 50_000  # ~350 КБ: несколько блоков загрузки


@pytest.fixture
def source(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> str:
    """Файл-«публикация» и подмена его размера и суммы."""
    path = tmp_path / "remote.pth"
    path.write_bytes(PAYLOAD)
    monkeypatch.setattr(models, "MODEL_SIZE", len(PAYLOAD))
    monkeypatch.setattr(models, "MODEL_MD5", hashlib.md5(PAYLOAD).hexdigest())
    return path.as_uri()


@pytest.fixture
def registry(tmp_path: Path) -> ModelRegistry:
    return ModelRegistry(tmp_path / "cache")


def test_downloads_once_and_verifies(registry, source):
    seen: list[tuple[int, int]] = []

    path, official = registry.ensure(lambda done, total: seen.append((done, total)), url=source)

    assert official
    assert path.read_bytes() == PAYLOAD
    assert seen[-1][0] == len(PAYLOAD)
    # Второй вызов файл не трогает — он уже на месте и проверен.
    mtime = path.stat().st_mtime_ns
    registry.ensure(url="file:///nonexistent")
    assert path.stat().st_mtime_ns == mtime


def test_corrupted_download_rejected(registry, source, monkeypatch):
    monkeypatch.setattr(models, "MODEL_MD5", "0" * 32)

    with pytest.raises(TranscriptionError, match="не совпадают"):
        registry.ensure(url=source)

    assert not registry.default_path.exists()


def test_truncated_file_rejected_without_hashing(registry, source):
    """Недокачанный файл: исходная библиотека проверяла лишь «больше 160 МБ»."""
    registry.default_path.parent.mkdir(parents=True)
    registry.default_path.write_bytes(PAYLOAD[:-1])
    assert not registry.verify(registry.default_path)
    assert registry.locate() is None


def test_network_error_is_explained(registry, tmp_path):
    with pytest.raises(TranscriptionError) as info:
        registry.ensure(url=(tmp_path / "missing.pth").as_uri())
    assert "audio.model_path" in info.value.user_message
    assert not registry.default_path.with_suffix(".pth.part").exists()


def test_cancel_removes_partial(registry, source):
    with pytest.raises(CancelledError):
        registry.ensure(cancel=lambda: True, url=source)
    assert not registry.default_path.exists()
    assert not registry.default_path.with_suffix(".pth.part").exists()


def test_custom_path_used_as_is(tmp_path, source):
    """Своя модель не обязана совпадать с официальной — например, дообученная."""
    own = tmp_path / "mine.pth"
    own.write_bytes(b"other")
    path, official = ModelRegistry(tmp_path / "cache", str(own)).ensure(url=source)
    assert path == own
    assert not official


def test_custom_path_missing(tmp_path):
    with pytest.raises(TranscriptionError, match="не найден"):
        ModelRegistry(tmp_path, str(tmp_path / "absent.pth")).ensure()


def test_cpu_choice_needs_no_torch():
    assert select_device("cpu") == ("cpu", None)


def test_backend_status_names_missing(monkeypatch):
    import importlib.util

    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None)
    message = backend_status()
    assert message is not None
    assert "torch" in message
    assert "audio-setup" in message
