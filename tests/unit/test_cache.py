"""Кэш скачанных роликов."""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from mir.ingest.cache import MediaCache

URL = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"


@pytest.fixture
def cache(tmp_path: Path) -> MediaCache:
    return MediaCache(tmp_path / "cache", max_size_gb=0.001)


def _make_file(path: Path, size: int = 1024) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"\0" * size)
    return path


class TestKey:
    def test_equivalent_urls_share_key(self) -> None:
        """Ключ строится по нормализованной ссылке."""
        assert MediaCache.key_for(URL) == MediaCache.key_for("https://youtu.be/dQw4w9WgXcQ?t=42")

    def test_different_videos_differ(self) -> None:
        assert MediaCache.key_for(URL) != MediaCache.key_for("https://youtu.be/aaaaaaaaaaa")


class TestPutGet:
    def test_miss_then_hit(self, cache: MediaCache, tmp_path: Path) -> None:
        assert cache.get(URL) is None
        cache.put(URL, _make_file(tmp_path / "video.mp4"))
        assert cache.get(URL) is not None

    def test_hit_by_equivalent_url(self, cache: MediaCache, tmp_path: Path) -> None:
        cache.put(URL, _make_file(tmp_path / "video.mp4"))
        assert cache.get("https://youtu.be/dQw4w9WgXcQ") is not None

    def test_index_survives_restart(self, tmp_path: Path) -> None:
        first = MediaCache(tmp_path / "c", max_size_gb=1.0)
        first.put(URL, _make_file(tmp_path / "v.mp4"))
        second = MediaCache(tmp_path / "c", max_size_gb=1.0)
        assert second.get(URL) is not None

    def test_deleted_file_drops_entry(self, cache: MediaCache, tmp_path: Path) -> None:
        stored = cache.put(URL, _make_file(tmp_path / "v.mp4"))
        stored.unlink()
        assert cache.get(URL) is None
        assert len(cache) == 0

    def test_broken_index_recovers(self, tmp_path: Path) -> None:
        root = tmp_path / "c"
        root.mkdir(parents=True)
        (root / "index.json").write_text("{ это не json", encoding="utf-8")
        assert len(MediaCache(root)) == 0


class TestEviction:
    def test_oldest_evicted_when_over_limit(self, tmp_path: Path) -> None:
        cache = MediaCache(tmp_path / "c", max_size_gb=2 / 1024**3)  # 2 байта
        cache.put("https://youtu.be/aaaaaaaaaaa", _make_file(tmp_path / "a.mp4", 1))
        time.sleep(0.01)
        cache.put("https://youtu.be/bbbbbbbbbbb", _make_file(tmp_path / "b.mp4", 1))
        time.sleep(0.01)
        cache.put("https://youtu.be/ccccccccccc", _make_file(tmp_path / "c.mp4", 1))
        assert cache.get("https://youtu.be/aaaaaaaaaaa") is None
        assert cache.get("https://youtu.be/ccccccccccc") is not None


class TestClear:
    def test_clear_all(self, cache: MediaCache, tmp_path: Path) -> None:
        cache.put(URL, _make_file(tmp_path / "v.mp4", 4096))
        assert cache.clear() == 4096
        assert len(cache) == 0

    def test_clear_respects_age(self, cache: MediaCache, tmp_path: Path) -> None:
        cache.put(URL, _make_file(tmp_path / "v.mp4"))
        assert cache.clear(older_than_days=1.0) == 0
        assert len(cache) == 1
