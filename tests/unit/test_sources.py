"""Распознавание и нормализация источников."""

from __future__ import annotations

from pathlib import Path

import pytest

from mir.common.enums import Platform
from mir.common.errors import UnsupportedSourceError
from mir.ingest.sources import (
    detect_platform,
    extract_video_id,
    is_url,
    normalize_url,
    resolve_source,
)


class TestDetectPlatform:
    @pytest.mark.parametrize(
        ("source", "expected"),
        [
            ("https://www.youtube.com/watch?v=dQw4w9WgXcQ", Platform.YOUTUBE),
            ("https://youtu.be/dQw4w9WgXcQ", Platform.YOUTUBE),
            ("https://youtube.com/shorts/dQw4w9WgXcQ", Platform.YOUTUBE),
            ("https://music.youtube.com/watch?v=dQw4w9WgXcQ", Platform.YOUTUBE),
            ("https://vk.com/video-123_456", Platform.VK_VIDEO),
            ("https://vkvideo.ru/video-123_456", Platform.VK_VIDEO),
            ("https://rutube.ru/video/abc123/", Platform.RUTUBE),
            ("./local.mp4", Platform.LOCAL_FILE),
            ("C:\\video\\file.mkv", Platform.LOCAL_FILE),
            ("/home/user/video.webm", Platform.LOCAL_FILE),
            ("https://example.com/video", Platform.UNKNOWN),
            ("", Platform.UNKNOWN),
        ],
    )
    def test_platforms(self, source: str, expected: Platform) -> None:
        assert detect_platform(source) is expected

    def test_youtube_needs_proxy(self) -> None:
        assert Platform.YOUTUBE.needs_proxy_in_russia
        assert not Platform.VK_VIDEO.needs_proxy_in_russia
        assert not Platform.RUTUBE.needs_proxy_in_russia


class TestIsUrl:
    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            ("https://youtube.com/watch?v=x", True),
            ("http://example.com", True),
            ("./file.mp4", False),
            ("/abs/path.mp4", False),
            ("ftp://host/file", False),
            ("", False),
        ],
    )
    def test_is_url(self, value: str, expected: bool) -> None:
        assert is_url(value) is expected


class TestExtractVideoId:
    @pytest.mark.parametrize(
        "url",
        [
            "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
            "https://youtu.be/dQw4w9WgXcQ",
            "https://youtu.be/dQw4w9WgXcQ?t=42",
            "https://www.youtube.com/shorts/dQw4w9WgXcQ",
            "https://www.youtube.com/embed/dQw4w9WgXcQ",
            "https://www.youtube.com/watch?v=dQw4w9WgXcQ&list=PLabc&index=3",
        ],
    )
    def test_youtube_forms(self, url: str) -> None:
        assert extract_video_id(url) == "dQw4w9WgXcQ"

    def test_rutube(self) -> None:
        assert extract_video_id("https://rutube.ru/video/abc123def/") == "abc123def"

    def test_vk(self) -> None:
        assert extract_video_id("https://vk.com/video-123_456789") == "video-123_456789"

    def test_unknown_returns_none(self) -> None:
        assert extract_video_id("https://example.com/x") is None


class TestNormalizeUrl:
    def test_strips_tracking_and_timestamp(self) -> None:
        messy = "https://www.youtube.com/watch?v=dQw4w9WgXcQ&list=PLabc&index=3&t=42s"
        assert normalize_url(messy) == "https://www.youtube.com/watch?v=dQw4w9WgXcQ"

    def test_short_form_expanded(self) -> None:
        assert (
            normalize_url("https://youtu.be/dQw4w9WgXcQ?si=xyz")
            == "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
        )

    def test_same_video_same_key(self) -> None:
        """Разные формы одной ссылки должны совпасть — иначе кэш скачает дважды."""
        forms = [
            "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
            "https://youtu.be/dQw4w9WgXcQ",
            "https://youtube.com/watch?v=dQw4w9WgXcQ&t=10",
            "https://www.youtube.com/watch?v=dQw4w9WgXcQ&feature=share",
        ]
        assert len({normalize_url(f) for f in forms}) == 1

    def test_rutube_keeps_path(self) -> None:
        assert "rutube.ru/video/abc123" in normalize_url(
            "https://rutube.ru/video/abc123/?utm_source=vk"
        )


class TestResolveSource:
    def test_local_file(self, tmp_path: Path) -> None:
        video = tmp_path / "clip.mp4"
        video.write_bytes(b"stub")
        platform, resolved = resolve_source(video)
        assert platform is Platform.LOCAL_FILE
        assert Path(resolved).is_absolute()

    def test_missing_file(self, tmp_path: Path) -> None:
        with pytest.raises(UnsupportedSourceError, match="не найден"):
            resolve_source(tmp_path / "absent.mp4")

    def test_directory_rejected(self, tmp_path: Path) -> None:
        with pytest.raises(UnsupportedSourceError, match="не файл"):
            resolve_source(tmp_path)

    def test_bad_extension(self, tmp_path: Path) -> None:
        doc = tmp_path / "notes.txt"
        doc.write_text("x")
        with pytest.raises(UnsupportedSourceError, match="расширение"):
            resolve_source(doc)

    def test_unknown_platform(self) -> None:
        with pytest.raises(UnsupportedSourceError, match="площадка"):
            resolve_source("https://example.com/video")

    def test_empty(self) -> None:
        with pytest.raises(UnsupportedSourceError):
            resolve_source("")

    def test_user_message_is_russian(self) -> None:
        """Сообщение для интерфейса должно быть человекочитаемым (F-35)."""
        with pytest.raises(UnsupportedSourceError) as info:
            resolve_source("https://example.com/video")
        assert "YouTube" in info.value.user_message
        assert info.value.user_message != info.value.technical
