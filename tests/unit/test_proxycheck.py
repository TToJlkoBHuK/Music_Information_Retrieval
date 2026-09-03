"""Диагностика прокси."""

from __future__ import annotations

import socket
import sys
import threading
from collections.abc import Iterator
from pathlib import Path

import pytest

from mir.common.errors import DemuxError
from mir.ingest import demuxer
from mir.ingest.demuxer import FFMPEG_DIR_ENV
from mir.ingest.proxycheck import COMMON_PROXY_PORTS, check_proxy


@pytest.fixture
def listening_port() -> Iterator[int]:
    """Поднять слушающий сокет на свободном порту."""
    server = socket.socket()
    server.bind(("127.0.0.1", 0))
    server.listen(1)
    port = server.getsockname()[1]
    threading.Thread(target=lambda: server.accept(), daemon=True).start()
    yield port
    server.close()


class TestCheckProxy:
    def test_open_port_detected(self, listening_port: int) -> None:
        probe = check_proxy("127.0.0.1", listening_port)
        assert probe.reachable
        assert probe.url == f"socks5://127.0.0.1:{listening_port}"

    def test_closed_port_reports_refusal(self) -> None:
        """Именно этот случай был у пользователя: VPN в режиме TUN."""
        probe = check_proxy("127.0.0.1", 1)
        assert not probe.reachable
        assert probe.error

    def test_unknown_host(self) -> None:
        probe = check_proxy("nonexistent.invalid", 1080, timeout=0.5)
        assert not probe.reachable


class TestCommonPorts:
    def test_covers_popular_clients(self) -> None:
        ports = {port for port, _ in COMMON_PROXY_PORTS}
        assert {10808, 2080, 7890, 12334} <= ports

    def test_every_port_documented(self) -> None:
        assert all(description for _, description in COMMON_PROXY_PORTS)


class TestFfmpegLookup:
    """FFmpeg должен находиться и вне PATH: winget обновляет его не сразу."""

    def test_env_variable_wins(self, tmp_path, monkeypatch):
        bin_dir = tmp_path / "bin"
        bin_dir.mkdir()
        name = "ffmpeg.exe" if sys.platform == "win32" else "ffmpeg"
        (bin_dir / name).write_text("", encoding="utf-8")

        monkeypatch.setenv(FFMPEG_DIR_ENV, str(bin_dir))
        monkeypatch.setattr("mir.ingest.demuxer.shutil.which", lambda _: None)

        assert demuxer._lookup("ffmpeg") == bin_dir / name

    def test_path_has_priority_over_guesses(self, monkeypatch):
        monkeypatch.setattr("mir.ingest.demuxer.shutil.which", lambda _: "/usr/bin/ffmpeg")

        assert demuxer._lookup("ffmpeg") == Path("/usr/bin/ffmpeg")

    def test_missing_tool_names_itself(self, monkeypatch):
        monkeypatch.setattr("mir.ingest.demuxer.shutil.which", lambda _: None)
        monkeypatch.setattr(demuxer, "_candidate_dirs", list)

        with pytest.raises(DemuxError) as excinfo:
            demuxer.find_ffmpeg()

        assert "FFmpeg" in excinfo.value.user_message
        assert FFMPEG_DIR_ENV in excinfo.value.user_message
