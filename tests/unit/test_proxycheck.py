"""Диагностика прокси."""

from __future__ import annotations

import socket
import threading

import pytest

from mir.ingest.proxycheck import COMMON_PROXY_PORTS, check_proxy


@pytest.fixture
def listening_port() -> int:
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
