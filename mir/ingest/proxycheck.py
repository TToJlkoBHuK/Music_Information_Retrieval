"""Автоопределение прокси (F-03).

Пользователь не должен ничего настраивать: включил VPN — программа работает.
Для готового приложения этого достаточно, потому что оно запускается обычным
процессом Windows, и режим TUN покрывает его автоматически.

Прокси нужен только там, где TUN не действует: в контейнере Docker (у WSL2
собственный сетевой стек) либо если VPN-клиент работает в режиме Proxy.
В этих случаях адрес подбирается автоматически перебором типичных портов —
вручную задавать ничего не требуется.
"""

from __future__ import annotations

import os
import socket
from dataclasses import dataclass
from pathlib import Path

from mir.common.logging import get_logger

__all__ = [
    "COMMON_PROXY_PORTS",
    "ProxyProbe",
    "autodetect_proxy",
    "check_proxy",
    "running_in_container",
    "scan_common_ports",
]

_log = get_logger(__name__)

COMMON_PROXY_PORTS: tuple[tuple[int, str], ...] = (
    (10808, "Happ, v2rayN, Xray — SOCKS5"),
    (10809, "v2rayN — HTTP"),
    (2080, "Nekoray, NekoBox — SOCKS5"),
    (2081, "Nekoray — HTTP"),
    (7890, "Clash, Clash Verge — смешанный"),
    (7891, "Clash — SOCKS5"),
    (12334, "Hiddify — SOCKS5"),
    (12335, "Hiddify — HTTP"),
    (1080, "стандартный порт SOCKS5"),
    (8080, "стандартный порт HTTP-прокси"),
    (9050, "Tor — SOCKS5"),
)
"""Порты популярных VPN-клиентов с пояснением, кому они принадлежат."""

DOCKER_HOST = "host.docker.internal"
LOCAL_HOST = "127.0.0.1"


@dataclass(frozen=True)
class ProxyProbe:
    """Результат проверки одного адреса.

    Attributes:
        host: Проверенный хост.
        port: Проверенный порт.
        reachable: Порт принимает соединения.
        description: Кому обычно принадлежит порт.
        error: Причина отказа.
    """

    host: str
    port: int
    reachable: bool
    description: str = ""
    error: str = ""

    @property
    def url(self) -> str:
        """Готовая строка для параметра `--proxy`."""
        return f"socks5://{self.host}:{self.port}"


def check_proxy(host: str, port: int, timeout: float = 1.5) -> ProxyProbe:
    """Проверить, принимает ли адрес TCP-соединения.

    Проверяется только доступность порта: полноценное рукопожатие SOCKS
    здесь не выполняется, потому что для диагностики достаточно понять,
    слушает ли кто-нибудь этот адрес.

    Args:
        host: Имя хоста или адрес.
        port: Порт.
        timeout: Таймаут соединения в секундах.
    """
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return ProxyProbe(host=host, port=port, reachable=True)
    except TimeoutError:
        return ProxyProbe(host, port, False, error="таймаут")
    except socket.gaierror as exc:
        return ProxyProbe(host, port, False, error=f"имя не разрешается ({exc.strerror})")
    except OSError as exc:
        return ProxyProbe(
            host,
            port,
            False,
            error="соединение отклонено" if exc.errno in (111, 61, 10061) else str(exc),
        )


def running_in_container() -> bool:
    """Определить, выполняется ли код внутри контейнера.

    От этого зависит, по какому адресу искать прокси: изнутри контейнера
    хост доступен как `host.docker.internal`, снаружи — как `127.0.0.1`.
    """
    if Path("/.dockerenv").exists():
        return True
    if os.environ.get("MIR_IN_CONTAINER"):
        return True
    try:
        return "docker" in Path("/proc/1/cgroup").read_text(encoding="utf-8")
    except OSError:
        return False


def _default_hosts() -> tuple[str, ...]:
    return (DOCKER_HOST, LOCAL_HOST) if running_in_container() else (LOCAL_HOST,)


def scan_common_ports(hosts: tuple[str, ...] | None = None) -> list[ProxyProbe]:
    """Перебрать типичные порты VPN-клиентов.

    Args:
        hosts: Адреса для проверки. По умолчанию подбираются по окружению.

    Returns:
        Только доступные адреса, в порядке проверки.
    """
    found: list[ProxyProbe] = []
    for host in hosts or _default_hosts():
        for port, description in COMMON_PROXY_PORTS:
            probe = check_proxy(host, port, timeout=0.7)
            if probe.reachable:
                found.append(ProxyProbe(host, port, True, description=description))
                _log.info("найден локальный прокси: %s (%s)", probe.url, description)
    return found


def autodetect_proxy() -> str | None:
    """Подобрать адрес прокси автоматически.

    Вызывается, когда прямое соединение не удалось. Возвращает первый
    отозвавшийся адрес — этого достаточно, потому что на машине обычно
    работает один VPN-клиент.

    Returns:
        Строка вида `socks5://127.0.0.1:10808` или `None`, если ничего не найдено.
    """
    found = scan_common_ports()
    return found[0].url if found else None
