"""Режимы работы с прокси (F-03).

Главное требование: пользователь ничего не настраивает. Включил VPN —
программа работает. Прокси подбирается сама и только при необходимости.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest

from mir.common.errors import ConfigError
from mir.config import load_config
from mir.ingest.cache import MediaCache
from mir.ingest.downloader import Downloader


def _config(mode: str, proxy: str = "") -> Any:
    overrides: dict[str, dict[str, object]] = {"ingest": {"proxy_mode": mode}}
    if proxy:
        overrides["ingest"]["proxy"] = proxy
    return load_config(overrides=overrides, use_user_config=False, use_env=False)


@pytest.fixture
def cache(tmp_path: Any) -> MediaCache:
    return MediaCache(tmp_path / "cache")


def _counting_probe(calls: dict[str, int]) -> Callable[[], str]:
    """Заглушка автопоиска прокси, считающая обращения к себе."""

    def probe() -> str:
        calls["count"] += 1
        return "socks5://x:1"

    return probe


class TestFirstAttempt:
    def test_auto_starts_direct(self, cache: MediaCache) -> None:
        """При VPN в режиме TUN прямого соединения достаточно."""
        downloader = Downloader(_config("auto").ingest, cache)
        assert downloader._active_proxy is None

    def test_none_stays_direct(self, cache: MediaCache) -> None:
        downloader = Downloader(_config("none").ingest, cache)
        assert downloader._active_proxy is None

    def test_manual_uses_given_address(self, cache: MediaCache) -> None:
        config = _config("manual", "socks5://127.0.0.1:10808")
        downloader = Downloader(config.ingest, cache)
        assert downloader._active_proxy == "socks5://127.0.0.1:10808"


class TestAutodetect:
    def test_auto_retries_after_network_error(
        self, cache: MediaCache, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Прямое соединение не прошло — прокси подбирается автоматически."""
        monkeypatch.setattr(
            "mir.ingest.proxycheck.autodetect_proxy", lambda: "socks5://127.0.0.1:10808"
        )
        downloader = Downloader(_config("auto").ingest, cache)

        attempts: list[str | None] = []

        def action() -> str:
            attempts.append(downloader._active_proxy)
            if len(attempts) == 1:
                raise OSError("Failed to establish a new connection: Connection refused")
            return "ok"

        assert downloader._run_with_fallback(action) == "ok"
        assert attempts == [None, "socks5://127.0.0.1:10808"]

    def test_none_mode_does_not_retry(
        self, cache: MediaCache, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            "mir.ingest.proxycheck.autodetect_proxy", lambda: "socks5://127.0.0.1:10808"
        )
        downloader = Downloader(_config("none").ingest, cache)

        def action() -> str:
            raise OSError("Connection refused")

        with pytest.raises(OSError):
            downloader._run_with_fallback(action)

    def test_no_proxy_found_reraises(
        self, cache: MediaCache, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr("mir.ingest.proxycheck.autodetect_proxy", lambda: None)
        downloader = Downloader(_config("auto").ingest, cache)

        def action() -> str:
            raise OSError("Connection refused")

        with pytest.raises(OSError):
            downloader._run_with_fallback(action)

    def test_non_network_error_not_retried(
        self, cache: MediaCache, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Приватное видео через прокси не откроется — повтор бессмыслен."""
        calls = {"count": 0}
        monkeypatch.setattr(
            "mir.ingest.proxycheck.autodetect_proxy",
            _counting_probe(calls),
        )
        downloader = Downloader(_config("auto").ingest, cache)

        def action() -> str:
            raise ValueError("Private video. Sign in if you've been granted access")

        with pytest.raises(ValueError):
            downloader._run_with_fallback(action)
        assert calls["count"] == 0

    def test_autodetect_runs_once(self, cache: MediaCache, monkeypatch: pytest.MonkeyPatch) -> None:
        """Повторный поиск при каждой ошибке только тормозил бы работу."""
        monkeypatch.setattr("mir.ingest.proxycheck.autodetect_proxy", lambda: "socks5://x:1")
        downloader = Downloader(_config("auto").ingest, cache)
        assert downloader._try_autodetect() is True
        assert downloader._try_autodetect() is False


class TestValidation:
    def test_unknown_mode_rejected(self) -> None:
        with pytest.raises(ConfigError, match="proxy_mode"):
            _config("magic")

    def test_manual_requires_address(self) -> None:
        with pytest.raises(ConfigError, match="manual"):
            _config("manual")
