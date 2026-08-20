"""Загрузка роликов через yt-dlp (F-01, F-03, F-04).

Ошибки yt-dlp переводятся в [`DownloadError`][mir.common.errors.DownloadError]
с русским объяснением: пользователь должен понять, нужен ли ему VPN,
а не читать traceback.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

from mir.common.enums import Platform, Stage
from mir.common.errors import DownloadError
from mir.common.logging import get_logger
from mir.common.types import ProgressCallback, VideoInfo
from mir.config import IngestConfig
from mir.ingest.cache import MediaCache
from mir.ingest.sources import resolve_source

__all__ = ["Downloader"]

_log = get_logger(__name__)

_PROXY_REFUSED = re.compile(
    r"socks|proxy.*(refused|failed)|connection refused|errno 111|10061", re.I
)

_NETWORK_FAILURE = re.compile(
    r"unable to download|failed to establish|connection|timed out|timeout|"
    r"resolve host|network is unreachable|tunnel|refused|socks",
    re.I,
)
"""Ошибки, при которых имеет смысл повторить попытку через прокси."""

_ERROR_HINTS: tuple[tuple[re.Pattern[str], str], ...] = (
    (
        re.compile(r"unable to download|network|timed out|connection|resolve host|tunnel", re.I),
        "Не удалось подключиться к сайту. Проверьте интернет, включите VPN "
        "или укажите прокси в настройках",
    ),
    (
        re.compile(r"private video|members-only|login required|sign in", re.I),
        "Видео закрыто автором: доступ только для подписчиков или по входу в аккаунт",
    ),
    (
        re.compile(r"video unavailable|removed|deleted|does not exist", re.I),
        "Видео недоступно: удалено или скрыто автором",
    ),
    (
        re.compile(r"age.?restricted|confirm your age|inappropriate", re.I),
        "Видео с возрастным ограничением скачать не удалось",
    ),
    (
        re.compile(r"geo|not available in your country|blocked", re.I),
        "Видео недоступно в вашем регионе. Попробуйте включить VPN или указать прокси",
    ),
    (
        re.compile(r"copyright|blocked it on copyright", re.I),
        "Видео заблокировано по требованию правообладателя",
    ),
    (
        re.compile(r"playlist", re.I),
        "Это ссылка на плейлист. Укажите ссылку на конкретное видео",
    ),
    (
        re.compile(r"live event|is live|premiere", re.I),
        "Прямые трансляции не поддерживаются. Дождитесь публикации записи",
    ),
)


def _explain(message: str, platform: Platform, proxy: str) -> str:
    """Подобрать объяснение ошибки для пользователя.

    Отдельно разбирается случай, когда прокси указан, но не отвечает:
    «включите VPN» здесь только сбивает с толку — VPN может быть включён,
    а порт прокси указан неверно.
    """
    if proxy and _PROXY_REFUSED.search(message):
        return (
            f"Прокси {proxy} не отвечает.\n\n"
            "Проверьте, что VPN включён и работает, либо укажите другой "
            "адрес прокси в настройках"
        )

    for pattern, hint in _ERROR_HINTS:
        if pattern.search(message):
            if platform.needs_proxy_in_russia and not proxy and "VPN" not in hint:
                return f"{hint}.\n\nYouTube из России требует VPN или прокси"
            return hint

    if platform.needs_proxy_in_russia and not proxy:
        return (
            "Не удалось загрузить видео с YouTube. Из России сайт доступен "
            "только через VPN — включите его или укажите прокси в настройках"
        )
    return "Не удалось загрузить видео"


class Downloader:
    """Загрузчик роликов с кэшированием и поддержкой прокси.

    Args:
        config: Параметры загрузки.
        cache: Кэш. Если не передан, создаётся по пути из конфигурации.

    Example:
        >>> from mir.config import load_config
        >>> dl = Downloader(load_config().ingest)   # doctest: +SKIP
        >>> path = dl.fetch("https://youtu.be/abc")  # doctest: +SKIP
    """

    def __init__(self, config: IngestConfig, cache: MediaCache | None = None) -> None:
        self.config = config
        self.cache = cache or MediaCache(config.cache_path, config.cache_max_gb)
        self._active_proxy: str | None = self._initial_proxy()
        self._autodetect_tried = False

    def _initial_proxy(self) -> str | None:
        """Прокси для первой попытки.

        В режиме `auto` первая попытка идёт напрямую: при VPN в режиме TUN
        соединение проходит, и подбирать ничего не нужно.
        """
        if self.config.proxy_mode == "none":
            return None
        if self.config.proxy_mode == "manual":
            return self.config.proxy or None
        return self.config.proxy or None

    def _try_autodetect(self) -> bool:
        """Подобрать прокси после неудачи. Возвращает True, если стоит повторить."""
        if self.config.proxy_mode != "auto" or self._autodetect_tried:
            return False
        self._autodetect_tried = True

        from mir.ingest.proxycheck import autodetect_proxy

        _log.info("прямое соединение не удалось, ищу локальный прокси")
        found = autodetect_proxy()
        if found is None:
            _log.info("локальный прокси не найден")
            return False
        _log.info("повторная попытка через %s", found)
        self._active_proxy = found
        return True

    def _base_options(self) -> dict[str, Any]:
        options: dict[str, Any] = {
            "quiet": True,
            "no_warnings": True,
            "noprogress": True,
            "socket_timeout": self.config.timeout_sec,
            "retries": self.config.retries,
            "noplaylist": True,
            "logger": _YtdlpLogger(),
        }
        if self._active_proxy:
            options["proxy"] = self._active_proxy
        return options

    def _format_selector(self) -> str:
        """Выбор формата: высокий fps важнее максимального разрешения.

        При 60 fps шаг между кадрами 16 мс против 33 мс при 30 fps —
        это вдвое лучшее временное разрешение трекинга.
        """
        height = self.config.max_height
        fps = self.config.prefer_fps
        return (
            f"bestvideo[height<={height}][fps>={fps}]+bestaudio/"
            f"bestvideo[height<={height}]+bestaudio/"
            f"best[height<={height}]/best"
        )

    def probe(self, source: str) -> VideoInfo:
        """Получить метаданные без скачивания.

        Нужно, чтобы показать пользователю название и оценить время
        обработки до начала загрузки.

        Raises:
            DownloadError: Ролик недоступен.
        """
        import yt_dlp

        platform, url = resolve_source(source)
        if platform is Platform.LOCAL_FILE:
            path = Path(url)
            return VideoInfo(title=path.stem, duration=0.0, platform=platform, url=str(path))

        def extract() -> Any:
            with yt_dlp.YoutubeDL(self._base_options()) as ydl:
                return ydl.extract_info(url, download=False)

        try:
            info = self._run_with_fallback(extract)
        except Exception as exc:  # yt-dlp бросает разнородные исключения
            raise DownloadError(
                f"probe не удался для {url}: {exc}",
                _explain(str(exc), platform, self._active_proxy or ""),
            ) from exc

        if info is None:
            raise DownloadError(f"пустой ответ для {url}")

        return VideoInfo(
            title=info.get("title") or "Без названия",
            duration=float(info.get("duration") or 0.0),
            platform=platform,
            url=url,
            width=info.get("width"),
            height=info.get("height"),
            fps=info.get("fps"),
            filesize_approx=info.get("filesize_approx"),
        )

    def fetch(
        self,
        source: str,
        progress: ProgressCallback | None = None,
        use_cache: bool = True,
    ) -> Path:
        """Получить видеофайл.

        Локальный файл возвращается как есть, ссылка — скачивается
        или берётся из кэша.

        Args:
            source: Ссылка или путь.
            progress: Колбэк прогресса загрузки.
            use_cache: Использовать кэш.

        Returns:
            Путь к видеофайлу на диске.

        Raises:
            DownloadError: Загрузка не удалась.
            UnsupportedSourceError: Источник не распознан.
        """
        import yt_dlp

        platform, url = resolve_source(source)

        if platform is Platform.LOCAL_FILE:
            _log.info("используется локальный файл: %s", url)
            if progress:
                progress(Stage.DOWNLOAD, 1.0)
            return Path(url)

        if use_cache:
            cached = self.cache.get(url)
            if cached is not None:
                if progress:
                    progress(Stage.DOWNLOAD, 1.0)
                return cached

        target = self.cache.path_for(url, ".%(ext)s")
        options = self._base_options()
        options.update(
            {
                "format": self._format_selector(),
                "outtmpl": str(target),
                "merge_output_format": "mp4",
                "progress_hooks": [_make_hook(progress)] if progress else [],
            }
        )

        _log.info("загрузка %s (%s)", url, platform.value)

        def download() -> Any:
            fresh = self._base_options()
            fresh.update(
                {
                    "format": options["format"],
                    "outtmpl": options["outtmpl"],
                    "merge_output_format": options["merge_output_format"],
                    "progress_hooks": options["progress_hooks"],
                }
            )
            with yt_dlp.YoutubeDL(fresh) as ydl:
                return ydl.extract_info(url, download=True)

        try:
            info = self._run_with_fallback(download)
        except Exception as exc:
            raise DownloadError(
                f"скачивание {url} не удалось: {exc}",
                _explain(str(exc), platform, self._active_proxy or ""),
            ) from exc

        if info is None:
            raise DownloadError(f"yt-dlp вернул пустой результат для {url}")

        downloaded = self._locate_downloaded(url)
        if downloaded is None:
            raise DownloadError(
                f"файл не найден после загрузки {url}",
                "Видео скачалось, но файл не найден на диске",
            )

        if use_cache:
            downloaded = self.cache.put(url, downloaded, title=info.get("title"))
        if progress:
            progress(Stage.DOWNLOAD, 1.0)
        return downloaded

    def _run_with_fallback(self, action: Callable[[], Any]) -> Any:
        """Выполнить сетевую операцию, при сбое подобрать прокси и повторить.

        Смысл в том, чтобы пользователь ничего не настраивал: сначала пробуем
        напрямую, и только если соединения нет — ищем локальный прокси.
        """
        try:
            return action()
        except Exception as exc:
            if not _NETWORK_FAILURE.search(str(exc)):
                raise
            if not self._try_autodetect():
                raise
            return action()

    def _locate_downloaded(self, url: str) -> Path | None:
        """Найти файл: yt-dlp сам подставляет расширение в шаблон."""
        stem = self.cache.key_for(url)
        candidates = sorted(
            self.cache.root.glob(f"{stem}.*"),
            key=lambda p: p.stat().st_size,
            reverse=True,
        )
        for path in candidates:
            if path.suffix not in {".part", ".ytdl", ".json", ".tmp"}:
                return path
        return None


def _make_hook(progress: ProgressCallback) -> Callable[[dict[str, Any]], None]:
    def hook(status: dict[str, Any]) -> None:
        if status.get("status") != "downloading":
            return
        total = status.get("total_bytes") or status.get("total_bytes_estimate")
        done = status.get("downloaded_bytes", 0)
        if total:
            progress(Stage.DOWNLOAD, min(done / total, 1.0))

    return hook


class _YtdlpLogger:
    """Перенаправление вывода yt-dlp в `logging` (NF-19)."""

    def debug(self, msg: str) -> None:
        if not msg.startswith("[debug]"):
            _log.debug("yt-dlp: %s", msg)

    def info(self, msg: str) -> None:
        _log.debug("yt-dlp: %s", msg)

    def warning(self, msg: str) -> None:
        _log.warning("yt-dlp: %s", msg)

    def error(self, msg: str) -> None:
        _log.error("yt-dlp: %s", msg)
