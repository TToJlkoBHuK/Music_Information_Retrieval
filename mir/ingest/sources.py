"""Распознавание источника видео (F-01, F-02, F-04).

Нормализация ссылки нужна не только для порядка: без неё кэш считает
`youtu.be/abc` и `youtube.com/watch?v=abc&t=30` разными роликами
и качает один и тот же файл дважды.
"""

from __future__ import annotations

import re
from pathlib import Path
from urllib.parse import parse_qs, urlparse, urlunparse

from mir.common.enums import Platform
from mir.common.errors import UnsupportedSourceError

__all__ = [
    "SUPPORTED_EXTENSIONS",
    "detect_platform",
    "extract_video_id",
    "is_url",
    "normalize_url",
    "resolve_source",
]

SUPPORTED_EXTENSIONS = frozenset({".mp4", ".mkv", ".webm", ".avi", ".mov", ".m4v"})
"""Контейнеры, которые открывает локальный режим."""

_HOST_PLATFORMS: dict[str, Platform] = {
    "youtube.com": Platform.YOUTUBE,
    "youtu.be": Platform.YOUTUBE,
    "music.youtube.com": Platform.YOUTUBE,
    "vk.com": Platform.VK_VIDEO,
    "vkvideo.ru": Platform.VK_VIDEO,
    "rutube.ru": Platform.RUTUBE,
}

_TRACKING_PARAMS = frozenset(
    {
        "t",
        "list",
        "index",
        "start_radio",
        "feature",
        "ab_channel",
        "pp",
        "si",
        "utm_source",
        "utm_medium",
        "utm_campaign",
        "utm_content",
        "utm_term",
    }
)

_YOUTUBE_ID = re.compile(r"^[A-Za-z0-9_-]{11}$")


def is_url(source: str) -> bool:
    """Отличить ссылку от пути к файлу."""
    parsed = urlparse(source.strip())
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def _hostname(url: str) -> str:
    host = (urlparse(url).hostname or "").lower()
    return host[4:] if host.startswith("www.") else host


def detect_platform(source: str) -> Platform:
    """Определить площадку по ссылке или пути.

    Args:
        source: Ссылка или путь к файлу.

    Example:
        >>> detect_platform("https://youtu.be/dQw4w9WgXcQ")
        <Platform.YOUTUBE: 'youtube'>
        >>> detect_platform("./video.mp4")
        <Platform.LOCAL_FILE: 'local'>
    """
    source = source.strip()
    if not source:
        return Platform.UNKNOWN
    if not is_url(source):
        return Platform.LOCAL_FILE
    return _HOST_PLATFORMS.get(_hostname(source), Platform.UNKNOWN)


def extract_video_id(url: str) -> str | None:
    """Вытащить идентификатор ролика.

    Поддерживаются форматы YouTube (`watch?v=`, `youtu.be/`, `shorts/`,
    `embed/`) и Rutube (`/video/<id>/`). Для VK идентификатор составной
    (`video-123_456`), поэтому возвращается как есть.
    """
    platform = detect_platform(url)
    parsed = urlparse(url)

    if platform is Platform.YOUTUBE:
        if _hostname(url) == "youtu.be":
            candidate = parsed.path.lstrip("/").split("/")[0]
            return candidate if _YOUTUBE_ID.match(candidate) else None
        query_id = parse_qs(parsed.query).get("v", [None])[0]
        if query_id and _YOUTUBE_ID.match(query_id):
            return query_id
        match = re.search(r"/(?:shorts|embed|live|v)/([A-Za-z0-9_-]{11})", parsed.path)
        return match.group(1) if match else None

    if platform is Platform.RUTUBE:
        match = re.search(r"/video/([A-Za-z0-9]+)", parsed.path)
        return match.group(1) if match else None

    if platform is Platform.VK_VIDEO:
        match = re.search(r"(video-?\d+_\d+)", url)
        return match.group(1) if match else None

    return None


def normalize_url(url: str) -> str:
    """Привести ссылку к каноническому виду.

    Отбрасывает трекинг-параметры и метку времени, разворачивает короткие
    ссылки YouTube. Результат используется как ключ кэша (F-05).

    Example:
        >>> normalize_url("https://youtu.be/dQw4w9WgXcQ?t=42")
        'https://www.youtube.com/watch?v=dQw4w9WgXcQ'
    """
    url = url.strip()
    platform = detect_platform(url)

    if platform is Platform.YOUTUBE:
        video_id = extract_video_id(url)
        if video_id:
            return f"https://www.youtube.com/watch?v={video_id}"

    parsed = urlparse(url)
    kept = {
        key: value for key, value in parse_qs(parsed.query).items() if key not in _TRACKING_PARAMS
    }
    query = "&".join(f"{k}={v[0]}" for k, v in sorted(kept.items()))
    host = parsed.netloc.lower()
    path = parsed.path.rstrip("/") or "/"
    return urlunparse((parsed.scheme, host, path, "", query, ""))


def resolve_source(source: str | Path) -> tuple[Platform, str]:
    """Разобрать пользовательский ввод.

    Args:
        source: Ссылка или путь к файлу.

    Returns:
        Площадка и нормализованная ссылка либо абсолютный путь строкой.

    Raises:
        UnsupportedSourceError: Ссылка не распознана, файл отсутствует
            или имеет неподдерживаемое расширение.
    """
    raw = str(source).strip()
    if not raw:
        raise UnsupportedSourceError("пустой источник")

    platform = detect_platform(raw)

    if platform is Platform.UNKNOWN:
        raise UnsupportedSourceError(
            f"неизвестная площадка: {raw}",
            "Поддерживаются ссылки на YouTube, VK Видео и Rutube. "
            "Локальный файл можно открыть кнопкой «Открыть файл»",
        )

    if platform is Platform.LOCAL_FILE:
        path = Path(raw).expanduser()
        if not path.exists():
            raise UnsupportedSourceError(f"файл не найден: {path}", f"Файл не найден: {path.name}")
        if not path.is_file():
            raise UnsupportedSourceError(
                f"это не файл: {path}", f"{path.name} — это папка, а не видеофайл"
            )
        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            supported = ", ".join(sorted(SUPPORTED_EXTENSIONS))
            raise UnsupportedSourceError(
                f"неподдерживаемое расширение {path.suffix}",
                f"Формат {path.suffix} не поддерживается. Доступны: {supported}",
            )
        return platform, str(path.resolve())

    return platform, normalize_url(raw)
