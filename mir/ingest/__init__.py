"""Загрузка ролика и подготовка материала — первый этап конвейера.

Превращает пользовательский ввод (ссылку или путь к файлу) в
[`MediaBundle`][mir.common.types.MediaBundle]: видеофайл, извлечённую
аудиодорожку и технические параметры.

Example:
    >>> from mir.config import load_config
    >>> from mir.ingest import ingest
    >>> bundle = ingest("https://youtu.be/abc", config=load_config())  # doctest: +SKIP
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from mir.common.logging import get_logger
from mir.common.types import MediaBundle, ProgressCallback, VideoInfo
from mir.config import MirConfig, load_config
from mir.ingest.cache import CacheEntry, MediaCache
from mir.ingest.demuxer import Demuxer, ProbeResult, find_ffmpeg
from mir.ingest.downloader import Downloader
from mir.ingest.sources import (
    SUPPORTED_EXTENSIONS,
    detect_platform,
    extract_video_id,
    is_url,
    normalize_url,
    resolve_source,
)

__all__ = [
    "SUPPORTED_EXTENSIONS",
    "CacheEntry",
    "Demuxer",
    "Downloader",
    "MediaCache",
    "ProbeResult",
    "detect_platform",
    "extract_video_id",
    "find_ffmpeg",
    "ingest",
    "is_url",
    "normalize_url",
    "probe",
    "resolve_source",
]

_log = get_logger(__name__)


def ingest(
    source: str | Path,
    config: MirConfig | None = None,
    work_dir: Path | None = None,
    progress: ProgressCallback | None = None,
    use_cache: bool = True,
) -> MediaBundle:
    """Выполнить этап загрузки целиком.

    Args:
        source: Ссылка на ролик или путь к локальному файлу.
        config: Конфигурация. По умолчанию загружается стандартная.
        work_dir: Каталог для промежуточных файлов. По умолчанию — временный.
        progress: Колбэк прогресса.
        use_cache: Использовать кэш скачанных роликов.

    Returns:
        Подготовленный материал для этапов `vision` и `audio`.

    Raises:
        UnsupportedSourceError: Источник не распознан.
        DownloadError: Не удалось скачать ролик.
        DemuxError: Файл повреждён или без звука.
    """
    config = config or load_config()

    # FFmpeg проверяется до скачивания, а не после. Он нужен и для склейки
    # раздельных потоков YouTube, и для извлечения аудиодорожки, поэтому
    # без него смысла качать нет: пользователь ждал бы минуту загрузки,
    # чтобы получить отказ на последнем шаге.
    find_ffmpeg()

    work_dir = work_dir or Path(tempfile.mkdtemp(prefix="mir_"))
    work_dir.mkdir(parents=True, exist_ok=True)

    downloader = Downloader(config.ingest)
    platform, resolved = resolve_source(source)

    title: str | None = None
    if platform.name != "LOCAL_FILE":
        try:
            title = downloader.probe(str(source)).title
        except Exception as exc:
            _log.debug("не удалось получить название заранее: %s", exc)

    video_path = downloader.fetch(str(source), progress=progress, use_cache=use_cache)

    demuxer = Demuxer(config.ingest)
    return demuxer.split(
        video_path,
        work_dir=work_dir,
        title=title,
        source_url=resolved if platform.name != "LOCAL_FILE" else None,
        platform=platform,
        progress=progress,
    )


def probe(source: str, config: MirConfig | None = None) -> VideoInfo:
    """Получить метаданные ролика без скачивания.

    Args:
        source: Ссылка или путь.
        config: Конфигурация.
    """
    config = config or load_config()
    return Downloader(config.ingest).probe(source)
