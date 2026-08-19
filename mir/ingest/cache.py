"""Кэш скачанных роликов (F-05).

При подборе параметров пользователь разбирает один ролик много раз.
Без кэша каждый запуск — это минуты ожидания и лишняя нагрузка на VPN.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from mir.common.logging import get_logger
from mir.ingest.sources import normalize_url

__all__ = ["CacheEntry", "MediaCache"]

_log = get_logger(__name__)
_INDEX_NAME = "index.json"


@dataclass(frozen=True)
class CacheEntry:
    """Запись индекса кэша."""

    key: str
    url: str
    filename: str
    size_bytes: int
    created_at: float
    title: str | None = None

    @property
    def age_days(self) -> float:
        """Возраст записи в днях."""
        return (time.time() - self.created_at) / 86400


class MediaCache:
    """Файловый кэш видео, адресуемый по нормализованной ссылке.

    Индекс хранится в `index.json` рядом с файлами. Ключ — sha256
    от нормализованной ссылки, поэтому разные формы одного адреса
    попадают в одну запись.

    Args:
        root: Каталог кэша.
        max_size_gb: Порог, при превышении которого вытесняются
            самые старые записи.
    """

    def __init__(self, root: Path, max_size_gb: float = 20.0) -> None:
        self.root = Path(root).expanduser()
        self.max_size_bytes = int(max_size_gb * 1024**3)
        self.root.mkdir(parents=True, exist_ok=True)
        self._index_path = self.root / _INDEX_NAME
        self._index: dict[str, CacheEntry] = self._load_index()

    @staticmethod
    def key_for(url: str) -> str:
        """Ключ кэша для ссылки."""
        return hashlib.sha256(normalize_url(url).encode("utf-8")).hexdigest()[:16]

    def _load_index(self) -> dict[str, CacheEntry]:
        if not self._index_path.exists():
            return {}
        try:
            raw = json.loads(self._index_path.read_text(encoding="utf-8"))
            return {k: CacheEntry(**v) for k, v in raw.items()}
        except (OSError, ValueError, TypeError) as exc:
            _log.warning("индекс кэша повреждён (%s), создаётся заново", exc)
            return {}

    def _save_index(self) -> None:
        payload = {k: asdict(v) for k, v in self._index.items()}
        tmp = self._index_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(self._index_path)

    def get(self, url: str) -> Path | None:
        """Найти скачанный ролик.

        Returns:
            Путь к файлу или `None`. Запись с исчезнувшим файлом
            автоматически убирается из индекса.
        """
        entry = self._index.get(self.key_for(url))
        if entry is None:
            return None
        path = self.root / entry.filename
        if not path.exists():
            _log.debug("файл %s пропал, запись удалена из индекса", entry.filename)
            del self._index[entry.key]
            self._save_index()
            return None
        _log.info("ролик найден в кэше: %s", entry.filename)
        return path

    def put(self, url: str, path: Path, title: str | None = None) -> Path:
        """Поместить файл в кэш.

        Файл переносится внутрь каталога кэша; если он уже там, копия
        не создаётся.

        Returns:
            Путь к файлу внутри кэша.
        """
        key = self.key_for(url)
        target = self.root / f"{key}{path.suffix}"
        if path.resolve() != target.resolve():
            shutil.move(str(path), str(target))

        self._index[key] = CacheEntry(
            key=key,
            url=normalize_url(url),
            filename=target.name,
            size_bytes=target.stat().st_size,
            created_at=time.time(),
            title=title,
        )
        self._save_index()
        self._evict_if_needed()
        return target

    def path_for(self, url: str, suffix: str = ".mp4") -> Path:
        """Путь, по которому будет лежать файл для этой ссылки."""
        return self.root / f"{self.key_for(url)}{suffix}"

    @property
    def total_size(self) -> int:
        """Суммарный объём кэша в байтах."""
        return sum(e.size_bytes for e in self._index.values())

    def _evict_if_needed(self) -> None:
        if self.total_size <= self.max_size_bytes:
            return
        by_age = sorted(self._index.values(), key=lambda e: e.created_at)
        freed = 0
        for entry in by_age:
            if self.total_size - freed <= self.max_size_bytes:
                break
            (self.root / entry.filename).unlink(missing_ok=True)
            del self._index[entry.key]
            freed += entry.size_bytes
            _log.info("вытеснено из кэша: %s", entry.filename)
        self._save_index()

    def clear(self, older_than_days: float | None = None) -> int:
        """Очистить кэш.

        Args:
            older_than_days: Удалять только записи старше указанного
                возраста. `None` — очистить всё.

        Returns:
            Освобождённый объём в байтах.
        """
        freed = 0
        for entry in list(self._index.values()):
            if older_than_days is not None and entry.age_days < older_than_days:
                continue
            (self.root / entry.filename).unlink(missing_ok=True)
            freed += entry.size_bytes
            del self._index[entry.key]
        self._save_index()
        _log.info("кэш очищен, освобождено %.1f МБ", freed / 1024**2)
        return freed

    def __len__(self) -> int:
        return len(self._index)
