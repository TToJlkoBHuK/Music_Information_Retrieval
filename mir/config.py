"""Загрузка и валидация конфигурации (NF-18).

Приоритет источников, от низшего к высшему:

1. `config/default.toml` в репозитории;
2. пользовательский файл `~/.mir/config.toml`;
3. переменные окружения `MIR_<СЕКЦИЯ>_<КЛЮЧ>`;
4. явные значения, переданные из интерфейса или CLI.

Валидация выполняется при загрузке: понятная ошибка сразу лучше,
чем падение в глубине конвейера через десять минут обработки.
"""

from __future__ import annotations

import os
import sys
from collections.abc import Mapping
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, TypeVar, get_type_hints

from mir.common.errors import ConfigError
from mir.common.logging import get_logger

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover - ветка зависит от версии интерпретатора
    import tomli as tomllib

__all__ = [
    "AudioConfig",
    "ExportConfig",
    "FusionConfig",
    "IngestConfig",
    "MirConfig",
    "NotationConfig",
    "VisionConfig",
    "load_config",
]

_log = get_logger(__name__)

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "default.toml"
USER_CONFIG_PATH = Path.home() / ".mir" / "config.toml"
ENV_PREFIX = "MIR_"

T = TypeVar("T")


@dataclass(frozen=True)
class IngestConfig:
    """Параметры загрузки и подготовки материала.

    Режимы `proxy_mode`:

    * `auto` — сначала прямое соединение (при включённом VPN в режиме TUN
      этого достаточно), при неудаче адрес прокси подбирается автоматически.
      Пользователю настраивать нечего;
    * `none` — только прямое соединение;
    * `manual` — использовать адрес из `proxy` и не искать альтернатив.
    """

    max_height: int = 1080
    prefer_fps: int = 60
    timeout_sec: int = 300
    cache_dir: str = "~/.mir_cache"
    cache_max_gb: float = 20.0
    proxy: str = ""
    proxy_mode: str = "auto"
    audio_sample_rate: int = 16000
    audio_channels: int = 1
    retries: int = 3

    @property
    def cache_path(self) -> Path:
        """Каталог кэша с раскрытым `~`."""
        return Path(self.cache_dir).expanduser()


@dataclass(frozen=True)
class KeyboardConfig:
    """Детекция клавиатуры в кадре."""

    sample_frames: int = 90
    min_key_width_px: float = 4.0
    black_key_height_ratio: float = 0.6


@dataclass(frozen=True)
class TrackerConfig:
    """Отслеживание подсветки клавиш.

    `off_threshold` ниже `on_threshold` — это гистерезис против дребезга.
    """

    on_threshold: float = 0.25
    off_threshold: float = 0.15
    min_frames_on: int = 2
    velocity_gamma: float = 0.7
    min_note_duration: float = 0.06


@dataclass(frozen=True)
class VisionConfig:
    """Параметры анализа видеоряда."""

    keyboard: KeyboardConfig = field(default_factory=KeyboardConfig)
    tracker: TrackerConfig = field(default_factory=TrackerConfig)


@dataclass(frozen=True)
class AudioConfig:
    """Параметры нейросетевой транскрипции аудио."""

    chunk_seconds: float = 60.0
    overlap_seconds: float = 2.0
    onset_threshold: float = 0.3
    offset_threshold: float = 0.3
    hop_length: int = 256


@dataclass(frozen=True)
class FusionConfig:
    """Параметры слияния видео- и аудиоканалов."""

    onset_tolerance: float = 0.08
    max_av_offset: float = 1.0
    min_offset_confidence: float = 0.3
    duration_weight: float = 0.3
    octave_artifact_ratio: float = 0.8


@dataclass(frozen=True)
class NotationConfig:
    """Параметры построения нотного текста."""

    quantize_strength: float = 0.7
    max_subdivision: int = 16
    allow_triplets: bool = True
    min_duration_beats: float = 0.0625
    hand_split_pitch: int = 60
    dynamics_window_beats: float = 4.0


@dataclass(frozen=True)
class ExportConfig:
    """Параметры выгрузки MIDI, MusicXML и PDF."""

    ppq: int = 480
    page_size: str = "A4"
    staff_size_mm: float = 7.0
    show_measure_numbers: bool = True


@dataclass(frozen=True)
class MirConfig:
    """Полная конфигурация проекта."""

    ingest: IngestConfig = field(default_factory=IngestConfig)
    vision: VisionConfig = field(default_factory=VisionConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)
    fusion: FusionConfig = field(default_factory=FusionConfig)
    notation: NotationConfig = field(default_factory=NotationConfig)
    export: ExportConfig = field(default_factory=ExportConfig)


def _build(cls: type[T], data: Mapping[str, Any], path: str = "") -> T:
    """Собрать датакласс из словаря, отвергая неизвестные ключи.

    Типы полей разрешаются через `get_type_hints`: из-за
    `from __future__ import annotations` аннотации хранятся строками,
    и вложенные секции иначе не распознаются.
    """
    assert is_dataclass(cls)
    known = {f.name for f in fields(cls)}
    unknown = set(data) - known
    if unknown:
        where = f" в секции [{path}]" if path else ""
        raise ConfigError(
            f"неизвестные параметры{where}: {', '.join(sorted(unknown))}",
            f"В настройках указаны неизвестные параметры: {', '.join(sorted(unknown))}",
        )

    hints = get_type_hints(cls)
    kwargs: dict[str, Any] = {}
    for name in known:
        if name not in data:
            continue
        value = data[name]
        field_type = hints.get(name)
        if isinstance(field_type, type) and is_dataclass(field_type) and isinstance(value, Mapping):
            nested = f"{path}.{name}" if path else name
            kwargs[name] = _build(field_type, value, nested)
        else:
            kwargs[name] = value
    return cls(**kwargs)


def _read_toml(path: Path) -> dict[str, Any]:
    try:
        with path.open("rb") as fh:
            data: dict[str, Any] = tomllib.load(fh)
        return data
    except OSError as exc:
        raise ConfigError(f"не удалось прочитать {path}: {exc}") from exc
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(
            f"синтаксическая ошибка в {path}: {exc}",
            f"Файл настроек {path.name} повреждён",
        ) from exc


def _deep_merge(base: dict[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(base)
    for key, value in override.items():
        current = result.get(key)
        if isinstance(current, dict) and isinstance(value, Mapping):
            result[key] = _deep_merge(current, value)
        else:
            result[key] = value
    return result


def _coerce(raw: str) -> Any:
    """Привести значение переменной окружения к типу Python."""
    low = raw.lower()
    if low in {"true", "false"}:
        return low == "true"
    try:
        return int(raw)
    except ValueError:
        pass
    try:
        return float(raw)
    except ValueError:
        return raw


def _env_overrides() -> dict[str, Any]:
    """Собрать переопределения из `MIR_<СЕКЦИЯ>_<КЛЮЧ>`.

    Example:
        `MIR_INGEST_PROXY=socks5://localhost:1080` переопределяет `ingest.proxy`.
    """
    known_sections = {f.name for f in fields(MirConfig)}
    out: dict[str, Any] = {}
    for env_key, raw in os.environ.items():
        if not env_key.startswith(ENV_PREFIX):
            continue
        _, _, rest = env_key.partition(ENV_PREFIX)
        section, _, key = rest.lower().partition("_")
        if section in known_sections and key:
            out.setdefault(section, {})[key] = _coerce(raw)
    return out


def validate(config: MirConfig) -> list[str]:
    """Проверить инварианты, невыразимые типами.

    Returns:
        Список описаний проблем. Пустой список — конфигурация корректна.
    """
    problems: list[str] = []
    tracker = config.vision.tracker
    if tracker.off_threshold >= tracker.on_threshold:
        problems.append(
            "vision.tracker.off_threshold должен быть меньше on_threshold, "
            "иначе гистерезис не работает и подсветка будет дребезжать"
        )
    if config.audio.overlap_seconds >= config.audio.chunk_seconds:
        problems.append("audio.overlap_seconds должен быть меньше chunk_seconds")
    if not 0.0 <= config.notation.quantize_strength <= 1.0:
        problems.append("notation.quantize_strength должен лежать в диапазоне 0..1")
    if not 21 <= config.notation.hand_split_pitch <= 108:
        problems.append("notation.hand_split_pitch вне диапазона фортепиано 21..108")
    if config.ingest.max_height < 240:
        problems.append("ingest.max_height ниже 240 — распознавание невозможно")
    if config.ingest.audio_sample_rate not in (16000, 22050, 44100, 48000):
        problems.append("ingest.audio_sample_rate: допустимы 16000, 22050, 44100, 48000")
    if config.export.ppq % 12 != 0:
        problems.append("export.ppq должен делиться на 12, иначе триоли округляются")
    if config.ingest.proxy_mode not in ("auto", "none", "manual"):
        problems.append("ingest.proxy_mode: допустимы значения auto, none, manual")
    if config.vision.tracker.min_note_duration <= 0:
        problems.append("vision.tracker.min_note_duration должен быть положительным")
    if config.ingest.proxy_mode == "manual" and not config.ingest.proxy:
        problems.append("ingest.proxy_mode=manual требует заполненного ingest.proxy")
    return problems


def load_config(
    path: Path | None = None,
    overrides: Mapping[str, Any] | None = None,
    use_env: bool = True,
    use_user_config: bool = True,
) -> MirConfig:
    """Загрузить конфигурацию с учётом всех источников.

    Args:
        path: Файл вместо `config/default.toml`.
        overrides: Значения высшего приоритета: `{"ingest": {"proxy": "..."}}`.
        use_env: Учитывать переменные окружения `MIR_*`.
        use_user_config: Учитывать `~/.mir/config.toml`.

    Raises:
        ConfigError: Неизвестный параметр или нарушенный инвариант.
    """
    base_path = path or DEFAULT_CONFIG_PATH
    data: dict[str, Any] = _read_toml(base_path) if base_path.exists() else {}
    if not data:
        _log.warning("файл конфигурации %s не найден, взяты значения по умолчанию", base_path)

    if use_user_config and USER_CONFIG_PATH.exists():
        data = _deep_merge(data, _read_toml(USER_CONFIG_PATH))
        _log.debug("применены пользовательские настройки из %s", USER_CONFIG_PATH)

    if use_env:
        data = _deep_merge(data, _env_overrides())

    if overrides:
        data = _deep_merge(data, overrides)

    config = _build(MirConfig, data)

    problems = validate(config)
    if problems:
        raise ConfigError(
            "конфигурация некорректна: " + "; ".join(problems),
            "Настройки заданы неверно:\n" + "\n".join(f"• {p}" for p in problems),
        )
    return config
