"""Загрузка и валидация конфигурации."""

from __future__ import annotations

from pathlib import Path

import pytest

from mir.common.errors import ConfigError
from mir.config import MirConfig, load_config, validate


@pytest.fixture
def clean_load():  # type: ignore[no-untyped-def]
    """Загрузка без пользовательского файла и переменных окружения."""

    def _load(**kwargs: object) -> MirConfig:
        kwargs.setdefault("use_user_config", False)
        kwargs.setdefault("use_env", False)
        return load_config(**kwargs)  # type: ignore[arg-type]

    return _load


class TestDefaults:
    def test_loads(self, clean_load) -> None:  # type: ignore[no-untyped-def]
        config = clean_load()
        assert config.ingest.max_height == 1080
        assert config.export.ppq == 480

    def test_defaults_are_valid(self, clean_load) -> None:  # type: ignore[no-untyped-def]
        assert validate(clean_load()) == []

    def test_cache_path_expands_home(self, clean_load) -> None:  # type: ignore[no-untyped-def]
        assert "~" not in str(clean_load().ingest.cache_path)


class TestOverrides:
    def test_override_applies(self, clean_load) -> None:  # type: ignore[no-untyped-def]
        config = clean_load(overrides={"ingest": {"proxy": "socks5://127.0.0.1:1080"}})
        assert config.ingest.proxy == "socks5://127.0.0.1:1080"

    def test_nested_override(self, clean_load) -> None:  # type: ignore[no-untyped-def]
        config = clean_load(overrides={"vision": {"tracker": {"min_frames_on": 5}}})
        assert config.vision.tracker.min_frames_on == 5
        assert config.vision.tracker.on_threshold == 0.25  # соседи не затёрты

    def test_env_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("MIR_INGEST_MAX_HEIGHT", "720")
        config = load_config(use_user_config=False)
        assert config.ingest.max_height == 720

    def test_env_types_coerced(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("MIR_NOTATION_ALLOW_TRIPLETS", "false")
        monkeypatch.setenv("MIR_NOTATION_QUANTIZE_STRENGTH", "0.5")
        config = load_config(use_user_config=False)
        assert config.notation.allow_triplets is False
        assert config.notation.quantize_strength == 0.5

    def test_explicit_beats_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("MIR_INGEST_MAX_HEIGHT", "720")
        config = load_config(overrides={"ingest": {"max_height": 480}}, use_user_config=False)
        assert config.ingest.max_height == 480


class TestValidation:
    def test_unknown_key_rejected(self, clean_load) -> None:  # type: ignore[no-untyped-def]
        with pytest.raises(ConfigError, match="неизвестные параметры"):
            clean_load(overrides={"ingest": {"nonexistent": 1}})

    def test_hysteresis_enforced(self, clean_load) -> None:  # type: ignore[no-untyped-def]
        """off_threshold >= on_threshold ломает антидребезг."""
        with pytest.raises(ConfigError, match="гистерезис"):
            clean_load(overrides={"vision": {"tracker": {"off_threshold": 0.9}}})

    def test_overlap_must_be_smaller(self, clean_load) -> None:  # type: ignore[no-untyped-def]
        with pytest.raises(ConfigError, match="overlap_seconds"):
            clean_load(overrides={"audio": {"overlap_seconds": 120.0}})

    def test_quantize_range(self, clean_load) -> None:  # type: ignore[no-untyped-def]
        with pytest.raises(ConfigError, match="quantize_strength"):
            clean_load(overrides={"notation": {"quantize_strength": 1.5}})

    def test_hand_split_within_piano(self, clean_load) -> None:  # type: ignore[no-untyped-def]
        with pytest.raises(ConfigError, match="hand_split_pitch"):
            clean_load(overrides={"notation": {"hand_split_pitch": 200}})

    def test_ppq_divisible_by_12(self, clean_load) -> None:  # type: ignore[no-untyped-def]
        """Иначе триоли не выражаются целым числом тиков."""
        with pytest.raises(ConfigError, match="ppq"):
            clean_load(overrides={"export": {"ppq": 100}})

    def test_sample_rate_whitelist(self, clean_load) -> None:  # type: ignore[no-untyped-def]
        with pytest.raises(ConfigError, match="audio_sample_rate"):
            clean_load(overrides={"ingest": {"audio_sample_rate": 12345}})

    def test_error_has_user_message(self, clean_load) -> None:  # type: ignore[no-untyped-def]
        with pytest.raises(ConfigError) as info:
            clean_load(overrides={"export": {"ppq": 100}})
        assert info.value.user_message != info.value.technical


class TestFileLoading:
    def test_custom_file(self, tmp_path: Path) -> None:
        path = tmp_path / "custom.toml"
        path.write_text("[ingest]\nmax_height = 480\n", encoding="utf-8")
        config = load_config(path=path, use_user_config=False, use_env=False)
        assert config.ingest.max_height == 480
        assert config.export.ppq == 480  # остальное — из значений по умолчанию

    def test_broken_toml(self, tmp_path: Path) -> None:
        path = tmp_path / "broken.toml"
        path.write_text("[ingest\nmax_height =", encoding="utf-8")
        with pytest.raises(ConfigError, match="синтаксическая ошибка"):
            load_config(path=path, use_user_config=False, use_env=False)
