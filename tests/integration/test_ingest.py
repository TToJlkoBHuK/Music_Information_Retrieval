"""Полный этап загрузки на реальных файлах. Требует FFmpeg."""

from __future__ import annotations

from pathlib import Path

import pytest

from mir.common.enums import Platform, Stage
from mir.common.errors import DemuxError
from mir.config import MirConfig
from mir.ingest import Demuxer, ingest

pytestmark = pytest.mark.skipif(
    __import__("shutil").which("ffmpeg") is None, reason="FFmpeg не установлен"
)


class TestProbe:
    def test_reads_parameters(self, sample_video: Path, config: MirConfig) -> None:
        info = Demuxer(config.ingest).probe(sample_video)
        assert info.width == 640
        assert info.height == 360
        assert info.has_audio
        assert info.duration == pytest.approx(2.0, abs=0.2)

    def test_fps_stays_fractional(self, sample_video: Path, config: MirConfig) -> None:
        """r_frame_rate = 30000/1001 не должен превратиться в 30."""
        info = Demuxer(config.ingest).probe(sample_video)
        assert info.fps == pytest.approx(29.97, abs=0.01)
        assert info.fps != 30.0

    def test_missing_file(self, tmp_path: Path, config: MirConfig) -> None:
        with pytest.raises(DemuxError, match="не найден"):
            Demuxer(config.ingest).probe(tmp_path / "absent.mp4")


class TestExtractAudio:
    def test_creates_wav(self, sample_video: Path, tmp_path: Path, config: MirConfig) -> None:
        out = Demuxer(config.ingest).extract_audio(sample_video, tmp_path / "a.wav")
        assert out.exists()
        assert out.stat().st_size > 1000

    def test_sample_rate_matches_model_requirement(
        self, sample_video: Path, tmp_path: Path, config: MirConfig
    ) -> None:
        """Модель ByteDance ждёт 16 кГц моно."""
        import wave

        out = Demuxer(config.ingest).extract_audio(sample_video, tmp_path / "a.wav")
        with wave.open(str(out)) as wav:
            assert wav.getframerate() == config.ingest.audio_sample_rate
            assert wav.getnchannels() == config.ingest.audio_channels


class TestIngestPipeline:
    def test_local_file(self, sample_video: Path, tmp_path: Path, config: MirConfig) -> None:
        bundle = ingest(sample_video, config=config, work_dir=tmp_path / "work")
        assert bundle.platform is Platform.LOCAL_FILE
        assert bundle.video_path == sample_video
        assert bundle.audio_path.exists()
        assert bundle.frame_count > 0
        assert bundle.source_url is None

    def test_progress_reported(self, sample_video: Path, tmp_path: Path, config: MirConfig) -> None:
        seen: list[tuple[Stage, float]] = []
        ingest(
            sample_video,
            config=config,
            work_dir=tmp_path / "work",
            progress=lambda stage, pct: seen.append((stage, pct)),
        )
        stages = {stage for stage, _ in seen}
        assert Stage.DOWNLOAD in stages
        assert Stage.DEMUX in stages
        assert all(0.0 <= pct <= 1.0 for _, pct in seen)

    def test_video_not_reencoded(
        self, sample_video: Path, tmp_path: Path, config: MirConfig
    ) -> None:
        """Исходный файл должен остаться нетронутым."""
        before = sample_video.stat().st_size
        bundle = ingest(sample_video, config=config, work_dir=tmp_path / "work")
        assert bundle.video_path.stat().st_size == before

    def test_silent_video_rejected(
        self, silent_video: Path, tmp_path: Path, config: MirConfig
    ) -> None:
        with pytest.raises(DemuxError, match="аудиодорожк"):
            ingest(silent_video, config=config, work_dir=tmp_path / "work")
