"""Разделение файла на видеоряд и аудиодорожку через FFmpeg (F-02).

Видео не перекодируется: `vision` читает исходный файл покадрово,
а перекодирование заняло бы минуты и ухудшило качество, важное
для детекции. Извлекается только аудио.

Вызовы FFmpeg всегда с `encoding="utf-8", errors="replace"`. Без этого
вывод декодируется в системной кодировке, и на Windows (cp1251) русское
название ролика в сообщении об ошибке превращается в мусор либо роняет
программу. `errors="replace"` нужен на случай, когда FFmpeg всё же выдаёт
байты в другой кодировке: нам важны код возврата и текст ошибки, а не
побайтовая точность.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from fractions import Fraction
from pathlib import Path

from mir.common.enums import Platform, Stage
from mir.common.errors import DemuxError
from mir.common.logging import get_logger
from mir.common.types import MediaBundle, ProgressCallback
from mir.config import IngestConfig

__all__ = ["Demuxer", "ProbeResult", "find_ffmpeg"]

_log = get_logger(__name__)


def find_ffmpeg() -> tuple[Path, Path]:
    """Найти ffmpeg и ffprobe.

    Returns:
        Пути к обеим утилитам.

    Raises:
        DemuxError: Утилиты не найдены в PATH.
    """
    ffmpeg = shutil.which("ffmpeg")
    ffprobe = shutil.which("ffprobe")
    if not ffmpeg or not ffprobe:
        raise DemuxError(
            "ffmpeg или ffprobe не найдены в PATH",
            "Не найдена программа FFmpeg. Установите её и перезапустите приложение",
        )
    return Path(ffmpeg), Path(ffprobe)


class ProbeResult:
    """Технические параметры видеофайла.

    Attributes:
        fps: Частота кадров. Дробная — округление на часовом ролике
            даёт рассинхрон в несколько секунд.
        duration: Длительность в секундах.
        width: Ширина кадра.
        height: Высота кадра.
        has_audio: Наличие звуковой дорожки.
        title: Название из метаданных контейнера.
    """

    __slots__ = ("duration", "fps", "has_audio", "height", "title", "width")

    def __init__(
        self,
        fps: float,
        duration: float,
        width: int,
        height: int,
        has_audio: bool,
        title: str | None = None,
    ) -> None:
        self.fps = fps
        self.duration = duration
        self.width = width
        self.height = height
        self.has_audio = has_audio
        self.title = title

    def __repr__(self) -> str:
        return (
            f"ProbeResult(fps={self.fps:.3f}, duration={self.duration:.1f}, "
            f"{self.width}x{self.height}, audio={self.has_audio})"
        )


class Demuxer:
    """Подготовка материала к анализу.

    Args:
        config: Параметры извлечения аудио.
    """

    def __init__(self, config: IngestConfig) -> None:
        self.config = config
        self.ffmpeg, self.ffprobe = find_ffmpeg()

    def probe(self, video_path: Path) -> ProbeResult:
        """Прочитать параметры файла через ffprobe.

        Raises:
            DemuxError: Файл не читается или не содержит видеопотока.
        """
        if not video_path.exists():
            raise DemuxError(f"файл не найден: {video_path}", f"Файл не найден: {video_path.name}")

        command = [
            str(self.ffprobe),
            "-v",
            "error",
            "-print_format",
            "json",
            "-show_streams",
            "-show_format",
            str(video_path),
        ]
        try:
            completed = subprocess.run(
                command,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=60,
                check=True,
            )
        except subprocess.CalledProcessError as exc:
            raise DemuxError(
                f"ffprobe завершился с ошибкой: {exc.stderr}",
                f"Не удалось прочитать файл {video_path.name}",
            ) from exc
        except subprocess.TimeoutExpired as exc:
            raise DemuxError("ffprobe не ответил за 60 секунд") from exc

        data = json.loads(completed.stdout)
        streams = data.get("streams", [])
        video = next((s for s in streams if s.get("codec_type") == "video"), None)
        if video is None:
            raise DemuxError(
                f"в {video_path.name} нет видеопотока",
                f"Файл {video_path.name} не содержит видео",
            )

        has_audio = any(s.get("codec_type") == "audio" for s in streams)
        duration = float(data.get("format", {}).get("duration") or video.get("duration") or 0.0)
        title = data.get("format", {}).get("tags", {}).get("title")

        return ProbeResult(
            fps=self._parse_fps(video.get("r_frame_rate", "0/0")),
            duration=duration,
            width=int(video.get("width", 0)),
            height=int(video.get("height", 0)),
            has_audio=has_audio,
            title=title,
        )

    @staticmethod
    def _parse_fps(raw: str) -> float:
        """Разобрать `r_frame_rate` вида `30000/1001`.

        Дробная запись сохраняется: 29.97 нельзя округлять до 30.
        """
        try:
            value = float(Fraction(raw))
        except (ValueError, ZeroDivisionError):
            raise DemuxError(f"не удалось разобрать частоту кадров: {raw!r}") from None
        if value <= 0:
            raise DemuxError(f"некорректная частота кадров: {value}")
        return value

    def extract_audio(self, video_path: Path, out_path: Path) -> Path:
        """Извлечь аудиодорожку в WAV.

        Параметры (16 кГц, моно) заданы требованием модели транскрипции.
        Пересэмплирование делает FFmpeg — быстрее и качественнее, чем librosa.

        Raises:
            DemuxError: FFmpeg завершился с ошибкой.
        """
        out_path.parent.mkdir(parents=True, exist_ok=True)
        command = [
            str(self.ffmpeg),
            "-y",
            "-i",
            str(video_path),
            "-vn",
            "-acodec",
            "pcm_s16le",
            "-ar",
            str(self.config.audio_sample_rate),
            "-ac",
            str(self.config.audio_channels),
            "-loglevel",
            "error",
            str(out_path),
        ]
        _log.info("извлечение аудио: %s → %s", video_path.name, out_path.name)
        try:
            subprocess.run(
                command,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                check=True,
            )
        except subprocess.CalledProcessError as exc:
            raise DemuxError(
                f"ffmpeg не смог извлечь аудио: {exc.stderr}",
                "Не удалось извлечь звуковую дорожку из видео",
            ) from exc

        if not out_path.exists() or out_path.stat().st_size == 0:
            raise DemuxError(
                "аудиофайл пуст после извлечения",
                "Звуковая дорожка пуста. Возможно, в видео нет звука",
            )
        return out_path

    def split(
        self,
        video_path: Path,
        work_dir: Path,
        title: str | None = None,
        source_url: str | None = None,
        platform: Platform = Platform.LOCAL_FILE,
        progress: ProgressCallback | None = None,
    ) -> MediaBundle:
        """Подготовить материал целиком.

        Args:
            video_path: Исходный видеофайл.
            work_dir: Каталог для извлечённого аудио.
            title: Название произведения.
            source_url: Исходная ссылка.
            platform: Площадка-источник.
            progress: Колбэк прогресса.

        Returns:
            Готовый [`MediaBundle`][mir.common.types.MediaBundle].

        Raises:
            DemuxError: Файл повреждён или не содержит звука.
        """
        if progress:
            progress(Stage.DEMUX, 0.0)

        info = self.probe(video_path)
        _log.info("параметры видео: %r", info)

        if not info.has_audio:
            raise DemuxError(
                f"в {video_path.name} нет аудиодорожки",
                "В этом видео нет звука. Работа только по видеоряду пока не поддерживается",
            )

        if progress:
            progress(Stage.DEMUX, 0.3)

        audio_path = self.extract_audio(video_path, work_dir / f"{video_path.stem}.wav")

        if progress:
            progress(Stage.DEMUX, 1.0)

        return MediaBundle(
            video_path=video_path,
            audio_path=audio_path,
            fps=info.fps,
            duration=info.duration,
            width=info.width,
            height=info.height,
            title=title or info.title,
            source_url=source_url,
            platform=platform,
        )
