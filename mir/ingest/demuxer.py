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

import glob
import json
import os
import shutil
import subprocess
import sys
from fractions import Fraction
from pathlib import Path

from mir.common.enums import Platform, Stage
from mir.common.errors import DemuxError
from mir.common.logging import get_logger
from mir.common.types import MediaBundle, ProgressCallback
from mir.config import IngestConfig

__all__ = ["FFMPEG_DIR_ENV", "Demuxer", "ProbeResult", "find_ffmpeg"]

_log = get_logger(__name__)


FFMPEG_DIR_ENV = "MIR_FFMPEG_DIR"
"""Переменная окружения с каталогом FFmpeg — последнее слово за пользователем."""

_WINDOWS_FFMPEG_GLOBS: tuple[str, ...] = (
    r"%LOCALAPPDATA%\Microsoft\WinGet\Links",
    r"%LOCALAPPDATA%\Microsoft\WinGet\Packages\Gyan.FFmpeg*\*\bin",
    r"%LOCALAPPDATA%\Microsoft\WinGet\Packages\*FFmpeg*\*\bin",
    r"%ProgramData%\chocolatey\bin",
    r"%ProgramFiles%\ffmpeg\bin",
    r"C:\ffmpeg\bin",
)
"""Куда попадает FFmpeg на Windows при обычных способах установки.

Ставится он чаще всего через winget или chocolatey, и те кладут его
в свои каталоги. PATH при этом обновляется не всегда: у winget ссылки
появляются в новом сеансе, а запущенный терминал их не подхватывает.
Просить пользователя настольного приложения править PATH руками — плохой
обмен, поэтому известные места проверяются самостоятельно.
"""

_UNIX_FFMPEG_DIRS: tuple[str, ...] = (
    "/usr/bin",
    "/usr/local/bin",
    "/opt/homebrew/bin",
    "/snap/bin",
)


def _candidate_dirs() -> list[Path]:
    """Каталоги, где стоит поискать FFmpeg помимо PATH."""
    dirs: list[Path] = []
    explicit = os.environ.get(FFMPEG_DIR_ENV)
    if explicit:
        dirs.append(Path(explicit))

    # Рядом с проектом: удобно для переносимой сборки и для проверяющего,
    # который не хочет ставить FFmpeg в систему.
    dirs.append(Path(__file__).resolve().parent.parent.parent / "tools" / "ffmpeg" / "bin")

    if sys.platform == "win32":
        for pattern in _WINDOWS_FFMPEG_GLOBS:
            expanded = os.path.expandvars(pattern)
            if "*" in expanded:
                dirs.extend(sorted(Path(match) for match in glob.glob(expanded)))
            else:
                dirs.append(Path(expanded))
    else:
        dirs.extend(Path(d) for d in _UNIX_FFMPEG_DIRS)

    return dirs


def _lookup(name: str) -> Path | None:
    """Найти утилиту в PATH, затем в известных каталогах."""
    found = shutil.which(name)
    if found:
        return Path(found)

    suffix = ".exe" if sys.platform == "win32" else ""
    for directory in _candidate_dirs():
        candidate = directory / f"{name}{suffix}"
        if candidate.is_file():
            return candidate
    return None


def find_ffmpeg() -> tuple[Path, Path]:
    """Найти ffmpeg и ffprobe.

    Ищет в PATH, затем в каталогах, куда FFmpeg попадает при установке
    через winget, chocolatey и распаковкой архива, затем в `tools/ffmpeg/bin`
    рядом с проектом. Каталог можно задать явно переменной `MIR_FFMPEG_DIR`.

    Returns:
        Пути к обеим утилитам.

    Raises:
        DemuxError: Утилиты не найдены.
    """
    ffmpeg = _lookup("ffmpeg")
    ffprobe = _lookup("ffprobe")
    if ffmpeg is None or ffprobe is None:
        missing = "ffmpeg" if ffmpeg is None else "ffprobe"
        raise DemuxError(
            f"{missing} не найден ни в PATH, ни в типичных местах установки",
            "Не найдена программа FFmpeg.\n\n"
            "Windows: winget install Gyan.FFmpeg, затем перезапустите терминал.\n"
            f"Если она уже установлена, укажите каталог: set {FFMPEG_DIR_ENV}=C:\\путь\\к\\bin",
        )
    if shutil.which("ffmpeg") is None:
        _log.info("FFmpeg найден вне PATH: %s", ffmpeg.parent)
    return ffmpeg, ffprobe


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
