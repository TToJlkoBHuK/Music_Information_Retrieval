"""Веса нейросети и выбор устройства (#26, NF-12, NF-14).

Веса не кладутся в дистрибутив: 172 МБ раздули бы установщик (NF-14),
а скачать их нужно один раз. Здесь же исправлены недостатки исходной
библиотеки `piano_transcription_inference`, мешающие на Windows:

* она качает веса через `os.system("wget ...")` — на Windows `wget` нет,
  и загрузка молча не происходит;
* целостность проверяется лишь по размеру больше 160 МБ — недокачанный
  на 99 % файл проходит проверку;
* выбор устройства не учитывает свободную видеопамять.
"""

from __future__ import annotations

import hashlib
import shutil
import urllib.request
from collections.abc import Callable
from pathlib import Path

from mir.common.errors import CancelledError, TranscriptionError
from mir.common.logging import get_logger
from mir.common.types import CancelCheck

__all__ = [
    "MODEL_FILENAME",
    "MODEL_MD5",
    "MODEL_SIZE",
    "MODEL_URL",
    "ModelRegistry",
    "backend_status",
    "select_device",
]

_log = get_logger(__name__)

MODEL_FILENAME = "CRNN_note_F1=0.9677_pedal_F1=0.9186.pth"
"""Имя файла весов в публикации авторов."""

MODEL_URL = (
    "https://zenodo.org/records/4034264/files/"
    "CRNN_note_F1%3D0.9677_pedal_F1%3D0.9186.pth?download=1"
)
"""Публикация авторов модели на Zenodo (Kong et al., 2021), лицензия CC BY 4.0."""

MODEL_SIZE = 171_966_578
"""Размер файла в байтах — по метаданным записи Zenodo."""

MODEL_MD5 = "22b961b77c1878239fec963362097045"
"""Контрольная сумма из метаданных Zenodo.

MD5 — то, что публикует сам Zenodo. Для защиты от порчи файла при
загрузке этого достаточно; подмену на стороне сервера не исключает
никакая сумма, полученная с того же сервера.
"""

_CHUNK = 1 << 20
_MIN_GPU_MEMORY = 2 << 30
"""Свободной видеопамяти нужно не меньше 2 ГБ.

Свойство модели, а не настройка: CRNN на сегменте 10 с с промежуточными
тензорами занимает около 1.5 ГБ. Меньше — падение с нехваткой памяти
посреди обработки, поэтому лучше сразу считать на процессоре.
"""

ByteProgress = Callable[[int, int], None]
"""Прогресс загрузки: скачано байт, всего байт."""


def backend_status() -> str | None:
    """Проверить, установлены ли библиотеки нейросетевой транскрипции.

    Returns:
        `None`, если всё на месте, иначе текст для пользователя.
    """
    import importlib.util

    missing = [
        name
        for name in ("torch", "piano_transcription_inference", "torchlibrosa")
        if importlib.util.find_spec(name) is None
    ]
    if not missing:
        return None
    return (
        f"Не установлены библиотеки распознавания звука: {', '.join(missing)}.\n"
        "Установите их командой\n"
        "    run audio-setup\n"
        "или вручную (вариант для процессора):\n"
        "    python -m pip install torch --index-url https://download.pytorch.org/whl/cpu\n"
        '    python -m pip install -e ".[transcription]"'
    )


def select_device(preference: str) -> tuple[str, str | None]:
    """Выбрать, на чём считать (NF-12).

    Args:
        preference: `auto`, `cpu` или `cuda` из настроек.

    Returns:
        Устройство и предупреждение для пользователя (или `None`).
    """
    if preference == "cpu":
        return "cpu", None

    import torch

    if not torch.cuda.is_available():
        if preference == "cuda":
            return "cpu", (
                "Видеокарта с CUDA недоступна, распознавание звука идёт на процессоре — "
                "это в 3-5 раз медленнее"
            )
        return "cpu", None

    free, _total = torch.cuda.mem_get_info()
    if free < _MIN_GPU_MEMORY:
        return "cpu", (
            f"На видеокарте свободно {free / 2**30:.1f} ГБ, нужно не меньше "
            f"{_MIN_GPU_MEMORY / 2**30:.0f} ГБ. Распознавание звука идёт на процессоре"
        )
    return "cuda", None


def _md5(path: Path) -> str:
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as fh:
        while block := fh.read(_CHUNK):
            digest.update(block)
    return digest.hexdigest()


class ModelRegistry:
    """Где лежат веса модели и как их получить.

    Args:
        cache_dir: Каталог кэша проекта (`ingest.cache_dir`).
        model_path: Путь из настроек. Пусто — веса берутся из кэша.
    """

    def __init__(self, cache_dir: Path, model_path: str = "") -> None:
        self.cache_dir = cache_dir.expanduser()
        self.model_path = Path(model_path).expanduser() if model_path else None

    @property
    def default_path(self) -> Path:
        """Куда кладутся скачанные веса."""
        return self.cache_dir / "models" / MODEL_FILENAME

    @staticmethod
    def verify(path: Path) -> bool:
        """Совпадает ли файл с публикацией авторов: размер, затем MD5.

        Размер проверяется первым: это мгновенно и отсекает недокачанный
        файл без чтения 172 МБ.
        """
        if not path.is_file() or path.stat().st_size != MODEL_SIZE:
            return False
        return _md5(path) == MODEL_MD5

    def locate(self) -> tuple[Path, bool] | None:
        """Найти веса на диске, ничего не скачивая.

        Returns:
            Путь и признак «совпадает с официальным файлом» или `None`.
        """
        if self.model_path is not None:
            if not self.model_path.is_file():
                return None
            return self.model_path, self.verify(self.model_path)
        if self.verify(self.default_path):
            return self.default_path, True
        return None

    def ensure(
        self,
        progress: ByteProgress | None = None,
        cancel: CancelCheck | None = None,
        url: str = MODEL_URL,
    ) -> tuple[Path, bool]:
        """Вернуть путь к весам, при необходимости скачав их.

        Свой путь в настройках (`audio.model_path`) не скачивается и не
        отвергается при несовпадении суммы: так можно подключить
        дообученную модель. Несовпадение лишь пишется в журнал.

        Returns:
            Путь и признак «совпадает с официальным файлом».

        Raises:
            TranscriptionError: Файла нет, загрузка не удалась или скачанный
                файл повреждён.
            CancelledError: Пользователь отменил загрузку.
        """
        if self.model_path is not None:
            found = self.locate()
            if found is None:
                raise TranscriptionError(
                    f"audio.model_path={self.model_path} не найден",
                    f"Файл весов модели не найден: {self.model_path}",
                )
            if not found[1]:
                _log.warning(
                    "веса %s не совпадают с официальными — используются как есть", found[0]
                )
            return found

        if self.verify(self.default_path):
            return self.default_path, True

        self._download(url, self.default_path, progress, cancel)
        if not self.verify(self.default_path):
            self.default_path.unlink(missing_ok=True)
            raise TranscriptionError(
                "скачанные веса не совпадают с опубликованными (размер или MD5)",
                "Файл модели скачался с ошибкой. Повторите попытку",
            )
        return self.default_path, True

    @staticmethod
    def _download(
        url: str, target: Path, progress: ByteProgress | None, cancel: CancelCheck | None
    ) -> None:
        """Скачать во временный файл и переименовать только целиком.

        Прерванная загрузка не оставляет на месте весов обрывок, который
        потом пришлось бы распознавать как испорченный.
        """
        target.parent.mkdir(parents=True, exist_ok=True)
        partial = target.with_suffix(target.suffix + ".part")
        _log.info("загрузка весов модели (%.0f МБ) из %s", MODEL_SIZE / 2**20, url)
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "mir/0.2"})
            with urllib.request.urlopen(request, timeout=60) as response, partial.open("wb") as fh:
                total = int(response.headers.get("Content-Length") or MODEL_SIZE)
                done = 0
                while block := response.read(_CHUNK):
                    if cancel is not None and cancel():
                        raise CancelledError("загрузка весов отменена")
                    fh.write(block)
                    done += len(block)
                    if progress is not None:
                        progress(done, total)
            shutil.move(str(partial), str(target))
        except CancelledError:
            partial.unlink(missing_ok=True)
            raise
        except OSError as exc:
            partial.unlink(missing_ok=True)
            raise TranscriptionError(
                f"не удалось скачать веса модели: {exc}",
                "Не удалось скачать модель распознавания звука. Проверьте интернет; "
                f"файл можно скачать вручную по ссылке {MODEL_URL} и указать путь "
                "к нему в настройках (audio.model_path)",
            ) from exc
