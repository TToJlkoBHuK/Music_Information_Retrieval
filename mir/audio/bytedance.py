"""Модель ByteDance: High-resolution Piano Transcription (Kong et al., 2021).

Из библиотеки `piano_transcription_inference` берутся только архитектура
сети и постобработка — то, что повторять нельзя, иначе разошлось бы
с опубликованным качеством (F1 по нотам 0.968 на MAESTRO). Загрузка весов,
проход по сегментам и выбор устройства — свои, см. `models.py`
и `transcriber.py`.

PyTorch и библиотека импортируются при создании модели, а не при импорте
модуля: без них конвейер работает дальше (темп и атаки считаются и так),
а сообщение об отсутствии будет понятным, а не `ModuleNotFoundError`.
"""

from __future__ import annotations

import io
import os
import pickle
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from mir.audio.models import backend_status
from mir.audio.transcriber import DecodeThresholds, Outputs
from mir.common.errors import TranscriptionError
from mir.common.logging import get_logger

__all__ = ["SAMPLE_RATE", "ByteDanceModel"]

_log = get_logger(__name__)

SAMPLE_RATE = 16000
"""Частота, на которой обучена модель."""

_FRAMES_PER_SECOND = 100
_SEGMENT_SECONDS = 10
_CLASSES = 88

_RECOMPUTED_PREFIXES = ("spectrogram_extractor.", "logmel_extractor.")
"""Параметры, которые не обязаны быть в файле весов.

Это фиксированные матрицы преобразования Фурье и мел-фильтров: они
строятся заново при создании сети и не обучаются. Всё остальное
обязано загрузиться, иначе сеть работает на случайных весах.
"""


def _load_checkpoint(path: Path, official: bool) -> dict[str, Any]:
    """Прочитать файл весов.

    Сначала безопасным режимом `weights_only`: он не исполняет код
    из файла. Если файл сохранён в старом формате и так не читается,
    обычный режим допускается только для официального файла, сверенного
    по MD5, — исполнять код из произвольного файла нельзя.
    """
    import torch

    try:
        data = torch.load(str(path), map_location="cpu", weights_only=True)
    except pickle.UnpicklingError:
        if not official:
            raise TranscriptionError(
                f"{path} не читается в безопасном режиме, а его сумма не совпадает с официальной",
                "Файл весов модели не подходит. Удалите его или укажите официальный",
            ) from None
        _log.info("веса в старом формате, чтение в обычном режиме (файл сверен по MD5)")
        data = torch.load(str(path), map_location="cpu", weights_only=False)

    if not isinstance(data, dict) or "model" not in data:
        raise TranscriptionError(
            f"{path}: в файле нет ключа 'model'",
            "Файл весов модели повреждён или имеет другой формат",
        )
    state: dict[str, Any] = data["model"]
    return state


def _load_weights(module: Any, state: dict[str, Any], part: str) -> None:
    """Загрузить веса части сети и проверить, что ничего не потеряно.

    Исходная библиотека грузит веса с `strict=False` и не смотрит на
    результат: файл от другой модели загрузится «успешно», а сеть будет
    выдавать мусор на случайных весах. Здесь недостающие обучаемые
    параметры — ошибка.
    """
    result = module.load_state_dict(state, strict=False)
    missing = [k for k in result.missing_keys if not k.startswith(_RECOMPUTED_PREFIXES)]
    if missing:
        raise TranscriptionError(
            f"в весах {part} нет {len(missing)} параметров, например {missing[0]}",
            "Файл весов не подходит к модели распознавания звука",
        )
    if result.unexpected_keys:
        _log.warning(
            "в весах %s лишние параметры (%d), они пропущены", part, len(result.unexpected_keys)
        )


class ByteDanceModel:
    """Сеть ByteDance для протокола [`SegmentModel`][mir.audio.transcriber.SegmentModel].

    Args:
        checkpoint: Файл весов.
        device: `cpu` или `cuda`.
        official: Файл сверен с публикацией авторов.

    Raises:
        TranscriptionError: Нет PyTorch или библиотеки, файл весов не подходит.
    """

    sample_rate = SAMPLE_RATE
    frames_per_second = _FRAMES_PER_SECOND
    segment_samples = SAMPLE_RATE * _SEGMENT_SECONDS

    def __init__(self, checkpoint: Path, device: str = "cpu", official: bool = True) -> None:
        problem = backend_status()
        if problem is not None:
            raise TranscriptionError("нет библиотек нейросетевой транскрипции", problem)

        import torch

        # Модуль архитектуры зачем-то импортирует matplotlib.pyplot. В окне
        # приложения на Qt это пытается поднять собственный графический
        # бэкенд; рисовать здесь нечего, поэтому бэкенд — без окон.
        os.environ.setdefault("MPLBACKEND", "Agg")
        with redirect_stdout(io.StringIO()):
            from piano_transcription_inference.models import Note_pedal

        state = _load_checkpoint(checkpoint, official)
        network = Note_pedal(frames_per_second=_FRAMES_PER_SECOND, classes_num=_CLASSES)
        for part in ("note_model", "pedal_model"):
            if part not in state:
                raise TranscriptionError(
                    f"в весах нет части {part}",
                    "Файл весов не подходит к модели распознавания звука",
                )
            _load_weights(getattr(network, part), state[part], part)

        self.device = device
        self._torch = torch
        self._network = network.to(device).eval()
        _log.info("модель транскрипции загружена: %s, %s", checkpoint.name, device)

    def forward(self, segments: npt.NDArray[np.float32]) -> Outputs:
        """Разметить пачку сегментов. См. протокол `SegmentModel`."""
        torch = self._torch
        with torch.inference_mode():
            batch = torch.from_numpy(np.ascontiguousarray(segments, dtype=np.float32))
            result = self._network(batch.to(self.device))
        return {name: value.float().cpu().numpy() for name, value in result.items()}

    def decode(
        self, outputs: Outputs, thresholds: DecodeThresholds
    ) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
        """Постобработка авторов модели: регрессия моментов начала и конца.

        Используется как есть — в ней и состоит вклад статьи: момент
        начала уточняется внутри кадра по форме регрессии, отсюда
        разрешение лучше 10-миллисекундного шага.
        """
        with redirect_stdout(io.StringIO()):
            from piano_transcription_inference.utilities import RegressionPostProcessor

        processor = RegressionPostProcessor(
            _FRAMES_PER_SECOND,
            classes_num=_CLASSES,
            onset_threshold=thresholds.onset,
            offset_threshold=thresholds.offset,
            frame_threshold=thresholds.frame,
            pedal_offset_threshold=thresholds.pedal_offset,
        )
        notes, pedals = processor.output_dict_to_note_pedal_arrays(dict(outputs))
        note_array = np.asarray(notes, dtype=np.float32).reshape(-1, 4)
        pedal_array = (
            np.empty((0, 2), dtype=np.float32)
            if pedals is None
            else np.asarray(pedals, dtype=np.float32).reshape(-1, 2)
        )
        return note_array, pedal_array
