"""Ноты в CSV: эталонная разметка и сохранённые результаты.

Один формат на все отладочные инструменты: ноты, найденные
`vision_cli analyze --csv`, читаются ключом `--truth` и командой
`audio_cli sync --video-csv` — без повторного разбора ролика.

```csv
pitch,onset,offset,velocity,hand,source,confidence
60,0.5000,0.9000,90,RIGHT,VIDEO,1.000
```

Обязательны первые три столбца; остальные можно опустить — так проще
размечать эталон вручную.

Рядом с нотами видео сохраняются блоки-кандидаты — то, что датчик видел,
но не смог доказать: `video.csv` → `video.candidates.csv`
([`candidates_path`][mir.eval.notes_io.candidates_path]). Слияние каналов
находит этот файл само.
"""

from __future__ import annotations

import csv
from pathlib import Path

from mir.common.enums import Hand, Source
from mir.common.types import NoteEvent

__all__ = ["candidates_path", "read_notes_csv", "write_notes_csv"]

_COLUMNS = ["pitch", "onset", "offset", "velocity", "hand", "source", "confidence"]


def candidates_path(notes_path: Path) -> Path:
    """Где лежат блоки-кандидаты для файла с нотами видео."""
    return notes_path.with_suffix(".candidates.csv")


def write_notes_csv(notes: list[NoteEvent], path: Path) -> None:
    """Сохранить ноты в CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(_COLUMNS)
        for note in notes:
            writer.writerow(
                [
                    note.pitch,
                    f"{note.onset:.4f}",
                    f"{note.offset:.4f}",
                    note.velocity,
                    note.hand.name,
                    note.source.name,
                    f"{note.confidence:.3f}",
                ]
            )


def read_notes_csv(path: Path) -> list[NoteEvent]:
    """Прочитать ноты из CSV.

    Raises:
        ValueError: Нет обязательного столбца или значение вне диапазона.
    """
    with path.open(encoding="utf-8", newline="") as fh:
        return [
            NoteEvent(
                pitch=int(row["pitch"]),
                onset=float(row["onset"]),
                offset=float(row["offset"]),
                velocity=int(row.get("velocity") or 64),
                hand=Hand[row.get("hand") or "UNKNOWN"],
                source=Source[row.get("source") or "VIDEO"],
                confidence=float(row.get("confidence") or 1.0),
            )
            for row in csv.DictReader(fh)
        ]
