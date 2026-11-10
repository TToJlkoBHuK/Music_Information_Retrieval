"""Выгрузка результата в MIDI, MusicXML и PDF.

Реализуется на этапе 6. Пока есть только исполнительский MIDI — для
проверки результата слияния на слух.

Подробное описание устройства — в README.md рядом с этим файлом.
"""

from __future__ import annotations

from mir.export.midi_writer import write_performance_midi

__all__ = ["write_performance_midi"]
