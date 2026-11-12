"""Что делать с нотой, которую нашёл только один канал.

Правила выведены из разбора трёх роликов (два визуализатора), где расхождения
каналов просмотрены по кадрам (`mir/fusion/README.md`). Главный вывод:
падающий блок — это сама нота, нарисованная из MIDI исполнения, а звук — её
распознавание. Поэтому нота видео, прошедшая обе полосы датчика, принимается,
а нота только из звука должна чем-то подтвердиться:

* **блоком-кандидатом** — видео видело на этой клавише блок, но не смогло
  его доказать (тусклый блок, дым в нижней полосе);
* **слившимся блоком** — в видео в этот момент звучит соседняя клавиша
  другого цвета: блоки белой и чёрной соседок перекрываются и видны одним;
* **собственной уверенностью** — нейросеть оценивает каждую атаку, и у
  настоящих нот оценка заметно выше, чем у ложных.

Последнее — главный признак. У нот звука, подтверждённых видео, уверенность
0.80–0.92 (10-й и 90-й перцентили, одинаково на всех трёх роликах); у нот
без пары — медиана 0.50–0.64: это повторные атаки на удерживаемой клавише,
обертоны на октаву выше, эхо записи. Порог берётся из самого ролика —
`audio_confidence_quantile` от уверенности подтверждённых нот.
"""

from __future__ import annotations

from bisect import bisect_left, bisect_right
from collections import defaultdict
from collections.abc import Iterable

from mir.common.types import NoteEvent, is_black_key
from mir.fusion.decisions import Reason

__all__ = ["CandidatePool", "VideoIndex", "resolve_audio_only"]

_HARMONIC_INTERVALS = (12, 19, 24)
"""Октава, дуодецима, две октавы — вторая, третья и четвёртая гармоники.

Транскрипция по спектру чаще всего ошибается именно на них: слышит обертон
как отдельную ноту или, наоборот, приписывает ноту октавой ниже.
"""

_ECHO_WINDOW = 0.35
"""Сколько после ноты видео повтор той же высоты в звуке считается эхом, секунды.

На ролике Interstellar у записи эффект задержки: после нот звук находит ту
же высоту ещё раз через 0.15–0.3 с, а блока в этот момент нет.
"""

_LOOKBACK = 60.0
"""Дольше минуты клавишу не держат: дальше назад держащую ноту не искать."""


class VideoIndex:
    """Ноты видео по высотам — быстрые ответы на вопрос «что в этот момент в видео»."""

    def __init__(self, notes: Iterable[NoteEvent]) -> None:
        grouped: dict[int, list[NoteEvent]] = defaultdict(list)
        for note in notes:
            grouped[note.pitch].append(note)
        self._notes = {
            pitch: sorted(group, key=lambda n: n.onset) for pitch, group in grouped.items()
        }
        self._onsets = {pitch: [n.onset for n in group] for pitch, group in self._notes.items()}

    def holder(self, pitch: int, time: float, tolerance: float) -> NoteEvent | None:
        """Нота, которая началась раньше `time` и в этот момент ещё держится."""
        notes = self._notes.get(pitch, [])
        index = bisect_left(self._onsets.get(pitch, []), time - tolerance)
        for note in reversed(notes[:index]):
            if time - note.onset > _LOOKBACK:
                break
            if note.offset > time:
                return note
        return None

    def sounding(self, pitch: int, time: float, tolerance: float) -> bool:
        """Звучит ли клавиша в момент `time` — считая и только что начатую ноту."""
        onsets = self._onsets.get(pitch, [])
        low = bisect_left(onsets, time - tolerance)
        high = bisect_right(onsets, time + tolerance)
        return high > low or self.holder(pitch, time, tolerance) is not None

    def started(self, pitch: int, earliest: float, latest: float) -> bool:
        """Началась ли нота этой высоты в промежутке `[earliest, latest]`."""
        onsets = self._onsets.get(pitch, [])
        return bisect_right(onsets, latest) > bisect_left(onsets, earliest)


class CandidatePool:
    """Блоки-кандидаты видео; каждый подтверждает не больше одной ноты звука."""

    def __init__(self, candidates: Iterable[NoteEvent]) -> None:
        self._free: dict[int, list[NoteEvent]] = defaultdict(list)
        for note in candidates:
            self._free[note.pitch].append(note)

    def take(self, pitch: int, time: float, tolerance: float) -> NoteEvent | None:
        """Забрать ближайшего по времени кандидата этой высоты в пределах допуска."""
        free = self._free.get(pitch)
        if not free:
            return None
        nearest = min(free, key=lambda n: abs(n.onset - time))
        if abs(nearest.onset - time) > tolerance:
            return None
        free.remove(nearest)
        return nearest


def resolve_audio_only(
    note: NoteEvent,
    video: VideoIndex,
    candidates: CandidatePool,
    threshold: float,
    tolerance: float,
    in_view: bool,
) -> tuple[Reason, NoteEvent | None]:
    """Решить судьбу ноты, которая есть в звуке и не нашла пары в видео.

    Правила проверяются по порядку, от самого сильного свидетельства:

    1. клавиша за кадром — видео её видеть не могло, нота принимается;
    2. есть блок-кандидат — два независимых слабых свидетельства;
    3. в видео звучит соседняя клавиша другого цвета — блок мог слиться;
    4. нота уверена не меньше подтверждённых — принимается. Если клавиша
       при этом в видео удерживается, это повторный удар: блоки повторов
       с зазором меньше кадра датчик видит одним;
    5. иначе отбрасывается, а причина записывается по тому, на что это
       похоже: повторная атака, обертон, эхо или просто слабая нота.

    Args:
        note: Нота звука.
        video: Все ноты видео в шкале звука.
        candidates: Блоки-кандидаты видео.
        threshold: Уверенность, с которой нота принимается без видео.
        tolerance: Допуск по времени, секунды.
        in_view: Клавиша ноты видна в кадре.

    Returns:
        Причину и связанную ноту видео: кандидата или держащую ноту.
    """
    if not in_view:
        return Reason.OUT_OF_VIEW, None

    candidate = candidates.take(note.pitch, note.onset, tolerance)
    if candidate is not None:
        return Reason.CANDIDATE, candidate

    holder = video.holder(note.pitch, note.onset, tolerance)
    if holder is None and _hidden_by_neighbour(note, video, tolerance):
        return Reason.HIDDEN, None

    if note.confidence >= threshold:
        return (Reason.RESTRIKE, holder) if holder is not None else (Reason.CONFIDENT, None)

    if holder is not None:
        return Reason.KEY_HELD, holder
    for interval in _HARMONIC_INTERVALS:
        for pitch in (note.pitch - interval, note.pitch + interval):
            if video.started(pitch, note.onset - tolerance, note.onset + tolerance):
                return Reason.HARMONIC, None
    if video.started(note.pitch, note.onset - _ECHO_WINDOW, note.onset - tolerance):
        return Reason.ECHO, None
    return Reason.WEAK, None


def _hidden_by_neighbour(note: NoteEvent, video: VideoIndex, tolerance: float) -> bool:
    """Звучит ли в видео соседняя клавиша другого цвета.

    Блок чёрной клавиши рисуется со сдвигом к соседней белой, и вместе они
    дают один светлый участок шириной 1.3 белой клавиши вместо 1.1 —
    датчик относит его к одной клавише. Так на Clair de lune видео показало
    фа, звук — соль-бемоль, а в кадре оба блока (проверено по краям участка).
    Соседние белые (ми — фа, си — до) не сливаются: их общий участок вдвое
    шире блока, и датчик делит его на два.
    """
    for pitch in (note.pitch - 1, note.pitch + 1):
        if is_black_key(pitch) != is_black_key(note.pitch) and video.sounding(
            pitch, note.onset, tolerance
        ):
            return True
    return False
