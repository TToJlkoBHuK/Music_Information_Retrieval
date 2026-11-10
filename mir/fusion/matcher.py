"""Сопоставление нот двух каналов: какая нота видео — та же, что нота звука.

Пара — это ноты одной высоты, начавшиеся почти одновременно. Каждая нота
входит не больше чем в одну пару: иначе одна нота звука «подтвердила бы»
сразу два повтора в видео.

**Почему не ближайший сосед.** Жадный проход — каждой ноте видео ближайшая
нота звука — ошибается на повторах одной клавиши: если звук пропустил один
повтор из трёх, жадный выбор сдвигает пары и оставляет без пары не ту ноту.
Здесь ищется наилучшее сопоставление целиком: сначала как можно больше пар,
затем — с наименьшей суммой расхождений.

**Почему не венгерский алгоритм.** Ноты разной высоты не сопоставляются
никогда, а внутри одной высоты всё лежит на одной оси времени. На прямой
лучшее сопоставление не содержит пересечений (поменяв местами две
пересекающиеся пары, сумму расхождений можно только уменьшить), а такое
находит динамическое программирование — точно и без матрицы на весь ролик.
Ноты одной высоты к тому же распадаются на группы, между которыми больше
допуска: группы решаются независимо и обычно состоят из одной-двух нот.

Длительности в сравнении не участвуют: у видео это время удержания клавиши,
у звука — время звучания вместе с педалью (на «Лунной сонате» в среднем на
0.8 с больше), и сравнивать их нельзя.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

from mir.common.types import NoteEvent

__all__ = ["MatchResult", "match_notes"]


@dataclass
class MatchResult:
    """Итог сопоставления.

    Attributes:
        pairs: Пары «нота видео — нота звука».
        video_only: Ноты видео без пары.
        audio_only: Ноты звука без пары.
    """

    pairs: list[tuple[NoteEvent, NoteEvent]] = field(default_factory=list)
    video_only: list[NoteEvent] = field(default_factory=list)
    audio_only: list[NoteEvent] = field(default_factory=list)


def match_notes(video: list[NoteEvent], audio: list[NoteEvent], tolerance: float) -> MatchResult:
    """Сопоставить ноты двух каналов по высоте и началу.

    Оба списка должны быть в одной шкале времени — рассинхрон снимается до
    сопоставления (`mir.fusion.aligner`).

    Args:
        video: Ноты видеоканала.
        audio: Ноты аудиоканала.
        tolerance: Наибольшее расхождение начал в паре, секунды.

    Returns:
        Пары и ноты, оставшиеся без пары, — по возрастанию времени.
    """
    by_pitch: dict[int, tuple[list[NoteEvent], list[NoteEvent]]] = defaultdict(lambda: ([], []))
    for note in video:
        by_pitch[note.pitch][0].append(note)
    for note in audio:
        by_pitch[note.pitch][1].append(note)

    result = MatchResult()
    for seen, heard in by_pitch.values():
        seen.sort(key=lambda n: n.onset)
        heard.sort(key=lambda n: n.onset)
        for group_seen, group_heard in _groups(seen, heard, tolerance):
            paired = _align(group_seen, group_heard, tolerance)
            used_seen = {i for i, _ in paired}
            used_heard = {j for _, j in paired}
            result.pairs += [(group_seen[i], group_heard[j]) for i, j in paired]
            result.video_only += [n for i, n in enumerate(group_seen) if i not in used_seen]
            result.audio_only += [n for j, n in enumerate(group_heard) if j not in used_heard]

    result.pairs.sort(key=lambda pair: (pair[1].onset, pair[1].pitch))
    result.video_only.sort(key=lambda n: (n.onset, n.pitch))
    result.audio_only.sort(key=lambda n: (n.onset, n.pitch))
    return result


def _groups(
    seen: list[NoteEvent], heard: list[NoteEvent], tolerance: float
) -> list[tuple[list[NoteEvent], list[NoteEvent]]]:
    """Разбить ноты одной высоты на группы, между которыми больше допуска.

    Ноты из разных групп в пару попасть не могут, поэтому группы решаются
    по отдельности.
    """
    merged = sorted(
        [(note.onset, 0, note) for note in seen] + [(note.onset, 1, note) for note in heard],
        key=lambda item: item[0],
    )
    groups: list[tuple[list[NoteEvent], list[NoteEvent]]] = []
    last = float("-inf")
    for onset, channel, note in merged:
        if onset - last > tolerance:
            groups.append(([], []))
        groups[-1][channel].append(note)
        last = onset
    return groups


def _align(
    seen: list[NoteEvent], heard: list[NoteEvent], tolerance: float
) -> list[tuple[int, int]]:
    """Лучшее сопоставление без пересечений: больше пар, затем меньше расхождение.

    `best[i][j]` — лучший результат для первых `i` нот видео и первых `j` нот
    звука: пара (число пар, минус сумма расхождений), сравнение по порядку.
    """
    if not seen or not heard:
        return []
    rows, columns = len(seen), len(heard)
    best = [[(0, 0.0)] * (columns + 1) for _ in range(rows + 1)]
    for i in range(1, rows + 1):
        for j in range(1, columns + 1):
            choice = max(best[i - 1][j], best[i][j - 1])
            gap = abs(seen[i - 1].onset - heard[j - 1].onset)
            if gap <= tolerance:
                pairs, cost = best[i - 1][j - 1]
                choice = max(choice, (pairs + 1, cost - gap))
            best[i][j] = choice

    paired: list[tuple[int, int]] = []
    i, j = rows, columns
    while i > 0 and j > 0:
        gap = abs(seen[i - 1].onset - heard[j - 1].onset)
        pairs, cost = best[i - 1][j - 1]
        if gap <= tolerance and best[i][j] == (pairs + 1, cost - gap):
            paired.append((i - 1, j - 1))
            i, j = i - 1, j - 1
        elif best[i][j] == best[i - 1][j]:
            i -= 1
        else:
            j -= 1
    paired.reverse()
    return paired
