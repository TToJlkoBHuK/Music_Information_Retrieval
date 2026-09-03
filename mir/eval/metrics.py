"""Метрики качества распознавания.

Считаются по правилам MIREX для задачи Note Tracking — тем же, по которым
меряются работы в этой области. Своя формула сделала бы числа
несопоставимыми с публикациями, а сравнение с аналогами — главный
аргумент в пользу проекта.

Правила сопоставления:

* нота засчитана, если высота совпала точно, а начало отличается не
  более чем на `onset_tolerance` (стандарт — 50 мс);
* соответствие взаимно однозначное: одна найденная нота закрывает ровно
  одну эталонную, дубли идут в ложные срабатывания;
* длительность проверяется отдельной, более строгой метрикой: она хуже
  определяется на слух и по видео, поэтому обычно приводится отдельно.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from mir.common.enums import Hand
from mir.common.types import NoteEvent

__all__ = ["TranscriptionScore", "evaluate", "match_notes"]

ONSET_TOLERANCE = 0.05
"""Допуск на момент начала ноты, секунды. Стандарт MIREX."""

OFFSET_TOLERANCE_RATIO = 0.2
"""Допуск на конец ноты как доля её длительности."""

OFFSET_TOLERANCE_MIN = 0.05
"""Нижняя граница допуска на конец ноты, секунды."""


@dataclass(frozen=True)
class TranscriptionScore:
    """Результат сравнения с эталоном.

    Attributes:
        reference_count: Нот в эталоне.
        estimated_count: Нот найдено.
        matched: Совпало нот.
        onset_error_avg: Средняя ошибка начала среди совпавших, секунды.
        onset_error_max: Наибольшая ошибка начала, секунды.
        duration_error_avg: Средняя ошибка длительности, секунды.
        offset_matched: Совпало нот с учётом конца звучания.
        hand_matched: Совпало нот с верно определённой рукой.
        hand_known: Совпавших нот, у которых рука вообще определена.
    """

    reference_count: int
    estimated_count: int
    matched: int
    onset_error_avg: float = 0.0
    onset_error_max: float = 0.0
    duration_error_avg: float = 0.0
    offset_matched: int = 0
    hand_matched: int = 0
    hand_known: int = 0

    @property
    def precision(self) -> float:
        """Доля верных среди найденных."""
        return self.matched / self.estimated_count if self.estimated_count else 0.0

    @property
    def recall(self) -> float:
        """Доля найденных среди эталонных."""
        return self.matched / self.reference_count if self.reference_count else 0.0

    @property
    def f1(self) -> float:
        """Гармоническое среднее точности и полноты.

        Гармоническое, а не обычное: система, нашедшая все ноты вместе
        с тысячей лишних, не должна получать половину балла.
        """
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if p + r else 0.0

    @property
    def f1_with_offset(self) -> float:
        """F-мера при дополнительной проверке конца звучания."""
        p = self.offset_matched / self.estimated_count if self.estimated_count else 0.0
        r = self.offset_matched / self.reference_count if self.reference_count else 0.0
        return 2 * p * r / (p + r) if p + r else 0.0

    @property
    def hand_accuracy(self) -> float:
        """Доля верно определённых рук среди совпавших нот."""
        return self.hand_matched / self.hand_known if self.hand_known else 0.0

    def format_report(self, title: str = "качество распознавания") -> str:
        """Человекочитаемый отчёт для консоли и логов."""
        lines = [
            f"=== {title} ===",
            f"эталон / найдено : {self.reference_count} / {self.estimated_count}",
            f"совпало          : {self.matched}",
            f"precision        : {self.precision:.3f}",
            f"recall           : {self.recall:.3f}",
            f"F1               : {self.f1:.3f}",
            f"F1 с длительностью: {self.f1_with_offset:.3f}",
            f"onset            : средняя {self.onset_error_avg * 1000:.1f} мс, "
            f"макс {self.onset_error_max * 1000:.1f} мс",
            f"длительность     : средняя ошибка {self.duration_error_avg * 1000:.1f} мс",
        ]
        if self.hand_known:
            lines.append(
                f"рука             : {self.hand_matched}/{self.hand_known} "
                f"({self.hand_accuracy:.3f})"
            )
        return "\n".join(lines)


def match_notes(
    reference: Sequence[NoteEvent],
    estimated: Sequence[NoteEvent],
    onset_tolerance: float = ONSET_TOLERANCE,
) -> list[tuple[int, int]]:
    """Сопоставить найденные ноты эталонным.

    Жадный проход по эталону в порядке времени: для каждой эталонной ноты
    берётся ближайшая свободная найденная той же высоты. Полное решение
    задачи о назначениях дало бы на доли процента больше совпадений ценой
    кубической сложности — на роликах в тысячи нот это неоправданно.

    Args:
        reference: Эталонные ноты.
        estimated: Найденные ноты.
        onset_tolerance: Допуск на момент начала, секунды.

    Returns:
        Пары индексов `(эталон, найденное)`.
    """
    by_pitch: dict[int, list[int]] = {}
    for index, note in enumerate(estimated):
        by_pitch.setdefault(note.pitch, []).append(index)

    used: set[int] = set()
    pairs: list[tuple[int, int]] = []

    for ref_index in sorted(range(len(reference)), key=lambda i: reference[i].onset):
        ref = reference[ref_index]
        best: int | None = None
        best_delta = onset_tolerance

        for est_index in by_pitch.get(ref.pitch, []):
            if est_index in used:
                continue
            delta = abs(estimated[est_index].onset - ref.onset)
            if delta <= best_delta:
                best, best_delta = est_index, delta

        if best is not None:
            used.add(best)
            pairs.append((ref_index, best))

    return pairs


def evaluate(
    reference: Sequence[NoteEvent],
    estimated: Sequence[NoteEvent],
    onset_tolerance: float = ONSET_TOLERANCE,
) -> TranscriptionScore:
    """Сравнить результат распознавания с эталоном.

    Args:
        reference: Эталонные ноты.
        estimated: Найденные ноты.
        onset_tolerance: Допуск на момент начала, секунды.

    Returns:
        Полный набор метрик.

    Example:
        >>> score = evaluate(truth, result)
        >>> print(score.format_report())
    """
    pairs = match_notes(reference, estimated, onset_tolerance)
    if not pairs:
        return TranscriptionScore(len(reference), len(estimated), 0)

    onset_errors: list[float] = []
    duration_errors: list[float] = []
    offset_matched = 0
    hand_matched = 0
    hand_known = 0

    for ref_index, est_index in pairs:
        ref, est = reference[ref_index], estimated[est_index]
        onset_errors.append(abs(est.onset - ref.onset))
        duration_errors.append(abs(est.duration - ref.duration))

        offset_allow = max(OFFSET_TOLERANCE_MIN, ref.duration * OFFSET_TOLERANCE_RATIO)
        if abs(est.offset - ref.offset) <= offset_allow:
            offset_matched += 1

        if ref.hand is not Hand.UNKNOWN and est.hand is not Hand.UNKNOWN:
            hand_known += 1
            if ref.hand is est.hand:
                hand_matched += 1

    return TranscriptionScore(
        reference_count=len(reference),
        estimated_count=len(estimated),
        matched=len(pairs),
        onset_error_avg=sum(onset_errors) / len(onset_errors),
        onset_error_max=max(onset_errors),
        duration_error_avg=sum(duration_errors) / len(duration_errors),
        offset_matched=offset_matched,
        hand_matched=hand_matched,
        hand_known=hand_known,
    )
