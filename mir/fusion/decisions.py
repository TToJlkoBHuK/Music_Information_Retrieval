"""Решения слияния: что стало с каждой нотой каждого канала и почему.

Итог слияния — не только список нот. По каждой исходной ноте записано
решение с причиной: это и отчёт для пользователя («сколько нот подтверждено
обоими каналами»), и материал для экспериментальной главы — какие ошибки
каналов встречаются и как часто.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

from mir.common.enums import Source, StrEnum
from mir.common.types import NoteEvent, QualityReport
from mir.fusion.aligner import OffsetCurve

__all__ = ["Decision", "FusionResult", "Reason"]


class Reason(StrEnum):
    """Почему нота принята или отброшена."""

    CONFIRMED = "confirmed"
    LATE_PAIR = "late_pair"
    VIDEO_ONLY = "video_only"
    CANDIDATE = "candidate"
    HIDDEN = "hidden"
    OUT_OF_VIEW = "out_of_view"
    RESTRIKE = "restrike"
    CONFIDENT = "confident"
    AUDIO_LEAD = "audio_lead"
    SINGLE_CHANNEL = "single_channel"

    OUTSIDE = "outside"
    KEY_HELD = "key_held"
    HARMONIC = "harmonic"
    ECHO = "echo"
    WEAK = "weak"
    SPLIT = "split"
    UNTRUSTED = "untrusted"

    @property
    def accepted(self) -> bool:
        """Нота с такой причиной вошла в итог."""
        return self in _ACCEPTED

    @property
    def title_ru(self) -> str:
        """Пояснение для отчёта."""
        return _TITLES[self]


_ACCEPTED = frozenset(
    {
        Reason.CONFIRMED,
        Reason.LATE_PAIR,
        Reason.VIDEO_ONLY,
        Reason.CANDIDATE,
        Reason.HIDDEN,
        Reason.OUT_OF_VIEW,
        Reason.RESTRIKE,
        Reason.CONFIDENT,
        Reason.AUDIO_LEAD,
        Reason.SINGLE_CHANNEL,
    }
)

_TITLES: dict[Reason, str] = {
    Reason.CONFIRMED: "оба канала: высота и начало совпали",
    Reason.LATE_PAIR: "оба канала: начало в видео съехало, взято из звука",
    Reason.VIDEO_ONLY: "только видео: блок прошёл обе полосы датчика",
    Reason.CANDIDATE: "звук и блок-кандидат видео",
    Reason.HIDDEN: "только звук: блок слился с блоком соседней клавиши",
    Reason.OUT_OF_VIEW: "только звук: клавиша за пределами кадра",
    Reason.RESTRIKE: "только звук: повторный удар, блоки в видео слились",
    Reason.CONFIDENT: "только звук: уверенная нота, блок в видео не найден",
    Reason.AUDIO_LEAD: "только звук: видеоканал ненадёжен",
    Reason.SINGLE_CHANNEL: "каналы не совмещены: взят один канал",
    Reason.OUTSIDE: "звук за пределами видеоряда (заставка, титры)",
    Reason.KEY_HELD: "звук: повторная атака на удерживаемой клавише",
    Reason.HARMONIC: "звук: на октаву или дуодециму от ноты видео",
    Reason.ECHO: "звук: повтор сразу после ноты видео",
    Reason.WEAK: "звук: неуверенная нота без блока в видео",
    Reason.SPLIT: "видео: разрыв длинного блока, слит с предыдущей нотой",
    Reason.UNTRUSTED: "только видео: видеоканал ненадёжен",
}


@dataclass(frozen=True)
class Decision:
    """Решение по одной ноте или паре нот.

    Каждая исходная нота каждого канала входит ровно в одно решение.

    Attributes:
        reason: Причина.
        video: Нота видео в шкале времени итога; `None`, если её не было.
        audio: Нота звука; `None`, если её не было.
        note: Итоговая нота; `None`, если отброшена.
        related: На что опиралось решение по ноте звука: блок-кандидат
            или нота видео, которая в этот момент держала клавишу.
    """

    reason: Reason
    video: NoteEvent | None = None
    audio: NoteEvent | None = None
    note: NoteEvent | None = None
    related: NoteEvent | None = None


@dataclass
class FusionResult:
    """Итог слияния каналов.

    Время итоговых нот — в шкале звука: в ней же сетка долей и педаль, а
    время видео переводится по кривой рассинхрона.

    Attributes:
        notes: Итоговые ноты по возрастанию времени.
        decisions: Решение по каждой ноте каждого канала.
        curve: Рассинхрон по ходу ролика.
        pitch_shift: На сколько полутонов сдвинуты ноты видео (октава
            обрезанной клавиатуры определена по звуку).
        lead: Ведущий канал: чьи одиночные ноты принимаются без условий.
        video_trust: Доля нот видео, подтверждённых звуком.
        audio_trust: Доля нот звука, подтверждённых видео.
        confidence_threshold: Уверенность, с которой нота только из звука
            принимается без подтверждения видео.
        warnings: Предупреждения для пользователя.
    """

    notes: list[NoteEvent]
    decisions: list[Decision]
    curve: OffsetCurve
    pitch_shift: int = 0
    lead: Source = Source.VIDEO
    video_trust: float = 0.0
    audio_trust: float = 0.0
    confidence_threshold: float = float("inf")
    warnings: list[str] = field(default_factory=list)

    @property
    def counts(self) -> Counter[Reason]:
        """Сколько решений с каждой причиной."""
        return Counter(decision.reason for decision in self.decisions)

    def rejected(self, channel: Source) -> list[Decision]:
        """Отброшенные ноты канала — для показа «сомнительных» нот."""
        return [
            decision
            for decision in self.decisions
            if decision.note is None
            and (decision.video if channel is Source.VIDEO else decision.audio) is not None
        ]

    def quality_report(self, base: QualityReport | None = None) -> QualityReport:
        """Заполнить отчёт о достоверности.

        Args:
            base: Отчёт с уже заполненными сведениями о клавиатуре; без него
                создаётся новый.
        """
        report = base or QualityReport()
        by_source = Counter(note.source for note in self.notes)
        report.notes_confirmed = by_source[Source.BOTH]
        report.notes_from_video_only = by_source[Source.VIDEO]
        report.notes_from_audio_only = by_source[Source.AUDIO]
        report.av_offset_ms = self.curve.overall.offset_ms if self.curve.reliable else 0.0
        report.warnings.extend(self.warnings)
        return report
