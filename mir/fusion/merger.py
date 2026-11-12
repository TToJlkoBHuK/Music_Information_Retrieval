"""Слияние каналов: из нот видео и нот звука — один список (#30, F-10).

Последовательность:

1. **октава** — у обрезанной клавиатуры видео может ошибиться на октаву;
   верный сдвиг тот, при котором каналы совмещаются;
2. **рассинхрон** — кривая сдвига звука по ходу ролика; ноты видео
   переводятся в шкалу звука;
3. **границы** — звук до первого блока и после последнего отбрасывается:
   это музыка заставки и титров (у Clair de lune — 65 нот в другой
   тональности);
4. **сопоставление** — пары одной высоты с близким началом; второй проход с
   допуском шире подбирает пары, где начало в видео съехало;
5. **одиночки** — правила из `resolver`;
6. **сборка** — у пары каждое свойство берётся из того канала, который
   знает его лучше.

| свойство | откуда | почему |
|---|---|---|
| высота | видео | положение блока над клавишей однозначно |
| начало | звук | в его шкале сетка долей и педаль; у пары расхождение 6–9 мс |
| конец | видео | длина блока — время удержания клавиши; звук тянется, пока держит педаль |
| громкость | звук | блок её не кодирует |
| рука | видео | по цвету блока, если руки окрашены по-разному |

**Ведущий канал.** Пока звук подтверждает хотя бы половину нот видео (или
видео — половину нот звука), ведущий — видео: его одиночные ноты
принимаются, одиночные ноты звука проходят правила. Если согласия нет —
так бывает, когда нот по блокам нет и подсветку клавиш портят руки
исполнителя (`docs/known-issues.md`, В-05), — ведущим становится звук, а
ноты только из видео отбрасываются.
"""

from __future__ import annotations

from bisect import bisect_left, bisect_right
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import replace

import numpy as np

from mir.common.enums import Hand, Source
from mir.common.logging import get_logger
from mir.common.types import KeyboardLayout, NoteEvent
from mir.config import FusionConfig
from mir.fusion.aligner import (
    OffsetCurve,
    OffsetEstimate,
    estimate_offset_curve,
    estimate_pitch_shift,
    transpose,
)
from mir.fusion.decisions import Decision, FusionResult, Reason
from mir.fusion.matcher import match_notes
from mir.fusion.resolver import CandidatePool, VideoIndex, resolve_audio_only

__all__ = ["fuse"]

_log = get_logger(__name__)

_MIN_DURATION = 0.01
"""Короче ноту не делаем: `NoteEvent` требует конец строго позже начала."""

_VELOCITY_WINDOW = 2.0
"""Громкость ноты без звука берётся у соседей по времени, секунды в обе стороны.

Блок громкость не кодирует, а динамические оттенки (этап нотации) усредняются
по фрагментам в несколько долей — двух секунд хватает, чтобы нота без пары
не выбивалась из своего фрагмента.
"""

_DEFAULT_VELOCITY = 64
"""Громкость, когда соседей нет вовсе: середина шкалы MIDI."""


def fuse(
    video_notes: Sequence[NoteEvent],
    audio_notes: Sequence[NoteEvent],
    config: FusionConfig,
    *,
    candidates: Sequence[NoteEvent] = (),
    layout: KeyboardLayout | None = None,
    video_reliable: bool = True,
) -> FusionResult:
    """Слить ноты двух каналов в один список.

    Args:
        video_notes: Ноты видеоканала, время — по шкале видео.
        audio_notes: Ноты аудиоканала.
        config: Параметры из секции `[fusion]`.
        candidates: Блоки-кандидаты видео — то, что датчик видел, но не
            смог доказать (`VisionResult.unconfirmed`).
        layout: Разметка клавиатуры: какие клавиши видны в кадре и могла ли
            быть ошибка в октаве. Без неё все клавиши считаются видимыми.
        video_reliable: Ноты видео получены по блокам. `False` — по
            подсветке клавиш: если каналы не совместятся, итогом будет звук.

    Returns:
        Ноты в шкале времени звука и решение по каждой исходной ноте.
    """
    video = sorted(video_notes, key=lambda n: (n.onset, n.pitch))
    audio = sorted(audio_notes, key=lambda n: (n.onset, n.pitch))
    if not video or not audio:
        return _single_channel(video, audio, video_first=bool(video), warnings=[])

    warnings: list[str] = []
    shift = 0
    if layout is None or layout.is_cropped:
        shift = estimate_pitch_shift(video, audio, config)
    if shift:
        video = transpose(video, shift)
        candidates = transpose(candidates, shift)
        warnings.append(
            f"Октава клавиатуры определена по звуку: ноты видео сдвинуты на {shift:+d} полутонов"
        )

    curve = estimate_offset_curve(video, audio, config)
    if not curve.reliable:
        result = _single_channel(
            video,
            audio,
            video_first=video_reliable,
            warnings=[
                *warnings,
                "Каналы не удалось совместить по времени: ноты взяты только из "
                + ("видео" if video_reliable else "звука"),
            ],
        )
        result.curve, result.pitch_shift = curve, shift
        return result

    seen = [_to_audio_time(note, curve) for note in video]
    pool = CandidatePool(_to_audio_time(note, curve) for note in candidates)
    tolerance = config.onset_tolerance

    start = min(note.onset for note in seen) - tolerance
    end = max(note.offset for note in seen) + tolerance
    inside = [note for note in audio if start <= note.onset <= end]
    outside = [note for note in audio if not start <= note.onset <= end]

    tight = match_notes(seen, inside, tolerance)
    video_trust = len(tight.pairs) / len(seen)
    audio_trust = len(tight.pairs) / len(inside) if inside else 0.0
    trusted = max(video_trust, audio_trust) >= config.min_channel_trust
    threshold = _confidence_threshold([a for _, a in tight.pairs], config)

    loose = match_notes(tight.video_only, tight.audio_only, config.loose_onset_tolerance)
    pairs = [(v, a, Reason.CONFIRMED) for v, a in tight.pairs]
    pairs += [(v, a, Reason.LATE_PAIR) for v, a in loose.pairs]

    builder = _Builder(seen, pairs, video_trust, audio_trust, threshold, trusted)
    decisions: list[Decision] = []
    if trusted:
        _resolve_video_lead(
            builder, loose.video_only, loose.audio_only, pool, layout, shift, config
        )
        decisions = [Decision(Reason.OUTSIDE, audio=note) for note in outside]
    else:
        # Границы видеоряда заданы нотами видео, а им веры нет: звук не обрезается.
        builder.accept_all_audio([*loose.audio_only, *outside])
        outside = []
        builder.reject_video(loose.video_only)
        warnings.append(
            f"Видеоканал ненадёжен: звук подтверждает лишь {video_trust:.0%} его нот. "
            "Итог взят из звука"
        )

    notes, built = builder.build()
    decisions += built
    if trusted and audio_trust < config.min_channel_trust:
        warnings.append(
            f"В звуке много нот без блоков в видео (подтверждено {audio_trust:.0%}): "
            "вероятно, сопровождение или эффекты записи"
        )
    if outside:
        warnings.append(
            f"Отброшено {len(outside)} нот звука за пределами видеоряда — "
            "музыка заставки или титров"
        )
    if not np.isfinite(threshold):
        warnings.append(
            "У нот звука нет оценки уверенности: ноты только из звука принимаются "
            "лишь с подтверждением видео"
        )

    result = FusionResult(
        notes=notes,
        decisions=sorted(decisions, key=_decision_time),
        curve=curve,
        pitch_shift=shift,
        lead=Source.VIDEO if trusted else Source.AUDIO,
        video_trust=video_trust,
        audio_trust=audio_trust,
        confidence_threshold=threshold,
        warnings=warnings,
    )
    counts = result.counts
    _log.info(
        "слияние: нот %d (оба канала %d, только видео %d, только звук %d), отброшено %d",
        len(notes),
        counts[Reason.CONFIRMED] + counts[Reason.LATE_PAIR] + counts[Reason.CANDIDATE],
        counts[Reason.VIDEO_ONLY],
        sum(1 for note in notes if note.source is Source.AUDIO),
        sum(1 for decision in result.decisions if decision.note is None),
    )
    return result


def _resolve_video_lead(
    builder: _Builder,
    video_only: list[NoteEvent],
    audio_only: list[NoteEvent],
    pool: CandidatePool,
    layout: KeyboardLayout | None,
    shift: int,
    config: FusionConfig,
) -> None:
    """Разобрать одиночные ноты, когда ведущий канал — видео."""
    tolerance = config.onset_tolerance
    index = VideoIndex(builder.seen)
    for note in audio_only:
        in_view = layout is None or layout.covers(note.pitch - shift)
        reason, related = resolve_audio_only(
            note, index, pool, builder.threshold, tolerance, in_view
        )
        builder.add_audio(note, reason, related)

    for note in video_only:
        previous = builder.split_source(note, tolerance)
        if previous is not None:
            builder.merge_split(previous, note)
        else:
            builder.add_video(note)


class _Builder:
    """Собирает итоговые ноты из решений.

    Решения по одиночкам меняют и ноты видео: повторный удар обрезает ноту,
    на которую пришёлся, а разрыв блока продлевает предыдущую. Поэтому
    итоговые ноты строятся в конце, когда все решения приняты.
    """

    def __init__(
        self,
        seen: list[NoteEvent],
        pairs: list[tuple[NoteEvent, NoteEvent, Reason]],
        video_trust: float,
        audio_trust: float,
        threshold: float,
        video_leads: bool,
    ) -> None:
        self.seen = seen
        self.threshold = threshold
        self._video_leads = video_leads
        self._video_trust = video_trust
        self._audio_trust = audio_trust
        self._pairs = pairs
        self._partner = {id(v): a for v, a, _ in pairs}
        self._video: list[tuple[NoteEvent, Reason]] = []
        self._audio: list[tuple[NoteEvent, Reason, NoteEvent | None]] = []
        self._ends: dict[int, float] = {}
        self._cuts: dict[int, list[float]] = defaultdict(list)
        self._by_pitch: dict[int, list[NoteEvent]] = defaultdict(list)
        for note in seen:
            self._by_pitch[note.pitch].append(note)
        ordered = sorted((a.onset, a.velocity) for _, a, _ in pairs)
        self._velocity_times = [time for time, _ in ordered]
        self._velocities = [velocity for _, velocity in ordered]

    def add_audio(self, note: NoteEvent, reason: Reason, related: NoteEvent | None) -> None:
        """Записать решение по одиночной ноте звука."""
        self._audio.append((note, reason, related))
        if reason is Reason.RESTRIKE and related is not None:
            self._cuts[id(related)].append(note.onset)

    def accept_all_audio(self, notes: list[NoteEvent]) -> None:
        """Принять все одиночные ноты звука: ведущий канал — звук."""
        self._audio += [(note, Reason.AUDIO_LEAD, None) for note in notes]

    def add_video(self, note: NoteEvent) -> None:
        """Принять одиночную ноту видео."""
        self._video.append((note, Reason.VIDEO_ONLY))

    def reject_video(self, notes: list[NoteEvent]) -> None:
        """Отбросить одиночные ноты видео: ведущий канал — звук."""
        self._video += [(note, Reason.UNTRUSTED) for note in notes]

    def split_source(self, note: NoteEvent, tolerance: float) -> NoteEvent | None:
        """Нота видео, продолжением которой является `note`, — если это разрыв блока.

        Свечение долгой ноты иногда уводит светлый участок к соседней
        клавише, и датчик рвёт блок надвое (В-10). Признаки: вторая нота
        начинается там же, где кончилась первая (обрывки могут и перекрыться
        на кадр), а звук в этот момент новой атаки не слышит — его нота,
        парная первой, продолжает звучать.
        """
        for previous in self._by_pitch[note.pitch]:
            end = self._ends.get(id(previous), previous.offset)
            partner = self._partner.get(id(previous))
            if (
                previous is not note
                and partner is not None
                and note.onset > previous.onset
                and abs(note.onset - end) <= tolerance
                and partner.offset > note.onset + tolerance
            ):
                return previous
        return None

    def merge_split(self, previous: NoteEvent, piece: NoteEvent) -> None:
        """Присоединить обрывок блока к предыдущей ноте."""
        self._ends[id(previous)] = piece.offset
        self._video.append((piece, Reason.SPLIT))

    def build(self) -> tuple[list[NoteEvent], list[Decision]]:
        """Собрать итоговые ноты и решения."""
        notes: list[NoteEvent] = []
        decisions: list[Decision] = []

        for video, audio, reason in self._pairs:
            # У пары со второго прохода начала расходятся заметно. Обычно
            # прав звук: в видео начало съезжает, когда блок входит в дым.
            # Но если нота звука сама неуверенная, а видео — ведущий канал,
            # вернее начало блока.
            unsure = reason is Reason.LATE_PAIR and audio.confidence < self.threshold
            onset = video.onset if unsure and self._video_leads else audio.onset
            note = NoteEvent(
                pitch=video.pitch,
                onset=onset,
                offset=self._end(video, onset),
                velocity=audio.velocity,
                hand=video.hand,
                source=Source.BOTH,
                confidence=1.0 - (1.0 - video.confidence) * (1.0 - audio.confidence),
            )
            notes.append(note)
            decisions.append(Decision(reason, video, audio, note))

        for video, reason in self._video:
            if not reason.accepted:
                decisions.append(Decision(reason, video=video))
                continue
            note = replace(
                video,
                offset=self._end(video, video.onset),
                velocity=self._velocity_near(video.onset),
                source=Source.VIDEO,
                confidence=video.confidence * self._video_trust,
            )
            notes.append(note)
            decisions.append(Decision(reason, video=video, note=note))

        for audio, reason, related in self._audio:
            if not reason.accepted:
                decisions.append(Decision(reason, audio=audio, related=related))
                continue
            note = self._from_audio(audio, reason, related)
            notes.append(note)
            decisions.append(Decision(reason, audio=audio, note=note, related=related))

        notes.sort(key=lambda n: (n.onset, n.pitch))
        return notes, decisions

    def _from_audio(self, audio: NoteEvent, reason: Reason, related: NoteEvent | None) -> NoteEvent:
        """Нота, принятая из звука."""
        if reason is Reason.CANDIDATE and related is not None:
            # Длина блока — из кандидата: это удержание клавиши, как у пар.
            return replace(
                audio,
                offset=audio.onset + max(related.duration, _MIN_DURATION),
                hand=related.hand,
                source=Source.BOTH,
                confidence=1.0 - (1.0 - related.confidence) * (1.0 - audio.confidence),
            )
        offset, hand = audio.offset, Hand.UNKNOWN
        if reason is Reason.RESTRIKE and related is not None:
            # Повторный удар — та же клавиша и та же рука; держится до конца
            # общего блока или до следующего удара.
            later = [cut for cut in self._cuts[id(related)] if cut > audio.onset]
            limit = min([self._ends.get(id(related), related.offset), *later])
            offset, hand = max(limit, audio.onset + _MIN_DURATION), related.hand
        return replace(
            audio,
            offset=offset,
            hand=hand,
            source=Source.AUDIO,
            confidence=audio.confidence * self._audio_trust,
        )

    def _end(self, video: NoteEvent, onset: float) -> float:
        """Конец ноты видео с учётом присоединённых обрывков и повторных ударов."""
        end = max(onset + video.duration, self._ends.get(id(video), 0.0))
        cuts = [cut for cut in self._cuts.get(id(video), []) if cut > onset]
        if cuts:
            end = min(end, min(cuts))
        return max(end, onset + _MIN_DURATION)

    def _velocity_near(self, time: float) -> int:
        """Громкость для ноты без звука: медиана у подтверждённых нот рядом."""
        low = bisect_left(self._velocity_times, time - _VELOCITY_WINDOW)
        high = bisect_right(self._velocity_times, time + _VELOCITY_WINDOW)
        nearby = self._velocities[low:high] or self._velocities
        return int(np.median(nearby)) if nearby else _DEFAULT_VELOCITY


def _confidence_threshold(matched: list[NoteEvent], config: FusionConfig) -> float:
    """Уверенность, начиная с которой нота звука принимается без видео.

    Квантиль уверенности нот звука, подтверждённых видео: принимается нота,
    уверенная не меньше, чем заданная доля заведомо настоящих. Порог из
    самого ролика, а не константа: запись с шумом или эффектами снижает
    уверенность всех нот разом.

    Если оценок нет (у всех нот одно значение — так читается CSV без
    столбца уверенности), порог бесконечен: без видео ноты не принимаются.
    """
    values = np.array([note.confidence for note in matched])
    if len(values) < config.min_offset_pairs or float(np.ptp(values)) == 0.0:
        return float("inf")
    return float(np.quantile(values, config.audio_confidence_quantile))


def _single_channel(
    video: list[NoteEvent], audio: list[NoteEvent], *, video_first: bool, warnings: list[str]
) -> FusionResult:
    """Итог из одного канала: второго нет или совместить их не удалось."""
    chosen, lead = (video, Source.VIDEO) if video_first else (audio, Source.AUDIO)
    if not chosen:
        chosen, lead = (audio, Source.AUDIO) if video_first else (video, Source.VIDEO)
    decisions = [
        Decision(
            Reason.SINGLE_CHANNEL,
            video=note if lead is Source.VIDEO else None,
            audio=note if lead is Source.AUDIO else None,
            note=note,
        )
        for note in chosen
    ]
    if video and audio:
        dropped = audio if lead is Source.VIDEO else video
        _log.warning("каналы не совмещены: %d нот второго канала не использованы", len(dropped))
    elif not warnings:
        missing = "звука" if lead is Source.VIDEO else "видео"
        warnings = [f"Нот {missing} нет: сверка каналов не проводилась"]
    curve = OffsetCurve.constant(_no_estimate(), reliable=False)
    return FusionResult(list(chosen), decisions, curve, lead=lead, warnings=warnings)


def _no_estimate() -> OffsetEstimate:
    return OffsetEstimate(0.0, 0.0, 0, "notes")


def _to_audio_time(note: NoteEvent, curve: OffsetCurve) -> NoteEvent:
    """Перевести ноту видео в шкалу звука; длительность сохраняется."""
    onset = curve.to_audio(note.onset)
    return replace(note, onset=onset, offset=onset + note.duration)


def _decision_time(decision: Decision) -> tuple[float, int]:
    note = decision.note or decision.audio or decision.video
    assert note is not None
    return note.onset, note.pitch
