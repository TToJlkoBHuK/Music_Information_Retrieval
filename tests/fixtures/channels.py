"""Модель ошибок двух каналов: из известных нот — «ноты видео» и «ноты звука».

Слиянию нужен эталон, а у настоящих роликов его нет. Здесь эталон задаётся,
а каналы получаются из него порчей — теми ошибками, которые найдены при
разборе трёх настоящих роликов (`mir/fusion/README.md`):

| канал | ошибка | откуда |
|---|---|---|
| видео | блок не доказан и остался кандидатом | тусклые блоки, дым в нижней полосе |
| видео | блок не замечен вовсе | тусклые блоки в облаке искр (В-13) |
| видео | повторы с малым зазором слиты в одну ноту | зазор меньше кадра |
| видео | долгая нота разорвана надвое | свечение (В-10) |
| видео | ложная нота | дым прошёл обе полосы |
| звук | нота пропущена | тихая нота в аккорде |
| звук | обертон на октаву выше | типичная ошибка транскрипции |
| звук | повторная атака на долгой ноте | то же |
| звук | эхо через 0.2 с | эффект задержки в записи (Interstellar) |
| звук | музыка после конца пьесы | заставка канала в титрах |

Уверенность нот звука разыгрывается по замеру: у настоящих — около
0.88 ± 0.04, у ложных — от 0.3 до 0.8.

Это модель, а не данные: она проверяет, что слияние правильно обращается
с ошибками известного вида, и не говорит, насколько часто они встречаются
в жизни. Доли ошибок в сценариях взяты того же порядка, что на роликах.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np

from mir.common.enums import Hand, Source
from mir.common.types import PITCH_MAX, NoteEvent

__all__ = ["SCENARIOS", "ChannelModel", "Channels", "piece", "simulate", "video_on_audio_clock"]

_NEIGHBOURS = (89, 90)
"""Фа и фа-диез третьей октавы: выше всей остальной фактуры пьесы, чтобы не совпасть с ней."""


@dataclass(frozen=True)
class ChannelModel:
    """Доли ошибок каналов. Значения по умолчанию — ролик вроде Rousseau."""

    offset: float = 0.11
    drift: float = 0.0
    """Насколько сдвиг звука растёт за секунду ролика."""
    pitch_shift: int = 0
    """На сколько полутонов видео ошиблось в октаве (обрезанная клавиатура)."""

    video_jitter: float = 0.006
    video_unproven: float = 0.01
    video_miss: float = 0.01
    video_merge_gap: float = 0.045
    video_split: float = 0.02
    video_false: float = 0.01
    smoke_candidates: float = 0.5
    """Ложных блоков-кандидатов на одну ноту: дым при низком пороге датчика."""

    audio_jitter: float = 0.010
    audio_miss: float = 0.03
    audio_ghost: float = 0.03
    audio_retrigger: float = 0.04
    audio_echo: float = 0.0
    audio_pedal: float = 0.8
    """Средняя прибавка к длительности: звук тянется, пока держит педаль."""
    outro_notes: int = 0


@dataclass
class Channels:
    """Два канала и эталон.

    Attributes:
        truth: Эталон в шкале звука.
        video: Ноты видео в шкале видео.
        candidates: Блоки-кандидаты видео.
        audio: Ноты звука.
    """

    truth: list[NoteEvent]
    video: list[NoteEvent] = field(default_factory=list)
    candidates: list[NoteEvent] = field(default_factory=list)
    audio: list[NoteEvent] = field(default_factory=list)


def piece(seed: int = 0, bars: int = 24, bar: float = 2.0) -> list[NoteEvent]:
    """Пьеса в две руки: бас октавами, аккорды, мелодия с повторами нот.

    В ней есть всё, на чём ошибаются каналы: долгие ноты (разрывы и ложные
    повторные атаки), октавы (обертоны), повторы одной клавиши (слияние
    блоков), соседние белая и чёрная клавиши.
    """
    rng = np.random.default_rng(seed)
    notes: list[NoteEvent] = []
    roots = [37, 35, 33, 30, 32, 37, 42, 44]
    for index in range(bars):
        start = 1.0 + index * bar
        root = roots[index % len(roots)]
        loud = int(rng.integers(45, 80))
        for pitch in (root, root + 12):  # бас октавой на весь такт
            notes.append(NoteEvent(pitch, start, start + bar * 0.95, loud, Hand.LEFT))
        chord = [root + 24, root + 28, root + 31]
        for step in range(8):  # восьмые: фигурация по аккорду
            onset = start + step * bar / 8
            pitch = chord[step % 3]
            notes.append(NoteEvent(pitch, onset, onset + bar / 8 * 0.9, loud - 10, Hand.RIGHT))
        melody = root + 36 + int(rng.integers(0, 5))
        for step in range(4):  # мелодия: повторы одной ноты с малым зазором
            onset = start + step * bar / 4
            gap = 0.03 if step % 2 else 0.12
            notes.append(NoteEvent(melody, onset, onset + bar / 4 - gap, loud + 10, Hand.RIGHT))
        if index % 3 == 0:  # малая секунда: белая и чёрная соседки вместе
            for pitch in _NEIGHBOURS:
                notes.append(NoteEvent(pitch, start + bar / 2, start + bar * 0.9, loud, Hand.RIGHT))
    return sorted(notes, key=lambda n: (n.onset, n.pitch))


def simulate(truth: list[NoteEvent], model: ChannelModel, seed: int = 0) -> Channels:
    """Получить из эталона ноты двух каналов с ошибками по модели."""
    rng = np.random.default_rng(seed)
    channels = Channels(truth=sorted(truth, key=lambda n: (n.onset, n.pitch)))
    _video(channels, model, rng)
    _audio(channels, model, rng)
    return channels


def _video_time(time: float, model: ChannelModel) -> float:
    """Шкала звука → шкала видео: сдвиг растёт со временем."""
    return (time - model.offset) / (1.0 + model.drift)


def _video(channels: Channels, model: ChannelModel, rng: np.random.Generator) -> None:
    shown = _merge_repeats(channels.truth, model.video_merge_gap)
    shown = _hide_neighbours(shown)
    end = max(note.offset for note in channels.truth)
    pitches = sorted({note.pitch for note in channels.truth})

    for note in shown:
        onset = _video_time(note.onset, model) + float(rng.normal(0.0, model.video_jitter))
        block = replace(
            note,
            pitch=note.pitch - model.pitch_shift,
            onset=onset,
            offset=onset + note.duration,
            velocity=64,
            source=Source.VIDEO,
            confidence=float(rng.uniform(0.8, 1.0)),
        )
        draw = float(rng.uniform())
        if draw < model.video_miss:
            continue
        if draw < model.video_miss + model.video_unproven:
            channels.candidates.append(replace(block, confidence=float(rng.uniform(0.1, 0.5))))
        elif note.duration > 1.0 and rng.uniform() < model.video_split:
            cut = onset + note.duration * float(rng.uniform(0.3, 0.7))
            channels.video.append(replace(block, offset=cut))
            channels.video.append(replace(block, onset=cut + 0.01))
        else:
            channels.video.append(block)

    for _ in range(round(model.video_false * len(shown))):
        onset = float(rng.uniform(1.0, end))
        pitch = int(rng.choice(pitches)) + int(rng.choice([-2, -1, 1, 2]))
        channels.video.append(
            NoteEvent(pitch, onset, onset + float(rng.uniform(0.08, 0.25)), 64, confidence=0.6)
        )
    for _ in range(round(model.smoke_candidates * len(shown))):
        onset = float(rng.uniform(1.0, end))
        channels.candidates.append(
            NoteEvent(int(rng.choice(pitches)), onset, onset + 0.2, 64, confidence=0.3)
        )
    channels.video.sort(key=lambda n: (n.onset, n.pitch))
    channels.candidates.sort(key=lambda n: (n.onset, n.pitch))


def _merge_repeats(notes: list[NoteEvent], gap: float) -> list[NoteEvent]:
    """Повторы одной клавиши с зазором меньше `gap` видео показывает одной нотой."""
    merged: list[NoteEvent] = []
    last: dict[int, int] = {}
    for note in notes:
        index = last.get(note.pitch)
        if index is not None and 0 <= note.onset - merged[index].offset < gap:
            merged[index] = replace(merged[index], offset=note.offset)
        else:
            last[note.pitch] = len(merged)
            merged.append(note)
    return merged


def _hide_neighbours(notes: list[NoteEvent]) -> list[NoteEvent]:
    """Из одновременных фа и фа-диеза видео видит только фа: блоки слились."""
    white, black = _NEIGHBOURS
    lower = {round(note.onset, 3) for note in notes if note.pitch == white}
    return [note for note in notes if not (note.pitch == black and round(note.onset, 3) in lower)]


def _audio(channels: Channels, model: ChannelModel, rng: np.random.Generator) -> None:
    def real() -> float:
        return float(np.clip(rng.normal(0.88, 0.04), 0.5, 0.99))

    def false() -> float:
        return float(rng.uniform(0.3, 0.8))

    truth = channels.truth
    sounding = {(note.pitch, round(note.onset, 2)) for note in truth}
    heard: list[NoteEvent] = []
    for note in truth:
        if rng.uniform() >= model.audio_miss:
            onset = note.onset + float(rng.normal(0.0, model.audio_jitter))
            tail = float(rng.exponential(model.audio_pedal))
            heard.append(
                replace(
                    note,
                    onset=onset,
                    offset=onset + note.duration + tail,
                    hand=Hand.UNKNOWN,
                    source=Source.AUDIO,
                    confidence=real(),
                )
            )
        ghost = note.pitch + 12
        if (
            rng.uniform() < model.audio_ghost
            and ghost <= PITCH_MAX
            and (ghost, round(note.onset, 2)) not in sounding
        ):
            heard.append(
                NoteEvent(
                    ghost, note.onset, note.onset + 0.4, 30, source=Source.AUDIO, confidence=false()
                )
            )
        if note.duration > 0.8 and rng.uniform() < model.audio_retrigger:
            onset = note.onset + float(rng.uniform(0.3, note.duration * 0.9))
            heard.append(
                NoteEvent(
                    note.pitch, onset, onset + 0.5, 28, source=Source.AUDIO, confidence=false()
                )
            )
        if rng.uniform() < model.audio_echo:
            onset = note.onset + 0.2
            heard.append(
                NoteEvent(
                    note.pitch, onset, onset + 0.3, 40, source=Source.AUDIO, confidence=false()
                )
            )
    end = max(note.offset for note in truth)
    for index in range(model.outro_notes):
        onset = end + 2.0 + index * 0.3
        heard.append(
            NoteEvent(
                52 + index % 12, onset, onset + 1.0, 60, source=Source.AUDIO, confidence=real()
            )
        )
    channels.audio = sorted(heard, key=lambda n: (n.onset, n.pitch))


def video_on_audio_clock(channels: Channels, model: ChannelModel) -> list[NoteEvent]:
    """Ноты видео, переведённые в шкалу звука истинным сдвигом.

    Для сравнения «видео в одиночку против слияния»: одиночному каналу
    рассинхрон прощается (на нотный лист он не влияет), ошибка в октаве — нет.
    """
    moved: list[NoteEvent] = []
    for note in channels.video:
        onset = note.onset * (1.0 + model.drift) + model.offset
        moved.append(replace(note, onset=onset, offset=onset + note.duration))
    return moved


SCENARIOS: dict[str, ChannelModel] = {
    "чистый ролик (вроде Rousseau)": ChannelModel(),
    "тусклые блоки в искрах (вроде Interstellar)": ChannelModel(
        video_unproven=0.12, video_miss=0.06, audio_echo=0.10, audio_retrigger=0.08
    ),
    "звук с эффектами и музыкой в титрах": ChannelModel(
        audio_echo=0.15, audio_ghost=0.08, audio_retrigger=0.10, outro_notes=30
    ),
    "рассинхрон плывёт (сто миллионных)": ChannelModel(offset=0.30, drift=1e-4),
    "обрезанная клавиатура, ошибка в октаве": ChannelModel(pitch_shift=-12),
    "видео сломано (подсветка под руками)": ChannelModel(
        video_miss=0.55, video_false=1.5, video_unproven=0.0
    ),
    "звук пропускает треть нот": ChannelModel(audio_miss=0.33),
}
"""Сценарии для проверки NF-10: слияние не хуже лучшего одиночного канала."""
