"""Полный разбор видеоряда на синтетических роликах.

Ролик рендерится из заранее известного списка нот, поэтому эталон
получается бесплатно и точно — ручная разметка не нужна и не может
ошибиться. Пороги здесь — это зафиксированные требования к качеству:
если правка алгоритма их ухудшит, тест упадёт.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from mir.common.enums import Hand, Source
from mir.common.types import NoteEvent
from mir.config import load_config
from mir.eval import evaluate
from mir.fusion import fuse
from mir.vision.analyzer import analyze_file
from tests.fixtures.synth import ColorScheme, SynthConfig, render_visualizer_video, simple_melody

pytestmark = pytest.mark.requires_ffmpeg


@pytest.fixture(scope="module")
def melody() -> list[NoteEvent]:
    return simple_melody()


@pytest.fixture(scope="module")
def clean_video(tmp_path_factory: pytest.TempPathFactory, melody: list[NoteEvent]) -> Path:
    out = tmp_path_factory.mktemp("vision") / "clean.mp4"
    return render_visualizer_video(melody, out)


def test_keyboard_range_recovered_exactly(clean_video: Path):
    """Диапазон определяется по узору чёрных клавиш, а не по краям кадра."""
    result = analyze_file(clean_video)

    assert result.layout.lowest_pitch == 36
    assert result.layout.highest_pitch == 96
    assert result.layout.confidence > 0.9
    assert result.layout.is_cropped is True


def test_calibration_finds_fall_speed_and_hit_line(clean_video: Path):
    config = SynthConfig()
    result = analyze_file(clean_video)

    assert result.profile.fall_speed == pytest.approx(config.fall_speed, rel=0.1)
    assert result.profile.hit_line_y == pytest.approx(config.keyboard_top, abs=6)
    assert result.profile.has_hand_colors


def test_all_notes_found_without_false_positives(clean_video: Path, melody: list[NoteEvent]):
    """Главный тест этапа: точность и полнота обязаны быть единичными."""
    result = analyze_file(clean_video)
    score = evaluate(melody, result.notes)

    print("\n" + score.format_report("синтетический ролик, чистые условия"))

    assert score.recall == pytest.approx(1.0)
    assert score.precision == pytest.approx(1.0)
    assert score.f1 == pytest.approx(1.0)


def test_onset_precision_beats_frame_step(clean_video: Path, melody: list[NoteEvent]):
    """Ошибка начала должна укладываться в стандартный допуск 50 мс."""
    result = analyze_file(clean_video)
    score = evaluate(melody, result.notes)

    assert score.onset_error_avg < 0.05
    assert score.onset_error_max < 0.05


def test_hands_are_separated(clean_video: Path, melody: list[NoteEvent]):
    """Партии рук должны быть разделены.

    Без этого нотный лист не разложить на два стана.
    """
    result = analyze_file(clean_video)
    score = evaluate(melody, result.notes)

    assert score.hand_known == len(melody)
    assert score.hand_accuracy == pytest.approx(1.0)


def test_survives_compression_noise(tmp_path: Path, melody: list[NoteEvent]):
    """Плохое качество записи — первая из заявленных проблем проекта."""
    video = render_visualizer_video(
        melody,
        tmp_path / "noisy.mp4",
        SynthConfig(width=854, height=480, noise_sigma=6.0, glow=True),
    )

    score = evaluate(melody, analyze_file(video).notes)
    print("\n" + score.format_report("шум сжатия, 480p, свечение"))

    assert score.f1 > 0.9


def test_survives_dark_colour_scheme(tmp_path: Path, melody: list[NoteEvent]):
    """Тёмная тема встречается у половины визуализаторов."""
    video = render_visualizer_video(
        melody, tmp_path / "dark.mp4", SynthConfig(colors=ColorScheme.dark())
    )

    score = evaluate(melody, analyze_file(video).notes)
    print("\n" + score.format_report("тёмная тема"))

    assert score.f1 > 0.9


def test_monochrome_scheme_loses_hands_but_keeps_notes(tmp_path: Path, melody: list[NoteEvent]):
    """Одноцветная схема: руки неразличимы, но ноты обязаны найтись.

    Признать, что цвет не различает руки, — честный результат,
    а не ошибка.
    """
    video = render_visualizer_video(
        melody, tmp_path / "mono.mp4", SynthConfig(colors=ColorScheme.monochrome())
    )

    result = analyze_file(video)
    score = evaluate(melody, result.notes)
    print("\n" + score.format_report("одноцветная схема"))

    assert score.f1 > 0.9
    assert not result.profile.has_hand_colors


def test_cropped_keyboard_keeps_absolute_pitch(tmp_path: Path):
    """Обрезанная клавиатура — вторая из заявленных проблем.

    Узкий диапазон не должен съезжать по высоте: аналоги здесь требуют
    ручной калибровки.
    """
    notes = [
        NoteEvent(
            pitch=pitch, onset=0.6 + i * 0.5, offset=1.0 + i * 0.5, velocity=90, hand=Hand.RIGHT
        )
        for i, pitch in enumerate((55, 57, 59, 60))
    ]
    video = render_visualizer_video(
        notes, tmp_path / "cropped.mp4", SynthConfig(lowest_pitch=53, highest_pitch=72)
    )

    result = analyze_file(video)

    assert result.layout.lowest_pitch == 53
    assert result.layout.highest_pitch == 72
    assert evaluate(notes, result.notes).recall == pytest.approx(1.0)


def test_watermark_does_not_create_notes(tmp_path: Path, melody: list[NoteEvent]):
    """Логотипы и подписи поверх кадра — обычное дело для роликов."""
    video = render_visualizer_video(
        melody, tmp_path / "watermark.mp4", SynthConfig(watermark="PIANO TUTORIAL")
    )

    score = evaluate(melody, analyze_file(video).notes)

    assert score.precision > 0.9


def test_result_reports_backend_and_counts(clean_video: Path):
    result = analyze_file(clean_video)

    assert result.frames_processed > 0
    assert result.elapsed_sec > 0
    assert result.notes_from_keys > 0
    assert result.notes_from_blocks > 0


def test_hands_over_keyboard_do_not_break_notes(tmp_path: Path, melody: list[NoteEvent]):
    """Руки исполнителя поверх клавиш — как в большинстве роликов (В-05).

    Подсветка клавиш под ладонями и пальцами ломается: тени и цвет кожи
    дают ложные нажатия. Ноты берутся с блоков над клавиатурой, где рук нет.
    """
    video = render_visualizer_video(melody, tmp_path / "hands.mp4", SynthConfig(hands=True))

    result = analyze_file(video)
    score = evaluate(melody, result.notes)
    print("\n" + score.format_report("руки поверх клавиатуры"))
    print(f"подсветка клавиш дала бы {result.notes_from_keys} нот при {len(melody)} в эталоне")

    assert result.source == "blocks"
    assert score.f1 == pytest.approx(1.0)
    assert score.onset_error_max < 0.05
    assert score.hand_accuracy == pytest.approx(1.0)


def test_without_blocks_falls_back_to_keys(tmp_path: Path, melody: list[NoteEvent]):
    """Визуализатор только с подсветкой клавиш: датчик выключается, ноты с клавиш."""
    video = render_visualizer_video(melody, tmp_path / "keys.mp4", SynthConfig(blocks=False))

    result = analyze_file(video)
    score = evaluate(melody, result.notes)

    assert result.source == "keys"
    assert result.sensor is not None and not result.sensor.enabled
    assert any("подсветки клавиш" in w for w in result.warnings)
    assert score.f1 > 0.9


def test_intro_is_skipped_automatically(tmp_path: Path, melody: list[NoteEvent]):
    """Заставка в начале ролика не должна требовать ручного ключа.

    У каждого автора она своей длины, и просить пользователя указывать её
    вручную — ровно то неудобство, которое проект и должен устранить.
    """
    intro = 2.0
    video = render_visualizer_video(
        melody, tmp_path / "intro.mp4", SynthConfig(intro_seconds=intro, outro_seconds=1.5)
    )
    # Заставка сдвигает всю музыку: эталон сдвигается вместе с ней.
    expected = [
        replace(note, onset=note.onset + intro, offset=note.offset + intro) for note in melody
    ]

    result = analyze_file(video)
    score = evaluate(expected, result.notes)
    print("\n" + score.format_report("ролик с заставкой и титрами"))

    assert score.recall == pytest.approx(1.0)
    assert score.precision > 0.9


def test_video_notes_fuse_with_audio(clean_video: Path, melody: list[NoteEvent]):
    """Ноты настоящего разбора ролика сливаются со звуком: этапы 3 и 5 вместе.

    Звук здесь — эталон со сдвигом 120 мс и педалью, плюс одна ложная нота:
    нейросеть в тесты не тянем. Проверяется стык: формат нот видео, шкалы
    времени, руки.
    """
    result = analyze_file(clean_video)
    delay = 0.12
    heard = [
        replace(
            note,
            onset=note.onset + delay,
            offset=note.offset + delay + 0.6,
            hand=Hand.UNKNOWN,
            source=Source.AUDIO,
            confidence=0.86 + 0.02 * (i % 4),
        )
        for i, note in enumerate(melody)
    ]
    ghost = NoteEvent(84, 3.0, 3.4, 30, source=Source.AUDIO, confidence=0.4)

    fused = fuse(
        result.notes,
        [*heard, ghost],
        load_config(use_user_config=False, use_env=False).fusion,
        candidates=result.unconfirmed,
        layout=result.layout,
        video_reliable=result.source == "blocks",
    )

    on_audio_clock = [replace(n, onset=n.onset + delay, offset=n.offset + delay) for n in melody]
    score = evaluate(on_audio_clock, fused.notes)
    assert score.f1 == pytest.approx(1.0)
    assert score.hand_accuracy == pytest.approx(1.0)
    assert all(note.source is Source.BOTH for note in fused.notes)
    assert fused.curve.at(3.0) == pytest.approx(delay, abs=0.02)
    assert 84 not in [note.pitch for note in fused.notes]
