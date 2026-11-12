"""NF-10: слияние каналов не хуже лучшего одиночного канала.

Эталона у настоящих роликов нет, поэтому проверка идёт на модели ошибок
каналов (`tests/fixtures/channels.py`): эталон известен, каналы испорчены
ошибками тех видов, что найдены на настоящих роликах. Тот же прогон
печатает `scripts/ablation.py` — это таблица для экспериментальной главы.
"""

from __future__ import annotations

import pytest

from mir.common.enums import Source
from mir.config import FusionConfig, load_config
from mir.eval import evaluate
from mir.fusion import fuse
from tests.fixtures.channels import SCENARIOS, ChannelModel, piece, simulate, video_on_audio_clock

SEEDS = (0, 1, 2)


@pytest.fixture(scope="module")
def config() -> FusionConfig:
    return load_config(use_user_config=False, use_env=False).fusion


def scores(model: ChannelModel, seed: int, config: FusionConfig) -> tuple[float, float, float]:
    """F1 видео, звука и слияния против эталона."""
    truth = piece(seed, bars=40)
    channels = simulate(truth, model, seed)
    fused = fuse(channels.video, channels.audio, config, candidates=channels.candidates)
    video = evaluate(truth, video_on_audio_clock(channels, model)).f1
    return video, evaluate(truth, channels.audio).f1, evaluate(truth, fused.notes).f1


@pytest.mark.parametrize("name", list(SCENARIOS))
@pytest.mark.parametrize("seed", SEEDS)
def test_fusion_is_not_worse_than_best_channel(name: str, seed: int, config: FusionConfig):
    video, audio, fused = scores(SCENARIOS[name], seed, config)
    assert fused >= max(video, audio) - 1e-9, f"{name}: видео {video:.3f}, звук {audio:.3f}"


@pytest.mark.parametrize("seed", SEEDS)
def test_fusion_beats_both_channels_on_typical_video(seed: int, config: FusionConfig):
    """На обычном ролике слияние не просто «не хуже», а заметно лучше обоих."""
    video, audio, fused = scores(ChannelModel(), seed, config)
    assert fused > max(video, audio) + 0.01
    assert fused > 0.97


def test_broken_video_does_not_spoil_audio(config: FusionConfig):
    """Видео с мусором вместо нот: итог — звук, а не смесь."""
    model = SCENARIOS["видео сломано (подсветка под руками)"]
    truth = piece(0, bars=40)
    channels = simulate(truth, model, 0)

    fused = fuse(channels.video, channels.audio, config, candidates=channels.candidates)

    assert fused.lead is Source.AUDIO
    assert evaluate(truth, fused.notes).precision > 0.95


def test_hands_survive_fusion(config: FusionConfig):
    """Рука берётся из видео: у подтверждённых нот она не теряется."""
    truth = piece(0, bars=40)
    channels = simulate(truth, ChannelModel(), 0)

    fused = fuse(channels.video, channels.audio, config, candidates=channels.candidates)
    score = evaluate(truth, fused.notes)

    assert score.hand_known > 0.9 * len(truth)
    assert score.hand_accuracy == pytest.approx(1.0)
