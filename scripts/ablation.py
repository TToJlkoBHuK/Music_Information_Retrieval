"""Проверка NF-10: слияние каналов не хуже лучшего одиночного канала.

```
python scripts/ablation.py
python scripts/ablation.py --seeds 10 --bars 80
```

У настоящих роликов эталона нет, поэтому сравнение идёт на модели ошибок
каналов (`tests/fixtures/channels.py`): эталон задан, «ноты видео» и «ноты
звука» получены из него порчей — теми ошибками, которые найдены при разборе
трёх настоящих роликов. Для каждого сценария печатается F1 видео, звука и
слияния против эталона (правила MIREX, допуск начала 50 мс).

Это проверка логики слияния, а не замер качества на живых данных: модель
говорит, что слияние верно обращается с ошибками известного вида, и ничего
не говорит о том, как часто они встречаются. Таблица идёт в экспериментальную
главу именно с этой оговоркой.

Код возврата 1, если хоть в одном прогоне слияние уступило одиночному каналу.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mir.config import load_config
from mir.eval import evaluate
from mir.fusion import fuse
from tests.fixtures.channels import SCENARIOS, piece, simulate, video_on_audio_clock


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Слияние против одиночных каналов (NF-10)")
    parser.add_argument("--seeds", type=int, default=5, help="сколько прогонов на сценарий")
    parser.add_argument("--bars", type=int, default=60, help="длина пьесы в тактах по 2 с")
    args = parser.parse_args(argv)

    config = load_config(use_user_config=False, use_env=False).fusion
    width = max(len(name) for name in SCENARIOS)
    print(f"{'сценарий':{width}s}  видео   звук  слияние  худший выигрыш  ведущий")
    failed = False
    for name, model in SCENARIOS.items():
        rows: list[tuple[float, float, float]] = []
        lead = ""
        for seed in range(args.seeds):
            truth = piece(seed, bars=args.bars)
            channels = simulate(truth, model, seed)
            result = fuse(channels.video, channels.audio, config, candidates=channels.candidates)
            rows.append(
                (
                    evaluate(truth, video_on_audio_clock(channels, model)).f1,
                    evaluate(truth, channels.audio).f1,
                    evaluate(truth, result.notes).f1,
                )
            )
            lead = "видео" if result.lead.name == "VIDEO" else "звук"
        mean = [sum(column) / len(rows) for column in zip(*rows, strict=True)]
        worst = min(fused - max(video, audio) for video, audio, fused in rows)
        failed = failed or worst < -1e-9
        print(
            f"{name:{width}s}  {mean[0]:.3f}  {mean[1]:.3f}   {mean[2]:.3f}      "
            f"{worst:+.3f}       {lead}"
        )
    print()
    print("NF-10 нарушено" if failed else "NF-10 выполнено: слияние нигде не хуже лучшего канала")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
