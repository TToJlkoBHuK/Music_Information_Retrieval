"""Команда `run fuse`: каналы из CSV → итог, MIDI, таблица решений, валик."""

from __future__ import annotations

import csv
import importlib.util
from pathlib import Path
from types import ModuleType

import pytest

from mir.eval import candidates_path, read_notes_csv, write_notes_csv
from tests.fixtures.channels import ChannelModel, piece, simulate

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "fusion_cli.py"


@pytest.fixture(scope="module")
def cli() -> ModuleType:
    spec = importlib.util.spec_from_file_location("fusion_cli", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture()
def channel_files(tmp_path: Path) -> tuple[Path, Path, int]:
    truth = piece(0, bars=12)
    channels = simulate(truth, ChannelModel(), 0)
    video, audio = tmp_path / "video.csv", tmp_path / "audio.csv"
    write_notes_csv(channels.video, video)
    write_notes_csv(channels.candidates, candidates_path(video))
    write_notes_csv(channels.audio, audio)
    return video, audio, len(truth)


def test_fuse_from_csv_writes_all_outputs(cli, channel_files, tmp_path: Path, capsys):
    video, audio, expected = channel_files
    out = tmp_path / "out"

    code = cli.main(
        [
            "fuse",
            "--video-csv", str(video),
            "--audio-csv", str(audio),
            "--csv", str(out / "fused.csv"),
            "--midi", str(out / "fused.mid"),
            "--decisions", str(out / "decisions.csv"),
            "--roll", str(out / "fused.png"),
        ]
    )  # fmt: skip

    assert code == 0
    fused = read_notes_csv(out / "fused.csv")
    assert abs(len(fused) - expected) <= 0.05 * expected
    assert (out / "fused.mid").read_bytes()[:4] == b"MThd"
    assert (out / "fused.png").stat().st_size > 1000
    with (out / "decisions.csv").open(encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert sum(row["verdict"] == "+" for row in rows) == len(fused)
    assert {"confirmed", "video_only"} <= {row["reason"] for row in rows}

    report = capsys.readouterr().out
    assert "ведущий канал   : видео" in report
    assert "оба канала: высота и начало совпали" in report
    assert "блоков-кандидатов" in report  # файл кандидатов найден сам


def test_missing_channel_without_source_is_explained(cli, channel_files, tmp_path: Path, capsys):
    """Нет файла канала и не из чего его разобрать — понятная ошибка, без traceback."""
    video, _, _ = channel_files

    code = cli.main(["fuse", "--video-csv", str(video), "--audio-csv", str(tmp_path / "no.csv")])

    assert code == 1
    assert "Нет файла с нотами звука" in capsys.readouterr().out
