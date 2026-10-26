"""bench baselines, doctor, profile and prep-model CLIs (fake backend)."""

import json
import subprocess
import sys

import pytest

from slugai.backend import FakeBackend
from slugai.bench import run_single, run_streaming
from slugai.cli import build_parser, main as cli_main
from tests.test_fake_backend import reference_generate

PROMPTS = ["the quick brown fox", "hi", "a much longer prompt than the others, indeed", "zebra", "x"]


@pytest.mark.parametrize("side", ["left", "right"])
def test_bench_single_matches_reference(side):
    be = FakeBackend(slow_ms=0, padding_side=side)
    ref = reference_generate(FakeBackend(slow_ms=0), "fake-L6", PROMPTS, 9)
    outs, stats = run_single(be, "fake-L6", PROMPTS, max_tokens=9, batch=2)
    assert outs == ref
    assert stats["tokens_generated"] == sum(len(g) for g in ref) and stats["tok_s"] > 0


def test_bench_streaming_matches_reference():
    be = FakeBackend(slow_ms=0)
    ref = reference_generate(be, "fake-L6", PROMPTS, 7)
    outs, stats = run_streaming(be, "fake-L6", PROMPTS, max_tokens=7, batch=3)
    assert outs == ref
    assert stats["layer_loads"] > 6


def test_bench_cli(tmp_path):
    p = tmp_path / "p.txt"
    p.write_text("\n".join(PROMPTS))
    rep = tmp_path / "rep.json"
    rc = cli_main(["bench", "single", "--model", "fake-L4", "--prompts", str(p), "--backend", "fake", "--max-tokens", "5", "--report", str(rep)])
    assert rc == 0
    stats = json.loads(rep.read_text())
    assert stats["prompts"] == len(PROMPTS) and stats["baseline"] == "single"
    lines = (tmp_path / "rep.jsonl").read_text().splitlines()
    assert len(lines) == len(PROMPTS)


def test_profile_cli(capsys):
    rc = cli_main(["profile", "--model", "fake-L4", "--backend", "fake", "--json"])
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert out["num_layers"] == 4 and out["layer_time"]


def test_doctor_runs():
    out = subprocess.run([sys.executable, "-m", "slugai.cli", "doctor"], capture_output=True, text=True)
    assert "python" in out.stdout and ("all good" in out.stdout or "problems found" in out.stdout)


def test_prep_model_local_dir(tmp_path, capsys):
    (tmp_path / "config.json").write_text("{}")
    (tmp_path / "model.safetensors").write_bytes(b"\0" * 10)
    rc = cli_main(["prep-model", str(tmp_path)])
    assert rc == 0
    assert "1 shard" in capsys.readouterr().out
