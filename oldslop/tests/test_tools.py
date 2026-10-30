"""bench, doctor and prep-model from the command line, and the built-in prompts."""

import json
import subprocess
import sys

from oldslop.slugai.cli import main as cli_main

PROMPTS = ["the quick brown fox", "hi", "a much longer prompt than the others, indeed", "zebra", "x"]


def test_bench_cli(tiny, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    p = tmp_path / "p.txt"
    p.write_text("\n".join(PROMPTS))
    rc = cli_main(["bench", "--model", str(tiny), "--prompts", str(p), "--max-tokens", "5", "--job", "b1"])
    assert rc == 0
    stats = json.loads((tmp_path / "outputs" / "b1-report.json").read_text())
    assert stats["job"] == "b1" and stats["prompts"] == len(PROMPTS) and stats["tok_s"] > 0
    lines = (tmp_path / "outputs" / "b1-completions.jsonl").read_text().splitlines()
    assert len(lines) == len(PROMPTS)


def test_doctor_runs():
    out = subprocess.run([sys.executable, "-m", "slugai", "doctor"], capture_output=True, text=True)
    assert "python" in out.stdout and ("all good" in out.stdout or "problems found" in out.stdout)


def test_prep_model_local_dir(tmp_path, capsys):
    (tmp_path / "config.json").write_text("{}")
    (tmp_path / "model.safetensors").write_bytes(b"\0" * 10)
    rc = cli_main(["prep-model", str(tmp_path)])
    assert rc == 0
    assert "1 shard" in capsys.readouterr().out


def test_builtin_sample_prompts(tiny, tmp_path, monkeypatch):
    from oldslop.slugai.prompts import load_prompts

    monkeypatch.chdir(tmp_path)
    sample = load_prompts()
    assert len(sample) == 300 and all(isinstance(p, str) and p for p in sample)
    assert load_prompts(limit=3) == sample[:3]
    # no --prompts: the sample is used
    rc = cli_main(["bench", "--model", str(tiny), "--max-tokens", "3", "--limit", "4", "--job", "b2"])
    assert rc == 0 and json.loads((tmp_path / "outputs" / "b2-report.json").read_text())["prompts"] == 4
