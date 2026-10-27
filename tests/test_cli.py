import subprocess, sys

from slugai.cli import build_parser


def test_parser_subcommands():
    p = build_parser()
    a = p.parse_args(["run", "--model", "m", "--prompts", "p.txt"])
    assert a.cmd == "run" and a.min_hosts == 3
    a = p.parse_args(["host", "--backend", "fake"])
    assert a.backend == "fake"
    a = p.parse_args(["bench", "single", "--model", "m", "--prompts", "p"])
    assert a.baseline == "single"


def test_help_runs():
    out = subprocess.run([sys.executable, "-m", "slugai", "--help"], capture_output=True, text=True)
    assert out.returncode == 0 and "host" in out.stdout
