import subprocess, sys

from oldslop.slugai.cli import build_parser


def test_parser_subcommands():
    p = build_parser()
    a = p.parse_args(["run", "--model", "m", "--prompts", "p.txt"])
    assert a.cmd == "run" and a.min_hosts == 3
    a = p.parse_args(["run", "--model", "m"])
    assert a.prompts is None  # the built-in sample prompts
    a = p.parse_args(["host", "--port", "47222"])
    assert a.port == 47222
    a = p.parse_args(["bench", "--model", "m", "--batch", "8"])
    assert a.cmd == "bench" and a.batch == 8


def test_help_runs():
    out = subprocess.run([sys.executable, "-m", "slugai", "--help"], capture_output=True, text=True)
    assert out.returncode == 0 and "host" in out.stdout
