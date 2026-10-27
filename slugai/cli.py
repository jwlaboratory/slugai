"""Command-line entry point: ``slugai``.

    slugai host                      start on each MacBook; waits for a server
    slugai run --model M --prompts P start a job on the server
    slugai bench <baseline>          run a baseline on the same prompts
    slugai prep-model M              stage model files on this Mac
    slugai profile                   profile this Mac
    slugai doctor                    check this Mac's setup
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys

from . import __version__

DEFAULT_PORT = 47111
SERVICE_TYPE = "_slugai._tcp.local."


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="slugai", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--version", action="version", version=f"slugai {__version__}")
    p.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    sub = p.add_subparsers(dest="cmd", required=True)

    h = sub.add_parser("host", help="run the host daemon on this MacBook")
    h.add_argument("--port", type=int, default=DEFAULT_PORT)
    h.add_argument("--name", default=None, help="host name advertised to the server (default: hostname)")
    h.add_argument("--models-dir", default=None, help="where prep-model staged model files (default ~/.slugai/models)")
    h.add_argument("--backend", default="auto", choices=["auto", "mlx", "fake"], help="fake = numpy toy model for tests")
    h.add_argument("--no-caffeinate", action="store_true", help="do not block sleep")
    h.add_argument("--no-advertise", action="store_true", help="do not advertise over Bonjour")

    r = sub.add_parser("run", help="start a job (server side)")
    r.add_argument("--model", required=True, help="model id, e.g. mlx-community/Qwen3-8B-4bit")
    r.add_argument("--prompts", required=True, help="prompts file: .txt (one per line) or .jsonl ({'prompt': ...})")
    r.add_argument("--job", default=None, help="job name (default: auto)")
    r.add_argument("--hosts", default=None, help="comma-separated host:port list (skips Bonjour discovery)")
    r.add_argument("--min-hosts", type=int, default=3)
    r.add_argument("--wait", type=float, default=30.0, help="seconds to wait for hosts to appear")
    r.add_argument("--max-tokens", type=int, default=256)
    r.add_argument("--micro-batch", type=int, default=None, help="force micro-batch size (default: chosen from profiles)")
    r.add_argument("--num-micro-batches", type=int, default=None, help="force number of micro-batches in flight")
    r.add_argument("--no-tui", action="store_true", help="plain log lines instead of the dashboard")
    r.add_argument("--report", default=None, help="write a JSON report here when done")
    r.add_argument("--output", default=None, help="write completions here (.jsonl); default <job>.jsonl")
    r.add_argument("--rebalance-every", type=float, default=20.0, help="seconds between rebalance checks")
    r.add_argument("--no-rebalance", action="store_true")
    r.add_argument("--limit", type=int, default=None, help="only run the first N prompts")

    b = sub.add_parser("bench", help="run a baseline on the same prompts")
    b.add_argument("baseline", choices=["single", "streaming"], help="single = one Mac, all layers in memory; streaming = FlexGen-style, reread layers from disk every step")
    b.add_argument("--model", required=True)
    b.add_argument("--prompts", required=True)
    b.add_argument("--max-tokens", type=int, default=256)
    b.add_argument("--batch", type=int, default=64)
    b.add_argument("--limit", type=int, default=None)
    b.add_argument("--backend", default="auto", choices=["auto", "mlx", "fake"])
    b.add_argument("--report", default=None)

    pm = sub.add_parser("prep-model", help="download and stage model files on this Mac")
    pm.add_argument("model")
    pm.add_argument("--models-dir", default=None)

    pr = sub.add_parser("profile", help="profile this Mac: time per layer, memory")
    pr.add_argument("--model", required=True)
    pr.add_argument("--models-dir", default=None)
    pr.add_argument("--backend", default="auto", choices=["auto", "mlx", "fake"])
    pr.add_argument("--json", action="store_true")

    d = sub.add_parser("doctor", help="check this Mac's setup")
    d.add_argument("--model", default=None)
    d.add_argument("--models-dir", default=None)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname).1s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    try:
        if args.cmd == "host":
            from .host import main as host_main
            return asyncio.run(host_main(args))
        if args.cmd == "run":
            from .server import main as server_main
            return asyncio.run(server_main(args))
        if args.cmd == "bench":
            from .bench import main as bench_main
            return bench_main(args)
        if args.cmd == "prep-model":
            from .models import prep_model
            return prep_model(args.model, args.models_dir)
        if args.cmd == "profile":
            from .profiling import profile_cli
            return profile_cli(args)
        if args.cmd == "doctor":
            from .doctor import main as doctor_main
            return doctor_main(args)
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        return 130
    return 2


if __name__ == "__main__":
    sys.exit(main())
