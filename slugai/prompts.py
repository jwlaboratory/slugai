"""Prompt files: ``.jsonl`` (``{"prompt": ...}`` per line) or anything else as text, one prompt per line.

Without a file, jobs use the built-in sample in ``slugai/data/sample_prompts.jsonl``
(300 short explain-this prompts), handy for trying the ring or benchmarking.
"""

from __future__ import annotations

import json
import logging
from importlib.resources import files
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)

SAMPLE_PROMPTS = "sample_prompts.jsonl"


def load_prompts(path: Optional[str] = None, limit: Optional[int] = None) -> list[str]:
    if path is None:
        text, jsonl = files("slugai").joinpath("data", SAMPLE_PROMPTS).read_text(encoding="utf-8"), True
    else:
        text, jsonl = Path(path).read_text(encoding="utf-8"), path.lower().endswith(".jsonl")
    lines = [line for line in text.splitlines() if line.strip()]
    prompts = [str(json.loads(line)["prompt"]) for line in lines] if jsonl else lines
    total = len(prompts)
    if limit is not None:
        prompts = prompts[:limit]
    if path is None:
        log.info("no --prompts given; using %d of the %d built-in sample prompts", len(prompts), total)
    return prompts
