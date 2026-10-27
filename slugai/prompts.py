"""Prompt files: ``.txt`` (one prompt per line) or ``.jsonl`` (``{"prompt": ...}`` per line).

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
        name, suffix = SAMPLE_PROMPTS, ".jsonl"
        text = files("slugai").joinpath("data", SAMPLE_PROMPTS).read_text(encoding="utf-8")
    else:
        p = Path(path)
        name, suffix = path, p.suffix.lower()
        text = p.read_text(encoding="utf-8")
    prompts: list[str] = []
    if suffix in (".jsonl", ".ndjson"):
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if isinstance(obj, str):
                prompts.append(obj)
            elif isinstance(obj, dict):
                for key in ("prompt", "text", "input", "question"):
                    if key in obj:
                        prompts.append(str(obj[key]))
                        break
                else:
                    raise ValueError(f"{name}: no prompt/text/input key in {list(obj)}")
    elif suffix == ".json":
        obj = json.loads(text)
        if isinstance(obj, dict):
            obj = obj.get("prompts", obj)
        prompts = [x if isinstance(x, str) else str(x.get("prompt", x)) for x in obj]
    else:
        prompts = [line for line in text.splitlines() if line.strip()]
    total = len(prompts)
    if limit is not None:
        prompts = prompts[:limit]
    if path is None:
        log.info("no --prompts given; using %d of the %d built-in sample prompts", len(prompts), total)
    return prompts
