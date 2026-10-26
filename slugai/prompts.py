"""Prompt files: ``.txt`` (one prompt per line) or ``.jsonl`` (``{"prompt": ...}`` per line)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional


def load_prompts(path: str, limit: Optional[int] = None) -> list[str]:
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    prompts: list[str] = []
    if p.suffix.lower() in (".jsonl", ".ndjson"):
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
                    raise ValueError(f"{path}: no prompt/text/input key in {list(obj)}")
    elif p.suffix.lower() == ".json":
        obj = json.loads(text)
        if isinstance(obj, dict):
            obj = obj.get("prompts", obj)
        prompts = [x if isinstance(x, str) else str(x.get("prompt", x)) for x in obj]
    else:
        prompts = [line for line in text.splitlines() if line.strip()]
    if limit is not None:
        prompts = prompts[:limit]
    return prompts
