"""Model files on a host (``slugai prep-model``).

Model files live in the Hugging Face cache, where mlx-lm expects them (set
``HF_HOME`` to move it).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)

def model_files(path: Path) -> dict:
    """What is present for a model directory: config, weights, tokenizer."""
    weights = sorted(path.glob("model*.safetensors"))
    return {
        "config": (path / "config.json").exists(),
        "weights": [w.name for w in weights],
        "weight_bytes": sum(w.stat().st_size for w in weights),
        "tokenizer": any((path / n).exists() for n in ("tokenizer.json", "tokenizer.model", "tokenizer_config.json")),
    }


def prep_model(model: str) -> int:
    """Download (or verify) model files so ``slugai host`` can load them without network."""
    local = Path(model).expanduser()
    if local.exists():
        info = model_files(local)
        print(f"{model}: local directory, {info['weight_bytes'] / 1e9:.2f} GB of weights, {len(info['weights'])} shard(s)")
        return 0 if info["config"] and info["weights"] else 1
    from mlx_lm.utils import _download  # missing mlx-lm: the CLI says how to install it

    print(f"staging {model} ...")
    path = _download(model)
    info = model_files(path)
    print(f"{model}: {path}")
    print(f"  {info['weight_bytes'] / 1e9:.2f} GB of weights in {len(info['weights'])} shard(s); config {'ok' if info['config'] else 'MISSING'}; tokenizer {'ok' if info['tokenizer'] else 'MISSING'}")
    return 0 if info["config"] and info["weights"] else 1


def resolve_model_path(model: str) -> Optional[Path]:
    local = Path(model).expanduser()
    if local.exists():
        return local
    try:
        from mlx_lm.utils import hf_repo_to_path

        return Path(hf_repo_to_path(model))
    except Exception:
        return None
