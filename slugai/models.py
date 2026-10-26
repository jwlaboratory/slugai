"""Model files on a host (``slugai prep-model``).

Model files live in the Hugging Face cache, where mlx-lm expects them.
``--models-dir`` points that cache somewhere else (it sets ``HF_HUB_CACHE``),
so every slugai command that takes ``--models-dir`` must apply it before MLX or
mlx-lm are imported.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)

DEFAULT_MODELS_DIR = Path.home() / ".slugai" / "models"


def apply_models_dir(models_dir: Optional[str]) -> Optional[Path]:
    if not models_dir:
        return None
    p = Path(models_dir).expanduser()
    p.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HUB_CACHE"] = str(p)
    os.environ.setdefault("HF_HOME", str(p.parent))
    return p


def dir_size(path: Path) -> int:
    total = 0
    for f in path.rglob("*"):
        try:
            if f.is_file() or f.is_symlink():
                total += f.stat().st_size
        except OSError:
            pass
    return total


def model_files(path: Path) -> dict:
    """What is present for a model directory: config, weights, tokenizer."""
    weights = sorted(path.glob("model*.safetensors"))
    return {
        "path": str(path),
        "config": (path / "config.json").exists(),
        "weights": [w.name for w in weights],
        "weight_bytes": sum(w.stat().st_size for w in weights),
        "index": (path / "model.safetensors.index.json").exists(),
        "tokenizer": any((path / n).exists() for n in ("tokenizer.json", "tokenizer.model", "tokenizer_config.json")),
    }


def prep_model(model: str, models_dir: Optional[str] = None) -> int:
    """Download (or verify) model files so ``slugai host`` can load them without network."""
    apply_models_dir(models_dir)
    local = Path(model).expanduser()
    if local.exists():
        info = model_files(local)
        print(f"{model}: local directory, {info['weight_bytes'] / 1e9:.2f} GB of weights, {len(info['weights'])} shard(s)")
        return 0 if info["config"] and info["weights"] else 1
    try:
        from mlx_lm.utils import _download
    except ImportError:
        print("mlx-lm is not installed; install slugai[mlx] on this Mac first")
        return 1
    print(f"staging {model} ...")
    path = _download(model)
    info = model_files(path)
    print(f"{model}: {path}")
    print(f"  {info['weight_bytes'] / 1e9:.2f} GB of weights in {len(info['weights'])} shard(s); config {'ok' if info['config'] else 'MISSING'}; tokenizer {'ok' if info['tokenizer'] else 'MISSING'}")
    if not info["config"] or not info["weights"]:
        return 1
    return 0


def resolve_model_path(model: str) -> Optional[Path]:
    local = Path(model).expanduser()
    if local.exists():
        return local
    try:
        from mlx_lm.utils import hf_repo_to_path

        return Path(hf_repo_to_path(model))
    except Exception:
        return None
