"""Stage execution backends.

* :mod:`.base`: the :class:`Stage` / :class:`Backend` interface;
* :mod:`.fake`: a numpy toy transformer, deterministic, for tests and
  ``slugai host --backend fake``;
* :mod:`.mlx`: real models on Apple silicon via mlx-lm (imported on demand).
"""

from __future__ import annotations

import os

from .base import Backend, ModelInfo, Stage, Tokenizer
from .fake import FakeBackend, FakeTokenizer

__all__ = ["Backend", "FakeBackend", "FakeTokenizer", "ModelInfo", "Stage", "Tokenizer", "get_backend"]


def get_backend(name: str = "auto") -> Backend:
    if name == "fake":
        return FakeBackend()
    if name in ("auto", "mlx"):
        try:
            from .mlx import MLXBackend
        except ImportError as e:
            if name == "mlx":
                raise
            if os.environ.get("SLUGAI_BACKEND") == "fake":
                return FakeBackend()
            raise RuntimeError(
                "MLX is not available on this machine; install slugai[mlx] on Apple silicon or use --backend fake"
            ) from e
        return MLXBackend()
    raise ValueError(f"unknown backend {name}")
