"""Tiny real models built on disk once per session (tests that use them need MLX)."""

import pytest


@pytest.fixture(scope="session")
def tiny(tmp_path_factory):
    """A 6-layer fp16 Qwen3 with a character tokenizer."""
    pytest.importorskip("mlx.core")
    from oldslop.tests.tiny_model import make_tiny_qwen3

    return make_tiny_qwen3(tmp_path_factory.mktemp("tiny") / "qwen3", num_layers=6)


@pytest.fixture(scope="session", params=["fp16", "q4", "next"])
def tiny_model(request, tmp_path_factory):
    """Each architecture the backend must handle: Qwen3 fp16, 4-bit, and hybrid Qwen3-Next."""
    pytest.importorskip("mlx.core")
    from oldslop.tests.tiny_model import make_tiny_qwen3, make_tiny_qwen3_next

    path = tmp_path_factory.mktemp("tiny") / request.param
    if request.param == "next":
        # hybrid linear-attention + MoE model, like the Qwen3-Next target
        return make_tiny_qwen3_next(path, num_layers=8)
    return make_tiny_qwen3(path, num_layers=6, quantize=request.param == "q4")
