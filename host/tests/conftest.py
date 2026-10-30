"""Tiny random models, built on disk once per test session."""

import json
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import pytest
from mlx.utils import tree_flatten, tree_map
from mlx_lm.models import qwen3, qwen3_next

QWEN3 = {
    "model_type": "qwen3",
    "hidden_size": 32,
    "num_hidden_layers": 6,
    "intermediate_size": 64,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": 8,
    "rms_norm_eps": 1e-6,
    "vocab_size": 64,
    "max_position_embeddings": 512,
    "rope_theta": 10000.0,
    "tie_word_embeddings": False,
}

# hybrid: linear attention with full attention every 4th layer, and MoE
QWEN3_NEXT = {
    **QWEN3,
    "model_type": "qwen3_next",
    "num_hidden_layers": 8,
    "full_attention_interval": 4,
    "linear_num_value_heads": 4,
    "linear_num_key_heads": 2,
    "linear_key_head_dim": 8,
    "linear_value_head_dim": 8,
    "linear_conv_kernel_dim": 4,
    "num_experts": 4,
    "num_experts_per_tok": 2,
    "decoder_sparse_step": 1,
    "shared_expert_intermediate_size": 32,
    "mlp_only_layers": [],
    "moe_intermediate_size": 32,
    "partial_rotary_factor": 0.25,
    "norm_topk_prob": True,
}

MODELS = {  # name -> (architecture, config, weights' dtype)
    "qwen3": (qwen3, QWEN3, mx.float16),
    "qwen3-bf16": (qwen3, QWEN3, mx.bfloat16),
    "qwen3-4bit": (qwen3, {**QWEN3, "quantization": {"group_size": 32, "bits": 4}}, mx.float16),
    "qwen3-tied": (qwen3, {**QWEN3, "tie_word_embeddings": True}, mx.float16),
    "qwen3-next": (qwen3_next, QWEN3_NEXT, mx.float32),  # MoE on the CPU build needs float32
}


def build(path: Path, arch, config: dict, dtype) -> Path:
    mx.random.seed(0)
    model = arch.Model(arch.ModelArgs.from_dict(config))
    # small random weights, so greedy decoding isn't degenerate
    model.update(tree_map(lambda v: (mx.random.normal(v.shape) / v.shape[-1] ** 0.5 if v.ndim > 1 else mx.ones(v.shape)).astype(dtype), model.parameters()))
    if "quantization" in config:
        nn.quantize(model, **config["quantization"])
    path.mkdir()
    mx.save_safetensors(str(path / "model.safetensors"), dict(tree_flatten(model.parameters())))
    (path / "config.json").write_text(json.dumps(config))
    return path


@pytest.fixture(scope="session", params=MODELS)
def tiny_model(request, tmp_path_factory) -> str:
    return str(build(tmp_path_factory.mktemp("models") / request.param, *MODELS[request.param]))
