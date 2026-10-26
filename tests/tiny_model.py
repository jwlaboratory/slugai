"""Build a tiny random Qwen3-architecture model on disk for MLX backend tests."""

from __future__ import annotations

import json
from pathlib import Path


def make_tiny_qwen3(path: Path, num_layers: int = 6, hidden: int = 32, vocab: int = 64, quantize: bool = False) -> Path:
    import mlx.core as mx
    import mlx.nn as nn
    from mlx.utils import tree_flatten
    from mlx_lm.models import qwen3
    from tokenizers import Tokenizer, models, pre_tokenizers, decoders

    path.mkdir(parents=True, exist_ok=True)
    config = {
        "model_type": "qwen3",
        "architectures": ["Qwen3ForCausalLM"],
        "hidden_size": hidden,
        "num_hidden_layers": num_layers,
        "intermediate_size": hidden * 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": hidden // 4,
        "rms_norm_eps": 1e-6,
        "vocab_size": vocab,
        "max_position_embeddings": 512,
        "rope_theta": 10000.0,
        "tie_word_embeddings": False,
        "eos_token_id": 2,
        "bos_token_id": 1,
        "pad_token_id": 0,
        "torch_dtype": "float16",
    }
    args = qwen3.ModelArgs.from_dict(config)
    mx.random.seed(1234)
    model = qwen3.Model(args)
    # re-init with a controlled scale so greedy decoding is not degenerate
    params = {}
    for k, v in tree_flatten(model.parameters()):
        if v.ndim == 2:
            params[k] = (mx.random.normal(v.shape) * (1.0 / v.shape[-1] ** 0.5)).astype(mx.float16)
        else:
            params[k] = mx.ones(v.shape, dtype=mx.float16)
    model.load_weights(list(params.items()))
    if quantize:
        nn.quantize(model, group_size=32, bits=4)
        config["quantization"] = {"group_size": 32, "bits": 4}
        config["quantization_config"] = {"group_size": 32, "bits": 4}
    weights = dict(tree_flatten(model.parameters()))
    mx.save_safetensors(str(path / "model.safetensors"), weights)
    (path / "config.json").write_text(json.dumps(config, indent=1))

    # byte-ish tokenizer: one token per character of a small alphabet
    alphabet = " abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.,;:!?'\"-()"
    vocab_map = {"<pad>": 0, "<s>": 1, "</s>": 2, "[UNK]": 3}
    for ch in alphabet:
        if len(vocab_map) >= vocab:
            break
        vocab_map[ch] = len(vocab_map)
    tok = Tokenizer(models.WordLevel(vocab_map, unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.Split("", "isolated")
    tok.decoder = decoders.Fuse()
    tok.save(str(path / "tokenizer.json"))
    (path / "tokenizer_config.json").write_text(
        json.dumps(
            {
                "tokenizer_class": "PreTrainedTokenizerFast",
                "bos_token": "<s>",
                "eos_token": "</s>",
                "pad_token": "<pad>",
                "unk_token": "[UNK]",
                "model_max_length": 512,
                "clean_up_tokenization_spaces": False,
            }
        )
    )
    return path


def make_tiny_qwen3_next(path: Path, num_layers: int = 8, hidden: int = 32, vocab: int = 64) -> Path:
    """Tiny hybrid model (linear attention + full attention every 4th layer, MoE).

    Saved in float32 because the CPU build's gather-matmul (MoE) only supports float32.
    """
    import mlx.core as mx
    from mlx.utils import tree_flatten
    from mlx_lm.models import qwen3_next

    path.mkdir(parents=True, exist_ok=True)
    config = {
        "model_type": "qwen3_next",
        "architectures": ["Qwen3NextForCausalLM"],
        "hidden_size": hidden,
        "num_hidden_layers": num_layers,
        "intermediate_size": hidden * 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": hidden // 4,
        "linear_num_value_heads": 4,
        "linear_num_key_heads": 2,
        "linear_key_head_dim": 8,
        "linear_value_head_dim": 8,
        "linear_conv_kernel_dim": 4,
        "num_experts": 4,
        "num_experts_per_tok": 2,
        "decoder_sparse_step": 1,
        "shared_expert_intermediate_size": hidden,
        "mlp_only_layers": [],
        "moe_intermediate_size": hidden,
        "rms_norm_eps": 1e-6,
        "vocab_size": vocab,
        "rope_theta": 10000.0,
        "partial_rotary_factor": 0.25,
        "max_position_embeddings": 512,
        "norm_topk_prob": True,
        "tie_word_embeddings": False,
        "full_attention_interval": 4,
        "eos_token_id": 2,
        "bos_token_id": 1,
        "pad_token_id": 0,
    }
    args = qwen3_next.ModelArgs.from_dict(config)
    mx.random.seed(99)
    model = qwen3_next.Model(args)
    params = {}
    for k, v in tree_flatten(model.parameters()):
        if v.ndim >= 2:
            params[k] = (mx.random.normal(v.shape) * (1.0 / v.shape[-1] ** 0.5)).astype(mx.float32)
        else:
            params[k] = mx.ones(v.shape, dtype=mx.float32)
    model.load_weights(list(params.items()))
    mx.save_safetensors(str(path / "model.safetensors"), dict(tree_flatten(model.parameters())))
    (path / "config.json").write_text(json.dumps(config, indent=1))
    _write_char_tokenizer(path, vocab)
    return path


def _write_char_tokenizer(path: Path, vocab: int) -> None:
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers

    alphabet = " abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.,;:!?'\"-()"
    vocab_map = {"<pad>": 0, "<s>": 1, "</s>": 2, "[UNK]": 3}
    for ch in alphabet:
        if len(vocab_map) >= vocab:
            break
        vocab_map[ch] = len(vocab_map)
    tok = Tokenizer(models.WordLevel(vocab_map, unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.Split("", "isolated")
    tok.decoder = decoders.Fuse()
    tok.save(str(path / "tokenizer.json"))
    (path / "tokenizer_config.json").write_text(
        json.dumps({"tokenizer_class": "PreTrainedTokenizerFast", "bos_token": "<s>", "eos_token": "</s>", "pad_token": "<pad>", "unk_token": "[UNK]", "model_max_length": 512, "clean_up_tokenization_spaces": False})
    )
