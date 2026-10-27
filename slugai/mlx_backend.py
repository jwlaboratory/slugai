"""MLX backend: run a contiguous range of a real model's layers on Apple silicon.

Built on mlx-lm's own model classes so every architecture it supports works
here without per-model code:

* the model is loaded *lazily* (``load_model(lazy=True)``), then
  ``model.model.layers`` is sliced to this stage's range and only the kept
  parameters are evaluated, so a stage never materialises the other stages'
  weights;
* a stage that is not first swaps the embedding for an identity so hidden
  states can be fed straight in; a stage that is not last swaps the final norm
  for an identity so hidden states come out un-normalised;
* KV caches come from mlx-lm (``make_prompt_cache`` merged into batch caches),
  which handle padding and per-sequence RoPE offsets — including hybrid
  models such as Qwen3-Next whose linear-attention layers keep state arrays
  instead of keys and values;
* hidden states cross the wire in the model's own dtype (bfloat16 as its raw
  16 bits, since numpy has no bfloat16), so the ring computes exactly what one
  machine would and large activations never overflow float16.

Prompts are **right**-padded for prefill (``padding_side = "right"``), exactly
as mlx-lm's batch generator does, and the caches are finalised into the
left-aligned layout afterwards. The host splits every prompt into
``prompt[:-1]`` (prefill) and its last token (a decode step) so that, after
prefill, all sequences in a micro-batch are aligned.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten
from mlx_lm.models.cache import BatchKVCache, make_prompt_cache
from mlx_lm.utils import _download, load_config, load_model, load_tokenizer

from .backend import Backend, ModelInfo, Stage, Tokenizer
from .placement import ModelSpec

log = logging.getLogger(__name__)


def _device_info() -> dict:
    try:
        return mx.device_info()
    except Exception:
        try:
            return mx.metal.device_info()
        except Exception:
            return {}


def _recommended_memory() -> int:
    info = _device_info()
    rec = info.get("max_recommended_working_set_size")
    if rec:
        return int(rec)
    mem = info.get("memory_size")
    return int(mem * 0.75) if mem else 16 << 30


def _active_memory() -> int:
    try:
        return int(mx.get_active_memory())
    except Exception:
        return int(mx.metal.get_active_memory())


def _nbytes(module: nn.Module) -> int:
    return sum(a.nbytes for _, a in tree_flatten(module.parameters()))


def _compute_dtype(layer: nn.Module):
    for _, a in tree_flatten(layer.parameters()):
        if a.dtype in (mx.float16, mx.bfloat16, mx.float32):
            return a.dtype
    return mx.float16


def _to_wire(h: mx.array) -> np.ndarray:
    if h.dtype == mx.bfloat16:
        return np.array(h.view(mx.uint16))
    return np.array(h)


def _from_wire(x: np.ndarray, dtype) -> mx.array:
    if x.dtype == np.uint16:  # bfloat16 bits
        return mx.array(x).view(mx.bfloat16).astype(dtype)
    return mx.array(x).astype(dtype)


def _cache_bytes(model, tokens: int) -> int:
    """Cache bytes for one sequence after ``tokens`` tokens, from a lazy forward pass.

    Nothing is evaluated (or read from disk): array shapes are known as soon as
    the graph is built.
    """
    caches = make_prompt_cache(model)
    model.model(mx.zeros((1, tokens), dtype=mx.int32), cache=caches)
    return sum(a.nbytes for c in caches for _, a in tree_flatten(c.state) if isinstance(a, mx.array))


def _cache_sizes(model, block: int) -> tuple[int, int]:
    """(KV bytes per token, fixed state bytes per sequence), each averaged over layers."""
    n = len(model.model.layers)
    one, two = _cache_bytes(model, 1), _cache_bytes(model, block + 1)
    per_token = (two - one) / block
    return int(per_token / n + 0.5), max(0, int((one - per_token * block) / n + 0.5))


class _Identity(nn.Module):
    def __call__(self, x):
        return x


class MLXTokenizer(Tokenizer):
    def __init__(self, wrapper, use_chat_template: bool = True):
        self.w = wrapper
        self.use_chat_template = use_chat_template and getattr(wrapper, "has_chat_template", False)

    def encode(self, text: str) -> list[int]:
        if self.use_chat_template:
            try:
                ids = self.w.apply_chat_template([{"role": "user", "content": text}], add_generation_prompt=True, tokenize=True)
                if ids and not isinstance(ids[0], int):
                    ids = ids["input_ids"] if isinstance(ids, dict) else list(ids)
                return [int(i) for i in ids]
            except Exception as e:  # unusual templates: fall back to raw text
                log.warning("chat template failed (%s); encoding raw text", e)
                self.use_chat_template = False
        ids = self.w.encode(text)
        return [int(i) for i in ids]

    def decode(self, ids: Sequence[int]) -> str:
        return self.w.decode([int(i) for i in ids], skip_special_tokens=True)


class MLXStage(Stage):
    padding_side = "right"

    def __init__(self, model, config: dict, layer_range: tuple[int, int], first: bool, last: bool):
        self.model = model
        self.inner = model.model
        self.config = config
        self.layer_range = layer_range
        self.first, self.last = first, last
        self.caches: dict[int, list] = {}
        self.tie = bool(getattr(model.args, "tie_word_embeddings", False))
        import inspect

        self.supports_input_embeddings = "input_embeddings" in inspect.signature(self.inner.__call__).parameters
        self.dtype = _compute_dtype(self.inner.layers[0])
        self.weight_bytes = _nbytes(self.inner.layers[0]) * len(self.inner.layers)
        if first or (last and self.tie):
            self.weight_bytes += _nbytes(self.inner.embed_tokens)
        if last:
            self.weight_bytes += _nbytes(self.inner.norm)
            if not self.tie and hasattr(model, "lm_head"):
                self.weight_bytes += _nbytes(model.lm_head)

    # -------------------------------------------------------------- caches

    def _new_caches(self, b: int) -> list:
        per_seq = [make_prompt_cache(self.model) for _ in range(b)]
        return [per_seq[0][i].merge([ps[i] for ps in per_seq]) for i in range(len(per_seq[0]))]

    def reset_slot(self, slot: int) -> None:
        self.caches.pop(slot, None)

    def free_slots(self) -> None:
        self.caches.clear()
        mx.clear_cache()

    def memory_used(self) -> int:
        return _active_memory()

    def unload(self) -> None:
        self.caches.clear()
        self.model = None
        self.inner = None
        mx.clear_cache()

    # -------------------------------------------------------------- forward

    def _forward(self, x, caches: list):
        if self.first:
            return self.inner(x, cache=caches)
        if self.supports_input_embeddings:
            dummy = mx.zeros(x.shape[:2], dtype=mx.int32)
            return self.inner(dummy, cache=caches, input_embeddings=x)
        return self.inner(x, cache=caches)  # embed_tokens was replaced by identity

    def _head(self, h):
        if self.tie:
            return self.inner.embed_tokens.as_linear(h)
        return self.model.lm_head(h)

    def run(self, slot: int, phase: str, x: np.ndarray, pad_lens: Optional[Sequence[int]] = None) -> np.ndarray:
        b, T = int(x.shape[0]), int(x.shape[1])
        padded = False
        if phase == "prefill":
            if pad_lens is None:
                raise ValueError("prefill needs pad_lens")
            caches = self._new_caches(b)
            if max(pad_lens) > 0:
                lengths = [T - int(p) for p in pad_lens]
                for c in caches:
                    c.prepare(lengths=lengths, right_padding=[int(p) for p in pad_lens])
                padded = True
            self.caches[slot] = caches
        elif slot not in self.caches:
            raise RuntimeError(f"slot {slot} has no cache; prefill first")
        caches = self.caches[slot]

        if self.first:
            inp = mx.array(x.astype(np.int32))
        else:
            inp = _from_wire(x, self.dtype)
        h = self._forward(inp, caches)
        if padded:
            for c in caches:
                c.finalize()
        if self.last:
            logits = self._head(h[:, -1, :])
            tokens = mx.argmax(logits, axis=-1).astype(mx.int32)
            mx.eval(tokens, [c.state for c in caches])
            out = np.array(tokens, dtype=np.int32)
        else:
            out = h.astype(self.dtype)
            mx.eval(out, [c.state for c in caches])
            out = _to_wire(out)
        return out


class MLXBackend(Backend):
    name = "mlx"
    padding_side = "right"

    def __init__(self, use_chat_template: bool = True):
        self.use_chat_template = use_chat_template
        self._paths: dict[str, Path] = {}
        self._configs: dict[str, dict] = {}
        self._tokenizers: dict[str, MLXTokenizer] = {}
        try:
            from mlx_lm.utils import maybe_set_recommended_wired_limit

            maybe_set_recommended_wired_limit()
        except Exception:
            pass

    # -------------------------------------------------------------- files

    def model_path(self, model_id: str) -> Path:
        if model_id not in self._paths:
            self._paths[model_id] = _download(model_id)
        return self._paths[model_id]

    def config(self, model_id: str) -> dict:
        if model_id not in self._configs:
            self._configs[model_id] = load_config(self.model_path(model_id))
        return self._configs[model_id]

    def tokenizer(self, model_id: str) -> Tokenizer:
        if model_id not in self._tokenizers:
            cfg = self.config(model_id)
            w = load_tokenizer(self.model_path(model_id), eos_token_ids=cfg.get("eos_token_id"))
            self._tokenizers[model_id] = MLXTokenizer(w, self.use_chat_template)
        return self._tokenizers[model_id]

    # -------------------------------------------------------------- info

    def model_info(self, model_id: str) -> ModelInfo:
        cfg = self.config(model_id)
        text_cfg = cfg.get("text_config", cfg)
        L = int(text_cfg["num_hidden_layers"])
        hidden = int(text_cfg["hidden_size"])
        n_heads = int(text_cfg.get("num_attention_heads", 1))
        n_kv = int(text_cfg.get("num_key_value_heads", n_heads))
        head_dim = int(text_cfg.get("head_dim") or hidden // max(n_heads, 1))
        # weight and cache sizes from a lazy model (no bytes are read)
        model, _ = load_model(self.model_path(model_id), lazy=True, strict=False)
        inner = model.model
        dtype = _compute_dtype(inner.layers[0])
        block = int(getattr(BatchKVCache, "step", 256))
        try:
            kv_bytes, state_bytes = _cache_sizes(model, block)
        except Exception as e:  # unusual forward pass: estimate from the config
            log.warning("could not measure cache sizes (%s); estimating from config", e)
            attn_fraction = 1.0  # hybrid models: only some layers keep a KV cache
            if "layer_types" in text_cfg:
                lt = text_cfg["layer_types"]
                attn_fraction = sum(1 for t in lt if "attention" in str(t) and "linear" not in str(t)) / max(len(lt), 1)
            elif "full_attention_interval" in text_cfg:
                attn_fraction = 1.0 / int(text_cfg["full_attention_interval"])
            kv_bytes, state_bytes = int(2 * n_kv * head_dim * dtype.size * attn_fraction), 0
        layer_bytes = int(sum(_nbytes(l) for l in inner.layers) / max(len(inner.layers), 1))
        embed_bytes = _nbytes(inner.embed_tokens)
        head_bytes = _nbytes(inner.norm)
        if not getattr(model.args, "tie_word_embeddings", False) and hasattr(model, "lm_head"):
            head_bytes += _nbytes(model.lm_head)
        else:
            head_bytes += embed_bytes  # tied: the last stage needs the table too
        tok = self.tokenizer(model_id)
        eos = sorted(int(t) for t in tok.w.eos_token_ids)
        pad = getattr(tok.w, "pad_token_id", None)
        if pad is None:
            pad = eos[0] if eos else 0
        spec = ModelSpec(
            num_layers=L,
            hidden_size=hidden,
            layer_bytes=layer_bytes,
            kv_bytes_per_token=kv_bytes,
            embed_bytes=embed_bytes,
            head_bytes=head_bytes,
            hidden_dtype_bytes=dtype.size,
            state_bytes_per_seq=state_bytes,
            kv_block_tokens=block,
        )
        return ModelInfo(model_id, spec, eos_token_ids=eos, pad_token_id=int(pad), vocab_size=int(text_cfg.get("vocab_size", 0)), extra={"model_type": cfg.get("model_type"), "quantization": cfg.get("quantization")})

    # -------------------------------------------------------------- memory

    def total_memory(self) -> int:
        return _recommended_memory()

    def free_memory(self) -> int:
        return max(0, _recommended_memory() - _active_memory() - (1 << 30))

    # -------------------------------------------------------------- stage

    def load_stage(self, model_id: str, layer_range: tuple[int, int], first: bool, last: bool) -> Stage:
        a, b = layer_range
        path = self.model_path(model_id)
        model, config = load_model(path, lazy=True, strict=False)
        inner = model.model
        all_layers = list(inner.layers)
        if not (0 <= a < b <= len(all_layers)):
            raise ValueError(f"layer range {layer_range} outside 0..{len(all_layers)}")
        inner.layers = all_layers[a:b]
        # hybrid models keep indices of their first attention / first linear layer for mask building
        if hasattr(inner, "fa_idx") or hasattr(inner, "ssm_idx"):
            inner.ssm_idx = next((i for i, l in enumerate(inner.layers) if getattr(l, "is_linear", False)), None)
            inner.fa_idx = next((i for i, l in enumerate(inner.layers) if not getattr(l, "is_linear", False)), None)
        tie = bool(getattr(model.args, "tie_word_embeddings", False))
        to_eval = [l.parameters() for l in inner.layers]
        if first or (last and tie):
            to_eval.append(inner.embed_tokens.parameters())
        if not first:
            import inspect

            if "input_embeddings" not in inspect.signature(inner.__call__).parameters and not (last and tie):
                inner.embed_tokens = _Identity()
            elif "input_embeddings" not in inspect.signature(inner.__call__).parameters:
                # need the table for the tied head but must not embed hidden states: keep a
                # reference for the head and bypass the embedding in the forward pass
                table = inner.embed_tokens
                inner.embed_tokens = _Identity()
                inner.embed_tokens.as_linear = table.as_linear  # type: ignore[attr-defined]
        if last:
            to_eval.append(inner.norm.parameters())
            if not tie and hasattr(model, "lm_head"):
                to_eval.append(model.lm_head.parameters())
        else:
            inner.norm = _Identity()
        t0 = time.perf_counter()
        mx.eval(to_eval)
        mx.clear_cache()
        log.info("MLX stage %s layers %d-%d loaded in %.1fs, active memory %.2f GB", model_id, a, b - 1, time.perf_counter() - t0, _active_memory() / 1e9)
        return MLXStage(model, config, layer_range, first, last)

    # -------------------------------------------------------------- profile

    def profile(self, model_id: str, batch_sizes: Sequence[int] = (1, 8, 32, 64), probe_layers: int = 2, context: int = 64) -> dict:
        """Marginal time per layer: time(n2 layers) - time(n1 layers) over (n2 - n1).

        Loading a stage and embedding tokens has fixed costs; measuring two
        depths and taking the difference leaves only the per-layer cost.
        """
        info = self.model_info(model_id)
        L = info.spec.num_layers
        n1 = max(1, min(probe_layers, L))
        n2 = min(L, max(n1 + 2, 2 * n1 + 2))
        if n2 == n1:
            return super().profile(model_id, batch_sizes, probe_layers, context)
        stage = self.load_stage(model_id, (0, n2), first=True, last=False)
        full_layers = list(stage.inner.layers)
        layer_time: dict[int, float] = {}
        prefill_per_token = 0.0
        try:
            for b in batch_sizes:
                times = {}
                for n in (n2, n1):
                    stage.inner.layers = full_layers[:n]
                    if hasattr(stage.inner, "fa_idx") or hasattr(stage.inner, "ssm_idx"):
                        stage.inner.ssm_idx = next((i for i, l in enumerate(stage.inner.layers) if getattr(l, "is_linear", False)), None)
                        stage.inner.fa_idx = next((i for i, l in enumerate(stage.inner.layers) if not getattr(l, "is_linear", False)), None)
                    toks = np.full((b, context), info.pad_token_id, dtype=np.int32)
                    t0 = time.perf_counter()
                    stage.run(0, "prefill", toks, pad_lens=[0] * b)
                    t_prefill = time.perf_counter() - t0
                    step = np.zeros((b, 1), dtype=np.int32)
                    stage.run(0, "decode", step)
                    reps = 4
                    t0 = time.perf_counter()
                    for _ in range(reps):
                        stage.run(0, "decode", step)
                    times[n] = ((time.perf_counter() - t0) / reps, t_prefill)
                    stage.reset_slot(0)
                    mx.clear_cache()
                dec = max((times[n2][0] - times[n1][0]) / (n2 - n1), times[n2][0] / n2 * 0.5)
                layer_time[int(b)] = dec
                if b == batch_sizes[0]:
                    prefill_per_token = max((times[n2][1] - times[n1][1]) / (n2 - n1), 0.0) / (b * context)
                log.info("profile b=%d: %.2f ms/layer", b, dec * 1e3)
        finally:
            stage.inner.layers = full_layers
            stage.unload()
        return {"layer_time": layer_time, "prefill_layer_time_per_token": prefill_per_token, "free_mem": self.free_memory()}
