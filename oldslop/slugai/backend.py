"""Run a contiguous range of a real model's layers on Apple silicon with MLX.

A *stage* holds a contiguous range of layers (plus the embedding on the first
stage and the final norm + lm_head on the last) and a KV cache per
micro-batch *slot*. The ring moves micro-batches through stages; each stage
only ever sees numpy arrays:

* first stage input: token ids ``(b, T)`` int32 (prefill) or ``(b, 1)`` (decode)
* other stages input / output: hidden states ``(b, T, D)``
* last stage output: greedy next-token ids ``(b,)`` int32

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
  machine would and large activations never overflow float16;
* a stage may *stream* some of its layers, FlexGen-style: their weights stay
  on disk and are read just before each use, then dropped, so a Mac can run
  more layers than fit in its memory (at the cost of reading them every step).
  Each streamed layer's weights are read while the one before it computes, so
  disk reads overlap compute.

Prompts are right-padded for prefill, exactly as mlx-lm's batch generator
does, and ``pad_lens[i]`` says how many pad tokens sequence ``i`` has; the
caches are finalised into the left-aligned layout afterwards. The host splits
every prompt into ``prompt[:-1]`` (prefill) and its last token (a decode step)
so that, after prefill, all sequences in a micro-batch are aligned.
"""

from __future__ import annotations

import json
import logging
import os
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten, tree_map, tree_unflatten
from mlx_lm.models.cache import BatchKVCache, make_prompt_cache
from mlx_lm.utils import _download, load_config, load_model, load_tokenizer

from .placement import ModelSpec

log = logging.getLogger(__name__)


@dataclass
class ModelInfo:
    spec: ModelSpec
    eos_token_ids: list[int]
    pad_token_id: int


def _recommended_memory() -> int:
    info = mx.device_info()
    rec = info.get("max_recommended_working_set_size")
    if rec:
        return int(rec)
    mem = info.get("memory_size")
    return int(mem * 0.75) if mem else 16 << 30


def _active_memory() -> int:
    return int(mx.get_active_memory())


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


PROFILE_BATCH_SIZES = (1, 8, 16, 32, 64, 128)
PROFILE_MAX_CONTEXT = 1024  # longer caches cost more to fill than they teach


def _non_decreasing(values: list[float]) -> list[float]:
    """The closest non-decreasing sequence (pool adjacent violators).

    A layer can't get faster per step as the batch grows; a profile that
    says so is noise, and this averages it away.
    """
    blocks: list[list[float]] = []  # [mean, count]
    for v in values:
        blocks.append([v, 1])
        while len(blocks) > 1 and blocks[-2][0] > blocks[-1][0]:
            m, n = blocks.pop()
            pm, pn = blocks[-1]
            blocks[-1] = [(pm * pn + m * n) / (pn + n), pn + n]
    return [m for m, n in blocks for _ in range(int(n))]


def _set_layers(inner, layers: list) -> None:
    inner.layers = layers
    # hybrid models keep indices of their first attention / first linear layer for mask building
    if hasattr(inner, "fa_idx") or hasattr(inner, "ssm_idx"):
        inner.ssm_idx = next((i for i, l in enumerate(layers) if getattr(l, "is_linear", False)), None)
        inner.fa_idx = next((i for i, l in enumerate(layers) if not getattr(l, "is_linear", False)), None)


def _weight_files(path: Path) -> dict[str, Path]:
    """Which safetensors file holds each weight (from the index, or each file's header)."""
    index = path / "model.safetensors.index.json"
    if index.exists():
        return {k: path / f for k, f in json.loads(index.read_text())["weight_map"].items()}
    return {k: f for f in sorted(path.glob("model*.safetensors")) for k in mx.load(str(f))}  # lazy: headers only


def _stream(model, layer, index: int, files: dict[str, Path]) -> dict:
    """Make ``layer`` (global number ``index``) read its weights from disk for each call and drop them after.

    Returns the layer's prefetch state: set ``state["next"]`` to the next
    streamed layer's state (after the stage's last streamed layer, the
    first's, for the next micro-batch). While this layer computes, the next
    one's weights are read from disk and the one after's file headers are
    parsed, so reads overlap compute and start without waiting on the parse.
    """
    prefix = f"model.layers.{index}."
    # unread weights (headers parsed); weights being read; the next streamed layer's state
    state: dict = {"parsed": None, "ready": None, "next": None}
    owning = sorted({f for k, f in files.items() if k.startswith(prefix)})

    def load() -> dict:  # the layer's weights as unread arrays: mx.load only parses headers
        w = {k: v for f in owning for k, v in mx.load(str(f)).items() if k.startswith(prefix)}
        if hasattr(model, "sanitize"):
            w = model.sanitize(w)
        return tree_unflatten([(k[len(prefix):], v) for k, v in w.items() if k.startswith(prefix)])

    want = {k: v.shape for k, v in tree_flatten(layer.parameters())}
    have = {k: v.shape for k, v in tree_flatten(load())}
    if want != have:
        raise ValueError(f"can't stream layer {index}: its weights on disk don't match the model ({sorted(set(want) ^ set(have))[:3]})")
    # what the layer holds between uses: empty arrays of the weights' dtypes. Cheaper to
    # swap in than unread ones, since mx.load parses the whole file's header every call
    empty = tree_map(lambda v: mx.zeros((0,), dtype=v.dtype), layer.parameters())
    base = type(layer)

    def parse() -> None:
        state["parsed"] = load()

    def read() -> None:
        state["ready"], state["parsed"] = state["parsed"] if state["parsed"] is not None else load(), None
        mx.async_eval(state["ready"])

    def __call__(self, *args, **kwargs):
        w, state["ready"] = state["ready"] if state["ready"] is not None else load(), None
        self.update(w)
        out = base.__call__(self, *args, **kwargs)
        cache = kwargs.get("cache", args[2] if len(args) > 2 else None)
        todo = [out, cache.state if cache is not None else []]
        mx.async_eval(todo)  # queue this layer first...
        nxt = state["next"]
        nxt["read"]()  # ...then read the next streamed layer while it computes
        nxt["next"]["parse"]()  # and parse the one after's headers (mx.load parses a whole file's)
        mx.eval(todo)  # finish with the weights...
        self.update(empty)  # ...then drop them, freeing the memory
        return out

    layer.__class__ = type(f"Streamed{base.__name__}", (base,), {"__call__": __call__})
    layer.update(empty)
    state.update(parse=parse, read=read, next=state)
    return state


def _disk_read_speed(files: list[Path], nbytes: int = 1 << 30) -> float:
    """Bytes/s reading the model's files, bypassing the file cache where macOS allows (F_NOCACHE)."""
    import fcntl

    done, t0 = 0, time.perf_counter()
    for f in files:
        fd = os.open(f, os.O_RDONLY)
        try:
            try:
                fcntl.fcntl(fd, getattr(fcntl, "F_NOCACHE", 48), 1)
            except OSError:
                pass
            while done < nbytes:
                chunk = os.read(fd, 16 << 20)
                if not chunk:
                    break
                done += len(chunk)
        finally:
            os.close(fd)
        if done >= nbytes:
            break
    return done / max(time.perf_counter() - t0, 1e-6)


class _Identity(nn.Module):
    def __call__(self, x):
        return x


class Tokenizer:
    def __init__(self, wrapper):
        self.w = wrapper
        self.use_chat_template = getattr(wrapper, "has_chat_template", False)

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


class Stage:
    """A loaded range of layers with per-slot KV caches."""

    def __init__(self, model, first: bool, last: bool):
        self.model = model
        self.inner = model.model
        self.first, self.last = first, last
        self.caches: dict[int, list] = {}
        self.tie = bool(getattr(model.args, "tie_word_embeddings", False))
        self.dtype = _compute_dtype(self.inner.layers[0])

    # -------------------------------------------------------------- caches

    def _new_caches(self, b: int) -> list:
        per_seq = [make_prompt_cache(self.model) for _ in range(b)]
        return [per_seq[0][i].merge([ps[i] for ps in per_seq]) for i in range(len(per_seq[0]))]

    def reset_slot(self, slot: int) -> None:
        self.caches.pop(slot, None)

    def memory_used(self) -> int:
        return _active_memory()

    def unload(self) -> None:
        self.caches.clear()
        self.model = None
        self.inner = None
        mx.clear_cache()

    # -------------------------------------------------------------- forward

    def _head(self, h):
        if self.tie:
            return self.inner.embed_tokens.as_linear(h)
        return self.model.lm_head(h)

    def run(self, slot: int, phase: str, x: np.ndarray, pad_lens: Optional[Sequence[int]] = None) -> np.ndarray:
        """Run this stage on one micro-batch.

        ``phase`` is ``"prefill"`` (resets the slot's cache; ``pad_lens`` required)
        or ``"decode"``. See the module docstring for shapes.
        """
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
        h = self.inner(inp, cache=caches)  # past stage 1 the embedding is an identity
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


class Backend:
    """Loads models: their sizes, tokenizer, a stage of layers, and this Mac's memory and speed.

    ``memory_limit`` caps the bytes this Mac offers for weights and caches
    (default: what macOS recommends for the GPU).
    """

    def __init__(self, memory_limit: Optional[int] = None):
        self.memory_limit = memory_limit
        self._paths: dict[str, Path] = {}
        self._configs: dict[str, dict] = {}
        self._tokenizers: dict[str, Tokenizer] = {}
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
            self._tokenizers[model_id] = Tokenizer(w)
        return self._tokenizers[model_id]

    # -------------------------------------------------------------- info

    def model_info(self, model_id: str) -> ModelInfo:
        cfg = self.config(model_id)
        text_cfg = cfg.get("text_config", cfg)
        # weight and cache sizes from a lazy model (no bytes are read)
        model, _ = load_model(self.model_path(model_id), lazy=True, strict=False)
        inner = model.model
        dtype = _compute_dtype(inner.layers[0])
        block = int(getattr(BatchKVCache, "step", 256))
        kv_bytes, state_bytes = _cache_sizes(model, block)
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
            num_layers=int(text_cfg["num_hidden_layers"]),
            hidden_size=int(text_cfg["hidden_size"]),
            layer_bytes=layer_bytes,
            kv_bytes_per_token=kv_bytes,
            embed_bytes=embed_bytes,
            head_bytes=head_bytes,
            hidden_dtype_bytes=dtype.size,
            state_bytes_per_seq=state_bytes,
            kv_block_tokens=block,
        )
        return ModelInfo(spec, eos_token_ids=eos, pad_token_id=int(pad))

    # -------------------------------------------------------------- memory

    def total_memory(self) -> int:
        rec = _recommended_memory()
        return min(rec, self.memory_limit) if self.memory_limit else rec

    def free_memory(self) -> int:
        """Bytes this Mac can give to weights + KV cache right now."""
        total = self.total_memory()
        return max(0, total - _active_memory() - min(1 << 30, total // 10))

    # -------------------------------------------------------------- stage

    def load_stage(self, model_id: str, layer_range: tuple[int, int], first: bool, last: bool, resident: Optional[int] = None) -> Stage:
        """Load layers ``[a, b)``. The first ``resident`` stay in memory (default: all);
        the rest are streamed from disk at every step."""
        a, b = layer_range
        path = self.model_path(model_id)
        model, _ = load_model(path, lazy=True, strict=False)
        inner = model.model
        all_layers = list(inner.layers)
        if not (0 <= a < b <= len(all_layers)):
            raise ValueError(f"layer range {layer_range} outside 0..{len(all_layers)}")
        _set_layers(inner, all_layers[a:b])
        keep = b - a if resident is None else max(0, min(resident, b - a))
        if keep < b - a:
            files = _weight_files(path)
            streams = [_stream(model, inner.layers[j], a + j, files) for j in range(keep, b - a)]
            for s, nxt in zip(streams, streams[1:] + streams[:1]):
                s["next"] = nxt
            streams[0]["read"]()  # the first micro-batch's first streamed layer
            streams[0]["next"]["parse"]()
        tie = bool(getattr(model.args, "tie_word_embeddings", False))
        to_eval = [l.parameters() for l in inner.layers[:keep]]
        if first or (last and tie):
            to_eval.append(inner.embed_tokens.parameters())
        if not first:  # hidden states come in: pass them straight through the embedding
            table = inner.embed_tokens
            inner.embed_tokens = _Identity()
            if last and tie:  # the tied head still needs the table
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
        log.info(
            "stage %s layers %d-%d loaded in %.1fs (%d streamed from disk), active memory %.2f GB",
            model_id, a, b - 1, time.perf_counter() - t0, b - a - keep, _active_memory() / 1e9,
        )
        return Stage(model, first, last)

    # -------------------------------------------------------------- profile

    def profile(self, model_id: str, batch_sizes: Sequence[int] = PROFILE_BATCH_SIZES, probe_layers: int = 2, context: int = 64, reps: int = 8) -> dict:
        """Seconds per layer per decode step for each micro-batch size, and free memory.

        Each number is a layer's marginal cost: the step time through n2
        layers minus n1 layers, over (n2 - n1), so fixed costs (embedding,
        copies) cancel. Steps run with ``context`` tokens already cached, the
        job's typical length, because attention over a longer cache costs
        more; inputs are random tokens so MoE layers route to a realistic mix
        of experts. After a GPU warm-up, each measurement drops two steps and
        takes the median of ``reps``, and the results are smoothed so time per
        layer never falls as the batch grows. Also measures how fast the
        model's files read from disk (for streamed layers). Returns
        ``{"layer_time": {b: seconds}, "disk_bw": bytes/s, "free_mem": bytes}``.
        """
        info = self.model_info(model_id)
        cfg = self.config(model_id)
        vocab = int(cfg.get("text_config", cfg).get("vocab_size") or 1000)
        L = info.spec.num_layers
        n1 = max(1, min(probe_layers, L))
        n2 = min(L, max(n1 + 2, 2 * n1 + 2))
        depths = (n2, n1) if n2 > n1 else (n2,)  # a model this shallow: whole-stage time per layer
        context = max(1, min(context, PROFILE_MAX_CONTEXT))
        rng = np.random.default_rng(0)
        stage = self.load_stage(model_id, (0, n2), first=True, last=False)
        full_layers = list(stage.inner.layers)

        def tokens(b: int, t: int) -> np.ndarray:
            return rng.integers(0, vocab, (b, t), dtype=np.int32)

        def per_layer(b: int) -> float:
            """Seconds per layer per decode step of ``b`` sequences.

            Steps through n2 and n1 layers alternate (one cache each), and the
            median of the paired differences is taken, so clock drift hits
            both sides alike and cancels.
            """
            chunk = max(1, min(context, 8192 // b))  # fill caches in chunks: no huge attention buffers
            for slot, n in enumerate(depths):
                _set_layers(stage.inner, full_layers[:n])
                stage.run(slot, "prefill", tokens(b, chunk), pad_lens=[0] * b)
                for done in range(chunk, context, chunk):
                    stage.run(slot, "decode", tokens(b, min(chunk, context - done)))
            steps = []
            for _ in range(reps + 2):
                t = []
                for slot, n in enumerate(depths):
                    _set_layers(stage.inner, full_layers[:n])
                    x = tokens(b, 1)
                    t0 = time.perf_counter()
                    stage.run(slot, "decode", x)
                    t.append(time.perf_counter() - t0)
                steps.append(t)
            for slot in range(len(depths)):
                stage.reset_slot(slot)
            mx.clear_cache()
            steps = steps[2:]  # the first two are warm-up
            whole = statistics.median(s[0] for s in steps) / n2
            if n2 == n1:
                return whole
            return max(statistics.median(s[0] - s[1] for s in steps) / (n2 - n1), whole * 0.5)

        layer_time: dict[int, float] = {}
        try:
            per_layer(max(batch_sizes))  # wake the GPU up before measuring anything
            for b in sorted(batch_sizes):
                layer_time[int(b)] = per_layer(b)
            bs = sorted(layer_time)
            layer_time = dict(zip(bs, _non_decreasing([layer_time[b] for b in bs])))
            log.info("profile at %d tokens of context: %s", context, ", ".join(f"b{b}={t * 1e3:.2f}ms" for b, t in layer_time.items()))
        finally:
            _set_layers(stage.inner, full_layers)
            stage.unload()
        disk_bw = _disk_read_speed(sorted(set(_weight_files(self.model_path(model_id)).values())))
        log.info("disk reads at %.1f GB/s", disk_bw / 1e9)
        return {"layer_time": layer_time, "disk_bw": disk_bw, "free_mem": self.free_memory()}
