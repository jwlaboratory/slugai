"""Fake backend: a tiny numpy transformer for tests and ``slugai host --backend fake``.

It is a real causal transformer (attention with a per-slot KV cache, padding
masks on either side), just small and deterministic, so the ring, scheduling,
rebalancing and placement can be exercised end to end without MLX.
"""

from __future__ import annotations

import os
import time
from typing import Optional, Sequence

import numpy as np

from ..placement import ModelSpec
from .base import Backend, ModelInfo, Stage, Tokenizer


class FakeTokenizer(Tokenizer):
    PAD, BOS, EOS = 0, 1, 2
    CHARS = " abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.,;:!?'\"-()\n"

    def __init__(self) -> None:
        self.itos = ["<pad>", "<s>", "</s>"] + list(self.CHARS)
        self.stoi = {c: i for i, c in enumerate(self.itos)}
        self.vocab_size = len(self.itos)

    def encode(self, text: str) -> list[int]:
        unk = self.stoi["?"]
        return [self.BOS] + [self.stoi.get(c, unk) for c in text]

    def decode(self, ids: Sequence[int]) -> str:
        out = []
        for i in ids:
            i = int(i)
            if i in (self.PAD, self.BOS):
                continue
            if i == self.EOS:
                break
            if 0 <= i < len(self.itos):
                out.append(self.itos[i])
        return "".join(out)


def _rms(x: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    return x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps)


def _parse_fake_id(model_id: str) -> tuple[int, int, int]:
    """``fake``, ``fake-L8``, ``fake-L8-D32`` -> (layers, dim, seed)."""
    layers, dim = 6, 32
    for part in model_id.split("-")[1:]:
        if part.startswith("L"):
            layers = int(part[1:])
        elif part.startswith("D"):
            dim = int(part[1:])
    seed = sum(ord(c) for c in model_id)
    return layers, dim, seed


class FakeStage(Stage):
    """Tiny single-head causal transformer in numpy.

    Hidden states are rounded to float16 after every layer so a single-process
    run and a pipelined run (which ships fp16 over the wire) are bit-identical.
    """

    def __init__(self, params: dict, layer_range: tuple[int, int], first: bool, last: bool, slow_ms: float = 0.0, padding_side: str = "left"):
        self.p = params
        self.layer_range = layer_range
        self.first, self.last = first, last
        self.slow_ms = slow_ms
        self.padding_side = padding_side
        self.caches: dict[int, dict] = {}
        self.D = params["embed"].shape[1]

    def reset_slot(self, slot: int) -> None:
        self.caches.pop(slot, None)

    def free_slots(self) -> None:
        self.caches.clear()

    def memory_used(self) -> int:
        n = 0
        for c in self.caches.values():
            for kv in c["kv"]:
                if kv is not None:
                    n += kv[0].nbytes + kv[1].nbytes
        for layer in self.p["layers"][self.layer_range[0] : self.layer_range[1]]:
            n += sum(w.nbytes for w in layer.values())
        return n

    def _attn(self, slot: int, li: int, h: np.ndarray, valid: np.ndarray) -> np.ndarray:
        W = self.p["layers"][li]
        c = self.caches[slot]
        q = h @ W["wq"]
        k = h @ W["wk"]
        v = h @ W["wv"]
        idx = li - self.layer_range[0]
        if c["kv"][idx] is None:
            K, V = k, v
        else:
            K = np.concatenate([c["kv"][idx][0], k], axis=1)
            V = np.concatenate([c["kv"][idx][1], v], axis=1)
        c["kv"][idx] = (K, V)
        b, T, _ = q.shape
        Tk = K.shape[1]
        scores = q @ K.transpose(0, 2, 1) / np.sqrt(self.D)  # (b, T, Tk)
        # causal: query at absolute position (Tk - T + i) sees keys <= that
        qpos = np.arange(Tk - T, Tk)[:, None]
        kpos = np.arange(Tk)[None, :]
        causal = kpos <= qpos  # (T, Tk)
        keymask = valid[:, None, :]  # (b, 1, Tk) valid keys
        allowed = causal[None] & (keymask | (kpos[None] == qpos[None]))
        scores = np.where(allowed, scores, -1e9)
        scores = scores - scores.max(-1, keepdims=True)
        w = np.exp(scores)
        w = w / w.sum(-1, keepdims=True)
        return (w @ V) @ W["wo"]

    def run(self, slot: int, phase: str, x: np.ndarray, pad_lens: Optional[Sequence[int]] = None) -> np.ndarray:
        a, bnd = self.layer_range
        if phase == "prefill":
            if pad_lens is None:
                raise ValueError("prefill needs pad_lens")
            self.caches[slot] = {"kv": [None] * (bnd - a), "pad": np.asarray(pad_lens, dtype=np.int64), "len": 0, "prefill_len": x.shape[1]}
        elif slot not in self.caches:
            raise RuntimeError(f"slot {slot} has no cache; prefill first")
        c = self.caches[slot]
        if self.first:
            h = self.p["embed"][x.astype(np.int64)].astype(np.float32)
        else:
            h = x.astype(np.float32)
        b, T, _ = h.shape
        c["len"] += T
        Tk = c["len"]
        pos = np.arange(Tk)[None, :]
        if self.padding_side == "left":
            valid = pos >= c["pad"][:, None]  # (b, Tk)
        else:  # right padding: pads sit between the prompt and the generated tokens
            Tp = c["prefill_len"]
            valid = (pos < (Tp - c["pad"])[:, None]) | (pos >= Tp)
        for li in range(a, bnd):
            W = self.p["layers"][li]
            h = h + self._attn(slot, li, _rms(h), valid)
            m = _rms(h) @ W["w1"]
            m = m * (1 / (1 + np.exp(-m)))  # silu
            h = h + m @ W["w2"]
            h = h.astype(np.float16).astype(np.float32)
            if self.slow_ms:
                time.sleep(self.slow_ms / 1000.0 * max(1, b) / 8)
        if self.last:
            logits = _rms(h[:, -1, :]) @ self.p["embed"].T
            return np.argmax(logits, axis=-1).astype(np.int32)
        return h.astype(np.float16)


class FakeBackend(Backend):
    name = "fake"

    def __init__(self, slow_ms: Optional[float] = None, free_mem: Optional[int] = None, padding_side: str = "left"):
        self.slow_ms = float(os.environ.get("SLUGAI_FAKE_SLOW_MS", "0")) if slow_ms is None else slow_ms
        self.padding_side = padding_side
        self._free_mem = (
            int(float(os.environ.get("SLUGAI_FAKE_MEM_GB", "16")) * (1 << 30)) if free_mem is None else free_mem
        )
        self._params: dict[str, dict] = {}
        self._tok = FakeTokenizer()

    def _params_for(self, model_id: str) -> dict:
        if model_id not in self._params:
            L, D, seed = _parse_fake_id(model_id)
            rng = np.random.default_rng(seed)
            V = self._tok.vocab_size
            s = 1.0 / np.sqrt(D)
            layers = []
            for _ in range(L):
                layers.append(
                    {
                        "wq": (rng.standard_normal((D, D)) * s).astype(np.float32),
                        "wk": (rng.standard_normal((D, D)) * s).astype(np.float32),
                        "wv": (rng.standard_normal((D, D)) * s).astype(np.float32),
                        "wo": (rng.standard_normal((D, D)) * s).astype(np.float32),
                        "w1": (rng.standard_normal((D, 4 * D)) * s).astype(np.float32),
                        "w2": (rng.standard_normal((4 * D, D)) * s * 0.5).astype(np.float32),
                    }
                )
            self._params[model_id] = {"embed": rng.standard_normal((V, D)).astype(np.float32), "layers": layers}
        return self._params[model_id]

    def model_info(self, model_id: str) -> ModelInfo:
        L, D, _ = _parse_fake_id(model_id)
        layer_bytes = 4 * (4 * D * D + 8 * D * D)
        spec = ModelSpec(
            num_layers=L,
            hidden_size=D,
            layer_bytes=layer_bytes,
            kv_bytes_per_token=2 * D * 4,
            embed_bytes=self._tok.vocab_size * D * 4,
            head_bytes=0,
        )
        return ModelInfo(
            model_id, spec, eos_token_ids=[FakeTokenizer.EOS], pad_token_id=FakeTokenizer.PAD, vocab_size=self._tok.vocab_size
        )

    def tokenizer(self, model_id: str) -> Tokenizer:
        return self._tok

    def load_stage(self, model_id: str, layer_range: tuple[int, int], first: bool, last: bool) -> Stage:
        return FakeStage(self._params_for(model_id), layer_range, first, last, slow_ms=self.slow_ms, padding_side=self.padding_side)

    def free_memory(self) -> int:
        return self._free_mem

    def total_memory(self) -> int:
        return self._free_mem
