"""The stage interface every backend implements.

A *stage* holds a contiguous range of layers (optionally the embedding on the
first stage and the final norm + lm_head on the last) and a KV cache per
micro-batch *slot*. The ring moves micro-batches through stages; each stage
only ever sees numpy arrays:

* first stage input: token ids ``(b, T)`` int32 (prefill) or ``(b, 1)`` (decode)
* other stages input: hidden states ``(b, T, D)`` in the backend's wire dtype
* last stage output: greedy next-token ids ``(b,)`` int32
* other stages output: hidden states ``(b, T, D)`` in the backend's wire dtype

Prompts are padded inside a micro-batch on the backend's ``padding_side``;
``pad_lens[i]`` is how many pad tokens sequence ``i`` has. Stages remember
that per slot so decode steps keep masking those keys.
"""

from __future__ import annotations

import abc
import time
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

from ..placement import ModelSpec


@dataclass
class ModelInfo:
    """What the server needs to know about a model, reported by a host."""

    model_id: str
    spec: ModelSpec
    eos_token_ids: list[int]
    pad_token_id: int
    vocab_size: int
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "model_id": self.model_id,
            "spec": self.spec.__dict__,
            "eos_token_ids": self.eos_token_ids,
            "pad_token_id": self.pad_token_id,
            "vocab_size": self.vocab_size,
            "extra": self.extra,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ModelInfo":
        return cls(
            model_id=d["model_id"],
            spec=ModelSpec(**d["spec"]),
            eos_token_ids=list(d["eos_token_ids"]),
            pad_token_id=d["pad_token_id"],
            vocab_size=d["vocab_size"],
            extra=d.get("extra", {}),
        )


class Tokenizer(abc.ABC):
    @abc.abstractmethod
    def encode(self, text: str) -> list[int]: ...

    @abc.abstractmethod
    def decode(self, ids: Sequence[int]) -> str: ...


class Stage(abc.ABC):
    """A loaded range of layers with per-slot KV caches."""

    first: bool
    last: bool
    layer_range: tuple[int, int]

    @abc.abstractmethod
    def run(self, slot: int, phase: str, x: np.ndarray, pad_lens: Optional[Sequence[int]] = None) -> np.ndarray:
        """Run this stage on one micro-batch.

        ``phase`` is ``"prefill"`` (resets the slot's cache; ``pad_lens`` required)
        or ``"decode"``. See the module docstring for shapes.
        """

    @abc.abstractmethod
    def reset_slot(self, slot: int) -> None: ...

    def free_slots(self) -> None:
        pass

    def memory_used(self) -> int:
        return 0

    def unload(self) -> None:
        pass


class Backend(abc.ABC):
    name: str = "base"
    padding_side: str = "left"  # how stage 1 pads prompts inside a micro-batch

    @abc.abstractmethod
    def model_info(self, model_id: str) -> ModelInfo: ...

    @abc.abstractmethod
    def tokenizer(self, model_id: str) -> Tokenizer: ...

    @abc.abstractmethod
    def load_stage(self, model_id: str, layer_range: tuple[int, int], first: bool, last: bool) -> Stage: ...

    @abc.abstractmethod
    def free_memory(self) -> int:
        """Bytes this machine can give to weights + KV cache right now."""

    def total_memory(self) -> int:
        return self.free_memory()

    def profile(
        self,
        model_id: str,
        batch_sizes: Sequence[int] = (1, 8, 32, 64),
        probe_layers: int = 2,
        context: int = 64,
    ) -> dict:
        """Measure seconds per layer per decode step for each micro-batch size.

        Loads ``probe_layers`` layers, prefills ``context`` tokens, then times
        decode steps. Returns ``{"layer_time": {b: seconds},
        "prefill_layer_time_per_token": s, "free_mem": bytes}``.
        """
        info = self.model_info(model_id)
        n = max(1, min(probe_layers, info.spec.num_layers))
        stage = self.load_stage(model_id, (0, n), first=True, last=False)
        layer_time: dict[int, float] = {}
        prefill_per_token = 0.0
        try:
            for b in batch_sizes:
                toks = np.full((b, context), info.pad_token_id + 1, dtype=np.int32)
                t0 = time.perf_counter()
                stage.run(0, "prefill", toks, pad_lens=[0] * b)
                t_prefill = time.perf_counter() - t0
                if b == batch_sizes[0]:
                    prefill_per_token = t_prefill / n / (b * context)
                step = np.zeros((b, 1), dtype=np.int32)
                stage.run(0, "decode", step)  # warm-up
                reps = 3
                t0 = time.perf_counter()
                for _ in range(reps):
                    stage.run(0, "decode", step)
                layer_time[int(b)] = (time.perf_counter() - t0) / reps / n
                stage.reset_slot(0)
        finally:
            stage.unload()
        return {
            "layer_time": layer_time,
            "prefill_layer_time_per_token": prefill_per_token,
            "free_mem": self.free_memory(),
        }
