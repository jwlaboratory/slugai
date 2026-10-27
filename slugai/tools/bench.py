"""``slugai bench``: baselines on the same prompts, same backend, same greedy decoding.

* ``single``: one Mac holds every layer (the model must fit). Prompts run in
  batches of ``--batch``; this is the "mlx-lm batched generation" number the
  ring is compared against for models that fit on one machine.
* ``streaming``: FlexGen-style for models that do *not* fit. Only one layer is
  resident at a time: every decode step walks the layers, reloading each from
  disk, running the whole batch through it, and dropping it. KV caches stay in
  memory. This is deliberately the "too much time loading layers in and out"
  baseline from the design doc.

Both write a ``.jsonl`` of completions and (optionally) a JSON report with
tokens/s, so numbers line up with ``slugai run``'s report.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Optional

import numpy as np

from ..backends import Backend, ModelInfo, Stage
from ..models import apply_models_dir
from ..prompts import load_prompts

log = logging.getLogger(__name__)


def _pad_batch(enc: list[list[int]], pad_id: int, side: str) -> tuple[np.ndarray, list[int]]:
    T = max(len(e) for e in enc)
    arr = np.full((len(enc), T), pad_id, dtype=np.int32)
    pads = []
    for i, e in enumerate(enc):
        if side == "left":
            arr[i, T - len(e):] = e
        else:
            arr[i, : len(e)] = e
        pads.append(T - len(e))
    return arr, pads


class Runner:
    """Greedy batched generation through a list of stages (one for ``single``)."""

    def __init__(self, backend: Backend, model: str, max_tokens: int):
        self.backend = backend
        self.model = model
        self.max_tokens = max_tokens
        self.info: ModelInfo = backend.model_info(model)
        self.tok = backend.tokenizer(model)
        self.side = getattr(backend, "padding_side", "left")
        self.tokens_generated = 0

    def generate_batch(self, prompts: list[str], step_fn) -> list[list[int]]:
        enc = [self.tok.encode(p) or [self.info.pad_token_id] for p in prompts]
        b = len(enc)
        eos = set(self.info.eos_token_ids)
        gen: list[list[int]] = [[] for _ in enc]
        done = [False] * b
        T = max(len(e) for e in enc)
        if T > 1:
            arr, pads = _pad_batch([e[:-1] for e in enc], self.info.pad_token_id, self.side)
            step_fn("prefill", arr, pads)
            nxt = step_fn("decode", np.array([[e[-1]] for e in enc], dtype=np.int32), None)
        else:
            nxt = step_fn("prefill", np.array([[e[-1]] for e in enc], dtype=np.int32), [0] * b)
        while True:
            for i, t in enumerate(nxt.tolist()):
                if not done[i]:
                    gen[i].append(int(t))
                    self.tokens_generated += 1
                    if t in eos or len(gen[i]) >= self.max_tokens:
                        done[i] = True
            if all(done):
                break
            nxt = step_fn("decode", nxt.reshape(b, 1).astype(np.int32), None)
        return gen


def run_single(backend: Backend, model: str, prompts: list[str], max_tokens: int, batch: int) -> tuple[list[list[int]], dict]:
    r = Runner(backend, model, max_tokens)
    L = r.info.spec.num_layers
    t0 = time.perf_counter()
    stage = backend.load_stage(model, (0, L), first=True, last=True)
    load_s = time.perf_counter() - t0
    outs: list[list[int]] = []
    t0 = time.perf_counter()
    for start in range(0, len(prompts), batch):
        chunk = prompts[start : start + batch]
        slot = start // batch

        def step(phase, x, pads, slot=slot):
            return stage.run(slot, phase, x, pad_lens=pads)

        outs.extend(r.generate_batch(chunk, step))
        stage.reset_slot(slot)
        log.info("%d/%d prompts, %.1f tok/s", min(start + batch, len(prompts)), len(prompts), r.tokens_generated / (time.perf_counter() - t0))
    gen_s = time.perf_counter() - t0
    stage.unload()
    return outs, {"baseline": "single", "load_seconds": load_s, "decode_seconds": gen_s, "tokens_generated": r.tokens_generated, "tok_s": r.tokens_generated / max(gen_s, 1e-9), "batch": batch}


def run_streaming(backend: Backend, model: str, prompts: list[str], max_tokens: int, batch: int) -> tuple[list[list[int]], dict]:
    """One layer resident at a time; KV caches kept per layer across reloads."""
    r = Runner(backend, model, max_tokens)
    L = r.info.spec.num_layers
    outs: list[list[int]] = []
    layer_loads = 0
    load_time = 0.0
    t0 = time.perf_counter()
    for start in range(0, len(prompts), batch):
        chunk = prompts[start : start + batch]
        saved: dict[int, dict] = {}  # layer -> that layer's slot caches

        def step(phase, x, pads):
            nonlocal layer_loads, load_time
            h = x
            for l in range(L):
                t1 = time.perf_counter()
                stage: Stage = backend.load_stage(model, (l, l + 1), first=(l == 0), last=(l == L - 1))
                load_time += time.perf_counter() - t1
                layer_loads += 1
                if l in saved:
                    stage.caches = saved[l]  # type: ignore[attr-defined]
                h = stage.run(0, phase, h, pad_lens=pads)
                saved[l] = stage.caches  # type: ignore[attr-defined]
                stage.caches = {}  # type: ignore[attr-defined]
                stage.unload()
            return h

        outs.extend(r.generate_batch(chunk, step))
        log.info("%d/%d prompts, %.2f tok/s, %.0f%% of time loading layers", min(start + batch, len(prompts)), len(prompts), r.tokens_generated / (time.perf_counter() - t0), 100 * load_time / (time.perf_counter() - t0))
    gen_s = time.perf_counter() - t0
    return outs, {"baseline": "streaming", "decode_seconds": gen_s, "tokens_generated": r.tokens_generated, "tok_s": r.tokens_generated / max(gen_s, 1e-9), "batch": batch, "layer_loads": layer_loads, "load_seconds": load_time}


def main(args) -> int:
    apply_models_dir(getattr(args, "models_dir", None))
    from ..backends import get_backend

    backend = get_backend(args.backend)
    prompts = load_prompts(args.prompts, limit=args.limit)
    if not prompts:
        print("no prompts")
        return 2
    fn = run_single if args.baseline == "single" else run_streaming
    log.info("bench %s: %s, %d prompts, batch %d, max_tokens %d", args.baseline, args.model, len(prompts), args.batch, args.max_tokens)
    outs, stats = fn(backend, args.model, prompts, args.max_tokens, args.batch)
    tok = backend.tokenizer(args.model)
    out_path = Path(args.report).with_suffix(".jsonl") if args.report else Path(f"bench-{args.baseline}.jsonl")
    with out_path.open("w") as fh:
        for i, (p, g) in enumerate(zip(prompts, outs)):
            fh.write(json.dumps({"id": i, "prompt": p, "completion": tok.decode(g), "tokens": len(g), "token_ids": g}, ensure_ascii=False) + "\n")
    stats.update(model=args.model, prompts=len(prompts), max_tokens=args.max_tokens, output=str(out_path), tok_h=stats["tok_s"] * 3600)
    if args.report:
        Path(args.report).write_text(json.dumps(stats, indent=2))
    print(f"{args.baseline}: {stats['tokens_generated']} tokens in {stats['decode_seconds']:.1f}s = {stats['tok_s']:.1f} tok/s ({stats['tok_h'] / 1000:.0f}k tok/h); wrote {out_path}")
    return 0
