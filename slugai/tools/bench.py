"""``slugai bench``: the baseline the ring is compared against.

One Mac holds every layer (the model must fit) and runs the same prompts with
the same greedy decoding, ``--batch`` prompts at a time: the "mlx-lm batched
generation" number. With ``--batch`` equal to the ring's micro-batch size the
tokens match the ring's exactly. Writes a ``.jsonl`` of completions and
(optionally) a JSON report with tokens/s, so numbers line up with
``slugai run``'s report.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import numpy as np

from ..backend import Backend
from ..prompts import load_prompts

log = logging.getLogger(__name__)


def run_single(backend: Backend, model: str, prompts: list[str], max_tokens: int, batch: int) -> tuple[list[list[int]], dict]:
    """Greedy tokens for every prompt, ``batch`` prompts at a time, and timing stats."""
    info = backend.model_info(model)
    tok = backend.tokenizer(model)
    eos = set(info.eos_token_ids)
    t0 = time.perf_counter()
    stage = backend.load_stage(model, (0, info.spec.num_layers), first=True, last=True)
    load_s = time.perf_counter() - t0
    outs: list[list[int]] = []
    generated = 0
    t0 = time.perf_counter()
    for slot, start in enumerate(range(0, len(prompts), batch)):
        # as the ring does: prefill prompt[:-1] right-padded, then decode from each last token
        enc = [tok.encode(p) or [info.pad_token_id] for p in prompts[start : start + batch]]
        last = np.array([[e[-1]] for e in enc], dtype=np.int32)
        T = max(len(e) for e in enc) - 1
        if T > 0:
            body = np.full((len(enc), T), info.pad_token_id, dtype=np.int32)
            for i, e in enumerate(enc):
                body[i, : len(e) - 1] = e[:-1]
            stage.run(slot, "prefill", body, pad_lens=[T - (len(e) - 1) for e in enc])
            nxt = stage.run(slot, "decode", last)
        else:
            nxt = stage.run(slot, "prefill", last, pad_lens=[0] * len(enc))
        gen: list[list[int]] = [[] for _ in enc]
        done = [False] * len(enc)
        while True:
            for i, t in enumerate(nxt.tolist()):
                if not done[i]:
                    gen[i].append(int(t))
                    generated += 1
                    done[i] = t in eos or len(gen[i]) >= max_tokens
            if all(done):
                break
            nxt = stage.run(slot, "decode", nxt.reshape(-1, 1).astype(np.int32))
        stage.reset_slot(slot)
        outs.extend(gen)
        log.info("%d/%d prompts, %.1f tok/s", len(outs), len(prompts), generated / (time.perf_counter() - t0))
    gen_s = time.perf_counter() - t0
    stage.unload()
    return outs, {"load_seconds": load_s, "decode_seconds": gen_s, "tokens_generated": generated, "tok_s": generated / max(gen_s, 1e-9), "batch": batch}


def main(args) -> int:
    backend = Backend()
    prompts = load_prompts(args.prompts, limit=args.limit)
    if not prompts:
        print("no prompts")
        return 2
    log.info("bench: %s, %d prompts, batch %d, max_tokens %d", args.model, len(prompts), args.batch, args.max_tokens)
    outs, stats = run_single(backend, args.model, prompts, args.max_tokens, args.batch)
    tok = backend.tokenizer(args.model)
    out_path = Path(args.report).with_suffix(".jsonl") if args.report else Path("bench.jsonl")
    with out_path.open("w") as fh:
        for i, (p, g) in enumerate(zip(prompts, outs)):
            fh.write(json.dumps({"id": i, "prompt": p, "completion": tok.decode(g), "tokens": len(g), "token_ids": g}, ensure_ascii=False) + "\n")
    stats.update(model=args.model, prompts=len(prompts), max_tokens=args.max_tokens, output=str(out_path), tok_h=stats["tok_s"] * 3600)
    if args.report:
        Path(args.report).write_text(json.dumps(stats, indent=2))
    print(f"bench: {stats['tokens_generated']} tokens in {stats['decode_seconds']:.1f}s = {stats['tok_s']:.1f} tok/s ({stats['tok_h'] / 1000:.0f}k tok/h); wrote {out_path}")
    return 0
