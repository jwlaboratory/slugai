"""Measure this Mac for the controller: time per layer, free memory, disk read speed."""

from __future__ import annotations

import argparse
import fcntl
import json
import statistics
import time
from dataclasses import asdict, dataclass
from itertools import accumulate
from pathlib import Path

import mlx.core as mx

from .model import Shard, model_path

BATCH_SIZES = (1, 8, 16, 32, 64, 128)
MAX_CONTEXT = 1024  # longer caches take longer to fill than they're worth


@dataclass
class Profile:
    layer_time: dict[int, float]  # micro-batch size -> seconds per layer per decode step
    free_memory: int  # bytes this Mac can give to weights and caches
    disk_speed: float  # bytes/s reading the model's files


def profile(model: str, context: int = 64, batch_sizes: tuple[int, ...] = BATCH_SIZES) -> Profile:
    """Profile this Mac on ``model``, with ``context`` tokens cached per sequence."""
    disk = disk_speed(model_path(model))  # before the timing pulls the first layers into the file cache
    free = free_memory()
    times = layer_times(model, min(context, MAX_CONTEXT), batch_sizes)
    mx.clear_cache()  # give back the probe layers' memory
    return Profile(times, free, disk)


def free_memory() -> int:
    """What macOS lets the GPU use, minus what's in use and 1 GB of headroom."""
    return max(0, mx.device_info()["max_recommended_working_set_size"] - mx.get_active_memory() - (1 << 30))


def disk_speed(path: Path, limit: int = 1 << 30) -> float:
    """Bytes/s reading up to ``limit`` bytes of the model's weight files, bypassing the file cache."""
    buf, done = bytearray(16 << 20), 0
    t0 = time.perf_counter()
    for file in sorted(path.glob("*.safetensors")):
        with open(file, "rb", buffering=0) as f:
            fcntl.fcntl(f, fcntl.F_NOCACHE, 1)
            while done < limit and (n := f.readinto(buf)):
                done += n
    return done / (time.perf_counter() - t0)


def layer_times(model: str, context: int, batch_sizes: tuple[int, ...], reps: int = 8) -> dict[int, float]:
    """Seconds per layer per decode step at each batch size.

    Each is a layer's marginal cost: decode steps through the model's first 6
    and first 2 layers alternate, and the median difference is divided by 4,
    so fixed costs (embedding, launching the step) cancel and clock drift hits
    both alike. Random tokens give MoE layers a realistic mix of experts. A
    layer can't get faster as the batch grows, so each time is at least the
    one before it.
    """
    deep, shallow = Shard(model, 0, 6), Shard(model, 0, 2)
    vocab = deep.model.args.vocab_size

    def step(shard: Shard, cache: list, b: int, tokens: int = 1, prefill: bool = False) -> float:
        x = mx.random.randint(0, vocab, (b, tokens))
        t0 = time.perf_counter()
        shard.forward(x, cache, [tokens] * b if prefill else None)
        return time.perf_counter() - t0

    def filled_cache(shard: Shard, b: int) -> list:
        cache, chunk = shard.new_cache(b), max(1, 8192 // b)  # fill in chunks: no huge attention buffers
        for done in range(0, context, chunk):
            step(shard, cache, b, min(chunk, context - done), prefill=done == 0)
        return cache

    def per_layer(b: int) -> float:
        deep_cache, shallow_cache = filled_cache(deep, b), filled_cache(shallow, b)
        steps = [(step(deep, deep_cache, b), step(shallow, shallow_cache, b)) for _ in range(reps + 2)][2:]  # 2 warm-ups
        whole = statistics.median(d for d, _ in steps) / 6
        return max(statistics.median(d - s for d, s in steps) / 4, whole / 2)  # noise can't take it below half the average

    sizes = sorted(batch_sizes)
    per_layer(sizes[-1])  # wake the GPU up
    return dict(zip(sizes, accumulate(map(per_layer, sizes), max)))


def main() -> None:
    p = argparse.ArgumentParser(description="Profile this Mac on a model and print the report.")
    p.add_argument("model", help="Hugging Face repo id or local model directory")
    p.add_argument("--context", type=int, default=64, help="tokens already cached per sequence (default: 64)")
    p.add_argument("--batch-sizes", default=",".join(map(str, BATCH_SIZES)), help="micro-batch sizes to time")
    args = p.parse_args()
    print(json.dumps(asdict(profile(args.model, args.context, tuple(map(int, args.batch_sizes.split(","))))), indent=1))


if __name__ == "__main__":
    main()
