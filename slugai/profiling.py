"""``slugai profile``: profile this Mac the way the server would.

Prints seconds per layer per decode step for several micro-batch sizes, the
prefill cost, and free memory. The host daemon runs the same measurement on
the server's request (``profile`` message) and link probes are measured
between hosts (``probe_link``); see :mod:`slugai.host`.
"""

from __future__ import annotations

import json
import platform
import time

from .models import apply_models_dir


def profile_cli(args) -> int:
    apply_models_dir(args.models_dir)
    from .backend import get_backend

    backend = get_backend(args.backend)
    t0 = time.perf_counter()
    info = backend.model_info(args.model)
    prof = backend.profile(args.model)
    elapsed = time.perf_counter() - t0
    spec = info.spec
    result = {
        "host": platform.node(),
        "backend": backend.name,
        "model": args.model,
        "num_layers": spec.num_layers,
        "hidden_size": spec.hidden_size,
        "layer_bytes": spec.layer_bytes,
        "kv_bytes_per_token": spec.kv_bytes_per_token,
        "embed_bytes": spec.embed_bytes,
        "head_bytes": spec.head_bytes,
        "layer_time": {str(k): v for k, v in prof["layer_time"].items()},
        "prefill_layer_time_per_token": prof["prefill_layer_time_per_token"],
        "free_mem": prof["free_mem"],
        "total_mem": backend.total_memory(),
        "profile_seconds": elapsed,
    }
    if args.json:
        print(json.dumps(result, indent=2))
        return 0
    print(f"{result['host']} · {backend.name} · {args.model}")
    print(f"  {spec.num_layers} layers · hidden {spec.hidden_size} · {spec.layer_bytes / 1e6:.0f} MB/layer · KV {spec.kv_bytes_per_token / 1024:.1f} KB/layer/token")
    print(f"  memory: {prof['free_mem'] / 1e9:.1f} GB free of {backend.total_memory() / 1e9:.1f} GB")
    print("  micro-batch   ms/layer/step   tokens/s/layer")
    for b, t in sorted(prof["layer_time"].items()):
        print(f"  {b:>11}   {t * 1e3:>13.2f}   {b / t:>14.0f}")
    whole = max(prof["layer_time"].items(), key=lambda kv: kv[0] / kv[1])
    print(f"  best: micro-batch {whole[0]} -> ~{whole[0] / (whole[1] * spec.num_layers):.1f} tok/s if this Mac held all {spec.num_layers} layers")
    print(f"  prefill: {prof['prefill_layer_time_per_token'] * 1e6:.1f} us/layer/token")
    print(f"  ({elapsed:.1f}s)")
    return 0
