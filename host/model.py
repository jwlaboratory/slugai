"""A contiguous range of a model's layers, run with a KV cache.

A shard holds layers [start, end) of a model, plus the embedding if it's the
first shard and the final norm and lm_head if it's the last; it never reads
the other layers' weights. What goes in and out of ``forward``:

* first shard in: token ids (b, T); other shards in: hidden states (b, T, D)
  in the model's dtype
* last shard out: logits for each row's last token (b, vocab); other shards
  out: hidden states

Prompts of different lengths are right-padded, as mlx-lm's batch generator
does. Prefill all but each prompt's last token, then run the last tokens as
the first decode step, so the rows line up from then on.

Streaming: only the first ``resident`` layers stay in memory. The rest stay on
disk: each is read while the layer before it computes and dropped after use,
so a Mac can hold more layers than fit in its memory, at the cost of reading
them on every step.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_map, tree_unflatten
from mlx_lm.models.cache import make_prompt_cache
from mlx_lm.utils import _download, load_config, load_model, load_tokenizer


def model_path(model: str) -> Path:
    """A local model directory, or a Hugging Face repo id (downloaded on first use)."""
    return _download(str(model))


class Shard:
    def __init__(self, model: str, start: int, end: int, resident: int | None = None):
        path = model_path(model)
        self.model, _ = load_model(path, lazy=True)  # lazy: weights are read only when evaluated
        inner = self.model.model
        n = len(inner.layers)
        if not 0 <= start < end <= n:
            raise ValueError(f"layers [{start}, {end}) are outside the model's {n}")
        self.first, self.last = start == 0, end == n

        layers = inner.layers[start:end]
        inner.layers = layers
        if hasattr(inner, "fa_idx"):  # hybrid models (Qwen3-Next) index their first attention and linear layers
            inner.fa_idx = next((i for i, l in enumerate(layers) if not l.is_linear), None)
            inner.ssm_idx = next((i for i, l in enumerate(layers) if l.is_linear), None)
        resident = len(layers) if resident is None else resident
        if resident < len(layers):
            _stream(self.model, path, layers[resident:], start + resident)

        tied = not hasattr(self.model, "lm_head")  # tied models use the embedding as the head
        head = inner.embed_tokens if tied else self.model.lm_head
        self._head = (head.as_linear if tied else head) if self.last else None
        to_load = [inner.parameters()] + ([head.parameters()] if self.last else [])
        if not self.first:
            inner.embed_tokens = _Identity()  # hidden states come in...
        if not self.last:
            inner.norm = _Identity()  # ...and go out un-normalised
        mx.eval(to_load)

    def new_cache(self, batch: int) -> list:
        """An empty cache for ``batch`` sequences."""
        return [c.merge([c] * batch) for c in make_prompt_cache(self.model)]

    def forward(self, x: mx.array, cache: list, lengths: list[int] | None = None) -> mx.array:
        """Run ``x`` through the shard, adding it to ``cache``.

        ``lengths``: the rows of ``x`` are right-padded prompts with this many
        real tokens each (prefill).
        """
        if lengths is not None:
            for c in cache:
                c.prepare(lengths=lengths, right_padding=[x.shape[1] - n for n in lengths])
        h = self.model.model(x, cache=cache)
        if lengths is not None:
            for c in cache:
                c.finalize()
        out = self._head(h[:, -1]) if self.last else h
        mx.eval(out, [c.state for c in cache])
        return out


class _Identity(nn.Module):
    def __call__(self, x):
        return x


def _stream(model, path: Path, layers: list, first: int) -> None:
    """Keep ``layers`` (numbered from ``first``) on disk, reading each one's weights while the one before computes.

    The last streamed layer reads the first, for the next micro-batch.
    """
    index = path / "model.safetensors.index.json"
    weight_map = json.loads(index.read_text())["weight_map"] if index.exists() else None
    ahead: dict[int, dict] = {}  # layer number -> its weights, being read

    def read(i: int) -> dict:
        prefix = f"model.layers.{i}."
        files = sorted({path / f for k, f in weight_map.items() if k.startswith(prefix)}) if weight_map else path.glob("*.safetensors")
        w = {k: v for f in files for k, v in mx.load(str(f)).items() if k.startswith(prefix)}  # unread: mx.load parses headers
        if hasattr(model, "sanitize"):
            w = model.sanitize(w)
        w = tree_unflatten([(k[len(prefix):], v) for k, v in w.items()])
        mx.async_eval(w)
        return w

    def stream(layer: nn.Module, i: int, after: int) -> None:
        base = type(layer)
        empty = tree_map(lambda a: mx.zeros((0,), a.dtype), layer.parameters())  # what it holds between uses

        def __call__(self, *args, **kwargs):
            self.update(ahead.pop(i) if i in ahead else read(i))
            out = base.__call__(self, *args, **kwargs)
            cache = kwargs.get("cache", args[2] if len(args) > 2 else None)
            done = [out, cache.state if cache is not None else []]
            mx.async_eval(done)  # queue this layer...
            ahead[after] = read(after)  # ...then read the next one while it computes
            mx.eval(done)
            self.update(empty)  # drop the weights, freeing the memory
            return out

        layer.__class__ = type(f"Streamed{base.__name__}", (base,), {"__call__": __call__})
        layer.update(empty)

    numbers = list(range(first, first + len(layers)))
    for layer, i, after in zip(layers, numbers, numbers[1:] + numbers[:1]):
        stream(layer, i, after)


def main() -> None:
    p = argparse.ArgumentParser(description="Try a split on this Mac: generate greedily through a chain of shards.")
    p.add_argument("model", help="Hugging Face repo id or local model directory")
    p.add_argument("--layers", help="each shard's layers, e.g. 0:18,18:36 (default: one shard with every layer)")
    p.add_argument("--resident", type=int, help="layers each shard keeps in memory; the rest stream from disk (default: all)")
    p.add_argument("--prompt", default="Name three rivers in Africa.")
    p.add_argument("--tokens", type=int, default=32, help="tokens to generate")
    args = p.parse_args()

    path = model_path(args.model)
    n = load_config(path)["num_hidden_layers"]
    ranges = [tuple(map(int, r.split(":"))) for r in args.layers.split(",")] if args.layers else [(0, n)]
    if [a for a, _ in ranges] != [0] + [b for _, b in ranges[:-1]] or ranges[-1][1] != n:
        p.error(f"--layers must cover 0 to {n} in order, e.g. 0:{n // 2},{n // 2}:{n}")
    t0 = time.perf_counter()
    shards = [Shard(args.model, a, b, args.resident) for a, b in ranges]
    print(f"loaded {args.layers or f'0:{n}'} in {time.perf_counter() - t0:.1f}s, {mx.get_active_memory() / 1e9:.2f} GB")

    tok, caches = load_tokenizer(path), [s.new_cache(1) for s in shards]

    def step(x: mx.array, lengths: list[int] | None = None) -> mx.array:
        for s, c in zip(shards, caches):
            x = s.forward(x, c, lengths)
        return mx.argmax(x, -1)[:, None]

    prompt = tok.encode(args.prompt)
    out = [step(mx.array([prompt]), [len(prompt)])]
    t0 = time.perf_counter()
    while len(out) < args.tokens:
        out.append(step(out[-1]))
    print(tok.decode(mx.concatenate(out, 1)[0].tolist()))
    print(f"{(time.perf_counter() - t0) / max(1, len(out) - 1) * 1e3:.1f} ms per token, peak memory {mx.get_peak_memory() / 1e9:.2f} GB")


if __name__ == "__main__":
    main()
