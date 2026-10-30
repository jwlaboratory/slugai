import mlx.core as mx
import pytest
from mlx_lm.models.cache import make_prompt_cache
from mlx_lm.utils import load_model

from host.model import Shard, model_path

PROMPTS = [[5, 9, 3, 7, 1, 12], [4, 2], [8, 8, 30, 2, 2, 2, 2, 2, 11]]


def num_layers(model: str) -> int:
    return len(load_model(model_path(model), lazy=True)[0].model.layers)


def generate(shards: list[Shard], steps: int = 4) -> list[mx.array]:
    """Each step's logits for PROMPTS as one right-padded micro-batch, decoded greedily."""
    caches = [s.new_cache(len(PROMPTS)) for s in shards]
    lengths = [len(p) - 1 for p in PROMPTS]
    x = mx.array([p[:-1] + [0] * (max(lengths) - n) for p, n in zip(PROMPTS, lengths)])
    for s, c in zip(shards, caches):
        x = s.forward(x, c, lengths)
    x, logits = mx.array([[p[-1]] for p in PROMPTS]), []
    for _ in range(steps):
        for s, c in zip(shards, caches):
            x = s.forward(x, c)
        logits.append(x)
        x = mx.argmax(x, -1)[:, None]
    return logits


def same(a: list[mx.array], b: list[mx.array]) -> bool:
    return all(mx.array_equal(x, y).item() for x, y in zip(a, b))


def test_any_split_gives_the_same_logits(tiny_model):
    n = num_layers(tiny_model)
    whole = generate([Shard(tiny_model, 0, n)])
    assert same(whole, generate([Shard(tiny_model, 0, 2), Shard(tiny_model, 2, n - 1), Shard(tiny_model, n - 1, n)]))


def test_streamed_layers_give_the_same_logits(tiny_model):
    n = num_layers(tiny_model)
    whole = generate([Shard(tiny_model, 0, n)])
    assert same(whole, generate([Shard(tiny_model, 0, 3, resident=1), Shard(tiny_model, 3, n, resident=0)]))


def test_streamed_layers_stay_on_disk(tiny_model):
    n = num_layers(tiny_model)
    resident = Shard(tiny_model, 0, n)
    held = mx.get_active_memory()
    del resident
    streamed = Shard(tiny_model, 0, n, resident=0)
    assert mx.get_active_memory() < held
    del streamed


def test_tokens_match_mlx_lm(tiny_model):
    """The padded batch picks the same tokens as mlx-lm running each prompt alone."""
    tokens = mx.argmax(mx.stack(generate([Shard(tiny_model, 0, num_layers(tiny_model))]), 1), -1).tolist()
    model, _ = load_model(model_path(tiny_model))
    for prompt, want in zip(PROMPTS, tokens):
        cache, x, got = make_prompt_cache(model), mx.array([prompt]), []
        for _ in want:
            x = mx.argmax(model(x, cache=cache)[:, -1], -1)[:, None]
            got.append(x.item())
        assert got == want


def test_layers_outside_the_model(tiny_model):
    with pytest.raises(ValueError):
        Shard(tiny_model, 2, num_layers(tiny_model) + 1)
