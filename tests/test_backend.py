"""The backend on tiny on-disk models: shapes, cache sizes, profiling, and agreement with mlx-lm.

Runs wherever ``mlx`` imports (Apple silicon, or the Linux CPU build).
"""

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
pytest.importorskip("mlx_lm")

from mlx.utils import tree_flatten  # noqa: E402
from slugai.backend import Backend  # noqa: E402
from tests.helpers import mlx_lm_reference  # noqa: E402

PROMPTS = ["hello world", "the quick brown fox", "zebra", "a", "once upon a time there was", "x y z"]


def test_model_info_and_stage_shapes(tiny_model):
    be = Backend(use_chat_template=False)
    info = be.model_info(str(tiny_model))
    assert info.spec.num_layers in (6, 8) and info.spec.hidden_size == 32
    assert info.spec.layer_bytes > 0 and info.spec.embed_bytes > 0 and info.spec.head_bytes > 0
    assert info.eos_token_ids == [2]
    tok = be.tokenizer(str(tiny_model))
    ids = tok.encode("hello")
    assert tok.decode(ids) == "hello"

    s1 = be.load_stage(str(tiny_model), (0, 2), first=True, last=False)
    s2 = be.load_stage(str(tiny_model), (2, 4), first=False, last=False)
    L = info.spec.num_layers
    s3 = be.load_stage(str(tiny_model), (4, L), first=False, last=True)
    assert len(s1.inner.layers) == 2 and len(s3.inner.layers) == L - 4
    x = np.array([[1, 5, 6, 7], [1, 5, 0, 0]], dtype=np.int32)  # right padded
    h = s1.run(0, "prefill", x, pad_lens=[0, 2])
    # hidden states leave a stage in the model's dtype (the tiny Qwen3-Next is float32)
    assert h.shape == (2, 4, 32) and h.dtype == (np.float32 if tiny_model.name == "next" else np.float16)
    h = s2.run(0, "prefill", h, pad_lens=[0, 2])
    t = s3.run(0, "prefill", h, pad_lens=[0, 2])
    assert t.shape == (2,) and t.dtype == np.int32
    # decode step keeps working on the same slot
    h = s1.run(0, "decode", t.reshape(2, 1))
    h = s2.run(0, "decode", h)
    t2 = s3.run(0, "decode", h)
    assert t2.shape == (2,)


def test_cache_sizes_match_allocation(tiny_model):
    """The spec's KV + state sizes (placement's memory model) match what a stage really allocates."""
    from mlx.utils import tree_flatten

    be = Backend(use_chat_template=False)
    sp = be.model_info(str(tiny_model)).spec
    assert (sp.state_bytes_per_seq > 0) == (tiny_model.name == "next")  # linear attention keeps state
    L, b, T = sp.num_layers, 3, 5
    stage = be.load_stage(str(tiny_model), (0, L), first=True, last=True)
    stage.run(0, "prefill", np.ones((b, T), dtype=np.int32), pad_lens=[0] * b)
    got = sum(a.nbytes for c in stage.caches[0] for _, a in tree_flatten(c.state) if isinstance(a, mx.array))
    want = L * b * sp.cache_bytes_per_seq(T)
    assert abs(got - want) <= 0.05 * want, (got, want)


def test_profile(tiny_model):
    be = Backend(use_chat_template=False)
    prof = be.profile(str(tiny_model), batch_sizes=(1, 2, 4), probe_layers=1, context=40)
    assert set(prof["layer_time"]) == {1, 2, 4}
    times = [prof["layer_time"][b] for b in (1, 2, 4)]
    assert all(v > 0 for v in times) and times == sorted(times)


def test_profile_noise_is_smoothed():
    from slugai.backend import _non_decreasing

    assert _non_decreasing([1.0, 2.0, 3.0]) == [1.0, 2.0, 3.0]
    # a real MacBook Air profile (ms/layer at batch 1, 8, 32, 64): batch 8 measured slower than 32
    out = _non_decreasing([1.2, 17.5, 13.1, 22.8])
    assert out == sorted(out) and out[0] == 1.2 and out[-1] == 22.8
    assert out[1] == out[2] == (17.5 + 13.1) / 2


def test_bench_matches_mlx_lm(tiny_model):
    from slugai.tools.bench import run_single

    ref = mlx_lm_reference(tiny_model, PROMPTS, 6)
    single, stats = run_single(Backend(use_chat_template=False), str(tiny_model), PROMPTS, max_tokens=6, batch=4)
    assert [[t for t in g if t != 2] for g in single] == ref
    assert stats["tokens_generated"] == sum(len(g) for g in single)


def test_streamed_layers_match_resident(tiny_model):
    """Layers streamed from disk at every step compute exactly what resident layers do."""
    be = Backend(use_chat_template=False)
    L = be.model_info(str(tiny_model)).spec.num_layers
    x = np.array([[1, 5, 6, 7], [1, 5, 0, 0]], dtype=np.int32)  # right padded
    outs = []
    for resident in (None, 1):  # all in memory, then all but one streamed
        stage = be.load_stage(str(tiny_model), (0, L), first=True, last=True, resident=resident)
        assert stage.streamed == (0 if resident is None else L - 1)
        t = stage.run(0, "prefill", x, pad_lens=[0, 2])
        seq = [t]
        for _ in range(4):
            t = stage.run(0, "decode", t.reshape(2, 1))
            seq.append(t)
        outs.append(np.stack(seq))
        if resident is not None:  # between steps, streamed layers hold no weights (the next one's are read ahead)
            assert all(v.size == 0 for l in stage.inner.layers[1:] for _, v in tree_flatten(l.parameters()))
    np.testing.assert_array_equal(outs[0], outs[1])
