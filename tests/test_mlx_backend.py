"""MLX backend against mlx-lm's own generator on a tiny on-disk Qwen3 model.

Runs wherever ``mlx`` imports (Apple silicon, or the Linux CPU build).
"""

import asyncio
import json
from pathlib import Path

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
pytest.importorskip("mlx_lm")

from slugai.net.discovery import HostAddr  # noqa: E402
from slugai.ring.host import HostDaemon  # noqa: E402
from slugai.ring.server import Server  # noqa: E402
from tests.tiny_model import make_tiny_qwen3, make_tiny_qwen3_next  # noqa: E402

PROMPTS = ["hello world", "the quick brown fox", "zebra", "a", "once upon a time there was", "x y z"]


@pytest.fixture(scope="module", params=["fp16", "q4", "next"])
def tiny_model(request, tmp_path_factory):
    path = tmp_path_factory.mktemp("tiny") / request.param
    if request.param == "next":
        # hybrid linear-attention + MoE model, like the Qwen3-Next target
        return make_tiny_qwen3_next(path, num_layers=8)
    return make_tiny_qwen3(path, num_layers=6, quantize=request.param == "q4")


def mlx_lm_reference(model_path, prompts, max_tokens):
    """Greedy token ids from mlx-lm's own generator (EOS stripped)."""
    from mlx_lm import load
    from mlx_lm.generate import stream_generate
    from mlx_lm.sample_utils import make_sampler

    model, tok = load(str(model_path))
    outs = []
    for p in prompts:
        ids = [r.token for r in stream_generate(model, tok, prompt=p, max_tokens=max_tokens, sampler=make_sampler(temp=0.0))]
        outs.append([t for t in ids if t not in tok.eos_token_ids])
    return outs


def test_model_info_and_stage_shapes(tiny_model):
    from slugai.backends.mlx import MLXBackend

    be = MLXBackend(use_chat_template=False)
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
    from slugai.backends.mlx import MLXBackend

    be = MLXBackend(use_chat_template=False)
    sp = be.model_info(str(tiny_model)).spec
    assert (sp.state_bytes_per_seq > 0) == (tiny_model.name == "next")  # linear attention keeps state
    L, b, T = sp.num_layers, 3, 5
    stage = be.load_stage(str(tiny_model), (0, L), first=True, last=True)
    stage.run(0, "prefill", np.ones((b, T), dtype=np.int32), pad_lens=[0] * b)
    got = sum(a.nbytes for c in stage.caches[0] for _, a in tree_flatten(c.state) if isinstance(a, mx.array))
    want = L * b * sp.cache_bytes_per_seq(T)
    assert abs(got - want) <= 0.05 * want, (got, want)


def test_profile(tiny_model):
    from slugai.backends.mlx import MLXBackend

    be = MLXBackend(use_chat_template=False)
    prof = be.profile(str(tiny_model), batch_sizes=(1, 2), probe_layers=1, context=8)
    assert set(prof["layer_time"]) == {1, 2}
    assert all(v > 0 for v in prof["layer_time"].values())


@pytest.mark.asyncio
async def test_ring_matches_mlx_lm_generate(tiny_model, tmp_path):
    from slugai.backends.mlx import MLXBackend

    max_tokens = 8
    ref = mlx_lm_reference(tiny_model, PROMPTS, max_tokens)
    hosts = []
    for i in range(3):
        d = HostDaemon(f"mlx{i}", 0, MLXBackend(use_chat_template=False), advertise=False, ip="127.0.0.1")
        await d.start()
        hosts.append(d)
    try:
        addrs = [HostAddr(d.name, "127.0.0.1", d.port) for d in hosts]
        server = Server(
            model=str(tiny_model), prompts=PROMPTS, job="tmlx", hosts=addrs, min_hosts=3, max_tokens=max_tokens,
            micro_batch=4, num_micro_batches=2, rebalance=False, output_path=str(tmp_path / "o.jsonl"),
            profile_batch_sizes=(1, 4),
        )
        report = await asyncio.wait_for(server.run(), 300)
        assert server.state.error is None
        assert report["sequences_done"] == len(PROMPTS)
        recs = [json.loads(l) for l in (tmp_path / "o.jsonl").read_text().splitlines()]
        got = [[t for t in r["token_ids"] if t != 2] for r in recs]
        assert got == ref
        assert any(len(g) > 0 for g in got)
    finally:
        for d in hosts:
            await d.stop()


def test_bench_baselines_match_mlx_lm(tiny_model):
    from slugai.tools.bench import run_single, run_streaming
    from slugai.backends.mlx import MLXBackend

    ref = mlx_lm_reference(tiny_model, PROMPTS, 6)
    be = MLXBackend(use_chat_template=False)
    single, _ = run_single(be, str(tiny_model), PROMPTS, max_tokens=6, batch=4)
    assert [[t for t in g if t != 2] for g in single] == ref
    streaming, stats = run_streaming(be, str(tiny_model), PROMPTS[:3], max_tokens=4, batch=3)
    assert [[t for t in g if t != 2] for g in streaming] == [g[:4] for g in ref[:3]]
    assert stats["layer_loads"] > 0
