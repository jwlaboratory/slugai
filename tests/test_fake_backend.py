import numpy as np

from slugai.backend import FakeBackend, FakeTokenizer


def reference_generate(backend, model_id, prompts, max_tokens):
    """Whole model in one stage; batch of 1 per prompt (no padding at all)."""
    info = backend.model_info(model_id)
    tok = backend.tokenizer(model_id)
    stage = backend.load_stage(model_id, (0, info.spec.num_layers), first=True, last=True)
    outs = []
    for i, p in enumerate(prompts):
        ids = np.array([tok.encode(p)], dtype=np.int32)
        nxt = stage.run(i, "prefill", ids, pad_lens=[0])
        gen = [int(nxt[0])]
        for _ in range(max_tokens - 1):
            if gen[-1] in info.eos_token_ids:
                break
            nxt = stage.run(i, "decode", nxt.reshape(1, 1))
            gen.append(int(nxt[0]))
        outs.append(gen)
    return outs


def pipelined_generate(backend, model_id, prompts, max_tokens, splits):
    """Same prompts as one left-padded micro-batch through several stages."""
    info = backend.model_info(model_id)
    tok = backend.tokenizer(model_id)
    stages = [
        backend.load_stage(model_id, r, first=(i == 0), last=(i == len(splits) - 1)) for i, r in enumerate(splits)
    ]
    enc = [tok.encode(p) for p in prompts]
    T = max(len(e) for e in enc)
    pad_lens = [T - len(e) for e in enc]
    ids = np.full((len(enc), T), info.pad_token_id, dtype=np.int32)
    for i, e in enumerate(enc):
        ids[i, T - len(e):] = e

    def ring(phase, x):
        for s in stages:
            x = s.run(0, phase, x, pad_lens=pad_lens if phase == "prefill" else None)
            if s.last:
                return x
            assert x.dtype == np.float16
        return x

    nxt = ring("prefill", ids)
    gens = [[int(t)] for t in nxt]
    for _ in range(max_tokens - 1):
        nxt = ring("decode", nxt.reshape(-1, 1))
        for g, t in zip(gens, nxt):
            g.append(int(t))
    out = []
    for g in gens:
        cut = []
        for t in g:
            cut.append(t)
            if t in info.eos_token_ids:
                break
        out.append(cut)
    return out


def test_tokenizer_roundtrip():
    tok = FakeTokenizer()
    s = "Hello, world 42!"
    assert tok.decode(tok.encode(s)) == s
    assert tok.encode("x")[0] == FakeTokenizer.BOS


def test_pipelined_matches_reference_with_padding():
    be = FakeBackend(slow_ms=0)
    prompts = ["the quick brown fox", "hi", "a much longer prompt than the others, indeed", "zebra"]
    ref = reference_generate(be, "fake-L6", prompts, 12)
    pipe = pipelined_generate(be, "fake-L6", prompts, 12, [(0, 2), (2, 4), (4, 6)])
    assert pipe == ref


def test_profile_runs():
    be = FakeBackend(slow_ms=0)
    prof = be.profile("fake-L6", batch_sizes=(1, 4), probe_layers=2, context=8)
    assert set(prof["layer_time"]) == {1, 4}
    assert all(v > 0 for v in prof["layer_time"].values())
