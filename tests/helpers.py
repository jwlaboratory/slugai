"""Shared test helpers."""

import numpy as np


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
