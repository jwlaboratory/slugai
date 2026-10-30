"""Shared test helpers (MLX test modules only)."""

import time

from oldslop.slugai.backend import Backend
from oldslop.slugai.net.discovery import HostAddr
from oldslop.slugai.ring.host import HostDaemon

PROMPTS = [
    "the quick brown fox",
    "hi",
    "a much longer prompt than the others, indeed",
    "zebra",
    "once upon a time",
    "numbers 1 2 3 4 5",
    "short",
    "Why is the sky blue? Tell me in one sentence.",
    "x",
    "The end.",
    "lorem ipsum dolor sit amet",
]


class SlowBackend(Backend):
    """The real backend plus a sleep of ``ms`` per layer per step.

    Puts step times under the test's control: raising ``ms`` mid-job throttles
    the host, and it stays slow across reloads because every stage reads it.
    """

    def __init__(self, ms: float = 0.0):
        super().__init__()
        self.ms = ms

    def load_stage(self, *args, **kwargs):
        stage = super().load_stage(*args, **kwargs)
        run = stage.run

        def slow_run(*a, **kw):
            out = run(*a, **kw)
            time.sleep(self.ms / 1000 * len(stage.inner.layers))
            return out

        stage.run = slow_run
        return stage


async def start_hosts(n, backend_factory=lambda i: Backend(), host_class=HostDaemon):
    hosts = []
    for i in range(n):
        d = host_class(f"mac{i}", 0, backend_factory(i), advertise=False, ip="127.0.0.1")
        await d.start()
        hosts.append(d)
    return hosts


def addrs(hosts):
    return [HostAddr(d.name, "127.0.0.1", d.port) for d in hosts]


def one_machine(model_path, prompts, max_tokens, batch):
    """Greedy tokens with the whole model on one stage, in the same batches the ring uses."""
    from oldslop.slugai.tools.bench import run_single

    outs, _ = run_single(Backend(), str(model_path), prompts, max_tokens=max_tokens, batch=batch)
    return outs


def mlx_lm_reference(model_path, prompts, max_tokens):
    """Greedy token ids from mlx-lm's own generator, one prompt at a time (EOS stripped)."""
    from mlx_lm import load
    from mlx_lm.generate import stream_generate
    from mlx_lm.sample_utils import make_sampler

    model, tok = load(str(model_path))
    outs = []
    for p in prompts:
        ids = [r.token for r in stream_generate(model, tok, prompt=p, max_tokens=max_tokens, sampler=make_sampler(temp=0.0))]
        outs.append([t for t in ids if t not in tok.eos_token_ids])
    return outs
