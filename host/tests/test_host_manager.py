import asyncio

import mlx.core as mx
import numpy as np

import wire
from host.host_manager import HostManager
from host.model import Shard
from test_model import PROMPTS, generate, num_layers

BATCHES = {0: PROMPTS, 1: [[7, 7, 1], [30, 4, 4, 9, 2]]}  # two micro-batches in flight at once
STEPS = 4


async def start_hosts(n: int) -> list[tuple[HostManager, str]]:
    hosts = []
    for _ in range(n):
        host = HostManager(stats_every=0.05)
        server = await host.serve(0)
        hosts.append((host, f"127.0.0.1:{server.sockets[0].getsockname()[1]}"))
    return hosts


async def recv(reader) -> tuple[dict, np.ndarray | None]:
    """The next message that isn't stats."""
    while True:
        msg, array = await wire.recv(reader)
        if msg["type"] != "stats":
            return msg, array


async def ring(model: str, layers: list[tuple]) -> dict[int, list[list[int]]]:
    """Generate BATCHES through one host per (start, end, resident), acting as the controller; returns each row's tokens."""
    addresses = [a for _, a in await start_hosts(len(layers))]
    conns = [await wire.connect(a) for a in addresses]
    for (start, end, resident), (reader, writer), nxt in zip(layers, conns, addresses[1:] + [None]):
        await wire.send(writer, {"type": "load", "model": model, "start": start, "end": end, "resident": resident, "next": nxt})
        assert (await recv(reader))[0]["type"] == "loaded"
    (_, first), (last, _) = conns[0], conns[-1]
    for slot, prompts in BATCHES.items():  # prefill all but each prompt's last token, then decode the last
        lengths = [len(p) - 1 for p in prompts]
        x = np.array([p[:-1] + [0] * (max(lengths) - n) for p, n in zip(prompts, lengths)], np.int32)
        await wire.send(first, {"type": "forward", "slot": slot, "lengths": lengths}, x)
        await wire.send(first, {"type": "forward", "slot": slot}, np.array([[p[-1]] for p in prompts], np.int32))
    tokens = {slot: [] for slot in BATCHES}
    while any(len(t) < STEPS for t in tokens.values()):
        msg, out = await recv(last)
        assert msg["type"] == "tokens", msg
        if "lengths" in msg:
            continue  # the prefill's: they follow prompts cut short, so they aren't real next tokens
        tokens[msg["slot"]].append(out.tolist())
        if len(tokens[msg["slot"]]) < STEPS:
            await wire.send(first, {"type": "forward", "slot": msg["slot"]}, out[:, None])
    return {slot: [list(row) for row in zip(*t)] for slot, t in tokens.items()}


def test_ring_matches_one_shard(tiny_model):
    """Three hosts, the middle one streaming its layers, two micro-batches in flight: one shard's tokens."""
    n = num_layers(tiny_model)
    got = asyncio.run(ring(tiny_model, [(0, 2, None), (2, n - 1, 0), (n - 1, n, None)]))
    whole = Shard(tiny_model, 0, n)
    for slot, prompts in BATCHES.items():
        assert got[slot] == mx.argmax(mx.stack(generate([whole], prompts, STEPS), 1), -1).tolist()


def test_requests(tiny_model):
    async def run():
        ((host, address),) = await start_hosts(1)
        reader, writer = await wire.connect(address)

        async def ask(msg: dict, array: np.ndarray | None = None) -> dict:
            await wire.send(writer, msg, array)
            return (await recv(reader))[0]

        report = await ask({"type": "profile", "model": tiny_model, "context": 8, "batch_sizes": [2, 1]})
        assert report["type"] == "profile" and list(report["layer_time"]) == ["1", "2"]
        assert (await ask({"type": "load", "model": tiny_model, "start": 0, "end": 2}))["type"] == "error"  # no next host
        assert (await ask({"type": "load", "model": tiny_model, "start": 0, "end": num_layers(tiny_model)}))["type"] == "loaded"
        error = await ask({"type": "forward", "slot": 5}, np.zeros((1, 1), np.int32))  # never prefilled
        assert error["type"] == "error" and error["slot"] == 5
        stats, _ = await wire.recv(reader)
        assert stats["type"] == "stats" and stats["memory"] > 0
        assert (await ask({"type": "unload"}))["type"] == "unloaded" and host.shard is None
        assert (await ask({"type": "load", "model": tiny_model, "start": 0, "end": num_layers(tiny_model)}))["type"] == "loaded"
        writer.close()  # the controller leaving unloads the shard
        while host.shard is not None:
            await asyncio.sleep(0.01)

    asyncio.run(asyncio.wait_for(run(), 30))
