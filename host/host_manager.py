"""The host process: holds one shard, runs every micro-batch sent to it and passes the result on.

The controller sends requests, each answered on the same connection (a failure
answers ``error {message}``):

* ``profile {model, context, batch_sizes}`` → ``profile {layer_time, free_memory, disk_speed}``
* ``load {model, start, end, resident, next}`` → ``loaded {seconds, memory}``: hold layers
  [start, end) and send outputs to ``next`` (``host:port``); the last shard has no next
* ``unload`` → ``unloaded``

Micro-batches come from the controller (to the first host) or the previous
host as ``forward {slot, lengths, ...}`` + an array: token ids for the first
shard, hidden states for the rest. Each slot has its own cache; ``lengths``
marks a prefill of right-padded prompts, which starts the slot's cache afresh.
The output goes on as the same message to the next host or, from the last
host, as ``tokens {slot, lengths, ...}`` + each row's greedy next token to the
controller. Other fields ride along, so the controller can tell what each
``tokens`` answers.

Every second the host pushes ``stats {busy, steps, memory}`` to the
controller; they double as heartbeats. All MLX work runs on one thread, so the
event loop keeps moving frames while the shard computes, and each output is
sent while the next micro-batch computes.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import socket
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict

import mlx.core as mx
import numpy as np
from mlx_lm.utils import maybe_set_recommended_wired_limit

import wire

from .model import Shard
from .profile import BATCH_SIZES, profile


class HostManager:
    def __init__(self, stats_every: float = 1.0):
        self.stats_every = stats_every
        self.shard: Shard | None = None
        self.caches: dict = {}  # slot -> its cache
        self.controller: asyncio.StreamWriter | None = None
        self.next: asyncio.StreamWriter | None = None
        self.work: asyncio.Queue = asyncio.Queue()  # forward frames, in arrival order
        self.outbox: asyncio.Queue = asyncio.Queue()  # (writer, message, array), in order
        self.mlx = ThreadPoolExecutor(1)
        self.busy, self.steps = 0.0, 0  # since the last stats push

    async def serve(self, port: int) -> asyncio.Server:
        self.tasks = [asyncio.create_task(t) for t in (self.compute(), self.send(), self.push_stats())]
        return await asyncio.start_server(self.connection, "0.0.0.0", port)

    def run(self, f, *args) -> asyncio.Future:
        """Run ``f`` on the MLX thread."""
        return asyncio.get_running_loop().run_in_executor(self.mlx, f, *args)

    async def connection(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            while True:
                msg, array = await wire.recv(reader)
                if msg["type"] == "forward":
                    self.work.put_nowait((msg, array))
                else:
                    self.controller = writer
                    await wire.send(writer, await self.request(msg))
        except (asyncio.IncompleteReadError, ConnectionError):
            pass  # closed
        finally:
            if writer is self.controller:  # the controller left: free the memory for the next one
                self.controller = None
                await self.unload()
            writer.close()

    async def request(self, msg: dict) -> dict:
        try:
            if msg["type"] == "profile":
                p = await self.run(profile, msg["model"], msg.get("context", 64), tuple(msg.get("batch_sizes", BATCH_SIZES)))
                return {"type": "profile", **asdict(p)}
            if msg["type"] == "load":
                return {"type": "loaded", **await self.load(msg)}
            if msg["type"] == "unload":
                await self.unload()
                return {"type": "unloaded"}
            raise ValueError(f"unknown request {msg['type']!r}")
        except Exception as e:
            return {"type": "error", "message": f"{type(e).__name__}: {e}"}

    async def load(self, msg: dict) -> dict:
        await self.unload()
        t0 = time.perf_counter()
        shard = await self.run(Shard, msg["model"], msg["start"], msg["end"], msg.get("resident"))
        if shard.last != (msg.get("next") is None):
            raise ValueError("every shard but the last needs a next host")
        if msg.get("next"):
            _, self.next = await wire.connect(msg["next"])
        self.shard = shard
        return {"seconds": time.perf_counter() - t0, "memory": mx.get_active_memory()}

    async def unload(self) -> None:
        for q in (self.work, self.outbox):
            while not q.empty():
                q.get_nowait()
        if self.next:
            self.next.close()
        self.shard, self.caches, self.next = None, {}, None
        await self.run(mx.clear_cache)  # after any step still running

    def step(self, msg: dict, x: np.ndarray) -> tuple[np.ndarray, bool]:
        """Run one micro-batch (on the MLX thread); returns the output and whether it's the last shard's tokens."""
        shard, lengths = self.shard, msg.get("lengths")
        if lengths is not None:
            self.caches[msg["slot"]] = shard.new_cache(len(x))
        a = mx.array(x).view(mx.bfloat16) if x.dtype == np.uint16 else mx.array(x)  # bfloat16 travels as its raw bits
        out = shard.forward(a, self.caches[msg["slot"]], lengths)
        if shard.last:
            return np.array(mx.argmax(out, -1).astype(mx.int32)), True
        return np.array(out.view(mx.uint16) if out.dtype == mx.bfloat16 else out), False

    async def compute(self) -> None:
        while True:
            msg, x = await self.work.get()
            t0 = time.perf_counter()
            try:
                out, last = await self.run(self.step, msg, x)
            except Exception as e:
                await self.error(msg, e)
                continue
            self.busy += time.perf_counter() - t0
            self.steps += 1
            to = self.controller if last else self.next
            if to:
                self.outbox.put_nowait((to, {**msg, "type": "tokens"} if last else msg, out))

    async def send(self) -> None:
        while True:
            writer, msg, out = await self.outbox.get()
            try:
                await wire.send(writer, msg, out)
            except ConnectionError as e:
                await self.error(msg, e)

    async def error(self, msg: dict, e: Exception) -> None:
        await self.tell_controller({"type": "error", "slot": msg.get("slot"), "message": f"{type(e).__name__}: {e}"})

    async def push_stats(self) -> None:
        last = time.perf_counter()
        while True:
            await asyncio.sleep(self.stats_every)
            now = time.perf_counter()
            stats = {"type": "stats", "busy": self.busy / (now - last), "steps": self.steps, "memory": mx.get_active_memory()}
            self.busy, self.steps, last = 0.0, 0, now
            await self.tell_controller(stats)

    async def tell_controller(self, msg: dict) -> None:
        try:
            if self.controller:
                await wire.send(self.controller, msg)
        except ConnectionError:
            pass  # it left; connection() cleans up


async def serve(port: int) -> None:
    server = await HostManager().serve(port)
    print(f"host listening on {socket.gethostname()}:{port}")
    await server.serve_forever()


def main() -> None:
    p = argparse.ArgumentParser(description="Host a shard for the controller.")
    p.add_argument("--port", type=int, default=47111)
    args = p.parse_args()
    maybe_set_recommended_wired_limit()  # keep weights in GPU memory
    subprocess.Popen(["caffeinate", "-i", "-s", "-w", str(os.getpid())])  # keep the Mac awake while hosting
    asyncio.run(serve(args.port))


if __name__ == "__main__":
    main()
