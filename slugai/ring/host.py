"""Host daemon (``slugai host``): runs on each MacBook and follows the server's orders.

It profiles itself, loads its layers, runs its pipeline stage and streams
stats. Two kinds of connection arrive on its port:

* **control** (first frame ``hello``) from the server: profile, probe_link,
  load, start, feed, pause, resume, unload, bye;
* **data** (first frame ``peer_hello``) from the previous stage in the ring
  (``hidden`` frames), from the last stage (``tokens`` frames to stage 1), or a
  link probe.

Work on the stage runs on one background thread so the asyncio loop keeps
moving bytes while the model computes.
"""

from __future__ import annotations

import asyncio
import logging
import os
import platform
import socket
import statistics
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .. import __version__
from ..backends import Backend, ModelInfo, Stage, Tokenizer, get_backend
from ..net.discovery import Advertiser
from ..net.protocol import Frame, pack
from ..net.transport import Connection, ConnectionClosed, connect, local_ip, serve

log = logging.getLogger(__name__)


@dataclass
class Work:
    slot: int
    phase: str
    x: np.ndarray
    pad_lens: Optional[list[int]]
    meta: dict


@dataclass
class SlotState:
    """Last-stage bookkeeping for one micro-batch slot."""

    ids: list[int]
    gen: list[list[int]] = field(default_factory=list)
    done: list[bool] = field(default_factory=list)
    started: float = field(default_factory=time.monotonic)
    prompt_tokens: Optional[list[int]] = None


class StatsWindow:
    def __init__(self, window: float = 5.0):
        self.window = window
        self.events: list[tuple[float, float, int, str]] = []  # (t_end, dt, b, phase)
        self.tokens_generated = 0
        self.total_busy = 0.0
        self.started = time.monotonic()

    def record(self, dt: float, b: int, phase: str) -> None:
        self.events.append((time.monotonic(), dt, b, phase))
        self.total_busy += dt
        cutoff = time.monotonic() - self.window
        while self.events and self.events[0][0] < cutoff:
            self.events.pop(0)

    def snapshot(self) -> dict:
        now = time.monotonic()
        cutoff = now - self.window
        recent = [e for e in self.events if e[0] >= cutoff]
        span = min(self.window, now - self.started) or 1e-9
        busy = sum(e[1] for e in recent) / span
        decode = [e[1] for e in recent if e[3] == "decode"]
        prefill = [e[1] for e in recent if e[3] == "prefill"]
        return {
            "busy": min(1.0, busy),
            "step": statistics.fmean(decode) if decode else (statistics.fmean(prefill) if prefill else 0.0),
            "prefill_step": statistics.fmean(prefill) if prefill else 0.0,
            "tokens_generated": self.tokens_generated,
            "total_busy": self.total_busy,
        }


class HostDaemon:
    def __init__(self, name: str, port: int, backend: Backend, advertise: bool = True, ip: Optional[str] = None):
        self.name = name
        self.port = port
        self.backend = backend
        self.advertise = advertise
        self.ip = ip or local_ip()
        self.stage: Optional[Stage] = None
        self.tok: Optional[Tokenizer] = None
        self.info: Optional[ModelInfo] = None
        self.job: dict = {}
        self.next_conn: Optional[Connection] = None
        self.server_conn: Optional[Connection] = None
        self.queue: asyncio.Queue[Work] = asyncio.Queue()
        self.worker_task: Optional[asyncio.Task] = None
        self.stats_task: Optional[asyncio.Task] = None
        self.paused = asyncio.Event()
        self.paused.set()
        self.running = False
        self.stats = StatsWindow()
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="stage")
        self.slot_state: dict[int, SlotState] = {}
        self.server: Optional[asyncio.base_events.Server] = None
        self._advertiser: Optional[Advertiser] = None

    # ------------------------------------------------------------------ lifecycle

    async def start(self) -> None:
        self.server = await serve(self._on_connection, "0.0.0.0", self.port)
        self.port = self.server.sockets[0].getsockname()[1]
        if self.advertise:
            self._advertiser = Advertiser(self.name, self.port, {"backend": self.backend.name, "version": __version__}, ip=self.ip)
            try:
                await self._advertiser.start()
            except Exception as e:  # Bonjour is optional (e.g. no multicast)
                log.warning("Bonjour advertising failed (%s); use --hosts on the server", e)
                self._advertiser = None
        log.info("host %s listening on %s:%d (backend %s)", self.name, self.ip, self.port, self.backend.name)

    async def stop(self) -> None:
        if self._advertiser:
            await self._advertiser.stop()
        await self._teardown()
        if self.server:
            self.server.close()
            await self.server.wait_closed()
        self.executor.shutdown(wait=False)

    async def serve_forever(self) -> None:
        await self.start()
        try:
            await asyncio.Event().wait()
        finally:
            await self.stop()

    async def _on_connection(self, conn: Connection) -> None:
        first = await conn.recv()
        if first.type == "hello":
            await self._serve_control(conn, first)
        elif first.type == "peer_hello":
            await self._serve_peer(conn, first)
        else:
            log.warning("unexpected first frame %s from %s", first.type, conn.peer)

    # ------------------------------------------------------------------ control

    async def _serve_control(self, conn: Connection, hello: Frame) -> None:
        conn.name = f"server@{conn.peer}"
        if self.server_conn is not None and not self.server_conn.closed:
            await conn.send(pack("error", message=f"host {self.name} is busy with job {self.job.get('job')}"))
            return
        self.server_conn = conn
        conn.start_heartbeat()
        log.info("server connected from %s (job %s)", conn.peer, hello.get("job"))
        await conn.send(
            pack(
                "welcome",
                name=self.name,
                ip=self.ip,
                port=self.port,
                backend=self.backend.name,
                version=__version__,
                free_mem=self.backend.free_memory(),
                total_mem=self.backend.total_memory(),
                platform=platform.platform(),
                machine=platform.machine(),
            )
        )
        # Keep reading while a long request (profile, load) runs, so heartbeats are
        # seen and the connection isn't timed out; requests still run in order.
        inbox: asyncio.Queue[Optional[Frame]] = asyncio.Queue()

        async def pump() -> None:
            try:
                while True:
                    await inbox.put(await conn.recv())
            except ConnectionClosed:
                await inbox.put(None)

        reader = asyncio.create_task(pump())
        try:
            while True:
                f = await inbox.get()
                if f is None:
                    raise ConnectionClosed(conn.name)
                try:
                    if await self._handle_control(conn, f):
                        break
                except ConnectionClosed:
                    raise
                except Exception as e:
                    log.exception("control %s failed", f.type)
                    await conn.send(pack("error", request=f.type, message=f"{type(e).__name__}: {e}"))
        except ConnectionClosed:
            log.info("server went away")
        finally:
            reader.cancel()
            if self.server_conn is conn:
                self.server_conn = None
            await self._teardown()

    async def _handle_control(self, conn: Connection, f: Frame) -> bool:
        """Returns True when the connection should end."""
        t = f.type
        if t == "profile":
            await conn.send(await self._profile(f))
        elif t == "probe_link":
            await conn.send(await self._probe_link(f))
        elif t == "load":
            await conn.send(await self._load(f))
        elif t == "start":
            self.running = True
            self.stats = StatsWindow()
            if self.stats_task is None:
                self.stats_task = asyncio.create_task(self._stats_loop())
            await conn.send(pack("started"))
        elif t == "feed":
            await self._feed(f)
        elif t == "pause":
            self.paused.clear()
        elif t == "resume":
            self.paused.set()
        elif t == "unload":
            await self._unload()
            await conn.send(pack("unloaded", free_mem=self.backend.free_memory()))
        elif t == "ping":
            await conn.send(pack("pong", t=f.get("t")))
        elif t == "bye":
            return True
        else:
            await conn.send(pack("error", request=t, message=f"unknown control message {t}"))
        return False

    async def _profile(self, f: Frame) -> Frame:
        model = f["model"]
        loop = asyncio.get_running_loop()
        kw = dict(
            batch_sizes=tuple(f.get("batch_sizes", (1, 8, 32, 64))),
            probe_layers=int(f.get("probe_layers", 2)),
            context=int(f.get("context", 64)),
        )
        log.info("profiling %s %s", model, kw)
        prof = await loop.run_in_executor(self.executor, lambda: self.backend.profile(model, **kw))
        info = self.backend.model_info(model)
        lengths = None
        if f.get("prompts") is not None:
            tok = self.backend.tokenizer(model)
            lengths = [len(tok.encode(p)) for p in f["prompts"]]
        return pack(
            "profile_result",
            name=self.name,
            layer_time={str(k): v for k, v in prof["layer_time"].items()},
            prefill_layer_time_per_token=prof["prefill_layer_time_per_token"],
            free_mem=prof["free_mem"],
            total_mem=self.backend.total_memory(),
            model_info=info.to_dict(),
            prompt_lengths=lengths,
        )

    async def _probe_link(self, f: Frame) -> Frame:
        """Measure latency and bandwidth from this host to another host."""
        host, port = f["host"], int(f["port"])
        nbytes = int(f.get("nbytes", 4 << 20))
        count = int(f.get("count", 3))
        conn = await connect(host, port, name=f"probe->{host}:{port}")
        try:
            await conn.send(pack("peer_hello", role="probe", sender=self.name))
            rtts = []
            for i in range(7):
                t0 = time.perf_counter()
                await conn.send(pack("link_probe", seq=i))
                ack = await conn.recv()
                assert ack.type == "link_probe_ack"
                rtts.append(time.perf_counter() - t0)
            latency = statistics.median(rtts) / 2
            payload = np.zeros((nbytes // 2,), dtype=np.float16)
            best = 0.0
            for i in range(count):
                t0 = time.perf_counter()
                await conn.send(pack("link_probe", seq=100 + i, tensor=payload))
                ack = await conn.recv()
                dt = time.perf_counter() - t0 - 2 * latency
                best = max(best, nbytes / max(dt, 1e-6))
        finally:
            await conn.send(pack("bye"))
            await conn.close()
        log.info("link %s -> %s:%d: %.1f MB/s, %.1f ms", self.name, host, port, best / 1e6, latency * 1e3)
        return pack("probe_result", source=self.name, host=host, port=port, bandwidth=best, latency=latency, nbytes=nbytes)

    async def _load(self, f: Frame) -> Frame:
        await self._unload()
        model = f["model"]
        a, b = f["layers"]
        stage_idx, num_stages = int(f["stage"]), int(f["num_stages"])
        first, last = stage_idx == 0, stage_idx == num_stages - 1
        self.job = dict(f.body)
        log.info("loading %s layers %d-%d (stage %d/%d)", model, a, b - 1, stage_idx + 1, num_stages)
        loop = asyncio.get_running_loop()
        t0 = time.perf_counter()
        self.stage = await loop.run_in_executor(self.executor, lambda: self.backend.load_stage(model, (a, b), first, last))
        self.tok = self.backend.tokenizer(model)
        self.info = self.backend.model_info(model)
        load_s = time.perf_counter() - t0
        nxt = f.get("next")
        if nxt and num_stages > 1:
            self.next_conn = await connect(nxt["host"], int(nxt["port"]), name=f"ring->{nxt.get('name', nxt['host'])}", timeout=30)
            await self.next_conn.send(pack("peer_hello", role="ring", sender=self.name, stage=stage_idx))
        self.slot_state.clear()
        if self.worker_task is None:
            self.worker_task = asyncio.create_task(self._worker())
        log.info("loaded in %.1fs, memory used %.2f GB", load_s, self.stage.memory_used() / 1e9)
        return pack("loaded", name=self.name, stage=stage_idx, layers=[a, b], load_seconds=load_s, mem_used=self.stage.memory_used(), free_mem=self.backend.free_memory())

    async def _unload(self) -> None:
        self.running = False
        while not self.queue.empty():
            self.queue.get_nowait()
        if self.next_conn is not None:
            try:
                await self.next_conn.send(pack("bye"))
            except ConnectionClosed:
                pass
            await self.next_conn.close()
            self.next_conn = None
        if self.stage is not None:
            stage, self.stage = self.stage, None
            # on the stage thread, so it waits for a step still running there
            await asyncio.get_running_loop().run_in_executor(self.executor, stage.unload)
        self.slot_state.clear()
        self.paused.set()

    async def _teardown(self) -> None:
        await self._unload()
        if self.stats_task is not None:
            self.stats_task.cancel()
            self.stats_task = None
        if self.worker_task is not None:
            self.worker_task.cancel()
            self.worker_task = None

    # ------------------------------------------------------------------ pipeline

    async def _feed(self, f: Frame) -> None:
        """Server hands stage 1 a micro-batch of prompts for a slot.

        Every prompt is split into ``prompt[:-1]`` (one padded prefill frame)
        and its last token (a decode frame), so after prefill all sequences in
        the micro-batch are aligned and the first generated token comes out of
        an ordinary decode step. The padding side is the backend's choice.
        """
        if self.stage is None or not self.stage.first:
            raise RuntimeError("feed sent to a host that is not stage 1")
        prompts, ids, slot = f["prompts"], f["ids"], int(f["slot"])
        enc = [self.tok.encode(p) for p in prompts]
        max_prompt = self.job.get("max_prompt_tokens")
        if max_prompt:
            enc = [e[-int(max_prompt):] for e in enc]
        enc = [e if e else [self.info.pad_token_id] for e in enc]
        side = getattr(self.backend, "padding_side", "left")
        base = {"slot": slot, "ids": list(ids), "prompt_tokens": [len(e) for e in enc]}
        T = max(len(e) for e in enc)
        if T > 1:
            body = [e[:-1] for e in enc]
            Tb = T - 1
            arr = np.full((len(body), Tb), self.info.pad_token_id, dtype=np.int32)
            pad_lens = []
            for i, e in enumerate(body):
                if side == "left":
                    arr[i, Tb - len(e):] = e
                else:
                    arr[i, : len(e)] = e
                pad_lens.append(Tb - len(e))
            meta = dict(base, phase="prefill", pad_lens=pad_lens, emit=False)
            await self.queue.put(Work(slot, "prefill", arr, pad_lens, meta))
            last = np.array([[e[-1]] for e in enc], dtype=np.int32)
            await self.queue.put(Work(slot, "decode", last, None, {"slot": slot, "phase": "decode", "emit": True}))
        else:
            arr = np.array([[e[-1]] for e in enc], dtype=np.int32)
            pad_lens = [0] * len(enc)
            meta = dict(base, phase="prefill", pad_lens=pad_lens, emit=True)
            await self.queue.put(Work(slot, "prefill", arr, pad_lens, meta))

    async def _worker(self) -> None:
        loop = asyncio.get_running_loop()
        try:
            while True:
                w = await self.queue.get()
                await self.paused.wait()
                stage = self.stage
                if stage is None:
                    continue
                t0 = time.perf_counter()
                out = await loop.run_in_executor(self.executor, stage.run, w.slot, w.phase, w.x, w.pad_lens)
                if self.stage is not stage:
                    continue  # unloaded while this step ran
                dt = time.perf_counter() - t0
                self.stats.record(dt, int(w.x.shape[0]), w.phase)
                if stage.last:
                    await self._after_last(w, out)
                else:
                    await self._send_next(pack("hidden", tensor=out, **w.meta))
        except asyncio.CancelledError:
            pass
        except ConnectionClosed as e:
            log.error("ring connection lost: %s", e)
            if self.server_conn:
                try:
                    await self.server_conn.send(pack("error", request="ring", message=str(e)))
                except ConnectionClosed:
                    pass
        except Exception:
            log.exception("stage worker crashed")
            if self.server_conn:
                try:
                    await self.server_conn.send(pack("error", request="worker", message="stage worker crashed; see host log"))
                except ConnectionClosed:
                    pass

    async def _send_next(self, frame: Frame) -> None:
        if self.next_conn is None:
            raise ConnectionClosed("no next stage connected")
        await self.next_conn.send(frame)

    async def _after_last(self, w: Work, tokens: np.ndarray) -> None:
        slot = w.slot
        if w.phase == "prefill":
            ids = list(w.meta.get("ids", range(len(tokens))))
            self.slot_state[slot] = SlotState(ids=ids, gen=[[] for _ in ids], done=[False] * len(ids))
            self.slot_state[slot].prompt_tokens = w.meta.get("prompt_tokens")
        if not w.meta.get("emit", True):
            return  # prefill of prompt[:-1]; the decode frame for the last token follows
        st = self.slot_state.get(slot)
        if st is None:
            log.warning("tokens for unknown slot %d", slot)
            return
        eos = set(self.info.eos_token_ids)
        max_tokens = int(self.job.get("max_tokens", 256))
        for i, t in enumerate(tokens.tolist()):
            if st.done[i]:
                continue
            st.gen[i].append(int(t))
            self.stats.tokens_generated += 1
            if t in eos or len(st.gen[i]) >= max_tokens:
                st.done[i] = True
        if all(st.done):
            texts = [self.tok.decode(g) for g in st.gen]
            if self.server_conn is not None:
                await self.server_conn.send(
                    pack("outputs", slot=slot, ids=st.ids, tokens=st.gen, texts=texts, prompt_tokens=st.prompt_tokens, seconds=time.monotonic() - st.started)
                )
            del self.slot_state[slot]
            await self._deliver_to_first(pack("tokens", slot=slot, done=True))
        else:
            await self._deliver_to_first(pack("tokens", tensor=tokens.astype(np.int32), slot=slot, done=False))

    async def _deliver_to_first(self, frame: Frame) -> None:
        if self.stage is not None and self.stage.first:
            await self._on_tokens(frame)
        else:
            await self._send_next(frame)

    async def _on_tokens(self, f: Frame) -> None:
        """Stage 1 receives the next tokens for a slot (or learns the slot is done)."""
        slot = int(f["slot"])
        if f.get("done"):
            if self.stage is not None:
                self.stage.reset_slot(slot)
            if self.server_conn is not None:
                await self.server_conn.send(pack("slot_free", slot=slot))
            return
        x = f.tensor.reshape(-1, 1).astype(np.int32)
        await self.queue.put(Work(slot, "decode", x, None, {"slot": slot, "phase": "decode"}))

    # ------------------------------------------------------------------ peers

    async def _serve_peer(self, conn: Connection, hello: Frame) -> None:
        role = hello.get("role", "ring")
        conn.name = f"{role}<-{hello.get('sender', conn.peer)}"
        if role == "probe":
            while True:
                f = await conn.recv()
                if f.type == "bye":
                    return
                if f.type == "link_probe":
                    await conn.send(pack("link_probe_ack", seq=f.get("seq")))
            return
        log.info("ring peer %s connected", conn.name)
        try:
            while True:
                f = await conn.recv()
                if f.type == "hidden":
                    if self.stage is None:
                        log.warning("hidden frame but no stage loaded; dropping")
                        continue
                    meta = dict(f.body)
                    await self.queue.put(Work(int(meta["slot"]), meta["phase"], f.tensor, meta.get("pad_lens"), meta))
                elif f.type == "tokens":
                    await self._on_tokens(f)
                elif f.type == "bye":
                    return
                else:
                    log.warning("unexpected ring frame %s", f.type)
        except ConnectionClosed:
            log.info("ring peer %s disconnected", conn.name)

    # ------------------------------------------------------------------ stats

    async def _stats_loop(self) -> None:
        try:
            while True:
                await asyncio.sleep(1.0)
                if self.server_conn is None or self.server_conn.closed:
                    continue
                snap = self.stats.snapshot()
                mem_used = self.stage.memory_used() if self.stage else 0
                try:
                    await self.server_conn.send(
                        pack(
                            "stats",
                            name=self.name,
                            stage=self.job.get("stage"),
                            busy=snap["busy"],
                            step=snap["step"],
                            prefill_step=snap["prefill_step"],
                            tokens_generated=snap["tokens_generated"],
                            total_busy=snap["total_busy"],
                            mem_used=mem_used,
                            mem_total=self.backend.total_memory(),
                            bytes_out=self.next_conn.bytes_sent if self.next_conn else 0,
                            queue=self.queue.qsize(),
                            slots=len(self.slot_state),
                            paused=not self.paused.is_set(),
                        )
                    )
                except ConnectionClosed:
                    pass
        except asyncio.CancelledError:
            pass


def _caffeinate() -> Optional[subprocess.Popen]:
    if platform.system() != "Darwin":
        return None
    try:
        return subprocess.Popen(["caffeinate", "-i", "-s", "-w", str(os.getpid())])
    except FileNotFoundError:
        return None


async def main(args) -> int:
    from ..models import apply_models_dir

    apply_models_dir(args.models_dir)
    backend = get_backend(args.backend)
    name = args.name or socket.gethostname().split(".")[0]
    daemon = HostDaemon(name, args.port, backend, advertise=not args.no_advertise)
    caff = None if args.no_caffeinate else _caffeinate()
    try:
        await daemon.serve_forever()
    finally:
        if caff:
            caff.terminate()
    return 0
