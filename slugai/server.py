"""Server (``slugai run``): decides everything.

Phases, run automatically for every job:

1. **discover** hosts (Bonjour, or ``--hosts``), connect, exchange hello/welcome
2. **profile** each host (time per layer vs micro-batch size, free memory) and
   every directed link (bandwidth, latency)
3. **place**: ring order, layer split, micro-batch size and count
4. **load** each stage and wire the ring
5. **prefill / decode**: feed prompts into free slots, collect outputs,
   rebalance at a batch boundary if a host slows down
6. **collect**: write completions and a report
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

from . import __version__
from .backend import ModelInfo
from .discovery import HostAddr, browse, parse_hosts
from .placement import (
    HostProfile,
    LinkProfile,
    ModelSpec,
    Placement,
    PlacementError,
    rebalance_with_measurements,
    should_rebalance,
    solve,
)
from .prompts import load_prompts
from .protocol import Frame, pack
from .transport import Connection, ConnectionClosed, connect

log = logging.getLogger(__name__)

PHASES = ["discover", "profile", "place", "load", "prefill", "decode", "collect"]


class HostClient:
    """One control connection to a host, with request/response and event dispatch."""

    def __init__(self, addr: HostAddr, conn: Connection, on_event: Callable[["HostClient", Frame], None]):
        self.addr = addr
        self.conn = conn
        self.name = addr.name
        self.welcome: dict = {}
        self.profile: Optional[HostProfile] = None
        self.model_info: Optional[ModelInfo] = None
        self.prompt_lengths: Optional[list[int]] = None
        self.stats: dict = {}
        self.stage: Optional[int] = None
        self._waiters: dict[str, asyncio.Future] = {}
        self._on_event = on_event
        self._reader = asyncio.create_task(self._read_loop())
        self.error: Optional[str] = None

    async def _read_loop(self) -> None:
        try:
            while True:
                f = await self.conn.recv()
                fut = self._waiters.pop(f.type, None)
                if fut is not None and not fut.done():
                    fut.set_result(f)
                    continue
                if f.type == "error":
                    req = f.get("request")
                    fut = self._waiters.pop(_expected_for(req), None) if req else None
                    if fut is not None and not fut.done():
                        fut.set_exception(RuntimeError(f"{self.name}: {f.get('message')}"))
                        continue
                    self.error = f.get("message")
                self._on_event(self, f)
        except ConnectionClosed:
            for fut in self._waiters.values():
                if not fut.done():
                    fut.set_exception(ConnectionClosed(self.name))
            self._on_event(self, pack("disconnected"))
        except asyncio.CancelledError:
            pass

    def expect(self, typ: str) -> asyncio.Future:
        fut: asyncio.Future = asyncio.get_running_loop().create_future()
        self._waiters[typ] = fut
        return fut

    async def request(self, frame: Frame, expect: str, timeout: float = 600.0) -> Frame:
        fut = self.expect(expect)
        await self.conn.send(frame)
        return await asyncio.wait_for(fut, timeout)

    async def send(self, frame: Frame) -> None:
        await self.conn.send(frame)

    async def close(self) -> None:
        self._reader.cancel()
        try:
            await self.conn.send(pack("bye"))
        except Exception:
            pass
        await self.conn.close()


def _expected_for(request: str) -> str:
    return {
        "profile": "profile_result",
        "probe_link": "probe_result",
        "load": "loaded",
        "unload": "unloaded",
        "start": "started",
    }.get(request, request)


@dataclass
class JobState:
    """Everything the dashboard shows; updated by the server, read by the UI."""

    job: str
    model: str
    phase: str = "discover"
    phase_index: int = 0
    started: float = field(default_factory=time.monotonic)
    hosts: dict[str, dict] = field(default_factory=dict)  # name -> {stage, layers, busy, step, mem_used, mem_total, bytes_out}
    links: list[dict] = field(default_factory=list)  # [{src, dst, used, capacity, carries}]
    placement: Optional[Placement] = None
    total_sequences: int = 0
    done_sequences: int = 0
    tokens_generated: int = 0
    tok_s: float = 0.0
    tok_s_window: deque = field(default_factory=lambda: deque(maxlen=30))
    decode_started: Optional[float] = None
    events: deque = field(default_factory=lambda: deque(maxlen=200))
    finished: bool = False
    error: Optional[str] = None
    paused: bool = False
    rebalances: int = 0
    message: str = ""

    def log(self, msg: str) -> None:
        self.events.append((time.time(), msg))
        log.info(msg)

    @property
    def elapsed(self) -> float:
        return time.monotonic() - self.started

    @property
    def eta(self) -> Optional[float]:
        if self.done_sequences == 0 or self.decode_started is None:
            return None
        rate = self.done_sequences / max(time.monotonic() - self.decode_started, 1e-9)
        return (self.total_sequences - self.done_sequences) / max(rate, 1e-9)

    @property
    def peak_tok_s(self) -> float:
        return self.placement.peak_tok_s if self.placement else 0.0


class Server:
    def __init__(
        self,
        model: str,
        prompts: list[str],
        job: Optional[str] = None,
        hosts: Optional[list[HostAddr]] = None,
        min_hosts: int = 3,
        wait: float = 30.0,
        max_tokens: int = 256,
        micro_batch: Optional[int] = None,
        num_micro_batches: Optional[int] = None,
        rebalance: bool = True,
        rebalance_every: float = 20.0,
        output_path: Optional[str] = None,
        report_path: Optional[str] = None,
        profile_batch_sizes: tuple[int, ...] = (1, 8, 32, 64),
    ):
        self.model = model
        self.prompts = prompts
        self.job = job or time.strftime("job-%Y%m%d-%H%M%S")
        self.fixed_hosts = hosts
        self.min_hosts = min_hosts
        self.wait = wait
        self.max_tokens = max_tokens
        self.micro_batch = micro_batch
        self.num_micro_batches = num_micro_batches
        self.rebalance_enabled = rebalance
        self.rebalance_every = rebalance_every
        self.output_path = output_path or f"{self.job}.jsonl"
        self.report_path = report_path
        self.profile_batch_sizes = profile_batch_sizes

        self.state = JobState(job=self.job, model=model, total_sequences=len(prompts))
        self.hosts: dict[str, HostClient] = {}
        self.links: dict[tuple[str, str], LinkProfile] = {}
        self.spec: Optional[ModelSpec] = None
        self.model_info: Optional[ModelInfo] = None
        self.placement: Optional[Placement] = None
        self.context: int = 0
        self.outputs: dict[int, dict] = {}
        self.pending: deque[int] = deque(range(len(prompts)))
        self.free_slots: deque[int] = deque()
        self.slot_prompt_ids: dict[int, list[int]] = {}
        self.busy_slots: set[int] = set()
        self._all_idle = asyncio.Event()
        self._draining = False
        self._stop = asyncio.Event()
        self.report: dict = {"job": self.job, "model": model, "version": __version__, "rebalances": []}
        self._last_tokens: dict[str, tuple[float, int]] = {}
        self._last_bytes: dict[str, tuple[float, int]] = {}
        self._tasks: list[asyncio.Task] = []

    # ------------------------------------------------------------------ phases

    def _set_phase(self, phase: str) -> None:
        self.state.phase = phase
        self.state.phase_index = PHASES.index(phase)
        self.state.log(f"phase {self.state.phase_index + 1}/{len(PHASES)} {phase}")

    async def run(self) -> dict:
        try:
            await self._discover()
            await self._profile()
            self._place()
            await self._load(self.placement)
            await self._start()
            await self._decode()
            await self._collect()
        except Exception as e:
            self.state.error = f"{type(e).__name__}: {e}"
            self.state.log(f"failed: {self.state.error}")
            raise
        finally:
            self.state.finished = True
            for t in self._tasks:
                t.cancel()
            await asyncio.gather(*[h.close() for h in self.hosts.values()], return_exceptions=True)
        return self.report

    async def _discover(self) -> None:
        self._set_phase("discover")
        addrs = self.fixed_hosts
        if not addrs:
            self.state.log(f"browsing Bonjour for {self.min_hosts}+ hosts (up to {self.wait:.0f}s)")
            addrs = await browse(timeout=self.wait, min_hosts=self.min_hosts)
        if len(addrs) < max(1, self.min_hosts):
            raise RuntimeError(f"found {len(addrs)} host(s), need {self.min_hosts}: {[str(a) for a in addrs]}")
        for a in addrs:
            conn = await connect(a.host, a.port, name=f"ctl->{a.name}")
            conn.start_heartbeat()
            client = HostClient(a, conn, self._on_event)
            fut = client.expect("welcome")
            await client.send(pack("hello", job=self.job, model=self.model, version=__version__))
            welcome = await asyncio.wait_for(fut, 30)
            client.welcome = welcome.body
            name = welcome["name"]
            if name in self.hosts:
                name = f"{name}-{len(self.hosts)}"
            client.name = name
            client.addr = HostAddr(name, welcome.get("ip", a.host), int(welcome.get("port", a.port)))
            self.hosts[name] = client
            self.state.hosts[name] = {"stage": None, "layers": None, "busy": 0.0, "step": 0.0, "mem_used": 0, "mem_total": welcome.get("total_mem", 0), "bytes_out": 0, "backend": welcome.get("backend")}
            self.state.log(f"host {name} ({welcome.get('backend')}, {welcome.get('free_mem', 0) / 1e9:.1f} GB free) at {client.addr.host}:{client.addr.port}")

    async def _profile(self) -> None:
        self._set_phase("profile")
        names = list(self.hosts)
        sample = self.prompts if len(self.prompts) <= 20000 else self.prompts[:20000]

        async def prof(i: int, h: HostClient) -> None:
            res = await h.request(
                pack("profile", model=self.model, batch_sizes=list(self.profile_batch_sizes), probe_layers=2, context=64, prompts=sample if i == 0 else None),
                "profile_result",
                timeout=1800,
            )
            h.profile = HostProfile(
                name=h.name,
                free_mem=int(res["free_mem"]),
                layer_time={int(k): float(v) for k, v in res["layer_time"].items()},
                prefill_layer_time_per_token=float(res.get("prefill_layer_time_per_token", 0.0)),
            )
            h.model_info = ModelInfo.from_dict(res["model_info"])
            if res.get("prompt_lengths") is not None:
                h.prompt_lengths = list(res["prompt_lengths"])
            self.state.hosts[h.name]["mem_total"] = res.get("total_mem", self.state.hosts[h.name]["mem_total"])
            lt = ", ".join(f"b{k}={v * 1e3:.1f}ms" for k, v in sorted(h.profile.layer_time.items()))
            self.state.log(f"{h.name}: {h.profile.free_mem / 1e9:.1f} GB free, per layer {lt}")

        await asyncio.gather(*[prof(i, self.hosts[n]) for i, n in enumerate(names)])
        self.model_info = self.hosts[names[0]].model_info
        self.spec = self.model_info.spec
        lengths = self.hosts[names[0]].prompt_lengths or [max(8, len(p) // 3) for p in self.prompts]
        self.context = max(lengths) + self.max_tokens
        self.state.log(f"model {self.model}: {self.spec.num_layers} layers, hidden {self.spec.hidden_size}, context budget {self.context}")

        # links, one at a time so probes don't disturb each other
        if len(names) == 1:
            n = names[0]
            self.links[(n, n)] = LinkProfile(n, n, 1e12, 0.0)
        for src in names:
            for dst in names:
                if src == dst:
                    continue
                d = self.hosts[dst]
                res = await self.hosts[src].request(pack("probe_link", host=d.addr.host, port=d.addr.port, nbytes=4 << 20, count=3), "probe_result", timeout=120)
                self.links[(src, dst)] = LinkProfile(src, dst, float(res["bandwidth"]), float(res["latency"]))
                self.state.log(f"link {src} > {dst}: {res['bandwidth'] / 1e6:.1f} MB/s, {res['latency'] * 1e3:.1f} ms")
        self.report["profiles"] = {n: self.hosts[n].profile.__dict__ for n in names}
        self.report["links"] = {f"{s}>{d}": l.__dict__ for (s, d), l in self.links.items()}
        self.report["model_spec"] = self.spec.__dict__

    def _place(self) -> None:
        self._set_phase("place")
        profiles = {n: h.profile for n, h in self.hosts.items()}
        self.placement = solve(
            self.spec, profiles, self.links, context=self.context, micro_batch=self.micro_batch, num_micro_batches=self.num_micro_batches
        )
        self._announce_placement(self.placement)
        self.report["placement"] = self.placement.to_dict()

    def _announce_placement(self, p: Placement) -> None:
        self.state.placement = p
        for i, name in enumerate(p.order):
            self.state.hosts[name]["stage"] = i
            self.state.hosts[name]["layers"] = p.layers[i]
            self.hosts[name].stage = i
        self.state.links = []
        S = len(p.order)
        for i in range(S):
            src, dst = p.order[i], p.order[(i + 1) % S]
            lp = self.links[(src, dst)]
            self.state.links.append({"src": src, "dst": dst, "used": 0.0, "capacity": lp.bandwidth, "carries": "token ids" if i == S - 1 else "hidden states"})
        self.state.log(
            "placement: "
            + "  ".join(f"{n}:{a}-{b - 1}" for n, (a, b) in zip(p.order, p.layers))
            + f"  micro-batch {p.micro_batch} x {p.num_micro_batches}  predicted {p.predicted_tok_s:.1f} tok/s (peak {p.peak_tok_s:.1f})"
        )

    async def _load(self, p: Placement) -> None:
        self._set_phase("load")
        S = len(p.order)
        max_prompt_tokens = max(1, self.context - self.max_tokens)

        async def load_one(i: int) -> None:
            name = p.order[i]
            nxt_name = p.order[(i + 1) % S]
            nxt = self.hosts[nxt_name].addr
            res = await self.hosts[name].request(
                pack(
                    "load",
                    job=self.job,
                    model=self.model,
                    layers=list(p.layers[i]),
                    stage=i,
                    num_stages=S,
                    next={"name": nxt_name, "host": nxt.host, "port": nxt.port} if S > 1 else None,
                    max_tokens=self.max_tokens,
                    max_prompt_tokens=max_prompt_tokens,
                    micro_batch=p.micro_batch,
                    num_micro_batches=p.num_micro_batches,
                ),
                "loaded",
                timeout=3600,
            )
            self.state.hosts[name]["mem_used"] = res.get("mem_used", 0)
            self.state.log(f"{name}: loaded layers {p.layers[i][0]}-{p.layers[i][1] - 1} in {res.get('load_seconds', 0):.1f}s")

        # Load in ring order so each stage's "next" is listening before we connect:
        # every host already listens on its port, so loading in parallel is safe.
        await asyncio.gather(*[load_one(i) for i in range(S)])

    async def _start(self) -> None:
        await asyncio.gather(*[h.request(pack("start"), "started", timeout=30) for h in self.hosts.values()])
        self.free_slots = deque(range(self.placement.num_micro_batches))
        self.busy_slots.clear()
        self._draining = False
        if self.state.decode_started is None:
            self.state.decode_started = time.monotonic()
        if not self._tasks:
            self._tasks.append(asyncio.create_task(self._stats_loop()))
            if self.rebalance_enabled and len(self.hosts) > 1:
                self._tasks.append(asyncio.create_task(self._rebalance_loop()))

    async def _decode(self) -> None:
        self._set_phase("prefill")
        await self._fill_slots()
        self._set_phase("decode")
        while True:
            if self._stop.is_set():
                self.state.log("stopping on request")
                return
            if not self.pending and not self.busy_slots:
                return
            if self._draining and not self.busy_slots:
                await self._do_rebalance()
                await self._fill_slots()
            await asyncio.sleep(0.1)

    async def _fill_slots(self) -> None:
        while self.pending and self.free_slots and not self._draining and not self._stop.is_set():
            slot = self.free_slots.popleft()
            await self._feed_slot(slot)

    async def _feed_slot(self, slot: int) -> None:
        b = self.placement.micro_batch
        ids = [self.pending.popleft() for _ in range(min(b, len(self.pending)))]
        self.slot_prompt_ids[slot] = ids
        self.busy_slots.add(slot)
        first = self.hosts[self.placement.order[0]]
        await first.send(pack("feed", slot=slot, ids=ids, prompts=[self.prompts[i] for i in ids]))

    async def _collect(self) -> None:
        self._set_phase("collect")
        out = Path(self.output_path)
        with out.open("w") as fh:
            for i in range(len(self.prompts)):
                rec = self.outputs.get(i, {"id": i, "prompt": self.prompts[i], "completion": None, "tokens": 0})
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        elapsed = self.state.elapsed
        decode_elapsed = time.monotonic() - (self.state.decode_started or self.state.started)
        self.report.update(
            {
                "sequences": len(self.prompts),
                "sequences_done": len(self.outputs),
                "tokens_generated": self.state.tokens_generated,
                "elapsed_seconds": elapsed,
                "decode_seconds": decode_elapsed,
                "tok_s": self.state.tokens_generated / max(decode_elapsed, 1e-9),
                "tok_h": self.state.tokens_generated / max(decode_elapsed, 1e-9) * 3600,
                "output": str(out),
                "hosts": self.state.hosts,
            }
        )
        if self.report_path:
            Path(self.report_path).parent.mkdir(parents=True, exist_ok=True)
            Path(self.report_path).write_text(json.dumps(self.report, indent=2, default=str))
        self.state.log(f"done: {len(self.outputs)}/{len(self.prompts)} sequences, {self.state.tokens_generated} tokens in {decode_elapsed:.0f}s = {self.report['tok_s']:.1f} tok/s; wrote {out}")

    # ------------------------------------------------------------------ events

    def _on_event(self, h: HostClient, f: Frame) -> None:
        t = f.type
        if t == "stats":
            self._on_stats(h, f)
        elif t == "outputs":
            self._on_outputs(h, f)
        elif t == "slot_free":
            slot = int(f["slot"])
            self.busy_slots.discard(slot)
            self.slot_prompt_ids.pop(slot, None)
            self.free_slots.append(slot)
            if self.pending and not self._draining and not self._stop.is_set():
                asyncio.create_task(self._fill_slots())
        elif t == "error":
            self.state.log(f"{h.name} error: {f.get('message')}")
            self.state.error = f"{h.name}: {f.get('message')}"
            self._stop.set()
        elif t == "disconnected":
            if not self.state.finished:
                self.state.log(f"{h.name} disconnected")
                self.state.error = f"{h.name} disconnected"
                self._stop.set()

    def _on_stats(self, h: HostClient, f: Frame) -> None:
        h.stats = f.body
        s = self.state.hosts.setdefault(h.name, {})
        s.update(busy=f.get("busy", 0.0), step=f.get("step", 0.0), mem_used=f.get("mem_used", 0), mem_total=f.get("mem_total", s.get("mem_total", 0)), queue=f.get("queue", 0))
        now = time.monotonic()
        # link usage from bytes_out deltas
        prev = self._last_bytes.get(h.name)
        cur = (now, int(f.get("bytes_out", 0)))
        self._last_bytes[h.name] = cur
        if prev and cur[0] > prev[0]:
            rate = (cur[1] - prev[1]) / (cur[0] - prev[0])
            for l in self.state.links:
                if l["src"] == h.name:
                    l["used"] = max(0.0, rate)
        # throughput from the last stage's generated-token counter
        if self.placement and h.name == self.placement.order[-1]:
            tg = int(f.get("tokens_generated", 0))
            prev = self._last_tokens.get(h.name)
            self._last_tokens[h.name] = (now, tg)
            if prev and now > prev[0]:
                self.state.tok_s_window.append((tg - prev[1]) / (now - prev[0]))
                recent = list(self.state.tok_s_window)[-10:]
                self.state.tok_s = sum(recent) / len(recent)

    def _on_outputs(self, h: HostClient, f: Frame) -> None:
        ids, toks, texts = f["ids"], f["tokens"], f["texts"]
        ptoks = f.get("prompt_tokens") or [None] * len(ids)
        for i, tk, tx, pt in zip(ids, toks, texts, ptoks):
            self.outputs[int(i)] = {"id": int(i), "prompt": self.prompts[int(i)], "completion": tx, "tokens": len(tk), "prompt_tokens": pt, "token_ids": [int(t) for t in tk]}
            self.state.tokens_generated += len(tk)
        self.state.done_sequences = len(self.outputs)

    # ------------------------------------------------------------------ loops

    async def _stats_loop(self) -> None:
        try:
            while not self.state.finished:
                await asyncio.sleep(1.0)
        except asyncio.CancelledError:
            pass

    async def _rebalance_loop(self) -> None:
        try:
            await asyncio.sleep(self.rebalance_every)
            while not self.state.finished:
                await asyncio.sleep(self.rebalance_every)
                if self.state.phase != "decode" or self._draining or not self.pending:
                    continue
                self.maybe_rebalance()
        except asyncio.CancelledError:
            pass

    def maybe_rebalance(self, force: bool = False) -> bool:
        """Compare live step times to the plan; if a Mac slowed down, drain and re-place."""
        if self.placement is None or self._draining:
            return False
        measured = {}
        for name in self.placement.order:
            st = self.hosts[name].stats
            if st and st.get("step", 0) > 0:
                measured[name] = float(st["step"])
        if len(measured) < len(self.placement.order):
            return False
        profiles = {n: h.profile for n, h in self.hosts.items()}
        try:
            proposed = rebalance_with_measurements(self.placement, self.spec, profiles, self.links, measured)
        except PlacementError as e:
            self.state.log(f"rebalance: {e}")
            return False
        if force or should_rebalance(self.placement, proposed):
            self._proposed = proposed
            self._draining = True
            self.state.log(
                "rebalance planned: " + "  ".join(f"{n}:{a}-{b - 1}" for n, (a, b) in zip(proposed.order, proposed.layers)) + " — draining slots"
            )
            return True
        return False

    async def _do_rebalance(self) -> None:
        proposed = getattr(self, "_proposed", None)
        self._draining = False
        if proposed is None:
            return
        self.state.rebalances += 1
        self.state.log("rebalancing: unloading stages")
        await asyncio.gather(*[h.request(pack("unload"), "unloaded", timeout=600) for h in self.hosts.values()])
        old = self.placement
        self.placement = proposed
        self._announce_placement(proposed)
        await self._load(proposed)
        await self._start()
        self._set_phase("decode")
        self.report["rebalances"].append({"at": self.state.elapsed, "from": old.to_dict(), "to": proposed.to_dict()})
        self._proposed = None

    async def pause(self) -> None:
        self.state.paused = True
        await asyncio.gather(*[h.send(pack("pause")) for h in self.hosts.values()], return_exceptions=True)

    async def resume(self) -> None:
        self.state.paused = False
        await asyncio.gather(*[h.send(pack("resume")) for h in self.hosts.values()], return_exceptions=True)

    def stop(self) -> None:
        self._stop.set()

    def save_report(self, path: Optional[str] = None) -> str:
        path = path or self.report_path or f"{self.job}-report.json"
        snap = dict(self.report)
        snap.update(
            sequences_done=len(self.outputs),
            tokens_generated=self.state.tokens_generated,
            elapsed_seconds=self.state.elapsed,
            tok_s=self.state.tok_s,
            hosts=self.state.hosts,
            links=[dict(l) for l in self.state.links],
            partial=not self.state.finished,
        )
        Path(path).write_text(json.dumps(snap, indent=2, default=str))
        self.state.log(f"report saved to {path}")
        return path


async def main(args) -> int:
    prompts = load_prompts(args.prompts, limit=args.limit)
    if not prompts:
        log.error("no prompts in %s", args.prompts)
        return 2
    hosts = parse_hosts(args.hosts, 47111) if args.hosts else None
    server = Server(
        model=args.model,
        prompts=prompts,
        job=args.job,
        hosts=hosts,
        min_hosts=args.min_hosts if not hosts else len(hosts),
        wait=args.wait,
        max_tokens=args.max_tokens,
        micro_batch=args.micro_batch,
        num_micro_batches=args.num_micro_batches,
        rebalance=not args.no_rebalance,
        rebalance_every=args.rebalance_every,
        output_path=args.output,
        report_path=args.report,
    )
    if args.no_tui:
        from .dashboard import run_plain

        return await run_plain(server)
    from .dashboard import run_tui

    return await run_tui(server)
