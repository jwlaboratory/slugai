"""End to end: a server and three hosts in one process, fake backend, real sockets."""

import asyncio
import json

import numpy as np
import pytest

from slugai.backends import FakeBackend
from slugai.net.discovery import HostAddr
from slugai.ring.host import HostDaemon
from slugai.ring.server import Server
from tests.helpers import reference_generate

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


def busiest(placement):
    """The host holding the most layers: slowing it down always leaves layers to move."""
    return max(zip(placement.order, placement.layers), key=lambda nl: nl[1][1] - nl[1][0])[0]


async def start_hosts(n, backend_factory):
    hosts = []
    for i in range(n):
        d = HostDaemon(f"mac{i}", 0, backend_factory(i), advertise=False, ip="127.0.0.1")
        await d.start()
        hosts.append(d)
    return hosts


@pytest.mark.asyncio
async def test_three_hosts_match_reference(tmp_path):
    hosts = await start_hosts(3, lambda i: FakeBackend(slow_ms=0, free_mem=1 << 30))
    try:
        addrs = [HostAddr(d.name, "127.0.0.1", d.port) for d in hosts]
        out = tmp_path / "out.jsonl"
        rep = tmp_path / "report.json"
        server = Server(
            model="fake-L6",
            prompts=PROMPTS,
            job="t3",
            hosts=addrs,
            min_hosts=3,
            max_tokens=10,
            micro_batch=4,
            num_micro_batches=3,
            rebalance=False,
            output_path=str(out),
            report_path=str(rep),
            profile_batch_sizes=(1, 4),
        )
        report = await asyncio.wait_for(server.run(), 60)
        assert server.state.error is None
        assert report["sequences_done"] == len(PROMPTS)
        assert len(report["placement"]["order"]) == 3
        # every host got a contiguous block and all 6 layers are covered
        layers = report["placement"]["layers"]
        assert layers[0][0] == 0 and layers[-1][1] == 6
        recs = [json.loads(l) for l in out.read_text().splitlines()]
        assert [r["id"] for r in recs] == list(range(len(PROMPTS)))
        ref = reference_generate(FakeBackend(slow_ms=0), "fake-L6", PROMPTS, 10)
        tok = FakeBackend().tokenizer("fake-L6")
        for r, g in zip(recs, ref):
            assert r["completion"] == tok.decode(g), r
            assert r["tokens"] == len(g)
        assert json.loads(rep.read_text())["tokens_generated"] == sum(len(g) for g in ref)
    finally:
        for d in hosts:
            await d.stop()


@pytest.mark.asyncio
async def test_right_padding_backend_matches_reference(tmp_path):
    """The MLX backend right-pads prompts; exercise that path of the host with the fake backend."""
    hosts = await start_hosts(3, lambda i: FakeBackend(slow_ms=0, free_mem=1 << 30, padding_side="right"))
    try:
        addrs = [HostAddr(d.name, "127.0.0.1", d.port) for d in hosts]
        server = Server(
            model="fake-L6", prompts=PROMPTS, job="trp", hosts=addrs, min_hosts=3, max_tokens=10,
            micro_batch=4, num_micro_batches=3, rebalance=False, output_path=str(tmp_path / "o.jsonl"),
            profile_batch_sizes=(1, 4),
        )
        report = await asyncio.wait_for(server.run(), 60)
        assert report["sequences_done"] == len(PROMPTS)
        ref = reference_generate(FakeBackend(slow_ms=0), "fake-L6", PROMPTS, 10)
        tok = FakeBackend().tokenizer("fake-L6")
        recs = [json.loads(l) for l in (tmp_path / "o.jsonl").read_text().splitlines()]
        assert [r["completion"] for r in recs] == [tok.decode(g) for g in ref]
    finally:
        for d in hosts:
            await d.stop()


@pytest.mark.asyncio
async def test_single_host_ring(tmp_path):
    hosts = await start_hosts(1, lambda i: FakeBackend(slow_ms=0, free_mem=1 << 30))
    try:
        addrs = [HostAddr(d.name, "127.0.0.1", d.port) for d in hosts]
        server = Server(
            model="fake-L4", prompts=PROMPTS[:5], job="t1", hosts=addrs, min_hosts=1, max_tokens=6,
            micro_batch=2, num_micro_batches=2, rebalance=False, output_path=str(tmp_path / "o.jsonl"),
            profile_batch_sizes=(1, 2),
        )
        report = await asyncio.wait_for(server.run(), 60)
        assert report["sequences_done"] == 5
        ref = reference_generate(FakeBackend(slow_ms=0), "fake-L4", PROMPTS[:5], 6)
        tok = FakeBackend().tokenizer("fake-L4")
        recs = [json.loads(l) for l in (tmp_path / "o.jsonl").read_text().splitlines()]
        assert [r["completion"] for r in recs] == [tok.decode(g) for g in ref]
    finally:
        for d in hosts:
            await d.stop()


@pytest.mark.asyncio
async def test_rebalance_moves_layers_when_a_host_slows(tmp_path):
    """Host 0 is profiled fast but then runs slow: a forced rebalance must re-place and finish correctly."""
    backends = [FakeBackend(slow_ms=0, free_mem=1 << 30) for _ in range(3)]
    hosts = await start_hosts(3, lambda i: backends[i])
    try:
        addrs = [HostAddr(d.name, "127.0.0.1", d.port) for d in hosts]
        prompts = PROMPTS * 8
        server = Server(
            model="fake-L6", prompts=prompts, job="trb", hosts=addrs, min_hosts=3, max_tokens=8,
            micro_batch=3, num_micro_batches=3, rebalance=True, rebalance_every=0.5,
            output_path=str(tmp_path / "o.jsonl"), profile_batch_sizes=(1, 3),
        )

        async def slow_down_later():
            # wait until decoding, then slow the host with the most layers (so some can move) and force a rebalance
            while server.state.phase != "decode":
                await asyncio.sleep(0.05)
            name = busiest(server.placement)
            d = next(h for h in hosts if h.name == name)
            d.stage.slow_ms = 100.0
            await asyncio.sleep(1.0)  # let stats reflect the slowdown
            server.maybe_rebalance(force=True)  # may already be draining from the periodic check

        t = asyncio.create_task(slow_down_later())
        report = await asyncio.wait_for(server.run(), 120)
        await t
        assert server.state.error is None
        assert report["sequences_done"] == len(prompts)
        assert len(report["rebalances"]) >= 1
        ref = reference_generate(FakeBackend(slow_ms=0), "fake-L6", prompts, 8)
        tok = FakeBackend().tokenizer("fake-L6")
        recs = [json.loads(l) for l in (tmp_path / "o.jsonl").read_text().splitlines()]
        assert [r["completion"] for r in recs] == [tok.decode(g) for g in ref]
    finally:
        for d in hosts:
            await d.stop()


@pytest.mark.asyncio
async def test_dashboard_render_and_discovery(tmp_path):
    from slugai.ring.dashboard import render
    from slugai.net.discovery import browse

    hosts = await start_hosts(2, lambda i: FakeBackend(slow_ms=0, free_mem=1 << 30))
    try:
        addrs = [HostAddr(d.name, "127.0.0.1", d.port) for d in hosts]
        server = Server(
            model="fake-L4", prompts=PROMPTS[:4], job="tdash", hosts=addrs, min_hosts=2, max_tokens=4,
            micro_batch=2, num_micro_batches=2, rebalance=False, output_path=str(tmp_path / "o.jsonl"),
            profile_batch_sizes=(1, 2),
        )
        await asyncio.wait_for(server.run(), 60)
        screen = render(server)
        assert "slugai · job tdash" in screen
        assert "phase 7/7 collect" in screen
        assert "hidden states" in screen and "token ids" in screen
        assert "4 / 4 sequences" in screen
        path = server.save_report(str(tmp_path / "r.json"))
        assert json.loads(open(path).read())["job"] == "tdash"
    finally:
        for d in hosts:
            await d.stop()

    # Bonjour: advertise one host and browse for it (skipped where multicast is unavailable)
    from slugai.net.discovery import Advertiser

    adv = Advertiser("zc-test-host", 12345, {"backend": "fake"}, ip="127.0.0.1")
    try:
        await adv.start()
    except Exception as e:  # pragma: no cover
        pytest.skip(f"zeroconf unavailable: {e}")
    try:
        found = await browse(timeout=8, min_hosts=1, settle=0.5)
    finally:
        await adv.stop()
    if not found:
        pytest.skip("no multicast in this environment")
    assert any(h.name == "zc-test-host" and h.port == 12345 for h in found)


@pytest.mark.asyncio
async def test_textual_dashboard_headless(tmp_path):
    from slugai.ring.dashboard import run_tui

    hosts = await start_hosts(2, lambda i: FakeBackend(slow_ms=0, free_mem=1 << 30))
    try:
        addrs = [HostAddr(d.name, "127.0.0.1", d.port) for d in hosts]
        server = Server(
            model="fake-L4", prompts=PROMPTS[:4], job="ttui", hosts=addrs, min_hosts=2, max_tokens=4,
            micro_batch=2, num_micro_batches=2, rebalance=False, output_path=str(tmp_path / "o.jsonl"),
            profile_batch_sizes=(1, 2),
        )
        rc = await asyncio.wait_for(run_tui(server, headless=True), 60)
        assert rc == 0 and server.state.finished and server.state.done_sequences == 4
    finally:
        for d in hosts:
            await d.stop()


@pytest.mark.asyncio
async def test_long_profile_keeps_control_connection(tmp_path, monkeypatch):
    """A profile (or load) that outlasts the heartbeat timeout must not drop the server."""
    import time

    import slugai.net.transport as transport

    monkeypatch.setattr(transport, "HEARTBEAT_INTERVAL", 0.1)
    monkeypatch.setattr(transport, "HEARTBEAT_TIMEOUT", 0.5)

    class SlowProfile(FakeBackend):
        def profile(self, *a, **kw):
            time.sleep(1.5)
            return super().profile(*a, **kw)

    hosts = await start_hosts(2, lambda i: SlowProfile(slow_ms=0, free_mem=1 << 30))
    try:
        addrs = [HostAddr(d.name, "127.0.0.1", d.port) for d in hosts]
        server = Server(
            model="fake-L4", prompts=PROMPTS[:4], job="thb", hosts=addrs, min_hosts=2, max_tokens=4,
            micro_batch=2, num_micro_batches=2, rebalance=False, output_path=str(tmp_path / "o.jsonl"),
            profile_batch_sizes=(1, 2),
        )
        report = await asyncio.wait_for(server.run(), 60)
        assert server.state.error is None and report["sequences_done"] == 4
    finally:
        for d in hosts:
            await d.stop()


@pytest.mark.asyncio
async def test_outputs_arriving_after_slot_free_are_kept(tmp_path):
    """The last stage's outputs can reach the server after stage 1's slot_free (different links)."""

    class LateOutputsHost(HostDaemon):
        async def _serve_control(self, conn, hello):
            send = conn.send

            async def late_send(frame):
                if frame.type != "outputs":
                    return await send(frame)

                async def later():
                    await asyncio.sleep(0.3)
                    await send(frame)

                asyncio.ensure_future(later())
                return 0

            conn.send = late_send
            await super()._serve_control(conn, hello)

    hosts = []
    for i in range(3):
        d = LateOutputsHost(f"mac{i}", 0, FakeBackend(slow_ms=0, free_mem=1 << 30), advertise=False, ip="127.0.0.1")
        await d.start()
        hosts.append(d)
    try:
        addrs = [HostAddr(d.name, "127.0.0.1", d.port) for d in hosts]
        server = Server(
            model="fake-L6", prompts=PROMPTS, job="tlate", hosts=addrs, min_hosts=3, max_tokens=6,
            micro_batch=4, num_micro_batches=3, rebalance=False, output_path=str(tmp_path / "o.jsonl"),
            profile_batch_sizes=(1, 4),
        )
        report = await asyncio.wait_for(server.run(), 60)
        assert report["sequences_done"] == len(PROMPTS)
        recs = [json.loads(l) for l in (tmp_path / "o.jsonl").read_text().splitlines()]
        assert all(r["completion"] is not None for r in recs)
    finally:
        for d in hosts:
            await d.stop()


@pytest.mark.asyncio
async def test_periodic_check_rebalances_once_when_a_host_throttles(tmp_path):
    """No forcing: the periodic check alone moves layers off a throttled Mac, then leaves them there."""
    backends = [FakeBackend(slow_ms=20, free_mem=1 << 30) for _ in range(3)]
    hosts = await start_hosts(3, lambda i: backends[i])
    try:
        addrs = [HostAddr(d.name, "127.0.0.1", d.port) for d in hosts]
        prompts = PROMPTS * 6
        server = Server(
            model="fake-L6", prompts=prompts, job="tpr", hosts=addrs, min_hosts=3, max_tokens=8,
            micro_batch=3, num_micro_batches=3, rebalance=True, rebalance_every=0.5,
            output_path=str(tmp_path / "o.jsonl"), profile_batch_sizes=(1, 3),
        )

        async def throttle_later():
            while server.state.phase != "decode":
                await asyncio.sleep(0.05)
            name = busiest(server.placement)
            d = next(h for h in hosts if h.name == name)
            d.backend.slow_ms = d.stage.slow_ms = 100.0  # stays slow across reloads
            return name

        t = asyncio.create_task(throttle_later())
        report = await asyncio.wait_for(server.run(), 120)
        slow = await t
        assert server.state.error is None
        assert len(report["rebalances"]) == 1, report["rebalances"]
        before, after = report["rebalances"][0]["from"], report["rebalances"][0]["to"]
        n = lambda p: next(b - a for name, (a, b) in zip(p["order"], p["layers"]) if name == slow)
        assert n(after) < n(before)
        ref = reference_generate(FakeBackend(slow_ms=0), "fake-L6", prompts, 8)
        tok = FakeBackend().tokenizer("fake-L6")
        recs = [json.loads(l) for l in (tmp_path / "o.jsonl").read_text().splitlines()]
        assert [r["completion"] for r in recs] == [tok.decode(g) for g in ref]
    finally:
        for d in hosts:
            await d.stop()


@pytest.mark.asyncio
async def test_ctrl_c_writes_partial_results(tmp_path):
    """--no-tui: the first Ctrl-C stops the job and still writes the completions so far."""
    import os
    import signal

    from slugai.ring.dashboard import run_plain

    hosts = await start_hosts(2, lambda i: FakeBackend(slow_ms=5, free_mem=1 << 30))
    try:
        addrs = [HostAddr(d.name, "127.0.0.1", d.port) for d in hosts]
        out = tmp_path / "o.jsonl"
        server = Server(
            model="fake-L4", prompts=PROMPTS * 6, job="tint", hosts=addrs, min_hosts=2, max_tokens=12,
            micro_batch=2, num_micro_batches=2, rebalance=False, output_path=str(out), profile_batch_sizes=(1, 2),
        )

        async def interrupt_after_some_outputs():
            while server.state.done_sequences < 2:
                await asyncio.sleep(0.02)
            os.kill(os.getpid(), signal.SIGINT)

        t = asyncio.create_task(interrupt_after_some_outputs())
        rc = await asyncio.wait_for(run_plain(server), 60)
        await t
        assert rc == 130
        recs = [json.loads(l) for l in out.read_text().splitlines()]
        done = [r for r in recs if r["completion"] is not None]
        assert len(recs) == len(PROMPTS) * 6 and 2 <= len(done) < len(recs)
    finally:
        for d in hosts:
            await d.stop()
