"""End to end: a server and its hosts in one process, tiny real models, real sockets."""

import asyncio
import json
import time

import pytest

pytest.importorskip("mlx.core")
pytest.importorskip("mlx_lm")

from slugai.backend import Backend  # noqa: E402
from slugai.ring.host import HostDaemon  # noqa: E402
from slugai.ring.server import Server  # noqa: E402
from tests.helpers import PROMPTS, SlowBackend, addrs, mlx_lm_reference, one_machine, start_hosts  # noqa: E402


def busiest(placement):
    """The host holding the most layers: slowing it down always leaves layers to move."""
    return max(zip(placement.order, placement.layers), key=lambda nl: nl[1][1] - nl[1][0])[0]


def records(path):
    return [json.loads(l) for l in path.read_text().splitlines()]


async def stop_all(hosts):
    for d in hosts:
        await d.stop()


@pytest.mark.asyncio
async def test_ring_matches_mlx_lm_generate(tiny_model, tmp_path):
    """Three hosts produce mlx-lm's own greedy tokens, for every architecture."""
    prompts = ["hello world", "the quick brown fox", "zebra", "a", "once upon a time there was", "x y z"]
    ref = mlx_lm_reference(tiny_model, prompts, 8)
    hosts = await start_hosts(3)
    try:
        out, rep = tmp_path / "o.jsonl", tmp_path / "report.json"
        server = Server(
            model=str(tiny_model), prompts=prompts, job="tring", hosts=addrs(hosts), min_hosts=3, max_tokens=8,
            micro_batch=4, num_micro_batches=2, rebalance=False, output_path=str(out), report_path=str(rep),
            profile_batch_sizes=(1, 4),
        )
        report = await asyncio.wait_for(server.run(), 300)
        assert server.state.error is None
        assert report["sequences_done"] == len(prompts)
        # every host got a contiguous block and all layers are covered
        layers = report["placement"]["layers"]
        assert len(layers) == 3 and layers[0][0] == 0 and layers[-1][1] == report["model_spec"]["num_layers"]
        recs = records(out)
        assert [r["id"] for r in recs] == list(range(len(prompts)))
        assert [[t for t in r["token_ids"] if t != 2] for r in recs] == ref
        assert json.loads(rep.read_text())["tokens_generated"] == sum(r["tokens"] for r in recs)
    finally:
        await stop_all(hosts)


@pytest.mark.asyncio
async def test_single_host_ring(tiny, tmp_path):
    hosts = await start_hosts(1)
    try:
        server = Server(
            model=str(tiny), prompts=PROMPTS[:5], job="t1", hosts=addrs(hosts), min_hosts=1, max_tokens=6,
            micro_batch=2, num_micro_batches=2, rebalance=False, output_path=str(tmp_path / "o.jsonl"),
            profile_batch_sizes=(1, 2),
        )
        report = await asyncio.wait_for(server.run(), 60)
        assert report["sequences_done"] == 5
        # hosts profile at the job's typical cache length: mean prompt + half the new tokens
        lengths = [len(Backend(use_chat_template=False).tokenizer(str(tiny)).encode(p)) for p in PROMPTS[:5]]
        assert report["profile_context"] == round(sum(lengths) / len(lengths)) + 6 // 2
        ref = one_machine(tiny, PROMPTS[:5], 6, batch=2)
        assert [r["token_ids"] for r in records(tmp_path / "o.jsonl")] == ref
    finally:
        await stop_all(hosts)


@pytest.mark.asyncio
async def test_rebalance_moves_layers_when_a_host_slows(tiny, tmp_path):
    """A host is profiled fast but then runs slow: a forced rebalance must re-place and finish correctly."""
    hosts = await start_hosts(3, lambda i: SlowBackend())
    try:
        prompts = PROMPTS * 4
        server = Server(
            model=str(tiny), prompts=prompts, job="trb", hosts=addrs(hosts), min_hosts=3, max_tokens=8,
            micro_batch=3, num_micro_batches=3, rebalance=True, rebalance_every=0.5,
            output_path=str(tmp_path / "o.jsonl"), profile_batch_sizes=(1, 3),
        )

        async def slow_down_later():
            # wait until decoding, then slow the host with the most layers (so some can move) and force a rebalance
            while server.state.phase != "decode":
                await asyncio.sleep(0.05)
            d = next(h for h in hosts if h.name == busiest(server.placement))
            d.backend.ms = 30.0
            await asyncio.sleep(1.0)  # let stats reflect the slowdown
            server.maybe_rebalance(force=True)  # may already be draining from the periodic check

        t = asyncio.create_task(slow_down_later())
        report = await asyncio.wait_for(server.run(), 120)
        await t
        assert server.state.error is None
        assert report["sequences_done"] == len(prompts)
        assert len(report["rebalances"]) >= 1
        assert [r["token_ids"] for r in records(tmp_path / "o.jsonl")] == one_machine(tiny, prompts, 8, batch=3)
    finally:
        await stop_all(hosts)


@pytest.mark.asyncio
async def test_periodic_check_rebalances_once_when_a_host_throttles(tiny, tmp_path):
    """No forcing: the periodic check alone moves layers off a throttled Mac, then leaves them there."""
    hosts = await start_hosts(3, lambda i: SlowBackend(ms=20))
    try:
        prompts = PROMPTS * 4
        server = Server(
            model=str(tiny), prompts=prompts, job="tpr", hosts=addrs(hosts), min_hosts=3, max_tokens=8,
            micro_batch=3, num_micro_batches=3, rebalance=True, rebalance_every=0.5,
            output_path=str(tmp_path / "o.jsonl"), profile_batch_sizes=(1, 3),
        )

        async def throttle_later():
            while server.state.phase != "decode":
                await asyncio.sleep(0.05)
            name = busiest(server.placement)
            next(h for h in hosts if h.name == name).backend.ms = 60.0  # stays slow across reloads
            return name

        t = asyncio.create_task(throttle_later())
        report = await asyncio.wait_for(server.run(), 120)
        slow = await t
        assert server.state.error is None
        assert len(report["rebalances"]) == 1, report["rebalances"]
        before, after = report["rebalances"][0]["from"], report["rebalances"][0]["to"]
        n = lambda p: next(b - a for name, (a, b) in zip(p["order"], p["layers"]) if name == slow)
        assert n(after) < n(before)
        assert [r["token_ids"] for r in records(tmp_path / "o.jsonl")] == one_machine(tiny, prompts, 8, batch=3)
    finally:
        await stop_all(hosts)


@pytest.mark.asyncio
async def test_dashboard_render_and_discovery(tiny, tmp_path):
    from slugai.net.discovery import Advertiser, browse
    from slugai.ring.dashboard import render

    hosts = await start_hosts(2)
    try:
        server = Server(
            model=str(tiny), prompts=PROMPTS[:4], job="tdash", hosts=addrs(hosts), min_hosts=2, max_tokens=4,
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
        await stop_all(hosts)

    # Bonjour: advertise one host and browse for it (skipped where multicast is unavailable)
    adv = Advertiser("zc-test-host", 12345, ip="127.0.0.1")
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
async def test_textual_dashboard_headless(tiny, tmp_path):
    from slugai.ring.dashboard import run_tui

    hosts = await start_hosts(2)
    try:
        server = Server(
            model=str(tiny), prompts=PROMPTS[:4], job="ttui", hosts=addrs(hosts), min_hosts=2, max_tokens=4,
            micro_batch=2, num_micro_batches=2, rebalance=False, output_path=str(tmp_path / "o.jsonl"),
            profile_batch_sizes=(1, 2),
        )
        rc = await asyncio.wait_for(run_tui(server, headless=True), 60)
        assert rc == 0 and server.state.finished and server.state.done_sequences == 4
    finally:
        await stop_all(hosts)


@pytest.mark.asyncio
async def test_long_profile_keeps_control_connection(tiny, tmp_path, monkeypatch):
    """A profile (or load) that outlasts the heartbeat timeout must not drop the server."""
    import slugai.net.transport as transport

    monkeypatch.setattr(transport, "HEARTBEAT_INTERVAL", 0.1)
    monkeypatch.setattr(transport, "HEARTBEAT_TIMEOUT", 0.5)

    class SlowProfile(Backend):
        def profile(self, *a, **kw):
            time.sleep(1.5)
            return super().profile(*a, **kw)

    hosts = await start_hosts(2, lambda i: SlowProfile(use_chat_template=False))
    try:
        server = Server(
            model=str(tiny), prompts=PROMPTS[:4], job="thb", hosts=addrs(hosts), min_hosts=2, max_tokens=4,
            micro_batch=2, num_micro_batches=2, rebalance=False, output_path=str(tmp_path / "o.jsonl"),
            profile_batch_sizes=(1, 2),
        )
        report = await asyncio.wait_for(server.run(), 60)
        assert server.state.error is None and report["sequences_done"] == 4
    finally:
        await stop_all(hosts)


@pytest.mark.asyncio
async def test_outputs_arriving_after_slot_free_are_kept(tiny, tmp_path):
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

    hosts = await start_hosts(3, host_class=LateOutputsHost)
    try:
        server = Server(
            model=str(tiny), prompts=PROMPTS, job="tlate", hosts=addrs(hosts), min_hosts=3, max_tokens=6,
            micro_batch=4, num_micro_batches=3, rebalance=False, output_path=str(tmp_path / "o.jsonl"),
            profile_batch_sizes=(1, 4),
        )
        report = await asyncio.wait_for(server.run(), 60)
        assert report["sequences_done"] == len(PROMPTS)
        assert [r["token_ids"] for r in records(tmp_path / "o.jsonl")] == one_machine(tiny, PROMPTS, 6, batch=4)
    finally:
        await stop_all(hosts)


@pytest.mark.asyncio
async def test_ctrl_c_writes_partial_results(tiny, tmp_path):
    """--no-tui: the first Ctrl-C stops the job and still writes the completions so far."""
    import os
    import signal

    from slugai.ring.dashboard import run_plain

    hosts = await start_hosts(2, lambda i: SlowBackend(ms=5))
    try:
        out = tmp_path / "o.jsonl"
        server = Server(
            model=str(tiny), prompts=PROMPTS * 6, job="tint", hosts=addrs(hosts), min_hosts=2, max_tokens=12,
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
        recs = records(out)
        done = [r for r in recs if r["completion"] is not None]
        assert len(recs) == len(PROMPTS) * 6 and 2 <= len(done) < len(recs)
    finally:
        await stop_all(hosts)


@pytest.mark.asyncio
async def test_ring_streams_layers_that_do_not_fit(tiny, tmp_path):
    """Hosts too small for the model stream the rest of their layers from disk, with identical output."""
    spec = Backend(use_chat_template=False).model_info(str(tiny)).spec
    # room for half the layers' caches (8 sequences, one 256-token block), the embedding and head,
    # and 2.5 layers of weights: one resident, one to stream through
    room = spec.embed_bytes + spec.head_bytes + spec.num_layers // 2 * 8 * spec.cache_bytes_per_seq(1) + int(2.5 * spec.layer_bytes)

    class SmallMemory(Backend):
        def free_memory(self):
            return room

    hosts = await start_hosts(2, lambda i: SmallMemory(use_chat_template=False))
    try:
        server = Server(
            model=str(tiny), prompts=PROMPTS, job="tstream", hosts=addrs(hosts), min_hosts=2, max_tokens=6,
            micro_batch=4, num_micro_batches=2, rebalance=False, output_path=str(tmp_path / "o.jsonl"),
            profile_batch_sizes=(1, 4),
        )
        report = await asyncio.wait_for(server.run(), 60)
        assert server.state.error is None and report["sequences_done"] == len(PROMPTS)
        p = report["placement"]
        assert sum(b - a for a, b in p["layers"]) - sum(p["resident"]) >= 3  # most layers stream
        assert [r["token_ids"] for r in records(tmp_path / "o.jsonl")] == one_machine(tiny, PROMPTS, 6, batch=4)
    finally:
        await stop_all(hosts)
