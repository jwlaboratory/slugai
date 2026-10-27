"""Server screen: a Textual dashboard (default) or plain log lines (``--no-tui``).

    slugai · job demo-01 · 70B 4-bit · 3 hosts
    phase 5/7 decode     elapsed 1:12:40   ETA 2:05:10

    HOST        STAGE  LAYERS  BUSY  STEP    MEMORY
    MacBook A   1      0-31    97%   1.52 s  35/48 GB
    ...
    LINK    USED        CAPACITY  CARRIES
    A > B   0.7 MB/s    48 MB/s   hidden states
    ...
    41 tok/s · 148k tok/h · 61% of peak
    1,212 / 3,000 sequences · 4 micro-batches x 64

    [r] rebalance  [s] save report  [q] quit
"""

from __future__ import annotations

import asyncio
import logging
import signal
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .server import Server

log = logging.getLogger(__name__)


def _hms(seconds: float | None) -> str:
    if seconds is None or seconds < 0 or seconds != seconds:
        return "--:--"
    s = int(seconds)
    h, rem = divmod(s, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def _rate(bps: float) -> str:
    if bps >= 1e6:
        return f"{bps / 1e6:.1f} MB/s"
    if bps >= 1e3:
        return f"{bps / 1e3:.1f} KB/s"
    return f"{bps:.0f} B/s"


def _gb(n: float) -> str:
    return f"{n / 1e9:.0f}" if n >= 10e9 else f"{n / 1e9:.1f}"


def render(server: "Server") -> str:
    st = server.state
    p = st.placement
    lines = []
    nh = len(st.hosts)
    lines.append(f"slugai · job {st.job} · {st.model} · {nh} host{'s' if nh != 1 else ''}")
    phase = f"phase {st.phase_index + 1}/7 {st.phase}"
    lines.append(f"{phase:<24} elapsed {_hms(st.elapsed):>8}   ETA {_hms(st.eta):>8}")
    lines.append("")
    name_w = max(10, max((len(n) for n in st.hosts), default=4))
    lines.append(f"{'HOST':<{name_w}}  STAGE  LAYERS   BUSY  STEP      MEMORY")
    order = p.order if p else list(st.hosts)
    for name in order:
        h = st.hosts.get(name, {})
        stage = h.get("stage")
        layers = h.get("layers")
        layers_s = f"{layers[0]}-{layers[1] - 1}" if layers else "-"
        busy = f"{h.get('busy', 0.0) * 100:3.0f}%"
        step = f"{h.get('step', 0.0):.2f} s" if h.get("step") else "-"
        mem = f"{_gb(h.get('mem_used', 0))}/{_gb(h.get('mem_total', 0))} GB" if h.get("mem_total") else "-"
        lines.append(f"{name:<{name_w}}  {str(stage + 1) if stage is not None else '-':<5}  {layers_s:<8} {busy:>4}  {step:<9} {mem}")
    lines.append("")
    if st.links:
        lw = max(7, max(len(f"{l['src']} > {l['dst']}") for l in st.links))
        lines.append(f"{'LINK':<{lw}}  USED        CAPACITY    CARRIES")
        for l in st.links:
            lines.append(f"{l['src'] + ' > ' + l['dst']:<{lw}}  {_rate(l['used']):<10}  {_rate(l['capacity']):<10}  {l['carries']}")
        lines.append("")
    pct = f" · {st.tok_s / st.peak_tok_s * 100:.0f}% of peak" if st.peak_tok_s else ""
    lines.append(f"{st.tok_s:.1f} tok/s · {st.tok_s * 3600 / 1000:.0f}k tok/h{pct}")
    mb = f" · {p.num_micro_batches} micro-batches x {p.micro_batch}" if p else ""
    lines.append(f"{st.done_sequences:,} / {st.total_sequences:,} sequences{mb} · {st.tokens_generated:,} tokens")
    if st.rebalances:
        lines.append(f"rebalanced {st.rebalances}x")
    if st.error:
        lines.append(f"ERROR: {st.error}")
    lines.append("")
    for ts, msg in list(st.events)[-6:]:
        lines.append(f"{time.strftime('%H:%M:%S', time.localtime(ts))}  {msg}")
    lines.append("")
    lines.append("[r] rebalance  [s] save report  [q] quit")
    return "\n".join(lines)


async def run_plain(server: "Server", interval: float = 5.0) -> int:
    """No dashboard: log phase changes and a status line every ``interval`` seconds.

    The first Ctrl-C stops the job and still writes the completions so far;
    a second one aborts without writing.
    """
    task = asyncio.create_task(server.run())
    loop = asyncio.get_running_loop()
    interrupted = False

    def on_sigint() -> None:
        nonlocal interrupted
        if interrupted:
            task.cancel()
            return
        interrupted = True
        server.state.log("stopping: writing the completions so far (Ctrl-C again to abort)")
        server.stop()

    try:
        loop.add_signal_handler(signal.SIGINT, on_sigint)
    except (NotImplementedError, RuntimeError, ValueError):  # no signals here (not the main thread)
        pass
    last = 0.0
    try:
        while not task.done():
            await asyncio.sleep(0.5)
            st = server.state
            if st.phase == "decode" and time.monotonic() - last >= interval:
                last = time.monotonic()
                log.info(
                    "%s/%s sequences · %.1f tok/s · %s elapsed · ETA %s · %s",
                    st.done_sequences,
                    st.total_sequences,
                    st.tok_s,
                    _hms(st.elapsed),
                    _hms(st.eta),
                    "  ".join(f"{n}:{h.get('busy', 0) * 100:.0f}%/{h.get('step', 0):.2f}s" for n, h in st.hosts.items()),
                )
        await task
    except asyncio.CancelledError:
        if not task.cancelled():
            raise
        log.warning("aborted; no results written")
        return 130
    finally:
        try:
            loop.remove_signal_handler(signal.SIGINT)
        except (NotImplementedError, RuntimeError, ValueError):
            pass
    if interrupted:
        return 130
    return 1 if server.state.error else 0


async def run_tui(server: "Server", headless: bool = False) -> int:
    try:
        from textual.app import App, ComposeResult
        from textual.widgets import Static
    except ImportError:
        log.warning("textual not installed; falling back to plain output")
        return await run_plain(server)

    class Dashboard(App):
        CSS = "Screen { background: $background; } #screen { padding: 1 2; }"
        BINDINGS = [("r", "rebalance", "rebalance"), ("s", "save", "save report"), ("q", "quit_job", "quit")]

        def compose(self) -> ComposeResult:
            yield Static(render(server), id="screen", markup=False)

        async def on_mount(self) -> None:
            self.set_interval(0.5, self.refresh_screen)
            self.job_task = asyncio.create_task(server.run())

        def refresh_screen(self) -> None:
            self.query_one("#screen", Static).update(render(server))
            if server.state.finished and self.job_task.done() and not getattr(self, "_exiting", False):
                self._exiting = True
                self.set_timer(0.2 if headless else 2.0, self.exit)

        def action_rebalance(self) -> None:
            if not server.maybe_rebalance(force=True):
                server.state.log("rebalance: nothing to change")

        def action_save(self) -> None:
            try:
                server.save_report()
            except Exception as e:
                server.state.log(f"save failed: {e}")

        async def action_quit(self) -> None:  # Textual's ctrl+q: stop the job properly too
            await self.action_quit_job()

        async def action_quit_job(self) -> None:
            server.stop()
            try:
                await asyncio.wait_for(self.job_task, 10)
            except Exception:
                pass
            self.exit()

    app = Dashboard()
    await app.run_async(headless=headless)
    # print the final screen so it stays in the terminal scrollback
    print(render(server))
    return 1 if server.state.error else 0
