"""Placement: which layers go on which Mac, in what ring order, with what micro-batches.

Everything here is pure arithmetic on measured profiles, so it is fast and
fully testable. The server runs it once after profiling and again whenever a
rebalance is triggered.

Model (per decode step, one micro-batch of ``b`` sequences):

* stage i computes its ``L_i`` layers in ``L_i * t_i(b)`` seconds,
* hop i -> i+1 carries hidden states (``b * hidden * 2`` bytes, fp16) in
  ``bytes / bandwidth + latency`` seconds,
* the return hop (last stage -> first) carries only token ids (``4 * b`` bytes).

With ``M`` micro-batches in flight the busiest resource is never idle once
``M * bottleneck >= lap``, where ``lap`` is one micro-batch's full trip around
the ring. Throughput is then ``b * M / max(M * bottleneck, lap)`` tokens/s.

Memory: stage i must hold its weights plus KV cache for every sequence in
flight: ``L_i * (w + B * (k * c + s)) <= m_i`` with ``B = b * M`` sequences,
context ``c`` (rounded up to the cache's allocation block) and ``s`` bytes of
fixed per-sequence state (linear-attention layers). Stage 1 also holds the
embedding table, the last stage the final norm and lm_head.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field, asdict
from typing import Iterable, Optional

import numpy as np

MICRO_BATCH_CANDIDATES = (128, 96, 64, 48, 32, 24, 16, 12, 8, 6, 4, 2, 1)


@dataclass
class ModelSpec:
    """Sizes the placement needs; derived from the model config by the hosts."""

    num_layers: int
    hidden_size: int
    layer_bytes: int  # w: weight bytes per layer
    kv_bytes_per_token: int  # k: KV bytes per layer per token
    embed_bytes: int = 0  # extra on stage 1
    head_bytes: int = 0  # extra on the last stage (norm + lm_head)
    hidden_dtype_bytes: int = 2  # bytes per hidden-state element on the wire
    state_bytes_per_seq: int = 0  # s: fixed cache bytes per layer per sequence (linear attention)
    kv_block_tokens: int = 1  # KV caches grow in blocks of this many tokens

    def hidden_bytes(self, b: int, tokens: int = 1) -> int:
        return b * tokens * self.hidden_size * self.hidden_dtype_bytes

    def cache_bytes_per_seq(self, context: int) -> int:
        """Cache bytes one sequence needs per layer at ``context`` tokens."""
        block = max(1, self.kv_block_tokens)
        return self.kv_bytes_per_token * (-(-context // block) * block) + self.state_bytes_per_seq


@dataclass
class HostProfile:
    name: str
    free_mem: int  # bytes usable for weights + KV
    # micro-batch size -> seconds per layer per decode step
    layer_time: dict[int, float]
    # optional: seconds per layer per 1k prompt tokens during prefill
    prefill_layer_time_per_token: float = 0.0
    extra: dict = field(default_factory=dict)

    def time_per_layer(self, b: int) -> float:
        """Linear interpolation (and extrapolation) of the profiled points."""
        pts = sorted((int(k), float(v)) for k, v in self.layer_time.items())
        if not pts:
            raise ValueError(f"host {self.name} has no layer timings")
        if len(pts) == 1:
            # assume time scales ~linearly with batch beyond the single point
            b0, t0 = pts[0]
            return t0 * max(b, 1) / max(b0, 1) if b > b0 else t0
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        if b <= xs[0]:
            return ys[0] if b == xs[0] else max(ys[0] * b / xs[0], ys[0] * 0.5)
        if b >= xs[-1]:
            (x0, y0), (x1, y1) = pts[-2], pts[-1]
            slope = (y1 - y0) / max(x1 - x0, 1)
            return max(y1 + slope * (b - x1), y1)
        return float(np.interp(b, xs, ys))


@dataclass
class LinkProfile:
    src: str
    dst: str
    bandwidth: float  # bytes per second
    latency: float  # seconds (one-way)

    def time(self, nbytes: int) -> float:
        return nbytes / max(self.bandwidth, 1.0) + self.latency


@dataclass
class Placement:
    order: list[str]  # host names in ring order
    layers: list[tuple[int, int]]  # [start, end) per stage
    micro_batch: int
    num_micro_batches: int
    step_time: float  # predicted seconds per decode round (all micro-batches once)
    predicted_tok_s: float
    peak_tok_s: float  # if links were free and no bubbles
    stage_compute: list[float]  # seconds per micro-batch per stage
    link_times: list[float]  # seconds per micro-batch per hop (last = return hop)
    context: int

    @property
    def sequences_in_flight(self) -> int:
        return self.micro_batch * self.num_micro_batches

    def stage_of(self, host: str) -> int:
        return self.order.index(host)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["layers"] = [list(x) for x in self.layers]
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Placement":
        d = dict(d)
        d["layers"] = [tuple(x) for x in d["layers"]]
        return cls(**d)


class PlacementError(RuntimeError):
    pass


# --------------------------------------------------------------------------- split


def best_split(times: list[float], caps: list[int], num_layers: int) -> Optional[tuple[list[int], float]]:
    """Contiguous split of ``num_layers`` over stages minimising ``max_i L_i * t_i``.

    ``caps[i]`` is the most layers stage i can hold. Every stage gets at least
    one layer. Returns (layers per stage, bottleneck seconds) or None if the
    caps cannot hold the model. Dynamic programme over (stage, layers so far),
    O(S * L^2); for 3 stages and 80 layers this is a few thousand steps.
    """
    S = len(times)
    if S == 0 or sum(caps) < num_layers or any(c < 1 for c in caps):
        return None
    INF = float("inf")
    # dp[s][n] = min bottleneck using first s stages for exactly n layers
    dp = [[INF] * (num_layers + 1) for _ in range(S + 1)]
    choice = [[0] * (num_layers + 1) for _ in range(S + 1)]
    dp[0][0] = 0.0
    for s in range(1, S + 1):
        t = times[s - 1]
        cap = caps[s - 1]
        for n in range(1, num_layers + 1):
            best = INF
            best_l = 0
            for l in range(1, min(cap, n) + 1):
                prev = dp[s - 1][n - l]
                if prev == INF:
                    continue
                cost = max(prev, l * t)
                if cost < best:
                    best, best_l = cost, l
            dp[s][n] = best
            choice[s][n] = best_l
    if dp[S][num_layers] == INF:
        return None
    out = []
    n = num_layers
    for s in range(S, 0, -1):
        l = choice[s][n]
        out.append(l)
        n -= l
    out.reverse()
    return out, dp[S][num_layers]


def _balanced_split_tiebreak(times: list[float], caps: list[int], num_layers: int, bottleneck: float) -> list[int]:
    """Among splits achieving ``bottleneck``, prefer the one whose stage times are most even.

    best_split returns a valid optimum; here we re-derive the split by filling
    each stage up to the bottleneck and then handing leftovers to the stage with
    the most slack, which spreads idle time instead of piling it on one Mac.
    """
    S = len(times)
    layers = [min(caps[i], int(math.floor(bottleneck / times[i] + 1e-9))) for i in range(S)]
    layers = [max(1, l) for l in layers]
    total = sum(layers)
    # too many: trim from the stages with most time
    while total > num_layers:
        i = max(range(S), key=lambda j: layers[j] * times[j] if layers[j] > 1 else -1)
        layers[i] -= 1
        total -= 1
    while total < num_layers:
        cands = [j for j in range(S) if layers[j] < caps[j]]
        if not cands:
            break
        i = min(cands, key=lambda j: (layers[j] + 1) * times[j])
        layers[i] += 1
        total += 1
    if total != num_layers or max(l * t for l, t in zip(layers, times)) > bottleneck * (1 + 1e-9):
        sp = best_split(times, caps, num_layers)
        return sp[0] if sp else layers
    return layers


# --------------------------------------------------------------------------- solve


def _extras_for_order(spec: ModelSpec, S: int) -> list[int]:
    ex = [0] * S
    ex[0] += spec.embed_bytes
    ex[-1] += spec.head_bytes
    return ex


def evaluate(
    spec: ModelSpec,
    hosts: dict[str, HostProfile],
    links: dict[tuple[str, str], LinkProfile],
    order: list[str],
    b: int,
    M: int,
    context: int,
) -> Optional[Placement]:
    """Best placement for a fixed ring order, micro-batch size and count; None if it doesn't fit."""
    S = len(order)
    B = b * M
    per_layer = spec.layer_bytes + B * spec.cache_bytes_per_seq(context)
    extras = _extras_for_order(spec, S)
    caps = []
    for i, name in enumerate(order):
        avail = hosts[name].free_mem - extras[i]
        caps.append(max(0, int(avail // per_layer)))
    times = [hosts[name].time_per_layer(b) for name in order]
    sp = best_split(times, caps, spec.num_layers)
    if sp is None:
        return None
    _, bottleneck = sp
    layers_n = _balanced_split_tiebreak(times, caps, spec.num_layers, bottleneck)
    return _assemble(spec, links, order, layers_n, times, b, M, context)


def _assemble(
    spec: ModelSpec,
    links: dict[tuple[str, str], LinkProfile],
    order: list[str],
    layers_n: list[int],
    times: list[float],
    b: int,
    M: int,
    context: int,
) -> Optional[Placement]:
    """Predicted timings and throughput for a fixed ring order and layer counts."""
    S = len(order)
    compute = [l * t for l, t in zip(layers_n, times)]
    hops = []
    for i in range(S):
        src, dst = order[i], order[(i + 1) % S]
        link = links.get((src, dst))
        if link is None:
            return None
        nbytes = 4 * b if i == S - 1 else spec.hidden_bytes(b)
        hops.append(link.time(nbytes))
    if S == 1:
        hops = [0.0]
    resource = max(max(compute), max(hops))
    lap = sum(compute) + sum(hops)
    step = max(M * resource, lap)
    tok_s = b * M / step
    peak = b / max(compute)
    bounds = []
    start = 0
    for l in layers_n:
        bounds.append((start, start + l))
        start += l
    return Placement(
        order=list(order),
        layers=bounds,
        micro_batch=b,
        num_micro_batches=M,
        step_time=step,
        predicted_tok_s=tok_s,
        peak_tok_s=peak,
        stage_compute=compute,
        link_times=hops,
        context=context,
    )


def predict(
    current: Placement,
    spec: ModelSpec,
    hosts: dict[str, HostProfile],
    links: dict[tuple[str, str], LinkProfile],
) -> Placement:
    """``current``'s layout (order, layers, micro-batches) re-predicted with other host timings."""
    b, M = current.micro_batch, current.num_micro_batches
    times = [hosts[name].time_per_layer(b) for name in current.order]
    layers_n = [e - a for a, e in current.layers]
    p = _assemble(spec, links, current.order, layers_n, times, b, M, current.context)
    if p is None:
        raise PlacementError("no link profile for the current ring")
    return p


def solve(
    spec: ModelSpec,
    hosts: dict[str, HostProfile],
    links: dict[tuple[str, str], LinkProfile],
    context: int,
    micro_batch: Optional[int] = None,
    num_micro_batches: Optional[int] = None,
    max_micro_batches: int = 8,
    candidates: Iterable[int] = MICRO_BATCH_CANDIDATES,
    max_sequences: Optional[int] = None,
) -> Placement:
    """Choose ring order, layer split, micro-batch size and count.

    Follows the design doc:

    1. try all ring orders (``S!``) — the slowest link naturally lands on the
       return hop because that hop carries only token ids;
    2. pick the largest micro-batch size every link can keep up with (no hop
       slower than the slowest stage) that fits in memory with ``S``
       micro-batches in flight;
    3. add micro-batches until no Mac idles (``M * bottleneck >= lap``) or KV
       memory runs out.

    Among everything that satisfies those rules the highest predicted
    throughput wins. ``micro_batch`` / ``num_micro_batches`` pin a choice.
    ``max_sequences`` (the job size) caps the micro-batch so a small job is
    still spread over ``S`` micro-batches instead of filling one.
    """
    names = list(hosts)
    S = len(names)
    if S == 0:
        raise PlacementError("no hosts")
    bs = [micro_batch] if micro_batch else [c for c in candidates]
    if not micro_batch and max_sequences:
        cap = max(1, -(-max_sequences // S))
        bs = sorted({c for c in bs if c <= cap} | {min(cap, max(bs))}, reverse=True)
    best: Optional[Placement] = None
    for order in itertools.permutations(names):
        order = list(order)
        for b in bs:
            Ms = [num_micro_batches] if num_micro_batches else range(S, max_micro_batches + 1)
            chosen_for_b: Optional[Placement] = None
            for M in Ms:
                p = evaluate(spec, hosts, links, order, b, M, context)
                if p is None:
                    break  # more micro-batches only need more memory
                links_keep_up = max(p.link_times) <= max(p.stage_compute) * 1.05 + 1e-9
                if not links_keep_up and not micro_batch:
                    chosen_for_b = None
                    break  # this b is too big for the links; try a smaller one
                chosen_for_b = p
                no_idle = M * max(max(p.stage_compute), max(p.link_times)) >= sum(p.stage_compute) + sum(p.link_times) - 1e-9
                if no_idle and not num_micro_batches:
                    break
            if chosen_for_b is None:
                continue
            if best is None or chosen_for_b.predicted_tok_s > best.predicted_tok_s * (1 + 1e-6):
                best = chosen_for_b
            if not micro_batch and best is not None and best.micro_batch == b:
                # larger micro-batches already failed; a smaller one is only
                # better if it fits links/memory better, which the loop handles
                pass
    if best is None:
        # fall back: ignore the "links keep up" rule, take anything that fits
        for order in itertools.permutations(names):
            for b in bs:
                for M in ([num_micro_batches] if num_micro_batches else range(S, max_micro_batches + 1)):
                    p = evaluate(spec, hosts, links, list(order), b, M, context)
                    if p is None:
                        break
                    if best is None or p.predicted_tok_s > best.predicted_tok_s:
                        best = p
    if best is None:
        need = spec.num_layers * spec.layer_bytes + spec.embed_bytes + spec.head_bytes
        have = sum(h.free_mem for h in hosts.values())
        raise PlacementError(
            f"model does not fit: needs >= {need / 1e9:.1f} GB of weights across hosts, "
            f"hosts have {have / 1e9:.1f} GB free in total (context {context})"
        )
    return best


def should_rebalance(current: Placement, proposed: Placement, min_gain: float = 0.10) -> bool:
    """Move layers only when the new placement is clearly better.

    Both placements must be predicted from the same timings: pass the current
    layout as re-predicted by :func:`rebalance_with_measurements`, not the
    original plan.
    """
    if proposed.predicted_tok_s <= 0:
        return False
    gain = proposed.predicted_tok_s / max(current.predicted_tok_s, 1e-9) - 1.0
    return not same_layout(current, proposed) and gain >= min_gain


def same_layout(a: Placement, b: Placement) -> bool:
    return a.order == b.order and [tuple(x) for x in a.layers] == [tuple(x) for x in b.layers]


def rebalance_with_measurements(
    current: Placement,
    spec: ModelSpec,
    hosts: dict[str, HostProfile],
    links: dict[tuple[str, str], LinkProfile],
    measured_step: dict[str, float],
) -> tuple[Placement, Placement]:
    """Fold live per-stage step times back into the host profiles and re-solve.

    ``measured_step[host]`` is the observed seconds per micro-batch on that host.
    We scale that host's profiled time-per-layer by observed / what the profile
    predicts for its current layers, so a Mac that is throttling gets fewer
    layers. ``hosts`` are always the original profiles, so the scale stays
    right after earlier rebalances.

    Returns ``(now, proposed)``: the current layout re-predicted with the
    measured speeds, and the best layout for them; compare the two with
    :func:`should_rebalance`.
    """
    b = current.micro_batch
    hosts2 = {}
    for name, h in hosts.items():
        scale = 1.0
        if name in measured_step and name in current.order:
            a, e = current.layers[current.order.index(name)]
            predicted = (e - a) * h.time_per_layer(b)
            if predicted > 0 and measured_step[name] > 0:
                scale = measured_step[name] / predicted
        hosts2[name] = HostProfile(
            name=h.name,
            free_mem=h.free_mem,
            layer_time={k: v * scale for k, v in h.layer_time.items()},
            prefill_layer_time_per_token=h.prefill_layer_time_per_token,
            extra=h.extra,
        )
    now = predict(current, spec, hosts2, links)
    proposed = solve(
        spec,
        hosts2,
        links,
        context=current.context,
        micro_batch=current.micro_batch,
        num_micro_batches=current.num_micro_batches,
    )
    return now, proposed
