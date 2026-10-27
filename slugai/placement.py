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

Memory: stage i must hold KV cache for every sequence in flight,
``L_i * B * (k * c + s)`` with ``B = b * M`` sequences, context ``c`` (rounded
up to the cache's allocation block) and ``s`` bytes of fixed per-sequence
state (linear-attention layers), plus its layers' weights ``w`` each. Stage 1
also holds the embedding table, the last stage the final norm and lm_head.

Streaming (FlexGen-style): when a Mac's layers don't all fit, it keeps
``R_i`` of them in memory and *streams* the rest, reading each from disk for
every micro-batch through one layer's worth of spare room:
``R_i * w + w + L_i * B * (k * c + s) <= m_i``, and the stage's time grows by
``(L_i - R_i) * w / disk_bw_i``. That lets the ring run a model bigger than
all its memory, slowly; the split minimises the slowest stage either way.
"""

from __future__ import annotations

import itertools
from dataclasses import asdict, dataclass
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
    disk_bw: float = 0.0  # bytes/s reading model files; 0: can't stream layers

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
    resident: list[int]  # layers per stage kept in memory; the rest stream from disk
    micro_batch: int
    num_micro_batches: int
    step_time: float  # predicted seconds per decode round (all micro-batches once)
    predicted_tok_s: float
    peak_tok_s: float  # if links were free and no bubbles
    stage_compute: list[float]  # seconds per micro-batch per stage (incl. reading streamed layers)
    link_times: list[float]  # seconds per micro-batch per hop (last = return hop)
    context: int

    @property
    def sequences_in_flight(self) -> int:
        return self.micro_batch * self.num_micro_batches

    def to_dict(self) -> dict:
        d = asdict(self)
        d["layers"] = [list(x) for x in self.layers]
        return d


class PlacementError(RuntimeError):
    pass


# --------------------------------------------------------------------------- split


INF = float("inf")


def best_split(costs: list[list[float]], num_layers: int) -> Optional[tuple[list[int], float]]:
    """Contiguous split of ``num_layers`` over stages minimising the slowest stage.

    ``costs[s][l]`` is stage s's seconds per micro-batch with ``l`` layers
    (inf if it can't hold them). Every stage gets at least one layer. Returns
    (layers per stage, bottleneck seconds) or None if no split fits. Dynamic
    programme over (stage, layers so far), O(S * L^2); for 3 stages and 80
    layers this is a few thousand steps.
    """
    S = len(costs)
    if S == 0:
        return None
    # dp[s][n] = min bottleneck using first s stages for exactly n layers
    dp = [[INF] * (num_layers + 1) for _ in range(S + 1)]
    choice = [[0] * (num_layers + 1) for _ in range(S + 1)]
    dp[0][0] = 0.0
    for s in range(1, S + 1):
        c = costs[s - 1]
        for n in range(1, num_layers + 1):
            for l in range(1, n + 1):
                cost = max(dp[s - 1][n - l], c[l])
                if cost < dp[s][n]:
                    dp[s][n], choice[s][n] = cost, l
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


def _stage_costs(spec: ModelSpec, host: HostProfile, b: int, B: int, context: int, extra: int) -> tuple[list[float], list[int]]:
    """For 0..L layers on ``host``: seconds per micro-batch (inf: can't fit) and how many stay in memory.

    Layers that don't fit next to the caches are streamed: kept on disk and
    read for every micro-batch, through one layer's worth of spare room.
    """
    L, w = spec.num_layers, spec.layer_bytes
    t = host.time_per_layer(b)
    read = w / host.disk_bw if host.disk_bw > 0 else INF
    cache = B * spec.cache_bytes_per_seq(context)
    cost, keep = [INF] * (L + 1), [0] * (L + 1)
    for l in range(1, L + 1):
        room = host.free_mem - extra - l * cache
        if room >= l * w:
            cost[l], keep[l] = l * t, l
        elif read < INF and room >= w:
            r = int(room // w) - 1
            cost[l], keep[l] = l * t + (l - r) * read, r
    return cost, keep


def _stage_time(spec: ModelSpec, host: HostProfile, b: int, layers: int, resident: int) -> float:
    """Seconds per micro-batch for ``layers`` layers on ``host``, ``resident`` of them in memory."""
    streamed = layers - resident
    return layers * host.time_per_layer(b) + (streamed * spec.layer_bytes / host.disk_bw if streamed > 0 else 0.0)


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
    extras = _extras_for_order(spec, len(order))
    tables = [_stage_costs(spec, hosts[name], b, b * M, context, extras[i]) for i, name in enumerate(order)]
    sp = best_split([cost for cost, _ in tables], spec.num_layers)
    if sp is None:
        return None
    layers_n = sp[0]
    compute = [tables[i][0][l] for i, l in enumerate(layers_n)]
    resident = [tables[i][1][l] for i, l in enumerate(layers_n)]
    return _assemble(spec, links, order, layers_n, compute, resident, b, M, context)


def _assemble(
    spec: ModelSpec,
    links: dict[tuple[str, str], LinkProfile],
    order: list[str],
    layers_n: list[int],
    compute: list[float],
    resident: list[int],
    b: int,
    M: int,
    context: int,
) -> Optional[Placement]:
    """Predicted timings and throughput for a fixed ring order, layer counts and stage times."""
    S = len(order)
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
        resident=list(resident),
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
    layers_n = [e - a for a, e in current.layers]
    compute = [_stage_time(spec, hosts[n], b, l, r) for n, l, r in zip(current.order, layers_n, current.resident)]
    p = _assemble(spec, links, current.order, layers_n, compute, current.resident, b, M, current.context)
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
    Ms = [num_micro_batches] if num_micro_batches else range(S, max_micro_batches + 1)
    best: Optional[Placement] = None  # links keep up with the stages
    fallback: Optional[Placement] = None  # anything that fits, used if nothing does
    for order in itertools.permutations(names):
        for b in bs:
            for M in Ms:
                p = evaluate(spec, hosts, links, list(order), b, M, context)
                if p is None:
                    break  # more micro-batches only need more memory
                if fallback is None or p.predicted_tok_s > fallback.predicted_tok_s:
                    fallback = p
                links_keep_up = max(p.link_times) <= max(p.stage_compute) * 1.05 + 1e-9
                if (micro_batch or links_keep_up) and (best is None or p.predicted_tok_s > best.predicted_tok_s * (1 + 1e-6)):
                    best = p
                no_idle = M * max(max(p.stage_compute), max(p.link_times)) >= sum(p.stage_compute) + sum(p.link_times) - 1e-9
                if no_idle and not num_micro_batches:
                    break  # more micro-batches would only use more memory
    best = best or fallback
    if best is None:
        need = spec.num_layers * spec.layer_bytes + spec.embed_bytes + spec.head_bytes
        have = sum(h.free_mem for h in hosts.values())
        why = "not even streaming layers from disk leaves room for the caches" if any(h.disk_bw > 0 for h in hosts.values()) else "and the hosts can't stream layers from disk"
        raise PlacementError(
            f"model does not fit: {need / 1e9:.1f} GB of weights, hosts have {have / 1e9:.1f} GB free in total, "
            f"{why} (context {context})"
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
    We scale that host's profile (compute and disk reads alike) by observed /
    what the profile predicts for its current layers, so a Mac that is
    throttling gets fewer layers. ``hosts`` are always the original profiles,
    so the scale stays right after earlier rebalances.

    Returns ``(now, proposed)``: the current layout re-predicted with the
    measured speeds, and the best layout for them; compare the two with
    :func:`should_rebalance`.
    """
    b = current.micro_batch
    hosts2 = {}
    for name, h in hosts.items():
        scale = 1.0
        if name in measured_step and name in current.order:
            i = current.order.index(name)
            a, e = current.layers[i]
            predicted = _stage_time(spec, h, b, e - a, current.resident[i])
            if predicted > 0 and measured_step[name] > 0:
                scale = measured_step[name] / predicted
        hosts2[name] = HostProfile(
            name=h.name,
            free_mem=h.free_mem,
            layer_time={k: v * scale for k, v in h.layer_time.items()},
            disk_bw=h.disk_bw / scale,
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
