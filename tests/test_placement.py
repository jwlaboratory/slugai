import itertools

import pytest

from slugai.placement import (
    HostProfile, LinkProfile, ModelSpec, Placement, PlacementError,
    best_split, rebalance_with_measurements, should_rebalance, solve,
)

GB = 1 << 30


def costs(times, caps, L):
    """Resident-only stage cost tables: l layers take l * t seconds, up to the cap."""
    return [[l * t if l <= cap else float("inf") for l in range(L + 1)] for t, cap in zip(times, caps)]


def spec_70b():
    # 70B-class dense, 4-bit: ~0.5 GB/layer, hidden 8192, KV = 2*8*128*2 bytes = 4 KB/layer/token (GQA)
    return ModelSpec(num_layers=80, hidden_size=8192, layer_bytes=int(0.5 * GB), kv_bytes_per_token=4 * 1024,
                     embed_bytes=int(0.6 * GB), head_bytes=int(0.6 * GB))


def hosts_abc():
    def prof(name, mem, t):
        return HostProfile(name, mem, {1: t * 0.6, 8: t * 0.7, 16: t * 0.8, 32: t * 0.9, 64: t})
    return {
        "A": prof("A", 48 * GB, 0.045),
        "B": prof("B", 36 * GB, 0.050),
        "C": prof("C", 24 * GB, 0.080),
    }


def links_abc(slow=("B", "A")):
    links = {}
    for s, d in itertools.permutations("ABC", 2):
        bw, lat = 48e6, 0.003
        if (s, d) == slow or (d, s) == slow:
            bw, lat = 9e6, 0.010
        links[(s, d)] = LinkProfile(s, d, bw, lat)
    return links


def test_best_split_equal_hosts():
    layers, bottleneck = best_split(costs([1.0, 1.0, 1.0], [80, 80, 80], 80), 80)
    assert sum(layers) == 80 and max(layers) - min(layers) <= 1
    assert bottleneck == 27.0


def test_best_split_respects_caps_and_speed():
    layers, bottleneck = best_split(costs([1.0, 2.0, 1.0], [40, 40, 10], 80), 80)
    assert sum(layers) == 80
    assert layers[2] <= 10
    assert layers[1] < layers[0]  # slower host gets fewer layers


def test_best_split_infeasible():
    assert best_split(costs([1.0, 1.0], [10, 10], 80), 80) is None


def test_solve_doc_example():
    spec, hosts, links = spec_70b(), hosts_abc(), links_abc()
    p = solve(spec, hosts, links, context=1024)
    assert isinstance(p, Placement)
    assert sorted(p.order) == ["A", "B", "C"]
    assert [a for a, _ in p.layers][0] == 0 and p.layers[-1][1] == 80
    for (a, b), (c, d) in zip(p.layers, p.layers[1:]):
        assert b == c
    # the slow link (A<->B) must be the return hop, which carries only token ids
    ret = (p.order[-1], p.order[0])
    assert set(ret) == {"A", "B"}, p.order
    # memory holds
    B = p.sequences_in_flight
    for i, name in enumerate(p.order):
        n = p.layers[i][1] - p.layers[i][0]
        extra = (spec.embed_bytes if i == 0 else 0) + (spec.head_bytes if i == len(p.order) - 1 else 0)
        assert n * (spec.layer_bytes + spec.kv_bytes_per_token * B * p.context) + extra <= hosts[name].free_mem
    # faster / bigger hosts carry more layers than C
    n = {name: p.layers[i][1] - p.layers[i][0] for i, name in enumerate(p.order)}
    assert n["A"] > n["C"] and n["B"] > n["C"]
    assert p.micro_batch >= 8 and p.num_micro_batches >= 3
    assert 0 < p.predicted_tok_s <= p.peak_tok_s + 1e-9


def test_links_keep_up_rule_shrinks_micro_batch():
    spec, hosts = spec_70b(), hosts_abc()
    fast = solve(spec, hosts, links_abc(), context=512)
    slow_links = {k: LinkProfile(k[0], k[1], 0.5e6, 0.02) for k in links_abc()}
    slow = solve(spec, hosts, slow_links, context=512)
    assert slow.micro_batch <= fast.micro_batch
    assert slow.predicted_tok_s < fast.predicted_tok_s


def test_pinned_choices():
    spec, hosts, links = spec_70b(), hosts_abc(), links_abc()
    p = solve(spec, hosts, links, context=512, micro_batch=16, num_micro_batches=4)
    assert p.micro_batch == 16 and p.num_micro_batches == 4


def test_infeasible_raises():
    spec = spec_70b()
    hosts = {n: HostProfile(n, 8 * GB, {1: 0.1}) for n in "ABC"}
    with pytest.raises(PlacementError):
        solve(spec, hosts, links_abc(), context=512)


def test_rebalance_moves_layers_off_throttled_host():
    spec, hosts, links = spec_70b(), hosts_abc(), links_abc()
    for h in hosts.values():
        h.free_mem = 96 * GB  # memory is not the binding constraint here
    p = solve(spec, hosts, links, context=512)
    iA = p.order.index("A")
    nA = p.layers[iA][1] - p.layers[iA][0]
    measured = {name: p.stage_compute[i] for i, name in enumerate(p.order)}
    measured["A"] *= 2.0  # A is throttling
    now, q = rebalance_with_measurements(p, spec, hosts, links, measured)
    jA = q.order.index("A")
    assert q.layers[jA][1] - q.layers[jA][0] < nA
    assert now.layers == p.layers and now.predicted_tok_s < p.predicted_tok_s
    assert should_rebalance(now, q)
    assert not should_rebalance(p, p)

    # A stays throttled on its new share: the next check keeps the layout (no ping-pong)
    measured2 = {name: q.layers[i][1] - q.layers[i][0] for i, name in enumerate(q.order)}
    measured2 = {name: n * hosts[name].time_per_layer(q.micro_batch) * (2.0 if name == "A" else 1.0) for name, n in measured2.items()}
    now2, q2 = rebalance_with_measurements(q, spec, hosts, links, measured2)
    assert not should_rebalance(now2, q2)


def test_no_rebalance_when_hosts_run_as_profiled():
    spec, hosts, links = spec_70b(), hosts_abc(), links_abc()
    p = solve(spec, hosts, links, context=512)
    measured = {name: p.stage_compute[i] for i, name in enumerate(p.order)}
    now, q = rebalance_with_measurements(p, spec, hosts, links, measured)
    assert now.predicted_tok_s == pytest.approx(p.predicted_tok_s)
    assert not should_rebalance(now, q)


def test_small_job_spreads_over_micro_batches():
    spec, hosts, links = spec_70b(), hosts_abc(), links_abc()
    assert solve(spec, hosts, links, context=512).micro_batch > 14
    p = solve(spec, hosts, links, context=512, max_sequences=40)  # 40 prompts, 3 hosts
    assert p.micro_batch <= 14 and p.num_micro_batches >= 3


def test_per_sequence_state_counts_against_memory():
    spec, hosts, links = spec_70b(), hosts_abc(), links_abc()
    lean = solve(spec, hosts, links, context=512)
    # e.g. Qwen3-Next linear attention: ~1.6 MB of state per layer per sequence
    spec.state_bytes_per_seq = int(1.6e6)
    spec.kv_block_tokens = 256
    heavy = solve(spec, hosts, links, context=300)
    assert heavy.sequences_in_flight < lean.sequences_in_flight
    B = heavy.sequences_in_flight
    for i, name in enumerate(heavy.order):
        n = heavy.layers[i][1] - heavy.layers[i][0]
        extra = (spec.embed_bytes if i == 0 else 0) + (spec.head_bytes if i == len(heavy.order) - 1 else 0)
        cache = spec.kv_bytes_per_token * 512 + spec.state_bytes_per_seq  # 300 tokens round up to a 512 block
        assert n * (spec.layer_bytes + B * cache) + extra <= hosts[name].free_mem


def test_single_host_degenerate():
    spec = ModelSpec(num_layers=4, hidden_size=16, layer_bytes=1000, kv_bytes_per_token=4)
    hosts = {"A": HostProfile("A", 10**9, {1: 0.001, 64: 0.002})}
    p = solve(spec, hosts, {("A", "A"): LinkProfile("A", "A", 1e9, 0.0)}, context=64)
    assert p.layers == [(0, 4)]


def test_streams_a_model_bigger_than_all_memory():
    """Like Qwen3-Next 80B (45 GB) on a 19 GB MacBook Pro and a 12.7 GB MacBook Air."""
    spec = ModelSpec(num_layers=48, hidden_size=2048, layer_bytes=int(0.92e9), kv_bytes_per_token=512,
                     embed_bytes=int(0.3e9), head_bytes=int(0.3e9), state_bytes_per_seq=int(1.65e6), kv_block_tokens=256)
    hosts = {
        "pro": HostProfile("pro", int(17.3e9), {1: 0.004, 64: 0.012}, disk_bw=5e9),
        "air": HostProfile("air", int(11e9), {1: 0.008, 64: 0.024}, disk_bw=2e9),
    }
    links = {(a, b): LinkProfile(a, b, 20e6, 0.004) for a in hosts for b in hosts if a != b}
    with pytest.raises(PlacementError):  # without streaming it can't run
        solve(spec, {n: HostProfile(n, h.free_mem, h.layer_time) for n, h in hosts.items()}, links, context=300)
    p = solve(spec, hosts, links, context=300)
    assert p.layers[0][0] == 0 and p.layers[-1][1] == 48
    streamed = {n: (e - a) - r for n, (a, e), r in zip(p.order, p.layers, p.resident)}
    assert sum(streamed.values()) > 0
    B = p.sequences_in_flight
    for i, name in enumerate(p.order):
        a, e = p.layers[i]
        extra = (spec.embed_bytes if i == 0 else 0) + (spec.head_bytes if i == len(p.order) - 1 else 0)
        buffer = 2 * spec.layer_bytes if streamed[name] else 0  # the layer in use and the next, read ahead
        used = p.resident[i] * spec.layer_bytes + buffer + (e - a) * B * spec.cache_bytes_per_seq(300) + extra
        assert used <= hosts[name].free_mem
    # the stage times include the part of each streamed layer's read that compute doesn't hide
    i = p.order.index("pro")
    a, e = p.layers[i]
    t = hosts["pro"].time_per_layer(p.micro_batch)
    assert p.stage_compute[i] == pytest.approx((e - a) * t + streamed["pro"] * max(0.0, spec.layer_bytes / 5e9 - t))
    assert streamed["pro"] * (spec.layer_bytes / 5e9 - t) > 0  # reads aren't hidden entirely here


def test_no_streaming_when_the_model_fits():
    spec, hosts, links = spec_70b(), hosts_abc(), links_abc()
    for h in hosts.values():
        h.disk_bw = 3e9
    p = solve(spec, hosts, links, context=512)
    assert p.resident == [e - a for a, e in p.layers]


def test_streaming_goes_to_the_faster_disk():
    spec = ModelSpec(num_layers=40, hidden_size=4096, layer_bytes=GB, kv_bytes_per_token=1024)
    prof = {1: 0.01, 64: 0.02}
    hosts = {"fast": HostProfile("fast", 12 * GB, dict(prof), disk_bw=6e9), "slow": HostProfile("slow", 12 * GB, dict(prof), disk_bw=1e9)}
    links = {(a, b): LinkProfile(a, b, 50e6, 0.003) for a in hosts for b in hosts if a != b}
    p = solve(spec, hosts, links, context=256)
    streamed = {n: (e - a) - r for n, (a, e), r in zip(p.order, p.layers, p.resident)}
    assert streamed["fast"] > streamed["slow"]


def test_rebalance_scales_disk_time_too():
    """A streaming layout measured exactly as predicted is left alone."""
    spec = ModelSpec(num_layers=40, hidden_size=4096, layer_bytes=GB, kv_bytes_per_token=1024)
    hosts = {n: HostProfile(n, 12 * GB, {1: 0.01, 64: 0.02}, disk_bw=3e9) for n in "AB"}
    links = {(a, b): LinkProfile(a, b, 50e6, 0.003) for a in hosts for b in hosts if a != b}
    p = solve(spec, hosts, links, context=256)
    assert sum(p.resident) < 40
    now, q = rebalance_with_measurements(p, spec, hosts, links, {n: p.stage_compute[i] for i, n in enumerate(p.order)})
    assert now.predicted_tok_s == pytest.approx(p.predicted_tok_s)
    assert not should_rebalance(now, q)
