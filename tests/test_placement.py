import itertools

import pytest

from slugai.placement import (
    HostProfile, LinkProfile, ModelSpec, Placement, PlacementError,
    best_split, evaluate, rebalance_with_measurements, should_rebalance, solve,
)

GB = 1 << 30


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
    layers, bottleneck = best_split([1.0, 1.0, 1.0], [80, 80, 80], 80)
    assert sum(layers) == 80 and max(layers) - min(layers) <= 1
    assert bottleneck == 27.0


def test_best_split_respects_caps_and_speed():
    layers, bottleneck = best_split([1.0, 2.0, 1.0], [40, 40, 10], 80)
    assert sum(layers) == 80
    assert layers[2] <= 10
    assert layers[1] < layers[0]  # slower host gets fewer layers


def test_best_split_infeasible():
    assert best_split([1.0, 1.0], [10, 10], 80) is None


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
    # roundtrip
    assert Placement.from_dict(p.to_dict()) == p


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
    q = rebalance_with_measurements(p, spec, hosts, links, measured)
    jA = q.order.index("A")
    assert q.layers[jA][1] - q.layers[jA][0] < nA
    assert should_rebalance(p, q) or not should_rebalance(p, p)
    assert not should_rebalance(p, p)


def test_single_host_degenerate():
    spec = ModelSpec(num_layers=4, hidden_size=16, layer_bytes=1000, kv_bytes_per_token=4)
    hosts = {"A": HostProfile("A", 10**9, {1: 0.001, 64: 0.002})}
    p = solve(spec, hosts, {("A", "A"): LinkProfile("A", "A", 1e9, 0.0)}, context=64)
    assert p.layers == [(0, 4)]
