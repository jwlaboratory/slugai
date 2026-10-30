import mlx.core as mx

from host.profile import profile


def test_profile(tiny_model):
    p = profile(tiny_model, context=16, batch_sizes=(8, 1, 4))
    assert list(p.layer_time) == [1, 4, 8]
    times = list(p.layer_time.values())
    assert times[0] > 0 and times == sorted(times)
    assert p.free_memory > 0 and p.disk_speed > 0
    assert mx.get_active_memory() < 1 << 20  # the probe layers are gone
