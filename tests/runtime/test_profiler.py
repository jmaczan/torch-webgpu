"""torch.webgpu.profiler: GPU timestamps per dispatch."""

import pytest
import torch
import torch_webgpu  # noqa: F401

D = "webgpu"
pytestmark = pytest.mark.skipif(not torch.webgpu.profiler.is_supported(), reason="no timestamp-query feature")


def test_records_every_dispatch():
    x = torch.randn(512, 512, device=D)
    torch.webgpu.synchronize()
    with torch.webgpu.profiler.profile() as p:
        y = (x @ x).relu()
        z = y.sum(1)
    names = [r.name for r in p.records]
    assert names[0] == "matmul" and "relu" in names and "sum" in names
    assert all(r.gpu_us > 0 for r in p.records)
    assert p.total_us == pytest.approx(sum(r.gpu_us for r in p.records))
    summary = dict((n, (c, t)) for n, c, t in p.summary())
    assert summary["matmul"][0] == 1
    assert "matmul" in p.table()
    # Results are unaffected by profiling.
    torch.testing.assert_close(z.cpu(), (x.cpu() @ x.cpu()).relu().sum(1), rtol=1e-4, atol=1e-2)


def test_many_dispatches_span_submits():
    """More dispatches than one query set holds (queries are resolved per
    submit)."""
    x = torch.zeros(16, device=D)
    with torch.webgpu.profiler.profile() as p:
        for _ in range(3000):
            x.add_(1)
    assert len(p.records) == 3000
    assert x.cpu().eq(3000).all()


def test_timing_scales_with_work():
    small, big = torch.randn(256, 256, device=D), torch.randn(2048, 2048, device=D)
    for t in (small, big):
        t @ t
    with torch.webgpu.profiler.profile() as p:
        small @ small
        big @ big
    t_small, t_big = (r.gpu_us for r in p.records)
    assert t_big > 4 * t_small


def test_stop_without_start_raises():
    with pytest.raises(RuntimeError):
        torch_webgpu._C._profiler_stop()


def test_readback_inside_profile():
    x = torch.arange(10.0, device=D)
    with torch.webgpu.profiler.profile() as p:
        v = (x * 2).sum().item()
    assert v == 90.0
    assert [r.name for r in p.records][:1] == ["mul"]
