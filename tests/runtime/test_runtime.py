"""Runtime and Python API of the webgpu device: device management,
transfers, memory, streams, RNG, serialization, autograd mechanics,
errors, autocast and threading."""

import copy
import io
import os
import pickle
import subprocess
import sys
import threading

import pytest
import torch
import torch_webgpu  # noqa: F401

D = "webgpu"

ALL_DTYPES = [torch.float32, torch.float16, torch.bfloat16, torch.float64, torch.int64, torch.int32, torch.int16,
              torch.int8, torch.uint8, torch.bool, torch.complex64]


def sample(dtype, shape=(3, 5)):
    g = torch.Generator().manual_seed(0)
    if dtype == torch.bool:
        return torch.rand(shape, generator=g) > 0.5
    if dtype.is_complex:
        return torch.randn(shape, dtype=dtype, generator=g)
    if dtype.is_floating_point:
        return torch.randn(shape, generator=g).to(dtype)
    info = torch.iinfo(dtype)
    return torch.randint(max(info.min, -1000), min(info.max, 1000), shape, generator=g).to(dtype)


# ------------------------------------------------------------------- device
class TestDevice:
    def test_availability(self):
        assert torch.webgpu.is_available()
        assert torch.webgpu.device_count() == 1
        assert torch.webgpu.current_device() == 0
        torch.webgpu.set_device(0)
        assert torch.webgpu.is_initialized()

    def test_device_objects(self):
        x = torch.ones(2, device=D)
        assert x.device.type == "webgpu" and x.device.index == 0
        assert x.is_webgpu
        assert torch.ones(2, device="webgpu:0").device == x.device
        assert torch.device("webgpu", 0) == x.device

    def test_invalid_device_index(self):
        with pytest.raises((RuntimeError, ValueError, AssertionError)):
            torch.ones(2, device="webgpu:3")
        with pytest.raises((RuntimeError, ValueError, AssertionError)):
            torch.webgpu.set_device(5)

    def test_generated_methods(self):
        x = torch.arange(4.0)
        y = x.webgpu()
        assert y.is_webgpu and torch.equal(y.cpu(), x)
        m = torch.nn.Linear(3, 2).webgpu()
        assert all(p.is_webgpu for p in m.parameters())
        s = torch.UntypedStorage(8).webgpu()
        assert s.device.type == "webgpu"

    def test_device_context(self):
        with torch.webgpu.device(0):
            assert torch.webgpu.current_device() == 0
        with torch.device(D):
            t = torch.zeros(3)
        assert t.is_webgpu

    def test_properties(self):
        props = torch.webgpu.get_device_properties()
        assert isinstance(props, dict) and props
        assert isinstance(torch.webgpu.get_device_name(), str)


# ---------------------------------------------------------------- transfers
class TestTransfers:
    @pytest.mark.parametrize("dtype", ALL_DTYPES, ids=str)
    def test_roundtrip_all_dtypes(self, dtype):
        x = sample(dtype)
        y = x.to(D)
        assert y.dtype == dtype and y.shape == x.shape
        assert torch.equal(y.cpu(), x)

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.int64, torch.bool, torch.uint8], ids=str)
    def test_noncontiguous(self, dtype):
        x = sample(dtype, (6, 8, 5))
        for v in [x.transpose(0, 2), x[::2, 1:7:3], x[:, 3], x.expand(2, 6, 8, 5)[1], x.permute(2, 0, 1)]:
            assert torch.equal(v.to(D).cpu(), v)
        # device -> host of non-contiguous device views
        d = x.to(D)
        for v, vc in [(d.transpose(0, 2), x.transpose(0, 2)), (d[::2, 1:7:3], x[::2, 1:7:3]), (d[:, 3], x[:, 3])]:
            assert torch.equal(v.cpu(), vc)

    def test_shapes(self):
        for x in [torch.tensor(3.5), torch.empty(0), torch.empty(0, 4), torch.ones(1, 1, 1)]:
            y = x.to(D)
            assert y.shape == x.shape and torch.equal(y.cpu(), x)

    def test_large(self):
        x = torch.randn(48 * 1024 * 1024 // 4)  # 48 MiB
        y = x.to(D)
        assert torch.equal(y.cpu(), x)
        assert torch.equal((y + 1).cpu(), x + 1)

    def test_non_blocking_and_copy(self):
        x = torch.randn(1000)
        y = x.to(D, non_blocking=True)
        torch.webgpu.synchronize()
        assert torch.equal(y.cpu(), x)
        z = torch.empty(4, 1000, device=D)
        z.copy_(x)  # broadcasting host -> device copy
        assert torch.equal(z.cpu(), x.expand(4, 1000))
        h = torch.empty(1000, dtype=torch.float16)
        h.copy_(y)  # device -> host with dtype conversion
        assert torch.equal(h, x.half())
        assert torch.equal(y.to(D, torch.int32).cpu(), x.to(torch.int32))

    def test_device_to_device_copy_and_clone(self):
        x = torch.randn(7, 9, device=D)
        y = x.clone()
        assert y.data_ptr() != x.data_ptr() and torch.equal(y.cpu(), x.cpu())
        z = torch.empty_like(x).copy_(x.t().contiguous().t())
        assert torch.equal(z.cpu(), x.cpu())


# -------------------------------------------------------------- views/memory
class TestViewsAndMemory:
    def test_view_aliasing(self):
        x = torch.zeros(4, 4, device=D)
        v = x[1:3, ::2]
        v.fill_(7)
        x.view(-1)[0] = 1
        expected = torch.zeros(4, 4)
        expected[1:3, ::2] = 7
        expected.view(-1)[0] = 1
        assert torch.equal(x.cpu(), expected)
        assert x.untyped_storage().data_ptr() == v.untyped_storage().data_ptr()

    def test_resize_and_set(self):
        x = torch.arange(6.0, device=D)
        x.resize_(10)
        assert x.shape == (10,) and torch.equal(x[:6].cpu(), torch.arange(6.0))
        y = torch.empty(0, device=D)
        y.set_(x.untyped_storage(), 2, (2, 2), (2, 1))
        assert torch.equal(y.cpu(), torch.tensor([[2.0, 3.0], [4.0, 5.0]]))

    def test_memory_stats(self):
        torch.webgpu.synchronize()
        torch.webgpu.empty_cache()
        base = torch.webgpu.memory_allocated()
        torch.webgpu.reset_peak_memory_stats()
        x = torch.empty(1024 * 1024, device=D)  # 4 MiB
        assert torch.webgpu.memory_allocated() >= base + 4 * 1024 * 1024
        assert torch.webgpu.max_memory_allocated() >= base + 4 * 1024 * 1024
        assert torch.webgpu.memory_reserved() >= torch.webgpu.memory_allocated()
        del x
        assert torch.webgpu.memory_allocated() == base
        assert torch.webgpu.max_memory_allocated() >= base + 4 * 1024 * 1024
        stats = torch.webgpu.memory_stats()
        assert isinstance(stats, dict)

    def test_empty_cache_releases(self):
        x = torch.empty(16 * 1024 * 1024, device=D)
        del x
        torch.webgpu.synchronize()
        before = torch.webgpu.memory_reserved()
        torch.webgpu.empty_cache()
        assert torch.webgpu.memory_reserved() <= before

    def test_out_of_memory_recovers(self):
        with pytest.raises((RuntimeError, torch.OutOfMemoryError)):
            torch.empty(1 << 46, dtype=torch.uint8, device=D)  # 64 TiB
        x = torch.ones(10, device=D)  # device still usable
        assert x.sum().item() == 10


# ----------------------------------------------------------- sync / streams
class TestStreams:
    def test_synchronize(self):
        x = torch.randn(100, device=D)
        for _ in range(50):
            x = x * 1.0001
        torch.webgpu.synchronize()
        torch.webgpu.synchronize(0)

    def test_stream_event_api(self):
        s = torch.webgpu.current_stream()
        assert s == torch.webgpu.default_stream()
        with torch.webgpu.stream(s):
            y = torch.ones(10, device=D) * 2
        e = s.record_event()
        e.synchronize()
        assert e.query()
        s.synchronize()
        assert y.sum().item() == 20

    def test_event_timing(self):
        start = torch.webgpu.Event(enable_timing=True)
        end = torch.webgpu.Event(enable_timing=True)
        start.record()
        x = torch.randn(512, 512, device=D)
        (x @ x).sum().item()
        end.record()
        end.synchronize()
        assert start.elapsed_time(end) >= 0


# ---------------------------------------------------------------------- RNG
class TestRNG:
    @pytest.mark.parametrize("fn", [
        lambda: torch.rand(1000, device=D),
        lambda: torch.randn(1000, device=D),
        lambda: torch.randint(0, 100, (1000,), device=D),
        lambda: torch.empty(1000, device=D).bernoulli_(0.3),
        lambda: torch.nn.functional.dropout(torch.ones(1000, device=D), 0.5),
        lambda: torch.empty(1000, device=D).normal_(2.0, 3.0),
        lambda: torch.randperm(100, device=D),
    ])
    def test_manual_seed_reproducible(self, fn):
        torch.manual_seed(123)
        a = fn().cpu()
        torch.manual_seed(123)
        b = fn().cpu()
        torch.manual_seed(124)
        c = fn().cpu()
        assert torch.equal(a, b)
        assert not torch.equal(a, c)

    def test_rng_state_roundtrip(self):
        torch.webgpu.manual_seed(7)
        state = torch.webgpu.get_rng_state()
        a = torch.randn(100, device=D).cpu()
        torch.webgpu.set_rng_state(state)
        b = torch.randn(100, device=D).cpu()
        assert torch.equal(a, b)
        assert torch.webgpu.initial_seed() == 7

    def test_successive_draws_differ(self):
        torch.manual_seed(0)
        assert not torch.equal(torch.rand(100, device=D).cpu(), torch.rand(100, device=D).cpu())

    def test_generator_object(self):
        g1 = torch.Generator(device=D).manual_seed(5)
        g2 = torch.Generator(device=D).manual_seed(5)
        a = torch.randn(50, device=D, generator=g1).cpu()
        b = torch.randn(50, device=D, generator=g2).cpu()
        assert torch.equal(a, b)
        assert g1.device.type == "webgpu"

    def test_fork_rng(self):
        torch.manual_seed(1)
        with torch.random.fork_rng(device_type="webgpu"):
            torch.randn(10, device=D)
        a = torch.randn(10, device=D).cpu()
        torch.manual_seed(1)
        b = torch.randn(10, device=D).cpu()
        assert torch.equal(a, b)

    def test_distribution_sanity(self):
        torch.manual_seed(0)
        x = torch.randn(1_000_000, device=D)
        assert abs(x.mean().item()) < 0.01 and abs(x.std().item() - 1) < 0.01
        u = torch.rand(1_000_000, device=D)
        assert 0 <= u.min().item() and u.max().item() < 1 and abs(u.mean().item() - 0.5) < 0.01


# ------------------------------------------------------------ serialization
class TestSerialization:
    def test_save_load_tensor(self, tmp_path):
        x = torch.randn(5, 6, device=D)
        path = tmp_path / "t.pt"
        torch.save(x, path)
        y = torch.load(path)
        assert y.is_webgpu and torch.equal(y.cpu(), x.cpu())
        z = torch.load(path, map_location="cpu")
        assert z.device.type == "cpu" and torch.equal(z, x.cpu())
        w = torch.load(path, map_location=D)
        assert w.is_webgpu

    def test_save_load_state_dict_and_views(self, tmp_path):
        m = torch.nn.Sequential(torch.nn.Linear(4, 8), torch.nn.ReLU(), torch.nn.Linear(8, 2)).to(D)
        buf = io.BytesIO()
        base = torch.arange(10.0, device=D)
        torch.save({"sd": m.state_dict(), "a": base, "b": base[2:5]}, buf)
        buf.seek(0)
        obj = torch.load(buf)
        m2 = torch.nn.Sequential(torch.nn.Linear(4, 8), torch.nn.ReLU(), torch.nn.Linear(8, 2)).to(D)
        m2.load_state_dict(obj["sd"])
        x = torch.randn(3, 4, device=D)
        assert torch.equal(m(x).cpu(), m2(x).cpu())
        obj["a"][3] = 100.0  # storage sharing preserved
        assert obj["b"][1].item() == 100.0

    def test_cpu_checkpoint_to_device(self, tmp_path):
        m = torch.nn.Linear(3, 3)
        buf = io.BytesIO()
        torch.save(m.state_dict(), buf)
        buf.seek(0)
        sd = torch.load(buf, map_location=D)
        assert all(v.is_webgpu for v in sd.values())

    def test_pickle_and_deepcopy(self):
        x = torch.randn(4, device=D)
        y = pickle.loads(pickle.dumps(x))
        assert y.is_webgpu and torch.equal(y.cpu(), x.cpu())
        m = torch.nn.Linear(2, 2).to(D)
        m2 = copy.deepcopy(m)
        assert m2.weight.is_webgpu and torch.equal(m2.weight.cpu(), m.weight.cpu())
        assert m2.weight.data_ptr() != m.weight.data_ptr()


# ------------------------------------------------------------ module moves
class TestModuleMoves:
    def _tied(self):
        emb = torch.nn.Embedding(10, 4)
        head = torch.nn.Linear(4, 10, bias=False)
        head.weight = emb.weight
        return torch.nn.ModuleDict({"emb": emb, "head": head})

    def test_tied_weights_survive_moves(self):
        m = self._tied()
        for target in [D, "cpu", D]:
            m.to(target)
            assert m["emb"].weight is m["head"].weight
            assert m["emb"].weight.device.type == torch.device(target).type

    def test_parameter_identity_kept_for_optimizers(self):
        m = torch.nn.Linear(3, 3)
        opt = torch.optim.SGD(m.parameters(), lr=0.5)
        before = m.weight
        m.to(D)
        assert m.weight is before  # optimizer created before .to() still works
        w0 = m.weight.detach().cpu().clone()
        m(torch.ones(2, 3, device=D)).sum().backward()
        opt.step()
        assert not torch.equal(m.weight.detach().cpu(), w0)

    def test_tied_gradient_accumulates_both_uses(self):
        m = self._tied()
        ref = copy.deepcopy(m)
        m.to(D)
        idx = torch.tensor([1, 3, 3])
        for mm, i in [(ref, idx), (m, idx.to(D))]:
            mm["head"](mm["emb"](i)).square().mean().backward()
        torch.testing.assert_close(m["emb"].weight.grad.cpu(), ref["emb"].weight.grad)

    def test_weakref_fallback_keeps_tying(self):
        import weakref
        m = self._tied()
        r = weakref.ref(m["emb"].weight)  # swapping is impossible with weakrefs
        m.to(D)
        assert m["emb"].weight is m["head"].weight and m["emb"].weight.is_webgpu
        del r

    def test_grads_and_buffers_move(self):
        m = torch.nn.BatchNorm1d(4)
        m(torch.randn(8, 4)).sum().backward()
        m.to(D)
        assert m.weight.grad.is_webgpu and m.running_mean.is_webgpu
        m.half()
        assert m.weight.dtype == torch.float16 and m.weight.is_webgpu


# ------------------------------------------------------------------ autograd
class TestAutograd:
    def test_backward_and_accumulation(self):
        w = torch.randn(3, 3, device=D, requires_grad=True)
        x = torch.randn(4, 3, device=D)
        (x @ w).sum().backward()
        g1 = w.grad.clone()
        (x @ w).sum().backward()
        assert torch.allclose(w.grad.cpu(), 2 * g1.cpu())
        assert torch.allclose(g1.cpu(), x.sum(0).cpu().unsqueeze(1).expand(3, 3))

    def test_no_grad_inference_mode_retain_graph(self):
        w = torch.randn(3, device=D, requires_grad=True)
        with torch.no_grad():
            assert not (w * 2).requires_grad
        with torch.inference_mode():
            y = w * 2
            assert y.is_inference()
        z = (w * w).sum()
        z.backward(retain_graph=True)
        z.backward()
        assert torch.allclose(w.grad.cpu(), 4 * w.detach().cpu())

    def test_hooks_and_cpu_mixing(self):
        w = torch.ones(3, device=D, requires_grad=True)
        seen = []
        w.register_hook(lambda g: seen.append(g.device.type))
        (w.cpu() * 3).sum().backward()  # graph crosses devices
        assert seen == ["webgpu"]
        assert torch.equal(w.grad.cpu(), torch.full((3,), 3.0))

    def test_training_step_matches_cpu(self):
        torch.manual_seed(0)
        m_cpu = torch.nn.Sequential(torch.nn.Linear(8, 16), torch.nn.GELU(), torch.nn.LayerNorm(16), torch.nn.Linear(16, 4))
        m_dev = copy.deepcopy(m_cpu).to(D)
        opt_c = torch.optim.AdamW(m_cpu.parameters(), lr=1e-2)
        opt_d = torch.optim.AdamW(m_dev.parameters(), lr=1e-2)
        x, y = torch.randn(32, 8), torch.randint(0, 4, (32,))
        for _ in range(5):
            for m, opt, xx, yy in [(m_cpu, opt_c, x, y), (m_dev, opt_d, x.to(D), y.to(D))]:
                opt.zero_grad()
                torch.nn.functional.cross_entropy(m(xx), yy).backward()
                opt.step()
        for pc, pd in zip(m_cpu.parameters(), m_dev.parameters()):
            torch.testing.assert_close(pd.detach().cpu(), pc.detach(), atol=1e-4, rtol=1e-4)


# -------------------------------------------------------------------- errors
class TestErrors:
    def test_index_out_of_bounds(self):
        x = torch.arange(5.0, device=D)
        with pytest.raises((IndexError, RuntimeError)):
            x[torch.tensor([7], device=D)].cpu()
        assert x.sum().item() == 10.0  # device usable afterwards

    def test_gather_out_of_bounds(self):
        x = torch.arange(5.0, device=D)
        with pytest.raises((IndexError, RuntimeError)):
            torch.gather(x, 0, torch.tensor([5], device=D)).cpu()
        assert torch.equal(x.cpu(), torch.arange(5.0))

    def test_device_mismatch(self):
        with pytest.raises(RuntimeError, match="device"):
            torch.ones(3, device=D) + torch.ones(3)
        # 0-dim CPU tensors are allowed as scalars
        assert torch.equal((torch.ones(3, device=D) + torch.tensor(2.0)).cpu(), torch.full((3,), 3.0))

    def test_shape_errors_are_python_errors(self):
        with pytest.raises(RuntimeError):
            torch.ones(2, 3, device=D) @ torch.ones(2, 3, device=D)

    def test_float64_runs_via_cpu(self):
        x = torch.arange(4, dtype=torch.float64, device=D)
        y = (x * 2 + 1).sin()
        # Warned once per op per process; the record says why it fell back.
        assert any(k.startswith("aten::sin") and "[Double]" in k for k in torch_webgpu._C._fallback_ops())
        assert y.dtype == torch.float64 and y.is_webgpu
        assert torch.equal(y.cpu(), (torch.arange(4, dtype=torch.float64) * 2 + 1).sin())

    def test_disable_fallback_env(self):
        code = ("import torch, torch_webgpu\n"
                "x = torch.ones(4, dtype=torch.float64, device='webgpu')\n"
                "try:\n    x.sin()\nexcept NotImplementedError as e:\n    print('raised', 'Double' in str(e))\n")
        env = dict(os.environ, TORCH_WEBGPU_DISABLE_CPU_FALLBACK="1")
        out = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=120)
        assert "raised True" in out.stdout, out.stderr[-2000:]


# ------------------------------------------------------------------- autocast
class TestAutocast:
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16], ids=str)
    def test_autocast_matmul(self, dtype):
        a = torch.randn(8, 8, device=D)
        b = torch.randn(8, 8, device=D)
        with torch.autocast(device_type="webgpu", dtype=dtype):
            c = a @ b
            s = torch.softmax(c, -1)
        assert c.dtype == dtype
        assert s.dtype == torch.float32
        torch.testing.assert_close(c.float().cpu(), (a.cpu() @ b.cpu()), atol=0.1, rtol=0.05)

    def test_amp_supported_dtypes(self):
        assert torch.float16 in torch.webgpu.get_amp_supported_dtype()
        assert torch.webgpu.is_bf16_supported()


# ------------------------------------------------------------------ threading
def test_concurrent_threads():
    errors = []

    def work(seed):
        try:
            g = torch.Generator().manual_seed(seed)
            a = torch.randn(64, 64, generator=g)
            b = torch.randn(64, 64, generator=g)
            for _ in range(20):
                r = (a.to(D) @ b.to(D)).relu().sum(0).cpu()
                torch.testing.assert_close(r, (a @ b).relu().sum(0), atol=1e-3, rtol=1e-4)
        except Exception as e:  # noqa: BLE001
            errors.append(e)

    threads = [threading.Thread(target=work, args=(i,)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors, errors[0]
