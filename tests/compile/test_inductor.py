"""torch.compile (Inductor) on webgpu: generated WGSL kernels vs CPU eager.

Covers pointwise / reduction codegen for every compute dtype class (float,
half types, int32, uint8, bool, exact int64), NaN semantics, indexing
(indirect loads, bounds errors, atomic accumulation), random numbers,
dynamic shapes, buffer aliasing, autograd + optimizer steps, and a small
transformer block.
"""

import copy

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_webgpu  # noqa: F401

D = "webgpu"


@pytest.fixture(autouse=True)
def fresh_dynamo(request):
    torch._dynamo.reset()
    before = set(torch_webgpu._C._fallback_ops())
    yield
    if not request.node.get_closest_marker("allow_fallback"):
        assert not set(torch_webgpu._C._fallback_ops()) - before, "unexpected CPU fallback"


def to_dev(a):
    return a.to(D) if isinstance(a, torch.Tensor) else a


def compare(fn, *args, tol=1e-4, dynamic=None):
    out = torch.compile(fn, dynamic=dynamic)(*map(to_dev, args))
    ref = fn(*args)
    outs = out if isinstance(out, (tuple, list)) else (out,)
    refs = ref if isinstance(ref, (tuple, list)) else (ref,)
    assert len(outs) == len(refs)
    for o, r in zip(outs, refs):
        assert o.device.type == "webgpu"
        assert o.dtype == r.dtype
        torch.testing.assert_close(o.cpu(), r, atol=tol, rtol=tol, equal_nan=True)


torch.manual_seed(0)
X = torch.randn(37, 129)
Y = torch.randn(37, 129)
W = torch.randn(129)
B = torch.randn(129)
I = torch.randint(-5, 5, (37, 129))
WITH_NAN = torch.where(torch.rand(37, 129) < 0.1, float("nan"), X)

FLOAT_CASES = {
    "pointwise": (lambda a, b: torch.sigmoid(a) * b.exp() - a.tanh(), (X, Y)),
    "transcendental": (lambda a: a.sin() + a.cos() + a.abs().log1p() + a.atan() + a.erf(), (X,)),
    "broadcast": (lambda a, w: a * w + 1, (X, W)),
    "transpose": (lambda a: a.t() * 2, (X,)),
    "sum": (lambda a: a.sum(1), (X,)),
    "sum_all": (lambda a: a.sum(), (X,)),
    "mean_dim0": (lambda a: a.mean(0), (X,)),
    "amax": (lambda a: a.amax(1), (X,)),
    "amin_dim0": (lambda a: a.amin(0), (X,)),
    "prod": (lambda a: (a * 0.1 + 1).prod(1), (X,)),
    "argmax": (lambda a: a.argmax(1), (X,)),
    "argmin_dim0": (lambda a: a.argmin(0), (X,)),
    "softmax": (lambda a: F.softmax(a, -1), (X,)),
    "log_softmax_dim0": (lambda a: F.log_softmax(a, 0), (X,)),
    "var_mean": (lambda a: torch.var_mean(a, 1), (X,)),
    "layer_norm": (lambda a, w, b: F.layer_norm(a, (129,), w, b), (X, W, B)),
    "rms_norm": (lambda a, w: a * torch.rsqrt(a.pow(2).mean(-1, keepdim=True) + 1e-6) * w, (X, W)),
    "gelu": (lambda a: F.gelu(a) + F.gelu(a, approximate="tanh"), (X,)),
    "silu_mul": (lambda a, b: F.silu(a) * b, (X, Y)),
    "clamp_pow": (lambda a: a.clamp(-0.5, 0.5) + a.abs().pow(1.7), (X,)),
    "cat": (lambda a, b: torch.cat([a, b], 1) * 2, (X, Y)),
    "pad": (lambda a: F.pad(a, (2, 3, 1, 1), value=-1.0), (X,)),
    "where_inf": (lambda a: torch.where(a > 0, a, float("-inf")).amax(0), (X,)),
    "cross_entropy": (lambda a, t: F.cross_entropy(a, t), (X, torch.randint(0, 129, (37,)))),
    "big_sum": (lambda a: a.sum(), (torch.randn(3_000_000),)),
}


@pytest.mark.parametrize("name", list(FLOAT_CASES))
def test_float(name):
    fn, args = FLOAT_CASES[name]
    compare(fn, *args, tol=1e-3 if name == "big_sum" else 1e-4)


NAN_CASES = {
    "max_propagates": lambda a: a.amax(1),
    "min_propagates": lambda a: a.amin(0),
    "argmax_nan": lambda a: a.argmax(1),
    "maximum": lambda a: torch.maximum(a, a.flip(0)),
    "isnan_sum": lambda a: a.isnan().sum(1),
    "ne_self": lambda a: a != a,
    "nan_to_num": lambda a: a.nan_to_num(0.5),
    "sum_nan": lambda a: a.sum(1),
}


@pytest.mark.parametrize("name", list(NAN_CASES))
def test_nan_semantics(name):
    compare(NAN_CASES[name], WITH_NAN)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_half_types(dtype):
    a = X.to(dtype)
    compare(lambda t: (t * 2).relu().sum(1), a, tol=2e-2)
    compare(lambda t: F.softmax(t, -1), a, tol=2e-2)
    compare(lambda t: (t.float() * 3).to(dtype), a, tol=0)  # rounding to the narrow dtype


INT_CASES = {
    "int64_arith": (lambda a: (a * 3 + 7) // 2 - a % 3, (I,)),
    "int64_sum": (lambda a: a.sum(1), (I,)),
    "int64_large": (lambda a: a * (2**40) + 2**35, (I,)),
    "int64_max": (lambda a: a.amax(0), (I,)),
    "int32": (lambda a: (a.int() * a.int()).sum(1), (I,)),
    "int32_div": (lambda a: torch.div(a.int(), 3, rounding_mode="floor") + torch.div(a.int(), 3, rounding_mode="trunc"), (I,)),
    "uint8_wrap": (lambda a: a.to(torch.uint8) + 250, (I,)),
    "int8_wrap": (lambda a: a.to(torch.int8) * 40, (I,)),
    "bitwise": (lambda a: (a & 3) | (a ^ 5) | (a << 2), (I,)),
    "bool_logic": (lambda a, b: ((a > 0) & (b < 0)) | ~(a > 1), (X, Y)),
    "any": (lambda a: (a > 2).any(1), (X,)),
    "all": (lambda a: (a > -3).all(0), (X,)),
    "float_to_int": (lambda a: (a * 10).to(torch.int64) + (a * 10).int(), (X,)),
    "mixed": (lambda a: a.to(torch.int64) + 1.5, (X,)),
    "arange": (lambda a: a + torch.arange(129, device=a.device), (I,)),
}


@pytest.mark.parametrize("name", list(INT_CASES))
def test_integer(name):
    fn, args = INT_CASES[name]
    compare(fn, *args, tol=0)


def test_indexing():
    emb = torch.randn(50, 16)
    idx = torch.randint(-50, 50, (8, 9))
    compare(lambda e, i: F.embedding(i.abs(), e) * 2, emb, idx)
    compare(lambda e, i: e[i] + 1, emb, idx)  # negative indices wrap
    compare(lambda a, i: torch.gather(a, 1, i), X, torch.randint(0, 129, (37, 5)))


def test_index_out_of_bounds_raises():
    f = torch.compile(lambda e, i: e[i] + 1)
    with pytest.raises(IndexError):
        f(torch.randn(10, 4, device=D), torch.tensor([1, 12], device=D))
        torch.webgpu.synchronize()
    # The device stays usable.
    assert torch.equal((torch.ones(3, device=D) + 1).cpu(), torch.full((3,), 2.0))


@pytest.mark.parametrize("dtype", [torch.float32, torch.int32])
def test_atomic_accumulate(dtype):
    idx = torch.randint(0, 10, (1000,))
    vals = torch.randint(-3, 4, (1000,)).to(dtype)
    compare(lambda z, i, v: z.index_put((i,), v, accumulate=True), torch.zeros(10, dtype=dtype), idx, vals, tol=1e-4)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.int32, torch.float32])
def test_atomic_accumulate_counts_exact(dtype):
    """Concurrent atomic adds lose no update, also for 16-bit types whose
    neighbours share a 32-bit word (compare-and-swap on the word)."""
    idx = torch.randint(0, 3, (600,))
    out = torch.compile(lambda z, i, v: z.index_put((i,), v, accumulate=True))(
        torch.zeros(4, dtype=dtype, device=D), idx.to(D), torch.ones(600, dtype=dtype, device=D))
    assert out.cpu().tolist() == torch.bincount(idx, minlength=4).to(dtype).tolist()


# Special values computed inside fused kernels. GPU drivers compile with
# fast-math; folding x / x -> 1, 0 / x -> 0, x - x -> 0 or arithmetic on
# visible NaN / Inf constants gives wrong results here.
SPECIAL_VALUE_CASES = {
    "x_div_x": lambda a: a / a,
    "x_div_0": lambda a: a / 0,
    "zero_div_x": lambda a: (a * 0) / a,
    "x_minus_x_inf": lambda a: (a + float("inf")) - (a + float("inf")),
    "inf_times_x": lambda a: a * float("inf"),
    "reciprocal": lambda a: a.reciprocal(),
    "empty_mean": lambda a: a[:0].mean(),
    "sum_div_0": lambda a: a.sum() / 0,
    "neg_inf_max": lambda a: torch.full_like(a, float("-inf")).amax() + a,
}


@pytest.mark.parametrize("name", list(SPECIAL_VALUE_CASES))
def test_special_values(name):
    a = torch.tensor([0.0, -0.0, 1.0, -2.0, float("inf"), float("-inf"), float("nan")])
    compare(SPECIAL_VALUE_CASES[name], a)


@pytest.mark.allow_fallback
def test_float64_falls_back():
    """float64 has no WGSL type: those nodes run eagerly (CPU fallback)
    while the rest of the graph is still compiled."""
    x = torch.randn(5, 7)
    torch._dynamo.reset()
    out = torch.compile(lambda t: (torch.softmax(t, 1, dtype=torch.float64) + 1, t.exp() * 2))(x.to(D))
    torch.testing.assert_close(out[0].cpu(), torch.softmax(x, 1, dtype=torch.float64) + 1, atol=1e-7, rtol=1e-7)
    torch.testing.assert_close(out[1].cpu(), x.exp() * 2)


def test_random():
    def f(a):
        r = torch.rand_like(a)
        n = torch.randn_like(a)
        return torch.stack([r.min(), r.max(), r.mean(), n.mean(), n.std()])

    stats = torch.compile(f)(torch.empty(1_000_000, device=D)).cpu()
    assert stats[0] >= 0 and stats[1] < 1
    assert abs(stats[2] - 0.5) < 0.01 and abs(stats[3]) < 0.01 and abs(stats[4] - 1) < 0.01
    torch.manual_seed(1)
    a = torch.compile(lambda t: F.dropout(t, 0.5, training=True))(torch.ones(10000, device=D)).cpu()
    assert set(a.unique().tolist()) <= {0.0, 2.0} and abs((a == 0).float().mean() - 0.5) < 0.03


def test_dynamic_shapes():
    f = lambda a: F.softmax(a * 2, -1).sum(0)  # noqa: E731
    cf = torch.compile(f, dynamic=True)
    for n in (5, 17, 300):
        a = torch.randn(n, n + 3)
        torch.testing.assert_close(cf(a.to(D)).cpu(), f(a), atol=1e-5, rtol=1e-5)


def test_aliased_buffers():
    """Views of one storage passed as separate inputs, one of them mutated."""

    def f(a, b):
        a.add_(b)
        return a * 2

    base = torch.randn(64)
    dev = base.to(D)
    out = torch.compile(f)(dev[:32], dev[32:])
    ref_base = base.clone()
    ref = f(ref_base[:32], ref_base[32:])
    torch.testing.assert_close(out.cpu(), ref)
    torch.testing.assert_close(dev.cpu(), ref_base)


def test_training_step():
    torch.manual_seed(0)
    # No conv bias: before BatchNorm its gradient is pure rounding noise,
    # which Adam's first step (~ lr * sign(grad)) turns into +-lr updates.
    m = nn.Sequential(nn.Conv2d(3, 16, 3, padding=1, bias=False), nn.BatchNorm2d(16), nn.ReLU(), nn.Flatten(), nn.Linear(1024, 10))
    mc = copy.deepcopy(m).to(D)
    x, t = torch.randn(4, 3, 8, 8), torch.randint(0, 10, (4,))
    opt = torch.optim.AdamW(m.parameters(), lr=1e-2)
    optc = torch.optim.AdamW(mc.parameters(), lr=1e-2)

    def step(mod, o, x, t):
        o.zero_grad()
        loss = F.cross_entropy(mod(x), t)
        loss.backward()
        o.step()
        return loss

    cstep = torch.compile(step)
    for _ in range(3):
        l1 = step(m, opt, x, t)
        l2 = cstep(mc, optc, x.to(D), t.to(D))
        torch.testing.assert_close(l2.cpu(), l1, atol=1e-4, rtol=1e-4)
    for a, b in zip(m.parameters(), mc.parameters()):
        torch.testing.assert_close(b.cpu(), a, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(mc[1].running_var.cpu(), m[1].running_var, atol=1e-5, rtol=1e-5)


def test_transformer_block():
    torch.manual_seed(0)
    layer = nn.TransformerEncoderLayer(64, 4, 128, dropout=0.0, batch_first=True, norm_first=True)
    # (train mode: eval takes the _transformer_encoder_layer_fwd fast path,
    # which has no webgpu kernel yet)
    x = torch.randn(2, 10, 64)
    with torch.no_grad():
        ref = layer(x)
        out = torch.compile(copy.deepcopy(layer).to(D))(x.to(D))
    torch.testing.assert_close(out.cpu(), ref, atol=1e-4, rtol=1e-4)
