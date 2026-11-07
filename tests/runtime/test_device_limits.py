"""Portability: run on a device created with the WebGPU spec *default* limits
(TORCH_WEBGPU_LIMITS=default: 128 MiB max binding, 256 MiB max buffer, 8
storage buffers per stage, 256 invocations per workgroup) - what the smallest
conforming devices offer - and use tensors larger than one binding.

Each check runs in a subprocess, since the device is created once per process.
"""

import os
import subprocess
import sys
import textwrap

import pytest

SCRIPT = textwrap.dedent(r"""
    import torch, torch_webgpu
    D = "webgpu"
    p = torch.webgpu.get_device_properties()
    assert p["max_storage_buffer_binding_size"] == 128 * 2**20, p
    torch.manual_seed(0)
    x = torch.randn(200000, 256)            # 200 MB > one binding
    xd = x.to(D)
    idx = torch.randint(0, 200000, (4096,))
    idd = idx.to(D)
    src = torch.randn(4096, 256)
    def close(a, b, tol=1e-3):
        return torch.allclose(a.float().cpu(), b.float(), atol=tol, rtol=tol)
    checks = {
        "elementwise": torch.equal((xd * 2 + 1).cpu(), x * 2 + 1),
        "cast": torch.equal(xd.half().cpu(), x.half()),
        "strided": torch.equal((xd[:, ::3] * 3).cpu(), x[:, ::3] * 3),
        "embedding": torch.equal(torch.nn.functional.embedding(idd, xd).cpu(), torch.nn.functional.embedding(idx, x)),
        "adv_index": torch.equal(xd[idd, idd % 256].cpu(), x[idx, idx % 256]),
        "index_add": close(xd.clone().index_add_(0, idd, src.to(D)), x.clone().index_add_(0, idx, src)),
        "sum_all": close(xd.sum(), x.double().sum(), 1e-2),
        "sum_dim0": close(xd.sum(0), x.sum(0)),
        "amax_dim1": close(xd.amax(1), x.amax(1)),
        "norm_all": close(torch.linalg.vector_norm(xd), torch.linalg.vector_norm(x.double()), 1e-4),
        "argmax_dim1": torch.equal(xd.argmax(1).cpu(), x.argmax(1)),
        "softmax": close(xd.softmax(1), x.softmax(1)),
        "log_softmax_strided": close(xd[:, :4].log_softmax(0), x[:, :4].log_softmax(0)),
        "layer_norm": close(torch.nn.functional.layer_norm(xd, (256,)), torch.nn.functional.layer_norm(x, (256,))),
        "lm_head": close(torch.ones(8, 256, device=D) @ xd.t(), torch.ones(8, 256) @ x.t(), 1e-2),
        "bmm": close(torch.bmm(xd.view(1000, 200, 256), torch.ones(1000, 256, 4, device=D)),
                     torch.bmm(x.view(1000, 200, 256), torch.ones(1000, 256, 4)), 1e-2),
    }
    try:
        xd.index_select(0, torch.tensor([200000], device=D)).cpu()
        checks["oob_raises"] = False
    except (IndexError, RuntimeError):
        checks["oob_raises"] = True
    try:
        torch.empty(300 * 2**20, dtype=torch.uint8, device=D)
        checks["max_buffer_error"] = False
    except RuntimeError as e:
        checks["max_buffer_error"] = "maxBufferSize" in str(e)
    bad = [k for k, v in checks.items() if not v]
    print("BAD" if bad else "ALL OK", bad)
""")


def test_large_tensors_under_spec_default_limits():
    env = dict(os.environ, TORCH_WEBGPU_LIMITS="default", TORCH_WEBGPU_FALLBACK_WARN="0")
    out = subprocess.run([sys.executable, "-c", SCRIPT], env=env, capture_output=True, text=True, timeout=900)
    assert "ALL OK" in out.stdout, out.stdout[-2000:] + out.stderr[-3000:]


@pytest.mark.slow
@pytest.mark.parametrize("target", ["tests/ops", "tests/opinfo/test_opinfo.py::test_opinfo"])
def test_suites_under_spec_default_limits(target):
    """The regular suites must pass on a minimum-limits device too."""
    if os.environ.get("TORCH_WEBGPU_LIMITS") == "default":
        pytest.skip("already running under default limits")
    root = os.path.join(os.path.dirname(__file__), "..", "..")
    env = dict(os.environ, TORCH_WEBGPU_LIMITS="default", TORCH_WEBGPU_FALLBACK_WARN="0")
    args = [sys.executable, "-m", "pytest", "-q", "-x", "-p", "no:cacheprovider", target]
    if "opinfo" in target:
        args += ["-k", "float32"]
    out = subprocess.run(args, cwd=root, env=env, capture_output=True, text=True, timeout=1800)
    assert out.returncode == 0, f"rc={out.returncode}\n{out.stdout[-3000:]}\n{out.stderr[-3000:]}"
