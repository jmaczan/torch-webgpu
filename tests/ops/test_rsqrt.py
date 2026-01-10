import torch
import torch_webgpu  # noqa: F401


def test_rsqrt_basic():
    a = torch.tensor([1.0, 4.0, 9.0, 16.0], device="webgpu")
    result = torch.rsqrt(a)
    expected = torch.rsqrt(a.to("cpu"))
    assert torch.allclose(result.to("cpu"), expected, rtol=1e-4, atol=1e-4)


def test_rsqrt_random():
    a = (torch.rand(32, 64) + 0.1).to("webgpu")  # Positive values only
    result = torch.rsqrt(a)
    expected = torch.rsqrt(a.to("cpu"))
    assert torch.allclose(result.to("cpu"), expected, rtol=1e-4, atol=1e-4)
