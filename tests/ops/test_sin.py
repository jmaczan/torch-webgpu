import torch
import torch_webgpu  # noqa: F401
import math


def test_sin_basic():
    a = torch.tensor([0.0, math.pi / 2, math.pi, math.pi * 1.5], device="webgpu")
    result = torch.sin(a)
    expected = torch.sin(a.to("cpu"))
    assert torch.allclose(result.to("cpu"), expected, rtol=1e-4, atol=1e-4)


def test_sin_random():
    a = torch.randn(32, 64).to("webgpu")
    result = torch.sin(a)
    expected = torch.sin(a.to("cpu"))
    assert torch.allclose(result.to("cpu"), expected, rtol=1e-4, atol=1e-4)
