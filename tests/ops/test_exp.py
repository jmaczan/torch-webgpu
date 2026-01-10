import torch
import torch_webgpu  # noqa: F401


def test_exp_basic():
    a = torch.tensor([0.0, 1.0, 2.0, -1.0], device="webgpu")
    result = torch.exp(a)
    expected = torch.exp(a.to("cpu"))
    assert torch.allclose(result.to("cpu"), expected, rtol=1e-4, atol=1e-4)


def test_exp_random():
    a = (torch.randn(32, 64) * 0.5).to("webgpu")  # Keep small to avoid overflow
    result = torch.exp(a)
    expected = torch.exp(a.to("cpu"))
    assert torch.allclose(result.to("cpu"), expected, rtol=1e-3, atol=1e-3)
