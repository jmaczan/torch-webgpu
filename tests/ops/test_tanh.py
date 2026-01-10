import torch
import torch_webgpu  # noqa: F401


def test_tanh_basic():
    a = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0], device="webgpu")
    result = torch.tanh(a)
    expected = torch.tanh(a.to("cpu"))
    assert torch.allclose(result.to("cpu"), expected, rtol=1e-4, atol=1e-4)


def test_tanh_random():
    a = torch.randn(32, 64).to("webgpu")
    result = torch.tanh(a)
    expected = torch.tanh(a.to("cpu"))
    assert torch.allclose(result.to("cpu"), expected, rtol=1e-4, atol=1e-4)
