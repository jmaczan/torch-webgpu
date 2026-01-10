import torch
import torch_webgpu  # noqa: F401


def test_abs_basic():
    a = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0], device="webgpu")
    result = torch.abs(a)
    expected = torch.tensor([2.0, 1.0, 0.0, 1.0, 2.0])
    assert torch.allclose(result.to("cpu"), expected)


def test_abs_random():
    a = torch.randn(32, 64).to("webgpu")
    result = torch.abs(a)
    expected = torch.abs(a.to("cpu"))
    assert torch.allclose(result.to("cpu"), expected, rtol=1e-4, atol=1e-4)
