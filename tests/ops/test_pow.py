import torch
import torch_webgpu  # noqa: F401


def test_pow_scalar():
    a = torch.tensor([1.0, 2.0, 3.0, 4.0], device="webgpu")
    result = torch.pow(a, 2.0)
    expected = torch.tensor([1.0, 4.0, 9.0, 16.0])
    assert torch.allclose(result.to("cpu"), expected, rtol=1e-4, atol=1e-4)


def test_pow_fractional():
    a = torch.tensor([1.0, 4.0, 9.0, 16.0], device="webgpu")
    result = torch.pow(a, 0.5)
    expected = torch.tensor([1.0, 2.0, 3.0, 4.0])
    assert torch.allclose(result.to("cpu"), expected, rtol=1e-4, atol=1e-4)


def test_pow_random():
    a = (torch.rand(32, 64) + 0.1).to("webgpu")  # Positive values
    result = torch.pow(a, 3.0)
    expected = torch.pow(a.to("cpu"), 3.0)
    assert torch.allclose(result.to("cpu"), expected, rtol=1e-3, atol=1e-3)
