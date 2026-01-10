import torch
import torch_webgpu  # noqa: F401


def test_div_basic():
    a = torch.tensor([10.0, 20.0, 30.0], device="webgpu")
    b = torch.tensor([2.0, 4.0, 5.0], device="webgpu")
    result = a / b
    expected = torch.tensor([5.0, 5.0, 6.0])
    assert torch.allclose(result.to("cpu"), expected)


def test_div_fractional():
    a = torch.tensor([1.0, 3.0, 5.0], device="webgpu")
    b = torch.tensor([3.0, 7.0, 2.0], device="webgpu")
    result = a / b
    expected = a.to("cpu") / b.to("cpu")
    assert torch.allclose(result.to("cpu"), expected, rtol=1e-4, atol=1e-4)


def test_div_negative():
    a = torch.tensor([-10.0, 20.0, -30.0], device="webgpu")
    b = torch.tensor([2.0, -4.0, -5.0], device="webgpu")
    result = a / b
    expected = torch.tensor([-5.0, -5.0, 6.0])
    assert torch.allclose(result.to("cpu"), expected)


def test_div_broadcast():
    a = torch.randn(3, 4, device="webgpu") + 1.0
    b = torch.randn(4, device="webgpu").abs() + 0.1
    result = a / b
    expected = a.to("cpu") / b.to("cpu")
    assert torch.allclose(result.to("cpu"), expected, rtol=1e-4, atol=1e-4)
