import torch
import torch_webgpu  # noqa: F401


# Mean tests
def test_mean_1d():
    a = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], device="webgpu")
    result = torch.mean(a)
    expected = torch.tensor(3.0)
    assert torch.allclose(result.to("cpu"), expected, rtol=1e-4, atol=1e-4)


def test_mean_2d():
    a = torch.randn(16, 32).to("webgpu")
    result = torch.mean(a)
    expected = torch.mean(a.to("cpu"))
    assert torch.allclose(result.to("cpu"), expected, rtol=1e-3, atol=1e-3)


def test_mean_large():
    a = torch.randn(64, 64).to("webgpu")
    result = torch.mean(a)
    expected = torch.mean(a.to("cpu"))
    assert torch.allclose(result.to("cpu"), expected, rtol=1e-3, atol=1e-3)


# Sum tests
def test_sum_1d():
    a = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], device="webgpu")
    result = torch.sum(a)
    expected = torch.tensor(15.0)
    assert torch.allclose(result.to("cpu"), expected, rtol=1e-4, atol=1e-4)


def test_sum_2d():
    a = torch.randn(16, 32).to("webgpu")
    result = torch.sum(a)
    expected = torch.sum(a.to("cpu"))
    assert torch.allclose(result.to("cpu"), expected, rtol=1e-2, atol=1e-2)


def test_sum_ones():
    a = torch.ones(10, 10, device="webgpu")
    result = torch.sum(a)
    expected = torch.tensor(100.0)
    assert torch.allclose(result.to("cpu"), expected, rtol=1e-4, atol=1e-4)
