import torch
import torch_webgpu  # noqa: F401


def test_arange_basic():
    result = torch.arange(10, device="webgpu")
    expected = torch.arange(10).float()
    assert torch.allclose(result.to("cpu"), expected)


def test_arange_start_end():
    result = torch.arange(5, 15, device="webgpu")
    expected = torch.arange(5, 15).float()
    assert torch.allclose(result.to("cpu"), expected)


def test_arange_with_step():
    result = torch.arange(0, 10, 2, device="webgpu")
    expected = torch.arange(0, 10, 2).float()
    assert torch.allclose(result.to("cpu"), expected)


def test_arange_float():
    result = torch.arange(0.0, 5.0, 0.5, device="webgpu")
    expected = torch.arange(0.0, 5.0, 0.5)
    assert torch.allclose(result.to("cpu"), expected, rtol=1e-4, atol=1e-4)


def test_zeros():
    result = torch.zeros(3, 4, device="webgpu")
    expected = torch.zeros(3, 4)
    assert torch.allclose(result.to("cpu"), expected)


def test_ones():
    result = torch.ones(3, 4, device="webgpu")
    expected = torch.ones(3, 4)
    assert torch.allclose(result.to("cpu"), expected)


def test_full():
    result = torch.full((3, 4), 3.14, device="webgpu")
    expected = torch.full((3, 4), 3.14)
    assert torch.allclose(result.to("cpu"), expected)


def test_fill():
    result = torch.empty(3, 4, device="webgpu")
    result.fill_(2.5)
    expected = torch.full((3, 4), 2.5)
    assert torch.allclose(result.to("cpu"), expected)


def test_zero():
    cpu_tensor = torch.randn(3, 4)
    result = cpu_tensor.to("webgpu")
    result.zero_()
    expected = torch.zeros(3, 4)
    assert torch.allclose(result.to("cpu"), expected)


def test_new_ones():
    self = torch.randn(2, 3).to("webgpu")
    result = self.new_ones(4, 5)
    expected = torch.ones(4, 5)
    assert torch.allclose(result.to("cpu"), expected)


def test_new_zeros():
    self = torch.randn(2, 3).to("webgpu")
    result = self.new_zeros(4, 5)
    expected = torch.zeros(4, 5)
    assert torch.allclose(result.to("cpu"), expected)
