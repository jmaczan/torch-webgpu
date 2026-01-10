import torch
import torch_webgpu  # noqa: F401


def test_arange_basic():
    # Create on CPU first, then move to WebGPU
    cpu_expected = torch.arange(10).float()
    result = cpu_expected.to("webgpu")
    assert torch.allclose(result.to("cpu"), cpu_expected)


def test_arange_start_end():
    cpu_expected = torch.arange(5, 15).float()
    result = cpu_expected.to("webgpu")
    assert torch.allclose(result.to("cpu"), cpu_expected)


def test_arange_with_step():
    cpu_expected = torch.arange(0, 10, 2).float()
    result = cpu_expected.to("webgpu")
    assert torch.allclose(result.to("cpu"), cpu_expected)


def test_arange_float():
    cpu_expected = torch.arange(0.0, 5.0, 0.5)
    result = cpu_expected.to("webgpu")
    assert torch.allclose(result.to("cpu"), cpu_expected, rtol=1e-4, atol=1e-4)


def test_zeros():
    cpu_expected = torch.zeros(3, 4)
    result = cpu_expected.to("webgpu")
    assert torch.allclose(result.to("cpu"), cpu_expected)


def test_ones():
    cpu_expected = torch.ones(3, 4)
    result = cpu_expected.to("webgpu")
    assert torch.allclose(result.to("cpu"), cpu_expected)


def test_full():
    cpu_expected = torch.full((3, 4), 3.14)
    result = cpu_expected.to("webgpu")
    assert torch.allclose(result.to("cpu"), cpu_expected)


def test_fill():
    cpu_tensor = torch.empty(3, 4)
    result = cpu_tensor.to("webgpu")
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
