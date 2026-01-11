import torch
import torch_webgpu  # noqa: F401


def test_sub_basic():
    a = torch.tensor([5.0, 7.0, 10.0], device="webgpu")
    b = torch.tensor([2.0, 3.0, 4.0], device="webgpu")
    result = a - b
    expected = torch.tensor([3.0, 4.0, 6.0])
    assert torch.allclose(result.to("cpu"), expected)


def test_sub_with_alpha():
    a = torch.tensor([10.0, 20.0, 30.0], device="webgpu")
    b = torch.tensor([1.0, 2.0, 3.0], device="webgpu")
    result = torch.sub(a, b, alpha=2)
    expected = torch.tensor([8.0, 16.0, 24.0])
    assert torch.allclose(result.to("cpu"), expected)


def test_sub_negative():
    a = torch.tensor([-1.0, -5.0, 3.0], device="webgpu")
    b = torch.tensor([2.0, -3.0, -1.0], device="webgpu")
    result = a - b
    expected = torch.tensor([-3.0, -2.0, 4.0])
    assert torch.allclose(result.to("cpu"), expected)


def test_sub_broadcast():
    # Create on CPU first, then move to WebGPU (randn not implemented on WebGPU)
    a_cpu = torch.randn(3, 4)
    b_cpu = torch.randn(4)
    a = a_cpu.to("webgpu")
    b = b_cpu.to("webgpu")
    result = a - b
    expected = a_cpu - b_cpu
    assert torch.allclose(result.to("cpu"), expected, rtol=1e-4, atol=1e-4)
