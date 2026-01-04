import pytest
import torch
import torch_webgpu  # noqa: F401

TOL = 1e-3


def _assert_close_webgpu_cpu(
    webgpu_tensor: torch.Tensor, cpu_tensor: torch.Tensor
) -> None:
    assert torch.allclose(webgpu_tensor.to("cpu"), cpu_tensor, rtol=TOL, atol=TOL)


def test_mm_small_square_happy_path():
    a_cpu = torch.tensor([[1.0, -2.0], [3.5, 0.5]], dtype=torch.float32)
    b_cpu = torch.tensor([[2.0, 1.5], [-1.0, 4.0]], dtype=torch.float32)

    expected = torch.mm(a_cpu, b_cpu)
    result = torch.mm(a_cpu.to(device="webgpu"), b_cpu.to(device="webgpu"))

    _assert_close_webgpu_cpu(result, expected)


def test_mm_rectangular_happy_path():
    a_cpu = torch.tensor([[1.0, 2.0, 3.0], [-1.0, 0.5, 4.0]], dtype=torch.float32)
    b_cpu = torch.tensor([[2.5, -1.0], [0.0, 3.0], [-2.0, 1.5]], dtype=torch.float32)

    expected = torch.mm(a_cpu, b_cpu)
    result = torch.mm(a_cpu.to(device="webgpu"), b_cpu.to(device="webgpu"))

    _assert_close_webgpu_cpu(result, expected)


def test_mm_large_random():
    torch.manual_seed(0)
    a_cpu = torch.randn(128, 96, dtype=torch.float32)
    b_cpu = torch.randn(96, 64, dtype=torch.float32)

    expected = torch.mm(a_cpu, b_cpu)
    result = torch.mm(a_cpu.to(device="webgpu"), b_cpu.to(device="webgpu"))

    _assert_close_webgpu_cpu(result, expected)


def test_mm_incompatible_shapes():
    a_cpu = torch.randn(4, 3)
    b_cpu = torch.randn(2, 4)

    a = a_cpu.to(device="webgpu")
    b = b_cpu.to(device="webgpu")

    with pytest.raises(RuntimeError):
        torch.mm(a, b)
