import pytest
import torch
import torch_webgpu  # noqa: F401


def test_reshape_happy_path():
    a = torch.arange(6, dtype=torch.float32)
    a = a.to(device="webgpu")
    result = torch.reshape(a, (2, 3))
    result = result.to("cpu")
    expected = torch.tensor([[0, 1, 2], [3, 4, 5]], dtype=torch.float32)
    assert torch.all(result == expected)


def test_reshape_with_minus_one():
    a = torch.arange(8, dtype=torch.float32)
    a = a.to(device="webgpu")
    result = torch.reshape(a, (2, -1))
    result = result.to("cpu")
    expected = torch.tensor([[0, 1, 2, 3], [4, 5, 6, 7]], dtype=torch.float32)
    assert torch.all(result == expected)


def test_reshape_to_flat():
    a = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32)
    a = a.to(device="webgpu")
    result = torch.reshape(a, (-1,))
    result = result.to("cpu")
    expected = torch.tensor([1, 2, 3, 4], dtype=torch.float32)
    assert torch.all(result == expected)


def test_reshape_invalid_shape():
    a = torch.arange(6, dtype=torch.float32)
    a = a.to(device="webgpu")
    with pytest.raises(RuntimeError):
        torch.reshape(a, (4, 2))


def test_reshape_zero_dim():
    a = torch.arange(6, dtype=torch.float32)
    a = a.to(device="webgpu")
    with pytest.raises(RuntimeError):
        torch.reshape(a, (0, 6))
