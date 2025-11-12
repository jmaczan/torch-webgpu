import pytest
import torch
import torch_webgpu


def test_mul_happy_path():
    a = torch.tensor([1, 2, 3], dtype=torch.float32)
    a = a.to(device="webgpu")
    b = torch.tensor([4, 5, 6], dtype=torch.float32)
    b = b.to(device="webgpu")
    result = torch.mul(a, b)
    result = result.to("cpu")
    expected = torch.tensor([4, 10, 18], dtype=torch.float32)
    expected = expected.to(device="cpu")
    assert torch.all(result == expected)


def test_mul_happy_path_asterix():
    a = torch.tensor([1, 2, 3], dtype=torch.float32)
    a = a.to(device="webgpu")
    b = torch.tensor([4, 5, 6], dtype=torch.float32)
    b = b.to(device="webgpu")
    result = a * b
    result = result.to("cpu")
    expected = torch.tensor([4, 10, 18], dtype=torch.float32)
    expected = expected.to(device="cpu")
    assert torch.all(result == expected)


def test_mul_broadcasting():
    a = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32)
    a = a.to(device="webgpu")
    b = torch.tensor([10, 20], dtype=torch.float32)
    b = b.to(device="webgpu")
    result = torch.mul(a, b)
    result = result.to("cpu")
    expected = torch.tensor([[10, 40], [30, 80]], dtype=torch.float32)
    expected = expected.to(device="cpu")
    assert torch.all(result == expected)


def test_mul_scalar():
    a = torch.tensor([1, -2, 3], dtype=torch.float32)
    a = a.to(device="webgpu")
    result = torch.mul(a, 2)
    result = result.to("cpu")
    expected = torch.tensor([2, -4, 6], dtype=torch.float32)
    expected = expected.to(device="cpu")
    assert torch.all(result == expected)


def test_mul_zero():
    a = torch.tensor([0, 1, 2], dtype=torch.float32)
    a = a.to(device="webgpu")
    b = torch.tensor([0, 0, 0], dtype=torch.float32)
    b = b.to(device="webgpu")
    result = torch.mul(a, b)
    result = result.to("cpu")
    expected = torch.tensor([0, 0, 0], dtype=torch.float32)
    expected = expected.to(device="cpu")
    assert torch.all(result == expected)


def test_mul_incompatible_shapes():
    a = torch.tensor([1, 2, 3], dtype=torch.float32)
    a = a.to(device="webgpu")
    b = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32)
    b = b.to(device="webgpu")
    with pytest.raises(RuntimeError):
        torch.mul(a, b)


# TODO: fix segfault
# def test_mul_empty():
#     try:
#         a = torch.tensor([], dtype=torch.float32)
#         a = a.to(device="webgpu")
#         b = torch.tensor([], dtype=torch.float32)
#         b = b.to(device="webgpu")
#         result = torch.mul(a, b)
#         result = result.to("cpu")
#         assert result.numel() == 0
#     except Exception as e:
#         print(e)
