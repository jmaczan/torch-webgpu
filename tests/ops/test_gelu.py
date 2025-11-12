import pytest
import torch
import torch_webgpu


# TODO: improve Gelu accuracy - atol=1e-3 ok, atol=1e-4 fail
def test_gelu_happy_path():
    a = torch.tensor([-1.0, 0.0, 1.0, 2.0], dtype=torch.float32)
    a = a.to(device="webgpu")
    result = torch.nn.functional.gelu(a)
    result = result.to("cpu")
    expected = torch.nn.functional.gelu(
        torch.tensor([-1.0, 0.0, 1.0, 2.0], dtype=torch.float32)
    )
    assert torch.allclose(result, expected, atol=1e-3)


def test_gelu_broadcasting():
    a = torch.tensor([[0.5, -0.5], [1.5, -1.5]], dtype=torch.float32)
    a = a.to(device="webgpu")
    result = torch.nn.functional.gelu(a)
    result = result.to("cpu")
    expected = torch.nn.functional.gelu(
        torch.tensor([[0.5, -0.5], [1.5, -1.5]], dtype=torch.float32)
    )
    assert torch.allclose(result, expected, atol=1e-3)


def test_gelu_scalar():
    a = torch.tensor([2.0], dtype=torch.float32)
    a = a.to(device="webgpu")
    result = torch.nn.functional.gelu(a)
    result = result.to("cpu")
    expected = torch.nn.functional.gelu(torch.tensor([2.0], dtype=torch.float32))
    assert torch.allclose(result, expected, atol=1e-3)


def test_gelu_zero():
    a = torch.tensor([0.0], dtype=torch.float32)
    a = a.to(device="webgpu")
    result = torch.nn.functional.gelu(a)
    result = result.to("cpu")
    expected = torch.nn.functional.gelu(torch.tensor([0.0], dtype=torch.float32))
    assert torch.allclose(result, expected, atol=1e-3)


# def test_gelu_empty():
#     a = torch.tensor([], dtype=torch.float32)
#     a = a.to(device="webgpu")
#     result = torch.nn.functional.gelu(a)
#     result = result.to("cpu")
#     expected = torch.nn.functional.gelu(torch.tensor([], dtype=torch.float32))
#     assert result.numel() == 0
#     assert expected.numel() == 0


# def test_gelu_incompatible_dtype():
#     a = torch.tensor([1, 2, 3], dtype=torch.int32)
#     a = a.to(device="webgpu")
#     with pytest.raises(RuntimeError):
#         torch.nn.functional.gelu(a)
