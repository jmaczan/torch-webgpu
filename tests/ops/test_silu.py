import torch
import torch_webgpu  # noqa: F401


def test_silu_happy_path():
    a = torch.tensor([-1.0, 0.0, 1.0, 2.0], dtype=torch.float32)
    a = a.to(device="webgpu")
    result = torch.nn.functional.silu(a)
    result = result.to("cpu")
    expected = torch.nn.functional.silu(
        torch.tensor([-1.0, 0.0, 1.0, 2.0], dtype=torch.float32)
    )
    assert torch.allclose(result, expected, atol=1e-6)


def test_silu_broadcasting():
    a = torch.tensor([[0.5, -0.5], [1.5, -1.5]], dtype=torch.float32)
    a = a.to(device="webgpu")
    result = torch.nn.functional.silu(a)
    result = result.to("cpu")
    expected = torch.nn.functional.silu(
        torch.tensor([[0.5, -0.5], [1.5, -1.5]], dtype=torch.float32)
    )
    assert torch.allclose(result, expected, atol=1e-6)


def test_silu_scalar():
    a = torch.tensor([2.0], dtype=torch.float32)
    a = a.to(device="webgpu")
    result = torch.nn.functional.silu(a)
    result = result.to("cpu")
    expected = torch.nn.functional.silu(
        torch.tensor(
            [2.0],
            dtype=torch.float32,
        )
    )
    assert torch.allclose(result, expected, atol=1e-6)


def test_silu_zero():
    a = torch.tensor([0.0], dtype=torch.float32)
    a = a.to(device="webgpu")
    result = torch.nn.functional.silu(a)
    result = result.to("cpu")
    expected = torch.nn.functional.silu(
        torch.tensor(
            [0.0],
            dtype=torch.float32,
        )
    )
    assert torch.allclose(result, expected, atol=1e-6)


# TODO: segfault
# def test_silu_empty():
#     a = torch.tensor([], dtype=torch.float32)
#     a = a.to(device="webgpu")
#     result = torch.nn.functional.silu(a)
#     result = result.to("cpu")
#     expected = torch.nn.functional.silu(torch.tensor([],
# dtype=torch.float32,))
#     assert result.numel() == 0
#     assert expected.numel() == 0


# def test_silu_incompatible_dtype():
#     a = torch.tensor([1, 2, 3], dtype=torch.int32)
#     a = a.to(device="webgpu")
#     with pytest.raises(RuntimeError):
#         torch.nn.functional.silu(a)
