import torch
import torch_webgpu  # noqa: F401


def test_add_happy_path():
    a = torch.tensor([-1.5, 2.7, 1.0, 2.0], device="webgpu")
    b = torch.tensor([-1.0, 0.9, 1.1, -2.1], device="webgpu")
    result = a + b
    expected = torch.tensor([-2.5, 3.6, 2.1, -0.1], device="cpu")
    assert torch.allclose(result.to("cpu"), expected)
