import torch
import torch_webgpu  # noqa: F401


def test_cos_happy_path():
    a = torch.tensor([-1.5, 2.7, 1.0, 2.0], device="webgpu")
    result = torch.cos(a)
    expected = torch.tensor([0.0707, -0.9041, 0.5403, -0.4161], device="cpu")
    assert torch.allclose(result.to("cpu"), expected, atol=1e-3)
