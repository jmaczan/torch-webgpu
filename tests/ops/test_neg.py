import torch
import torch_webgpu  # noqa: F401


def test_neg_basic():
    a = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0], device="webgpu")
    result = torch.neg(a)
    expected = torch.tensor([2.0, 1.0, 0.0, -1.0, -2.0])
    assert torch.allclose(result.to("cpu"), expected)


def test_neg_random():
    a = torch.randn(32, 64).to("webgpu")
    result = torch.neg(a)
    expected = torch.neg(a.to("cpu"))
    assert torch.allclose(result.to("cpu"), expected, rtol=1e-4, atol=1e-4)
