import torch
import torch_webgpu  # noqa: F401


def test_softmax_1d():
    a = torch.tensor([1.0, 2.0, 3.0, 4.0], device="webgpu")
    result = torch.softmax(a, 0)
    expected = torch.softmax(a.to("cpu"), 0)
    assert torch.allclose(result.to("cpu"), expected, rtol=1e-4, atol=1e-4)


def test_softmax_2d_last_dim():
    a = torch.randn(4, 8).to("webgpu")
    result = torch.softmax(a, -1)
    expected = torch.softmax(a.to("cpu"), -1)
    assert torch.allclose(result.to("cpu"), expected, rtol=1e-4, atol=1e-4)


def test_softmax_2d_first_dim():
    a = torch.randn(4, 8).to("webgpu")
    result = torch.softmax(a, 0)
    expected = torch.softmax(a.to("cpu"), 0)
    assert torch.allclose(result.to("cpu"), expected, rtol=1e-4, atol=1e-4)


def test_softmax_3d():
    a = torch.randn(2, 4, 8).to("webgpu")
    result = torch.softmax(a, -1)
    expected = torch.softmax(a.to("cpu"), -1)
    assert torch.allclose(result.to("cpu"), expected, rtol=1e-4, atol=1e-4)


def test_softmax_sums_to_one():
    a = torch.randn(4, 8).to("webgpu")
    result = torch.softmax(a, -1)
    sums = result.sum(-1).to("cpu")
    expected = torch.ones(4)
    assert torch.allclose(sums, expected, rtol=1e-4, atol=1e-4)


def test_log_softmax():
    a = torch.randn(4, 8).to("webgpu")
    result = torch.log_softmax(a, -1)
    expected = torch.log_softmax(a.to("cpu"), -1)
    assert torch.allclose(result.to("cpu"), expected, rtol=1e-4, atol=1e-4)
