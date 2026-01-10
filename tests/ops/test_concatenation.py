import torch
import torch_webgpu  # noqa: F401


# Cat tests
def test_cat_1d():
    a = torch.tensor([1.0, 2.0, 3.0], device="webgpu")
    b = torch.tensor([4.0, 5.0, 6.0], device="webgpu")
    result = torch.cat([a, b], 0)
    expected = torch.cat([a.to("cpu"), b.to("cpu")], 0)
    assert result.shape == expected.shape
    assert torch.allclose(result.to("cpu"), expected, rtol=1e-4, atol=1e-4)


def test_cat_2d_dim0():
    a = torch.randn(2, 4).to("webgpu")
    b = torch.randn(3, 4).to("webgpu")
    result = torch.cat([a, b], 0)
    expected = torch.cat([a.to("cpu"), b.to("cpu")], 0)
    assert result.shape == expected.shape
    assert torch.allclose(result.to("cpu"), expected, rtol=1e-4, atol=1e-4)


def test_cat_2d_dim1():
    a = torch.randn(3, 2).to("webgpu")
    b = torch.randn(3, 4).to("webgpu")
    result = torch.cat([a, b], 1)
    expected = torch.cat([a.to("cpu"), b.to("cpu")], 1)
    assert result.shape == expected.shape
    assert torch.allclose(result.to("cpu"), expected, rtol=1e-4, atol=1e-4)


def test_cat_multiple():
    a = torch.randn(2, 4).to("webgpu")
    b = torch.randn(2, 4).to("webgpu")
    c = torch.randn(2, 4).to("webgpu")
    result = torch.cat([a, b, c], 0)
    expected = torch.cat([a.to("cpu"), b.to("cpu"), c.to("cpu")], 0)
    assert result.shape == expected.shape
    assert torch.allclose(result.to("cpu"), expected, rtol=1e-4, atol=1e-4)


# Stack tests
def test_stack_1d():
    a = torch.tensor([1.0, 2.0, 3.0], device="webgpu")
    b = torch.tensor([4.0, 5.0, 6.0], device="webgpu")
    result = torch.stack([a, b], 0)
    expected = torch.stack([a.to("cpu"), b.to("cpu")], 0)
    assert result.shape == expected.shape
    assert torch.allclose(result.to("cpu"), expected, rtol=1e-4, atol=1e-4)


def test_stack_2d():
    a = torch.randn(3, 4).to("webgpu")
    b = torch.randn(3, 4).to("webgpu")
    result = torch.stack([a, b], 0)
    expected = torch.stack([a.to("cpu"), b.to("cpu")], 0)
    assert result.shape == expected.shape
    assert torch.allclose(result.to("cpu"), expected, rtol=1e-4, atol=1e-4)


def test_stack_dim1():
    a = torch.randn(3, 4).to("webgpu")
    b = torch.randn(3, 4).to("webgpu")
    result = torch.stack([a, b], 1)
    expected = torch.stack([a.to("cpu"), b.to("cpu")], 1)
    assert result.shape == expected.shape
    assert torch.allclose(result.to("cpu"), expected, rtol=1e-4, atol=1e-4)


# Narrow tests
def test_narrow():
    a = torch.arange(20).reshape(4, 5).float().to("webgpu")
    result = a.narrow(0, 1, 2)
    expected = a.to("cpu").narrow(0, 1, 2)
    assert result.shape == expected.shape
    assert torch.allclose(result.to("cpu"), expected, rtol=1e-4, atol=1e-4)


def test_narrow_dim1():
    a = torch.arange(20).reshape(4, 5).float().to("webgpu")
    result = a.narrow(1, 2, 3)
    expected = a.to("cpu").narrow(1, 2, 3)
    assert result.shape == expected.shape
    assert torch.allclose(result.to("cpu"), expected, rtol=1e-4, atol=1e-4)
