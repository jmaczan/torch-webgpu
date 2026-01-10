import torch
import torch_webgpu  # noqa: F401


# Transpose tests
def test_transpose_2d():
    a = torch.randn(3, 4).to("webgpu")
    result = torch.transpose(a, 0, 1)
    expected = torch.transpose(a.to("cpu"), 0, 1)
    assert result.shape == expected.shape
    assert torch.allclose(result.to("cpu"), expected, rtol=1e-4, atol=1e-4)


def test_transpose_3d():
    a = torch.randn(2, 3, 4).to("webgpu")
    result = torch.transpose(a, 0, 2)
    expected = torch.transpose(a.to("cpu"), 0, 2)
    assert result.shape == expected.shape
    assert torch.allclose(result.to("cpu").contiguous(), expected.contiguous(), rtol=1e-4, atol=1e-4)


# Permute tests
def test_permute_3d():
    a = torch.randn(2, 3, 4).to("webgpu")
    result = a.permute(2, 0, 1)
    expected = a.to("cpu").permute(2, 0, 1)
    assert result.shape == expected.shape
    assert torch.allclose(result.to("cpu").contiguous(), expected.contiguous(), rtol=1e-4, atol=1e-4)


def test_permute_4d():
    a = torch.randn(2, 3, 4, 5).to("webgpu")
    result = a.permute(3, 1, 2, 0)
    expected = a.to("cpu").permute(3, 1, 2, 0)
    assert result.shape == expected.shape
    assert torch.allclose(result.to("cpu").contiguous(), expected.contiguous(), rtol=1e-4, atol=1e-4)


# Unsqueeze tests
def test_unsqueeze():
    a = torch.randn(3, 4).to("webgpu")
    result = torch.unsqueeze(a, 0)
    expected = torch.unsqueeze(a.to("cpu"), 0)
    assert result.shape == expected.shape
    assert torch.allclose(result.to("cpu"), expected, rtol=1e-4, atol=1e-4)


def test_unsqueeze_last():
    a = torch.randn(3, 4).to("webgpu")
    result = torch.unsqueeze(a, -1)
    expected = torch.unsqueeze(a.to("cpu"), -1)
    assert result.shape == expected.shape
    assert torch.allclose(result.to("cpu"), expected, rtol=1e-4, atol=1e-4)


# Squeeze tests
def test_squeeze():
    a = torch.randn(1, 3, 1, 4).to("webgpu")
    result = torch.squeeze(a)
    expected = torch.squeeze(a.to("cpu"))
    assert result.shape == expected.shape
    assert torch.allclose(result.to("cpu"), expected, rtol=1e-4, atol=1e-4)


def test_squeeze_dim():
    a = torch.randn(1, 3, 4).to("webgpu")
    result = torch.squeeze(a, 0)
    expected = torch.squeeze(a.to("cpu"), 0)
    assert result.shape == expected.shape
    assert torch.allclose(result.to("cpu"), expected, rtol=1e-4, atol=1e-4)


# Expand tests
def test_expand():
    a = torch.randn(1, 4).to("webgpu")
    result = a.expand(3, 4)
    expected = a.to("cpu").expand(3, 4)
    assert result.shape == expected.shape
    assert torch.allclose(result.to("cpu"), expected, rtol=1e-4, atol=1e-4)


def test_expand_add_dims():
    a = torch.randn(4).to("webgpu")
    result = a.expand(2, 3, 4)
    expected = a.to("cpu").expand(2, 3, 4)
    assert result.shape == expected.shape
    assert torch.allclose(result.to("cpu"), expected, rtol=1e-4, atol=1e-4)


# Slice tests
def test_slice():
    a = torch.arange(20).reshape(4, 5).float().to("webgpu")
    result = a[1:3]
    expected = a.to("cpu")[1:3]
    assert result.shape == expected.shape
    assert torch.allclose(result.to("cpu"), expected, rtol=1e-4, atol=1e-4)


def test_slice_with_step():
    a = torch.arange(20).reshape(4, 5).float().to("webgpu")
    result = a[:, ::2]
    expected = a.to("cpu")[:, ::2]
    assert result.shape == expected.shape
    assert torch.allclose(result.to("cpu"), expected, rtol=1e-4, atol=1e-4)


# Select tests
def test_select():
    a = torch.arange(20).reshape(4, 5).float().to("webgpu")
    result = a.select(0, 2)
    expected = a.to("cpu").select(0, 2)
    assert result.shape == expected.shape
    assert torch.allclose(result.to("cpu"), expected, rtol=1e-4, atol=1e-4)


def test_select_negative():
    a = torch.arange(20).reshape(4, 5).float().to("webgpu")
    result = a.select(0, -1)
    expected = a.to("cpu").select(0, -1)
    assert result.shape == expected.shape
    assert torch.allclose(result.to("cpu"), expected, rtol=1e-4, atol=1e-4)


# T tests
def test_t():
    a = torch.randn(3, 4).to("webgpu")
    result = a.t()
    expected = a.to("cpu").t()
    assert result.shape == expected.shape
    assert torch.allclose(result.to("cpu"), expected, rtol=1e-4, atol=1e-4)


# Clone tests
def test_clone():
    a = torch.randn(3, 4).to("webgpu")
    result = a.clone()
    assert result.shape == a.shape
    assert torch.allclose(result.to("cpu"), a.to("cpu"), rtol=1e-4, atol=1e-4)


# Contiguous tests
def test_contiguous():
    a = torch.randn(3, 4).to("webgpu").transpose(0, 1)
    result = a.contiguous()
    assert result.is_contiguous()
    assert torch.allclose(result.to("cpu"), a.to("cpu").contiguous(), rtol=1e-4, atol=1e-4)
