import torch
import torch_webgpu  # noqa: F401


# Linear tests
def test_linear_2d():
    input = torch.randn(4, 8).to("webgpu")
    weight = torch.randn(16, 8).to("webgpu")
    bias = torch.randn(16).to("webgpu")
    result = torch.nn.functional.linear(input, weight, bias)
    expected = torch.nn.functional.linear(input.to("cpu"), weight.to("cpu"), bias.to("cpu"))
    assert result.shape == expected.shape
    assert torch.allclose(result.to("cpu"), expected, rtol=1e-3, atol=1e-3)


def test_linear_no_bias():
    input = torch.randn(4, 8).to("webgpu")
    weight = torch.randn(16, 8).to("webgpu")
    result = torch.nn.functional.linear(input, weight)
    expected = torch.nn.functional.linear(input.to("cpu"), weight.to("cpu"))
    assert result.shape == expected.shape
    assert torch.allclose(result.to("cpu"), expected, rtol=1e-3, atol=1e-3)


def test_linear_3d():
    input = torch.randn(2, 4, 8).to("webgpu")
    weight = torch.randn(16, 8).to("webgpu")
    bias = torch.randn(16).to("webgpu")
    result = torch.nn.functional.linear(input, weight, bias)
    expected = torch.nn.functional.linear(input.to("cpu"), weight.to("cpu"), bias.to("cpu"))
    assert result.shape == expected.shape
    assert torch.allclose(result.to("cpu"), expected, rtol=1e-3, atol=1e-3)


# Addmm tests
def test_addmm():
    m = torch.randn(4, 16).to("webgpu")
    mat1 = torch.randn(4, 8).to("webgpu")
    mat2 = torch.randn(8, 16).to("webgpu")
    result = torch.addmm(m, mat1, mat2)
    expected = torch.addmm(m.to("cpu"), mat1.to("cpu"), mat2.to("cpu"))
    assert result.shape == expected.shape
    assert torch.allclose(result.to("cpu"), expected, rtol=1e-3, atol=1e-3)


def test_addmm_with_alpha_beta():
    m = torch.randn(4, 16).to("webgpu")
    mat1 = torch.randn(4, 8).to("webgpu")
    mat2 = torch.randn(8, 16).to("webgpu")
    result = torch.addmm(m, mat1, mat2, beta=0.5, alpha=2.0)
    expected = torch.addmm(m.to("cpu"), mat1.to("cpu"), mat2.to("cpu"), beta=0.5, alpha=2.0)
    assert result.shape == expected.shape
    assert torch.allclose(result.to("cpu"), expected, rtol=1e-3, atol=1e-3)


# BMM tests
def test_bmm():
    a = torch.randn(4, 8, 16).to("webgpu")
    b = torch.randn(4, 16, 32).to("webgpu")
    result = torch.bmm(a, b)
    expected = torch.bmm(a.to("cpu"), b.to("cpu"))
    assert result.shape == expected.shape
    assert torch.allclose(result.to("cpu"), expected, rtol=1e-3, atol=1e-3)


def test_bmm_small():
    a = torch.randn(2, 3, 4).to("webgpu")
    b = torch.randn(2, 4, 5).to("webgpu")
    result = torch.bmm(a, b)
    expected = torch.bmm(a.to("cpu"), b.to("cpu"))
    assert result.shape == expected.shape
    assert torch.allclose(result.to("cpu"), expected, rtol=1e-3, atol=1e-3)


# Matmul tests
def test_matmul_vector_vector():
    a = torch.randn(8).to("webgpu")
    b = torch.randn(8).to("webgpu")
    result = torch.matmul(a, b)
    expected = torch.matmul(a.to("cpu"), b.to("cpu"))
    assert result.shape == expected.shape
    assert torch.allclose(result.to("cpu"), expected, rtol=1e-3, atol=1e-3)


def test_matmul_matrix_vector():
    a = torch.randn(4, 8).to("webgpu")
    b = torch.randn(8).to("webgpu")
    result = torch.matmul(a, b)
    expected = torch.matmul(a.to("cpu"), b.to("cpu"))
    assert result.shape == expected.shape
    assert torch.allclose(result.to("cpu"), expected, rtol=1e-3, atol=1e-3)


def test_matmul_matrix_matrix():
    a = torch.randn(4, 8).to("webgpu")
    b = torch.randn(8, 16).to("webgpu")
    result = torch.matmul(a, b)
    expected = torch.matmul(a.to("cpu"), b.to("cpu"))
    assert result.shape == expected.shape
    assert torch.allclose(result.to("cpu"), expected, rtol=1e-3, atol=1e-3)


def test_matmul_batched():
    a = torch.randn(2, 4, 8).to("webgpu")
    b = torch.randn(2, 8, 16).to("webgpu")
    result = torch.matmul(a, b)
    expected = torch.matmul(a.to("cpu"), b.to("cpu"))
    assert result.shape == expected.shape
    assert torch.allclose(result.to("cpu"), expected, rtol=1e-3, atol=1e-3)


def test_matmul_4d():
    a = torch.randn(2, 3, 4, 8).to("webgpu")
    b = torch.randn(2, 3, 8, 16).to("webgpu")
    result = torch.matmul(a, b)
    expected = torch.matmul(a.to("cpu"), b.to("cpu"))
    assert result.shape == expected.shape
    assert torch.allclose(result.to("cpu"), expected, rtol=1e-3, atol=1e-3)
