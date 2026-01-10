import torch
import torch_webgpu  # noqa: F401


def test_embedding_basic():
    weight = torch.randn(10, 8).to("webgpu")  # 10 vocab, 8 dim
    indices = torch.tensor([0, 2, 5, 9], dtype=torch.int).to("webgpu")
    result = torch.nn.functional.embedding(indices, weight)
    expected = torch.nn.functional.embedding(indices.to("cpu"), weight.to("cpu"))
    assert result.shape == expected.shape
    assert torch.allclose(result.to("cpu"), expected, rtol=1e-4, atol=1e-4)


def test_embedding_long_indices():
    weight = torch.randn(100, 16).to("webgpu")
    indices = torch.tensor([0, 10, 50, 99], dtype=torch.long).to("webgpu")
    result = torch.nn.functional.embedding(indices, weight)
    expected = torch.nn.functional.embedding(indices.to("cpu"), weight.to("cpu"))
    assert result.shape == expected.shape
    assert torch.allclose(result.to("cpu"), expected, rtol=1e-4, atol=1e-4)


def test_embedding_2d_indices():
    weight = torch.randn(50, 32).to("webgpu")
    indices = torch.tensor([[0, 1, 2], [10, 20, 30]], dtype=torch.int).to("webgpu")
    result = torch.nn.functional.embedding(indices, weight)
    expected = torch.nn.functional.embedding(indices.to("cpu"), weight.to("cpu"))
    assert result.shape == expected.shape
    assert torch.allclose(result.to("cpu"), expected, rtol=1e-4, atol=1e-4)


def test_embedding_larger():
    weight = torch.randn(1000, 64).to("webgpu")
    indices = torch.randint(0, 1000, (16, 32), dtype=torch.int).to("webgpu")
    result = torch.nn.functional.embedding(indices, weight)
    expected = torch.nn.functional.embedding(indices.to("cpu"), weight.to("cpu"))
    assert result.shape == expected.shape
    assert torch.allclose(result.to("cpu"), expected, rtol=1e-4, atol=1e-4)


def test_embedding_sequential():
    weight = torch.randn(20, 8).to("webgpu")
    indices = torch.arange(0, 10, dtype=torch.int).to("webgpu")
    result = torch.nn.functional.embedding(indices, weight)
    expected = torch.nn.functional.embedding(indices.to("cpu"), weight.to("cpu"))
    assert result.shape == expected.shape
    assert torch.allclose(result.to("cpu"), expected, rtol=1e-4, atol=1e-4)
