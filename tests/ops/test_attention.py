import torch
import torch_webgpu  # noqa: F401


def test_scaled_dot_product_attention_basic():
    # [batch, heads, seq_len, head_dim]
    query = torch.randn(1, 4, 8, 16).to("webgpu")
    key = torch.randn(1, 4, 8, 16).to("webgpu")
    value = torch.randn(1, 4, 8, 16).to("webgpu")

    result = torch.nn.functional.scaled_dot_product_attention(query, key, value)
    expected = torch.nn.functional.scaled_dot_product_attention(
        query.to("cpu"), key.to("cpu"), value.to("cpu")
    )

    assert result.shape == expected.shape
    assert torch.allclose(result.to("cpu"), expected, rtol=1e-3, atol=1e-3)


def test_scaled_dot_product_attention_small():
    query = torch.randn(1, 2, 4, 8).to("webgpu")
    key = torch.randn(1, 2, 4, 8).to("webgpu")
    value = torch.randn(1, 2, 4, 8).to("webgpu")

    result = torch.nn.functional.scaled_dot_product_attention(query, key, value)
    expected = torch.nn.functional.scaled_dot_product_attention(
        query.to("cpu"), key.to("cpu"), value.to("cpu")
    )

    assert result.shape == expected.shape
    assert torch.allclose(result.to("cpu"), expected, rtol=1e-3, atol=1e-3)


def test_scaled_dot_product_attention_batch():
    query = torch.randn(2, 4, 8, 16).to("webgpu")
    key = torch.randn(2, 4, 8, 16).to("webgpu")
    value = torch.randn(2, 4, 8, 16).to("webgpu")

    result = torch.nn.functional.scaled_dot_product_attention(query, key, value)
    expected = torch.nn.functional.scaled_dot_product_attention(
        query.to("cpu"), key.to("cpu"), value.to("cpu")
    )

    assert result.shape == expected.shape
    assert torch.allclose(result.to("cpu"), expected, rtol=1e-3, atol=1e-3)


def test_scaled_dot_product_attention_causal():
    query = torch.randn(1, 4, 8, 16).to("webgpu")
    key = torch.randn(1, 4, 8, 16).to("webgpu")
    value = torch.randn(1, 4, 8, 16).to("webgpu")

    # is_causal = True
    result = torch.nn.functional.scaled_dot_product_attention(
        query, key, value, is_causal=True
    )
    expected = torch.nn.functional.scaled_dot_product_attention(
        query.to("cpu"), key.to("cpu"), value.to("cpu"), is_causal=True
    )

    assert result.shape == expected.shape
    assert torch.allclose(result.to("cpu"), expected, rtol=1e-3, atol=1e-3)


def test_scaled_dot_product_attention_custom_scale():
    query = torch.randn(1, 4, 8, 16).to("webgpu")
    key = torch.randn(1, 4, 8, 16).to("webgpu")
    value = torch.randn(1, 4, 8, 16).to("webgpu")

    # Custom scale
    scale = 0.5
    result = torch.nn.functional.scaled_dot_product_attention(
        query, key, value, scale=scale
    )
    expected = torch.nn.functional.scaled_dot_product_attention(
        query.to("cpu"), key.to("cpu"), value.to("cpu"), scale=scale
    )

    assert result.shape == expected.shape
    assert torch.allclose(result.to("cpu"), expected, rtol=1e-3, atol=1e-3)
