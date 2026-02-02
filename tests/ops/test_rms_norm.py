import torch
import torch_webgpu  # noqa: F401


def rms_norm_reference(x, weight, eps=1e-6):
    """Reference implementation of RMSNorm using PyTorch ops."""
    variance = x.pow(2).mean(-1, keepdim=True)
    x_normed = x * torch.rsqrt(variance + eps)
    return x_normed * weight


def test_rms_norm_basic():
    """Test basic RMSNorm on small tensor."""
    hidden_size = 16
    x = torch.randn(4, hidden_size, dtype=torch.float32)
    weight = torch.randn(hidden_size, dtype=torch.float32)
    eps = 1e-6

    # Move to WebGPU
    x_webgpu = x.to(device="webgpu")
    weight_webgpu = weight.to(device="webgpu")

    # Run WebGPU RMSNorm
    result = torch.ops.webgpu.rms_norm(x_webgpu, weight_webgpu, eps)
    result_cpu = result.to("cpu")

    # Compare with reference
    expected = rms_norm_reference(x, weight, eps)
    assert torch.allclose(result_cpu, expected, atol=1e-5, rtol=1e-5), \
        f"Max diff: {(result_cpu - expected).abs().max()}"


def test_rms_norm_qwen_hidden_size():
    """Test RMSNorm with Qwen2.5-0.5B hidden size (896)."""
    hidden_size = 896
    batch_size = 8
    x = torch.randn(batch_size, hidden_size, dtype=torch.float32)
    weight = torch.randn(hidden_size, dtype=torch.float32)
    eps = 1e-6

    x_webgpu = x.to(device="webgpu")
    weight_webgpu = weight.to(device="webgpu")

    result = torch.ops.webgpu.rms_norm(x_webgpu, weight_webgpu, eps)
    result_cpu = result.to("cpu")

    expected = rms_norm_reference(x, weight, eps)
    assert torch.allclose(result_cpu, expected, atol=1e-5, rtol=1e-5), \
        f"Max diff: {(result_cpu - expected).abs().max()}"


def test_rms_norm_large_hidden_size():
    """Test RMSNorm with large hidden size (uses parallel kernel)."""
    hidden_size = 4096
    batch_size = 4
    x = torch.randn(batch_size, hidden_size, dtype=torch.float32)
    weight = torch.randn(hidden_size, dtype=torch.float32)
    eps = 1e-6

    x_webgpu = x.to(device="webgpu")
    weight_webgpu = weight.to(device="webgpu")

    result = torch.ops.webgpu.rms_norm(x_webgpu, weight_webgpu, eps)
    result_cpu = result.to("cpu")

    expected = rms_norm_reference(x, weight, eps)
    assert torch.allclose(result_cpu, expected, atol=1e-5, rtol=1e-5), \
        f"Max diff: {(result_cpu - expected).abs().max()}"


def test_rms_norm_3d_input():
    """Test RMSNorm with 3D input (batch, seq_len, hidden)."""
    batch_size = 2
    seq_len = 32
    hidden_size = 896
    x = torch.randn(batch_size, seq_len, hidden_size, dtype=torch.float32)
    weight = torch.randn(hidden_size, dtype=torch.float32)
    eps = 1e-6

    x_webgpu = x.to(device="webgpu")
    weight_webgpu = weight.to(device="webgpu")

    result = torch.ops.webgpu.rms_norm(x_webgpu, weight_webgpu, eps)
    result_cpu = result.to("cpu")

    expected = rms_norm_reference(x, weight, eps)
    assert torch.allclose(result_cpu, expected, atol=1e-5, rtol=1e-5), \
        f"Max diff: {(result_cpu - expected).abs().max()}"


def test_rms_norm_single_batch():
    """Test RMSNorm with single batch element."""
    hidden_size = 896
    x = torch.randn(1, hidden_size, dtype=torch.float32)
    weight = torch.randn(hidden_size, dtype=torch.float32)
    eps = 1e-6

    x_webgpu = x.to(device="webgpu")
    weight_webgpu = weight.to(device="webgpu")

    result = torch.ops.webgpu.rms_norm(x_webgpu, weight_webgpu, eps)
    result_cpu = result.to("cpu")

    expected = rms_norm_reference(x, weight, eps)
    assert torch.allclose(result_cpu, expected, atol=1e-5, rtol=1e-5), \
        f"Max diff: {(result_cpu - expected).abs().max()}"


def test_rms_norm_numerical_stability():
    """Test RMSNorm with values that could cause numerical issues."""
    hidden_size = 512
    # Test with large values
    x = torch.randn(4, hidden_size, dtype=torch.float32) * 100
    weight = torch.ones(hidden_size, dtype=torch.float32)
    eps = 1e-6

    x_webgpu = x.to(device="webgpu")
    weight_webgpu = weight.to(device="webgpu")

    result = torch.ops.webgpu.rms_norm(x_webgpu, weight_webgpu, eps)
    result_cpu = result.to("cpu")

    expected = rms_norm_reference(x, weight, eps)
    # Allow slightly larger tolerance for numerical stability test
    assert torch.allclose(result_cpu, expected, atol=1e-4, rtol=1e-4), \
        f"Max diff: {(result_cpu - expected).abs().max()}"


def test_rms_norm_different_eps():
    """Test RMSNorm with different epsilon values."""
    hidden_size = 256
    x = torch.randn(4, hidden_size, dtype=torch.float32)
    weight = torch.randn(hidden_size, dtype=torch.float32)

    for eps in [1e-5, 1e-6, 1e-8]:
        x_webgpu = x.to(device="webgpu")
        weight_webgpu = weight.to(device="webgpu")

        result = torch.ops.webgpu.rms_norm(x_webgpu, weight_webgpu, eps)
        result_cpu = result.to("cpu")

        expected = rms_norm_reference(x, weight, eps)
        assert torch.allclose(result_cpu, expected, atol=1e-5, rtol=1e-5), \
            f"Failed for eps={eps}, Max diff: {(result_cpu - expected).abs().max()}"
