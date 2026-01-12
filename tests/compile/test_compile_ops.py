"""Tests for torch.compile with webgpu backend on individual ops."""
import pytest
import torch
import torch_webgpu
from torch_webgpu import webgpu_backend


class TestCompileBasicOps:
    """Test torch.compile with basic arithmetic ops."""

    @pytest.mark.skip(reason="Compiler lowering not yet implemented for all ops")
    def test_compile_add(self):
        """Test torch.compile with add operation."""
        @torch.compile(backend=webgpu_backend)
        def fn():
            a = torch.tensor([1.0, 2.0, 3.0], device="webgpu")
            b = torch.tensor([4.0, 5.0, 6.0], device="webgpu")
            result = torch.add(a, b)
            return result.to("cpu")

        result = fn()
        expected = torch.tensor([5.0, 7.0, 9.0])
        assert torch.allclose(result, expected)

    @pytest.mark.skip(reason="Compiler lowering not yet implemented for all ops")
    def test_compile_mul(self):
        """Test torch.compile with mul operation."""
        @torch.compile(backend=webgpu_backend)
        def fn():
            a = torch.tensor([1.0, 2.0, 3.0], device="webgpu")
            b = torch.tensor([4.0, 5.0, 6.0], device="webgpu")
            result = torch.mul(a, b)
            return result.to("cpu")

        result = fn()
        expected = torch.tensor([4.0, 10.0, 18.0])
        assert torch.allclose(result, expected)

    @pytest.mark.skip(reason="Compiler lowering not yet implemented for all ops")
    def test_compile_sub(self):
        """Test torch.compile with sub operation."""
        @torch.compile(backend=webgpu_backend)
        def fn():
            a = torch.tensor([4.0, 5.0, 6.0], device="webgpu")
            b = torch.tensor([1.0, 2.0, 3.0], device="webgpu")
            result = torch.sub(a, b)
            return result.to("cpu")

        result = fn()
        expected = torch.tensor([3.0, 3.0, 3.0])
        assert torch.allclose(result, expected)


class TestCompileMatrixOps:
    """Test torch.compile with matrix ops."""

    @pytest.mark.skip(reason="Compiler lowering not yet implemented for all ops")
    def test_compile_mm(self):
        """Test torch.compile with mm operation."""
        @torch.compile(backend=webgpu_backend)
        def fn():
            a = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device="webgpu")
            b = torch.tensor([[5.0, 6.0], [7.0, 8.0]], device="webgpu")
            result = torch.mm(a, b)
            return result.to("cpu")

        result = fn()
        expected = torch.tensor([[19.0, 22.0], [43.0, 50.0]])
        assert torch.allclose(result, expected)

    @pytest.mark.skip(reason="Compiler lowering not yet implemented for all ops")
    def test_compile_matmul(self):
        """Test torch.compile with matmul operation."""
        @torch.compile(backend=webgpu_backend)
        def fn():
            a = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device="webgpu")
            b = torch.tensor([[5.0, 6.0], [7.0, 8.0]], device="webgpu")
            result = torch.matmul(a, b)
            return result.to("cpu")

        result = fn()
        expected = torch.tensor([[19.0, 22.0], [43.0, 50.0]])
        assert torch.allclose(result, expected)


class TestCompileActivations:
    """Test torch.compile with activation functions."""

    @pytest.mark.skip(reason="Compiler lowering not yet implemented for all ops")
    def test_compile_relu(self):
        """Test torch.compile with relu operation."""
        @torch.compile(backend=webgpu_backend)
        def fn():
            a = torch.tensor([-1.0, 0.0, 1.0, 2.0], device="webgpu")
            result = torch.relu(a)
            return result.to("cpu")

        result = fn()
        expected = torch.tensor([0.0, 0.0, 1.0, 2.0])
        assert torch.allclose(result, expected)

    @pytest.mark.skip(reason="Compiler lowering not yet implemented for all ops")
    def test_compile_silu(self):
        """Test torch.compile with silu operation."""
        @torch.compile(backend=webgpu_backend)
        def fn():
            a = torch.tensor([-1.0, 0.0, 1.0, 2.0], device="webgpu")
            result = torch.nn.functional.silu(a)
            return result.to("cpu")

        result = fn()
        # SiLU(x) = x * sigmoid(x)
        x = torch.tensor([-1.0, 0.0, 1.0, 2.0])
        expected = x * torch.sigmoid(x)
        assert torch.allclose(result, expected, atol=1e-5)


class TestCompileUnaryMath:
    """Test torch.compile with unary math ops."""

    @pytest.mark.skip(reason="Compiler lowering not yet implemented for all ops")
    def test_compile_cos(self):
        """Test torch.compile with cos operation."""
        @torch.compile(backend=webgpu_backend)
        def fn():
            a = torch.tensor([0.0, 3.14159/2, 3.14159], device="webgpu")
            result = torch.cos(a)
            return result.to("cpu")

        result = fn()
        expected = torch.cos(torch.tensor([0.0, 3.14159/2, 3.14159]))
        assert torch.allclose(result, expected, atol=1e-5)

    @pytest.mark.skip(reason="Compiler lowering not yet implemented for all ops")
    def test_compile_sin(self):
        """Test torch.compile with sin operation."""
        @torch.compile(backend=webgpu_backend)
        def fn():
            a = torch.tensor([0.0, 3.14159/2, 3.14159], device="webgpu")
            result = torch.sin(a)
            return result.to("cpu")

        result = fn()
        expected = torch.sin(torch.tensor([0.0, 3.14159/2, 3.14159]))
        assert torch.allclose(result, expected, atol=1e-5)

    @pytest.mark.skip(reason="Compiler lowering not yet implemented for all ops")
    def test_compile_rsqrt(self):
        """Test torch.compile with rsqrt operation."""
        @torch.compile(backend=webgpu_backend)
        def fn():
            a = torch.tensor([1.0, 4.0, 9.0, 16.0], device="webgpu")
            result = torch.rsqrt(a)
            return result.to("cpu")

        result = fn()
        expected = torch.tensor([1.0, 0.5, 1/3, 0.25])
        assert torch.allclose(result, expected, atol=1e-5)


class TestCompileReductions:
    """Test torch.compile with reduction ops."""

    @pytest.mark.skip(reason="Compiler lowering not yet implemented for all ops")
    def test_compile_sum(self):
        """Test torch.compile with sum operation."""
        @torch.compile(backend=webgpu_backend)
        def fn():
            a = torch.tensor([1.0, 2.0, 3.0, 4.0], device="webgpu")
            result = torch.sum(a)
            return result.to("cpu")

        result = fn()
        expected = torch.tensor(10.0)
        assert torch.allclose(result, expected)

    @pytest.mark.skip(reason="Compiler lowering not yet implemented for all ops")
    def test_compile_mean(self):
        """Test torch.compile with mean operation."""
        @torch.compile(backend=webgpu_backend)
        def fn():
            a = torch.tensor([1.0, 2.0, 3.0, 4.0], device="webgpu")
            result = torch.mean(a)
            return result.to("cpu")

        result = fn()
        expected = torch.tensor(2.5)
        assert torch.allclose(result, expected)


class TestCompileSoftmax:
    """Test torch.compile with softmax."""

    @pytest.mark.skip(reason="Compiler lowering not yet implemented for all ops")
    def test_compile_softmax(self):
        """Test torch.compile with softmax operation."""
        @torch.compile(backend=webgpu_backend)
        def fn():
            a = torch.tensor([[1.0, 2.0, 3.0]], device="webgpu")
            result = torch.softmax(a, dim=-1)
            return result.to("cpu")

        result = fn()
        expected = torch.softmax(torch.tensor([[1.0, 2.0, 3.0]]), dim=-1)
        assert torch.allclose(result, expected, atol=1e-5)
