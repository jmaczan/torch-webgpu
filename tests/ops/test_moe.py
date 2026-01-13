"""
Tests for MoE (Mixture of Experts) operations on WebGPU.
These ops are essential for running MoE models like Llama-4-Scout.
"""

import torch
import torch_webgpu  # noqa: F401
import pytest


class TestTopk:
    """Tests for torch.topk operation."""

    def test_topk_basic(self):
        cpu_input = torch.tensor([3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0])
        webgpu_input = cpu_input.to("webgpu")

        values, indices = torch.topk(webgpu_input, 3)
        expected_values, expected_indices = torch.topk(cpu_input, 3)

        assert torch.allclose(values.to("cpu"), expected_values)
        assert torch.equal(indices.to("cpu"), expected_indices)

    def test_topk_with_dim(self):
        cpu_input = torch.tensor([[3.0, 1.0, 4.0], [1.0, 5.0, 9.0]])
        webgpu_input = cpu_input.to("webgpu")

        values, indices = torch.topk(webgpu_input, 2, dim=1)
        expected_values, expected_indices = torch.topk(cpu_input, 2, dim=1)

        assert torch.allclose(values.to("cpu"), expected_values)
        assert torch.equal(indices.to("cpu"), expected_indices)

    def test_topk_smallest(self):
        cpu_input = torch.tensor([3.0, 1.0, 4.0, 1.0, 5.0, 9.0])
        webgpu_input = cpu_input.to("webgpu")

        values, indices = torch.topk(webgpu_input, 2, largest=False)
        expected_values, expected_indices = torch.topk(cpu_input, 2, largest=False)

        assert torch.allclose(values.to("cpu"), expected_values)
        assert torch.equal(indices.to("cpu"), expected_indices)

    def test_topk_sorted(self):
        cpu_input = torch.randn(10)
        webgpu_input = cpu_input.to("webgpu")

        values, indices = torch.topk(webgpu_input, 5, sorted=True)
        expected_values, expected_indices = torch.topk(cpu_input, 5, sorted=True)

        assert torch.allclose(values.to("cpu"), expected_values)
        assert torch.equal(indices.to("cpu"), expected_indices)


class TestScatter:
    """Tests for torch.scatter operation."""

    def test_scatter_src(self):
        cpu_self = torch.zeros(3, 5)
        cpu_index = torch.tensor([[0, 1, 2, 0, 0], [2, 0, 0, 1, 2]], dtype=torch.long)
        cpu_src = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0], [6.0, 7.0, 8.0, 9.0, 10.0]])

        webgpu_self = cpu_self.to("webgpu")
        webgpu_index = cpu_index.to("webgpu")
        webgpu_src = cpu_src.to("webgpu")

        result = webgpu_self.scatter(0, webgpu_index, webgpu_src)
        expected = cpu_self.scatter(0, cpu_index, cpu_src)

        assert torch.allclose(result.to("cpu"), expected)

    def test_scatter_value(self):
        cpu_self = torch.zeros(3, 5)
        cpu_index = torch.tensor([[0, 1, 2, 0, 0], [2, 0, 0, 1, 2]], dtype=torch.long)

        webgpu_self = cpu_self.to("webgpu")
        webgpu_index = cpu_index.to("webgpu")

        result = webgpu_self.scatter(0, webgpu_index, 1.0)
        expected = cpu_self.scatter(0, cpu_index, 1.0)

        assert torch.allclose(result.to("cpu"), expected)


class TestScatterAdd:
    """Tests for torch.scatter_add operation."""

    def test_scatter_add_basic(self):
        cpu_self = torch.zeros(3, 5)
        cpu_index = torch.tensor([[0, 1, 2, 0, 0], [2, 0, 0, 1, 2]], dtype=torch.long)
        cpu_src = torch.ones(2, 5)

        webgpu_self = cpu_self.to("webgpu")
        webgpu_index = cpu_index.to("webgpu")
        webgpu_src = cpu_src.to("webgpu")

        result = webgpu_self.scatter_add(0, webgpu_index, webgpu_src)
        expected = cpu_self.scatter_add(0, cpu_index, cpu_src)

        assert torch.allclose(result.to("cpu"), expected)

    def test_scatter_add_accumulate(self):
        """Test that scatter_add properly accumulates when multiple indices collide."""
        cpu_self = torch.zeros(5)
        cpu_index = torch.tensor([0, 0, 0, 1, 1], dtype=torch.long)
        cpu_src = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])

        webgpu_self = cpu_self.to("webgpu")
        webgpu_index = cpu_index.to("webgpu")
        webgpu_src = cpu_src.to("webgpu")

        result = webgpu_self.scatter_add(0, webgpu_index, webgpu_src)
        expected = cpu_self.scatter_add(0, cpu_index, cpu_src)

        # Index 0 should have 1+2+3=6, index 1 should have 4+5=9
        assert torch.allclose(result.to("cpu"), expected)
        assert result.to("cpu")[0].item() == 6.0
        assert result.to("cpu")[1].item() == 9.0


class TestAny:
    """Tests for torch.any operation."""

    def test_any_true(self):
        cpu_input = torch.tensor([False, False, True, False])
        webgpu_input = cpu_input.to("webgpu")

        result = torch.any(webgpu_input)
        expected = torch.any(cpu_input)

        assert torch.equal(result.to("cpu"), expected)

    def test_any_false(self):
        cpu_input = torch.tensor([False, False, False, False])
        webgpu_input = cpu_input.to("webgpu")

        result = torch.any(webgpu_input)
        expected = torch.any(cpu_input)

        assert torch.equal(result.to("cpu"), expected)

    def test_any_dim(self):
        cpu_input = torch.tensor([[False, True, False], [False, False, False]])
        webgpu_input = cpu_input.to("webgpu")

        result = torch.any(webgpu_input, dim=1)
        expected = torch.any(cpu_input, dim=1)

        assert torch.equal(result.to("cpu"), expected)


class TestAll:
    """Tests for torch.all operation."""

    def test_all_true(self):
        cpu_input = torch.tensor([True, True, True, True])
        webgpu_input = cpu_input.to("webgpu")

        result = torch.all(webgpu_input)
        expected = torch.all(cpu_input)

        assert torch.equal(result.to("cpu"), expected)

    def test_all_false(self):
        cpu_input = torch.tensor([True, True, False, True])
        webgpu_input = cpu_input.to("webgpu")

        result = torch.all(webgpu_input)
        expected = torch.all(cpu_input)

        assert torch.equal(result.to("cpu"), expected)

    def test_all_dim(self):
        cpu_input = torch.tensor([[True, True, True], [True, False, True]])
        webgpu_input = cpu_input.to("webgpu")

        result = torch.all(webgpu_input, dim=1)
        expected = torch.all(cpu_input, dim=1)

        assert torch.equal(result.to("cpu"), expected)


class TestNonzero:
    """Tests for torch.nonzero operation."""

    def test_nonzero_1d(self):
        cpu_input = torch.tensor([0.0, 1.0, 0.0, 2.0, 0.0])
        webgpu_input = cpu_input.to("webgpu")

        result = torch.nonzero(webgpu_input)
        expected = torch.nonzero(cpu_input)

        assert torch.equal(result.to("cpu"), expected)

    def test_nonzero_2d(self):
        cpu_input = torch.tensor([[0.0, 1.0, 0.0], [2.0, 0.0, 3.0]])
        webgpu_input = cpu_input.to("webgpu")

        result = torch.nonzero(webgpu_input)
        expected = torch.nonzero(cpu_input)

        assert torch.equal(result.to("cpu"), expected)


class TestMaskedSelect:
    """Tests for torch.masked_select operation."""

    def test_masked_select_1d(self):
        cpu_input = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        cpu_mask = torch.tensor([True, False, True, False, True])

        webgpu_input = cpu_input.to("webgpu")
        webgpu_mask = cpu_mask.to("webgpu")

        result = torch.masked_select(webgpu_input, webgpu_mask)
        expected = torch.masked_select(cpu_input, cpu_mask)

        assert torch.allclose(result.to("cpu"), expected)

    def test_masked_select_2d(self):
        cpu_input = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        cpu_mask = torch.tensor([[True, False, True], [False, True, False]])

        webgpu_input = cpu_input.to("webgpu")
        webgpu_mask = cpu_mask.to("webgpu")

        result = torch.masked_select(webgpu_input, webgpu_mask)
        expected = torch.masked_select(cpu_input, cpu_mask)

        assert torch.allclose(result.to("cpu"), expected)


class TestMoERouting:
    """Integration tests simulating MoE routing patterns."""

    def test_moe_routing_simulation(self):
        """Simulate basic MoE routing using topk and scatter."""
        # Router logits for 4 tokens across 4 experts
        cpu_logits = torch.tensor([
            [0.1, 0.9, 0.3, 0.2],  # Token 0: expert 1 best
            [0.8, 0.1, 0.2, 0.3],  # Token 1: expert 0 best
            [0.2, 0.3, 0.1, 0.9],  # Token 2: expert 3 best
            [0.4, 0.1, 0.8, 0.2],  # Token 3: expert 2 best
        ])
        webgpu_logits = cpu_logits.to("webgpu")

        # Select top-2 experts per token
        values, indices = torch.topk(webgpu_logits, 2, dim=1)

        cpu_values = values.to("cpu")
        cpu_indices = indices.to("cpu")

        # Verify shapes
        assert cpu_values.shape == (4, 2)
        assert cpu_indices.shape == (4, 2)

        # Verify correctness against CPU
        expected_values, expected_indices = torch.topk(cpu_logits, 2, dim=1)
        assert torch.allclose(cpu_values, expected_values)
        assert torch.equal(cpu_indices, expected_indices)

    def test_moe_expert_mask(self):
        """Test creating expert masks using topk and any."""
        num_tokens = 8
        num_experts = 4
        top_k = 2

        # Random router scores
        cpu_scores = torch.randn(num_tokens, num_experts)
        webgpu_scores = cpu_scores.to("webgpu")

        # Get top-k expert indices
        _, indices = torch.topk(webgpu_scores, top_k, dim=1)

        # Verify indices are valid
        cpu_indices = indices.to("cpu")
        assert cpu_indices.shape == (num_tokens, top_k)
        assert (cpu_indices >= 0).all()
        assert (cpu_indices < num_experts).all()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
