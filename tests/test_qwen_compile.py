"""
Test Qwen/Qwen2.5-0.5B-Instruct compilation and execution with WebGPU backend.

This test verifies that the Qwen 0.5B model can be:
1. Compiled with torch.compile(backend=webgpu_backend)
2. Run forward pass successfully
3. Produce outputs matching the non-compiled reference
"""

import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import torch_webgpu
from torch_webgpu.compiler.webgpu_compiler import webgpu_backend


MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"


@pytest.fixture(scope="module")
def model_and_tokenizer():
    """Load the Qwen model and tokenizer once for all tests."""
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float32)
    model.eval()
    return model, tokenizer


@pytest.fixture(scope="module")
def compiled_model(model_and_tokenizer):
    """Compile the model with WebGPU backend."""
    model, _ = model_and_tokenizer
    return torch.compile(model, backend=webgpu_backend, dynamic=False)


@pytest.fixture
def sample_input(model_and_tokenizer):
    """Create sample input for testing."""
    _, tokenizer = model_and_tokenizer
    prompt = "Hello, how are you?"
    inputs = tokenizer(prompt, return_tensors="pt")
    return inputs["input_ids"]


class TestQwenCompilation:
    """Tests for Qwen model compilation with WebGPU backend."""

    def test_model_compiles(self, compiled_model):
        """Test that the model compiles without errors."""
        assert compiled_model is not None

    def test_forward_pass(self, compiled_model, sample_input):
        """Test that forward pass executes successfully."""
        with torch.no_grad():
            outputs = compiled_model(sample_input)

        assert outputs is not None
        assert hasattr(outputs, "logits")
        assert outputs.logits is not None

    def test_output_shape(self, compiled_model, sample_input, model_and_tokenizer):
        """Test that output shape is correct."""
        model, _ = model_and_tokenizer
        vocab_size = model.config.vocab_size
        batch_size, seq_len = sample_input.shape

        with torch.no_grad():
            outputs = compiled_model(sample_input)

        expected_shape = (batch_size, seq_len, vocab_size)
        assert outputs.logits.shape == expected_shape, (
            f"Expected shape {expected_shape}, got {outputs.logits.shape}"
        )

    def test_output_matches_reference(self, model_and_tokenizer, compiled_model, sample_input):
        """Test that compiled model output matches non-compiled reference."""
        model, _ = model_and_tokenizer

        # Get reference output
        with torch.no_grad():
            ref_outputs = model(sample_input)
            ref_logits = ref_outputs.logits

        # Get compiled output
        with torch.no_grad():
            comp_outputs = compiled_model(sample_input)
            comp_logits = comp_outputs.logits

        # Compare outputs
        max_diff = (ref_logits - comp_logits).abs().max().item()
        mean_diff = (ref_logits - comp_logits).abs().mean().item()

        # Outputs should be very close (currently using PyTorch ops under the hood)
        assert max_diff < 1e-3, f"Max difference {max_diff} exceeds tolerance"
        assert mean_diff < 1e-4, f"Mean difference {mean_diff} exceeds tolerance"

    def test_next_token_prediction(self, model_and_tokenizer, compiled_model, sample_input):
        """Test that next token prediction matches reference."""
        model, tokenizer = model_and_tokenizer

        with torch.no_grad():
            ref_outputs = model(sample_input)
            ref_next_token = ref_outputs.logits[0, -1].argmax()

            comp_outputs = compiled_model(sample_input)
            comp_next_token = comp_outputs.logits[0, -1].argmax()

        assert ref_next_token == comp_next_token, (
            f"Next token mismatch: reference={ref_next_token.item()} "
            f"({tokenizer.decode([ref_next_token])}), "
            f"compiled={comp_next_token.item()} ({tokenizer.decode([comp_next_token])})"
        )


class TestQwenDifferentInputs:
    """Test Qwen compilation with various input configurations."""

    @pytest.mark.parametrize("prompt", [
        "Hello",
        "What is the capital of France?",
        "The quick brown fox jumps over the lazy dog.",
    ])
    def test_various_prompts(self, compiled_model, model_and_tokenizer, prompt):
        """Test with various input prompts."""
        _, tokenizer = model_and_tokenizer
        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"]

        with torch.no_grad():
            outputs = compiled_model(input_ids)

        assert outputs.logits is not None
        assert outputs.logits.shape[0] == 1  # batch size
        assert outputs.logits.shape[1] == input_ids.shape[1]  # sequence length


class TestQwenGeneration:
    """Test basic text generation with compiled Qwen model."""

    def test_greedy_generation(self, model_and_tokenizer, compiled_model):
        """Test greedy token-by-token generation."""
        model, tokenizer = model_and_tokenizer

        prompt = "The meaning of life is"
        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"]

        # Generate a few tokens using greedy decoding
        generated_ids = input_ids.clone()
        num_new_tokens = 5

        with torch.no_grad():
            for _ in range(num_new_tokens):
                outputs = compiled_model(generated_ids)
                next_token = outputs.logits[0, -1].argmax().unsqueeze(0).unsqueeze(0)
                generated_ids = torch.cat([generated_ids, next_token], dim=1)

        # Decode and verify we got valid output
        generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        assert len(generated_text) > len(prompt)
        assert generated_text.startswith(prompt)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
