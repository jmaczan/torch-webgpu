"""
Test script to compile and run Meta's Llama-3.2-3B on WebGPU backend.

Llama-3.2-3B specs:
- 3.21B parameters
- GQA (Grouped Query Attention)
- 128k context length
- ~6GB VRAM in BF16, ~12GB in FP32
"""

import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaConfig, LlamaForCausalLM

from torch_webgpu.compiler.webgpu_compiler import webgpu_backend


MODEL_NAME = "meta-llama/Llama-3.2-3B"


class TestLlama3Discovery:
    """Discover which ops Llama-3.2-3B needs."""

    def test_trace_llama3_tiny(self):
        """
        Create a Llama-3.2 style model with tiny config to test architecture.
        Uses the same architecture as Llama-3.2 but with minimal dimensions.
        """
        # Llama-3.2-3B-like config but tiny
        config = LlamaConfig(
            vocab_size=1000,
            hidden_size=128,
            intermediate_size=256,
            num_hidden_layers=2,
            num_attention_heads=8,
            num_key_value_heads=4,  # GQA: 8 query heads, 4 KV heads
            max_position_embeddings=256,
            rms_norm_eps=1e-5,
            rope_theta=500000.0,  # Llama 3.2 uses higher rope_theta
            attention_bias=False,
            attention_dropout=0.0,
            mlp_bias=False,
        )

        print("Creating tiny Llama-3.2-style model...")
        model = LlamaForCausalLM(config)
        model.eval()

        print("Compiling with webgpu_backend...")
        compiled = torch.compile(model, backend=webgpu_backend, dynamic=False)

        # Test forward pass
        input_ids = torch.randint(0, config.vocab_size, (1, 16))
        print(f"Input shape: {input_ids.shape}")

        with torch.no_grad():
            outputs = compiled(input_ids)

        print(f"Output logits shape: {outputs.logits.shape}")
        assert outputs.logits.shape == (1, 16, config.vocab_size)
        print("SUCCESS: Tiny Llama-3.2-style model compiled and ran!")


class TestLlama3Compilation:
    """Tests for compiling Llama-3.2-3B with WebGPU backend."""

    @pytest.mark.skip(reason="Requires HuggingFace access to meta-llama/Llama-3.2-3B")
    def test_compile_llama3_3b(self):
        """Test compiling the full Llama-3.2-3B model."""
        print(f"Loading {MODEL_NAME}...")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.float32,  # Use FP32 for WebGPU compatibility
            device_map="cpu",
        )
        model.eval()

        print("Compiling with webgpu_backend...")
        compiled = torch.compile(model, backend=webgpu_backend, dynamic=False)

        # Test with a simple prompt
        prompt = "The meaning of life is"
        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"]

        print(f"Input: '{prompt}' -> shape {input_ids.shape}")

        with torch.no_grad():
            outputs = compiled(input_ids)

        print(f"Output logits shape: {outputs.logits.shape}")

        # Get next token prediction
        next_token_id = outputs.logits[0, -1].argmax().item()
        next_token = tokenizer.decode([next_token_id])
        print(f"Next token prediction: '{next_token}'")

        print("SUCCESS: Llama-3.2-3B compiled and ran!")


class TestLlama3Generation:
    """Tests for text generation with Llama-3.2-3B."""

    @pytest.mark.skip(reason="Requires HuggingFace access to meta-llama/Llama-3.2-3B")
    def test_generate_llama3_3b(self):
        """Test text generation with compiled Llama-3.2-3B."""
        print(f"Loading {MODEL_NAME}...")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.float32,
            device_map="cpu",
        )
        model.eval()

        print("Compiling with webgpu_backend...")
        compiled = torch.compile(model, backend=webgpu_backend, dynamic=False)

        # Generate text
        prompt = "The capital of France is"
        inputs = tokenizer(prompt, return_tensors="pt")
        generated_ids = inputs["input_ids"].clone()

        print(f"Prompt: '{prompt}'")
        print("Generating...")

        max_new_tokens = 20
        with torch.no_grad():
            for i in range(max_new_tokens):
                outputs = compiled(generated_ids)
                next_token = outputs.logits[0, -1].argmax().unsqueeze(0).unsqueeze(0)
                generated_ids = torch.cat([generated_ids, next_token], dim=1)

                # Check for EOS
                if next_token.item() == tokenizer.eos_token_id:
                    break

                if (i + 1) % 5 == 0:
                    print(f"  Generated {i + 1} tokens...")

        generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        print(f"Generated: '{generated_text}'")

        assert len(generated_text) > len(prompt)
        print("SUCCESS: Text generation completed!")


class TestLlama3OnDevice:
    """Tests for running Llama-3.2-3B on WebGPU device."""

    @pytest.mark.skip(reason="Requires HuggingFace access and WebGPU device support")
    def test_llama3_on_webgpu_device(self):
        """Test moving Llama-3.2-3B to WebGPU device directly."""
        print(f"Loading {MODEL_NAME}...")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.float32,
        )
        model.eval()

        # Move to WebGPU device
        print("Moving model to WebGPU device...")
        device = torch.device("webgpu")
        model = model.to(device)

        prompt = "Hello"
        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(device)

        print(f"Input on device: {input_ids.device}")

        with torch.no_grad():
            outputs = model(input_ids)

        logits = outputs.logits.to("cpu")
        print(f"Output shape: {logits.shape}")
        print("SUCCESS: Model ran on WebGPU device!")


if __name__ == "__main__":
    print("=" * 60)
    print("Testing Llama-3.2-3B on WebGPU Backend")
    print("=" * 60)

    # Run the tiny model test first
    test = TestLlama3Discovery()
    try:
        test.test_trace_llama3_tiny()
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "=" * 60)

    # Try full model if available
    print("\nAttempting to load full Llama-3.2-3B model...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.float32,
            device_map="cpu",
        )
        model.eval()
        print(f"Model loaded! Parameters: {sum(p.numel() for p in model.parameters()):,}")

        print("\nCompiling with webgpu_backend...")
        compiled = torch.compile(model, backend=webgpu_backend, dynamic=False)

        prompt = "The key to happiness is"
        inputs = tokenizer(prompt, return_tensors="pt")

        print(f"Running forward pass with prompt: '{prompt}'")
        with torch.no_grad():
            outputs = compiled(inputs["input_ids"])

        next_token = outputs.logits[0, -1].argmax().item()
        next_word = tokenizer.decode([next_token])
        print(f"Next token prediction: '{next_word}'")

        # Generate a few more tokens
        print("\nGenerating 10 tokens...")
        generated_ids = inputs["input_ids"].clone()
        with torch.no_grad():
            for _ in range(10):
                outputs = compiled(generated_ids)
                next_token = outputs.logits[0, -1].argmax().unsqueeze(0).unsqueeze(0)
                generated_ids = torch.cat([generated_ids, next_token], dim=1)

        generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        print(f"Generated: '{generated_text}'")

        print("\nSUCCESS! Llama-3.2-3B running on WebGPU backend!")

    except Exception as e:
        print(f"Could not load full model: {e}")
        print("(This is expected if you don't have access to the gated model)")
        import traceback
        traceback.print_exc()
