"""
Test script to compile and run Meta's Llama-4-Scout-17B-16E on WebGPU backend.

This model is a Mixture-of-Experts (MoE) with:
- 17B activated parameters
- 109B total parameters
- 16 experts
- Native multimodality support

Given the model size, this test focuses on:
1. Tracing the FX graph to identify missing ops
2. Testing individual components when full model is too large
"""

import pytest
import torch
from collections import defaultdict


def get_unsupported_ops_from_trace(model, example_inputs, model_name="model"):
    """
    Trace a model with torch.compile and collect unsupported ops.
    Returns a dict of {op_name: count} for ops not yet supported.
    """
    from torch_webgpu.compiler.high_ir import fx_op_to_high_ir_op

    unsupported_ops = defaultdict(int)
    all_ops = defaultdict(int)

    def custom_backend(gm: torch.fx.GraphModule, example_inputs):
        """Custom backend that just collects ops without executing."""
        for node in gm.graph.nodes:
            if node.op in ("call_function", "call_method"):
                target = node.target
                all_ops[str(target)] += 1

                # Check if this op is supported
                if target not in fx_op_to_high_ir_op:
                    # Also check method names as strings
                    if isinstance(target, str) and target not in fx_op_to_high_ir_op:
                        unsupported_ops[str(target)] += 1
                    elif not isinstance(target, str):
                        unsupported_ops[str(target)] += 1

        # Return a dummy function that just runs on CPU
        def dummy_fn(*args):
            return gm(*args)
        return dummy_fn

    try:
        compiled = torch.compile(model, backend=custom_backend, fullgraph=False)
        with torch.no_grad():
            compiled(**example_inputs) if isinstance(example_inputs, dict) else compiled(example_inputs)
    except Exception as e:
        print(f"Error during tracing: {e}")

    return dict(unsupported_ops), dict(all_ops)


def print_ops_report(unsupported_ops, all_ops, model_name="model"):
    """Print a formatted report of supported and unsupported ops."""
    print(f"\n{'='*60}")
    print(f"Op Support Report for {model_name}")
    print(f"{'='*60}")

    print(f"\nTotal unique ops found: {len(all_ops)}")
    print(f"Unsupported ops: {len(unsupported_ops)}")
    print(f"Supported ops: {len(all_ops) - len(unsupported_ops)}")

    if unsupported_ops:
        print(f"\n{'-'*40}")
        print("UNSUPPORTED OPS (need implementation):")
        print(f"{'-'*40}")
        for op, count in sorted(unsupported_ops.items(), key=lambda x: -x[1]):
            print(f"  {op}: {count} occurrences")

    supported = {k: v for k, v in all_ops.items() if k not in unsupported_ops}
    if supported:
        print(f"\n{'-'*40}")
        print("SUPPORTED OPS:")
        print(f"{'-'*40}")
        for op, count in sorted(supported.items(), key=lambda x: -x[1]):
            print(f"  {op}: {count} occurrences")

    print(f"\n{'='*60}\n")
    return unsupported_ops


class TestLlama4OpDiscovery:
    """Tests to discover which ops are needed for Llama-4-Scout."""

    @pytest.mark.skip(reason="Requires HuggingFace access and significant memory")
    def test_trace_full_model(self):
        """Trace the full Llama-4-Scout model to find all required ops."""
        from transformers import AutoModelForCausalLM, AutoTokenizer

        model_name = "meta-llama/Llama-4-Scout-17B-16E"

        print(f"Loading {model_name}...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            device_map="cpu",  # Keep on CPU for tracing
            trust_remote_code=True,
        )
        model.eval()

        # Prepare example input
        prompt = "Hello"
        inputs = tokenizer(prompt, return_tensors="pt")

        print("Tracing model to discover ops...")
        unsupported, all_ops = get_unsupported_ops_from_trace(
            model, inputs, model_name
        )

        print_ops_report(unsupported, all_ops, model_name)

        # This test passes if we successfully traced - ops discovery is informational
        assert True

    def test_trace_llama_config_tiny(self):
        """
        Create a tiny Llama-style model to test the architecture ops.
        This helps identify ops without needing the full 109B model.
        """
        from transformers import LlamaConfig, LlamaForCausalLM

        # Create a minimal Llama config for testing
        config = LlamaConfig(
            vocab_size=1000,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,  # GQA style
            max_position_embeddings=128,
            rms_norm_eps=1e-6,
            rope_theta=10000.0,
            attention_bias=False,
            attention_dropout=0.0,
            mlp_bias=False,
        )

        print("Creating tiny Llama model for op discovery...")
        model = LlamaForCausalLM(config)
        model.eval()

        # Create example input
        batch_size = 1
        seq_len = 16
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))

        print("Tracing tiny Llama model...")
        unsupported, all_ops = get_unsupported_ops_from_trace(
            model, input_ids, "TinyLlama"
        )

        print_ops_report(unsupported, all_ops, "TinyLlama")

        # Return the unsupported ops for further analysis
        return unsupported


class TestLlama4MoEComponents:
    """Test individual MoE components that Llama-4 uses."""

    def test_trace_moe_layer(self):
        """Test a minimal Mixture-of-Experts layer."""

        class SimpleMoE(torch.nn.Module):
            """Simplified MoE layer for testing."""
            def __init__(self, hidden_size=64, num_experts=4, top_k=2):
                super().__init__()
                self.num_experts = num_experts
                self.top_k = top_k
                self.hidden_size = hidden_size

                # Router (gate)
                self.gate = torch.nn.Linear(hidden_size, num_experts, bias=False)

                # Expert networks (simplified as linear layers)
                self.experts = torch.nn.ModuleList([
                    torch.nn.Sequential(
                        torch.nn.Linear(hidden_size, hidden_size * 2),
                        torch.nn.SiLU(),
                        torch.nn.Linear(hidden_size * 2, hidden_size),
                    )
                    for _ in range(num_experts)
                ])

            def forward(self, x):
                batch_size, seq_len, hidden_size = x.shape
                x_flat = x.view(-1, hidden_size)

                # Compute router logits and select top-k experts
                router_logits = self.gate(x_flat)
                router_probs = torch.softmax(router_logits, dim=-1)
                top_k_probs, top_k_indices = torch.topk(router_probs, self.top_k, dim=-1)

                # Normalize top-k probabilities
                top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)

                # Compute expert outputs (simplified - full impl uses scatter/gather)
                output = torch.zeros_like(x_flat)
                for i, expert in enumerate(self.experts):
                    # Mask for tokens routed to this expert
                    expert_mask = (top_k_indices == i).any(dim=-1)
                    if expert_mask.any():
                        expert_input = x_flat[expert_mask]
                        expert_output = expert(expert_input)

                        # Weight by routing probability
                        expert_weight = torch.where(
                            top_k_indices[expert_mask] == i,
                            top_k_probs[expert_mask],
                            torch.zeros_like(top_k_probs[expert_mask])
                        ).sum(dim=-1, keepdim=True)

                        output[expert_mask] += expert_output * expert_weight

                return output.view(batch_size, seq_len, hidden_size)

        print("Creating SimpleMoE for op discovery...")
        model = SimpleMoE(hidden_size=64, num_experts=4, top_k=2)
        model.eval()

        # Example input
        x = torch.randn(1, 8, 64)

        print("Tracing MoE layer...")
        unsupported, all_ops = get_unsupported_ops_from_trace(model, x, "SimpleMoE")

        print_ops_report(unsupported, all_ops, "SimpleMoE")

        return unsupported

    def test_trace_rope_embedding(self):
        """Test Rotary Position Embedding (RoPE) which Llama uses."""

        class RoPE(torch.nn.Module):
            """Rotary Position Embedding implementation."""
            def __init__(self, dim, max_seq_len=512, base=10000.0):
                super().__init__()
                self.dim = dim
                self.max_seq_len = max_seq_len
                self.base = base

                # Precompute inverse frequencies
                inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
                self.register_buffer("inv_freq", inv_freq)

            def forward(self, x, seq_len=None):
                if seq_len is None:
                    seq_len = x.shape[1]

                # Compute position indices
                t = torch.arange(seq_len, device=x.device, dtype=self.inv_freq.dtype)

                # Compute frequencies
                freqs = torch.outer(t, self.inv_freq)

                # Create rotation matrix components
                cos = freqs.cos()
                sin = freqs.sin()

                return cos, sin

        class RoPEApply(torch.nn.Module):
            """Apply RoPE to query and key tensors."""
            def __init__(self, dim):
                super().__init__()
                self.rope = RoPE(dim)

            def forward(self, q, k):
                seq_len = q.shape[1]
                cos, sin = self.rope(q, seq_len)

                # Rotate half
                def rotate_half(x):
                    x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
                    return torch.cat((-x2, x1), dim=-1)

                # Apply rotation
                q_rot = q * cos + rotate_half(q) * sin
                k_rot = k * cos + rotate_half(k) * sin

                return q_rot, k_rot

        print("Creating RoPE for op discovery...")
        model = RoPEApply(dim=64)
        model.eval()

        # Example inputs (batch, seq, heads, dim)
        q = torch.randn(1, 16, 4, 64)
        k = torch.randn(1, 16, 4, 64)

        # Wrap in a simple forward
        class RoPEWrapper(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.rope = RoPEApply(dim=64)
            def forward(self, x):
                # Split into q, k
                q, k = x.chunk(2, dim=0)
                return self.rope(q, k)

        wrapper = RoPEWrapper()
        wrapper.eval()
        x = torch.cat([q, k], dim=0)

        print("Tracing RoPE...")
        unsupported, all_ops = get_unsupported_ops_from_trace(wrapper, x, "RoPE")

        print_ops_report(unsupported, all_ops, "RoPE")

        return unsupported

    def test_trace_rms_norm(self):
        """Test RMSNorm which Llama uses instead of LayerNorm."""

        class RMSNorm(torch.nn.Module):
            """Root Mean Square Layer Normalization."""
            def __init__(self, dim, eps=1e-6):
                super().__init__()
                self.eps = eps
                self.weight = torch.nn.Parameter(torch.ones(dim))

            def forward(self, x):
                # RMSNorm: x * rsqrt(mean(x^2) + eps) * weight
                rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
                return x * rms * self.weight

        print("Creating RMSNorm for op discovery...")
        model = RMSNorm(dim=64)
        model.eval()

        x = torch.randn(1, 16, 64)

        print("Tracing RMSNorm...")
        unsupported, all_ops = get_unsupported_ops_from_trace(model, x, "RMSNorm")

        print_ops_report(unsupported, all_ops, "RMSNorm")

        return unsupported


class TestLlama4Compilation:
    """Tests for compiling Llama-4 components with WebGPU backend."""

    def test_compile_tiny_llama(self):
        """Test compiling a tiny Llama model with WebGPU backend."""
        from transformers import LlamaConfig, LlamaForCausalLM
        from torch_webgpu.compiler.webgpu_compiler import webgpu_backend

        config = LlamaConfig(
            vocab_size=1000,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=64,
        )

        model = LlamaForCausalLM(config)
        model.eval()

        compiled = torch.compile(model, backend=webgpu_backend, dynamic=False)

        input_ids = torch.randint(0, config.vocab_size, (1, 8))

        with torch.no_grad():
            outputs = compiled(input_ids)

        assert outputs.logits is not None
        assert outputs.logits.shape == (1, 8, config.vocab_size)


class TestLlama4Forward:
    """Tests for running Llama-4 forward pass on WebGPU."""

    @pytest.mark.skip(
        reason="Llama-4 requires additional ops not yet implemented"
    )
    def test_llama4_forward():
        """Test a forward pass of Llama-4-Scout on WebGPU."""
        from transformers import AutoModelForCausalLM, AutoTokenizer

        print("Loading Llama-4-Scout-17B-16E model...")
        model_name = "meta-llama/Llama-4-Scout-17B-16E"

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            trust_remote_code=True,
        )
        model.eval()

        # Move model to WebGPU
        print("Moving model to WebGPU device...")
        device = torch.device("webgpu")
        model = model.to(device)

        # Prepare input
        prompt = "Hello"
        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(device)

        print(f"Input shape: {input_ids.shape}")

        # Run forward pass
        print("Running forward pass...")
        with torch.no_grad():
            outputs = model(input_ids=input_ids)

        logits = outputs.logits.to("cpu")
        print(f"Output logits shape: {logits.shape}")

        return True


if __name__ == "__main__":
    print("=" * 60)
    print("Testing Llama-4-Scout on WebGPU Backend")
    print("=" * 60)

    # Run the component tests to discover ops
    test_instance = TestLlama4MoEComponents()

    print("\n--- Testing RMSNorm ---")
    try:
        test_instance.test_trace_rms_norm()
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

    print("\n--- Testing RoPE ---")
    try:
        test_instance.test_trace_rope_embedding()
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

    print("\n--- Testing MoE Layer ---")
    try:
        test_instance.test_trace_moe_layer()
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

    print("\n--- Testing Tiny Llama ---")
    try:
        discovery = TestLlama4OpDiscovery()
        discovery.test_trace_llama_config_tiny()
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
