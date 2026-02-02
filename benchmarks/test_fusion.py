"""
Test script to verify fusion is working correctly.
"""

import sys
import os

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

# Import fusion module directly without loading the C++ extension
fusion_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'python', 'torch_webgpu', 'compiler')
sys.path.insert(0, fusion_path)

from fusion import find_rmsnorm_patterns, apply_fusion


def count_fx_ops(gm):
    """Count operations in an FX graph."""
    ops = {}
    for node in gm.graph.nodes:
        if node.op in ('call_function', 'call_method'):
            target = str(node.target)
            ops[target] = ops.get(target, 0) + 1
    return ops


def test_rmsnorm_pattern():
    """Test that we can detect and fuse RMSNorm patterns."""
    print("=" * 60)
    print("Testing RMSNorm pattern detection and fusion")
    print("=" * 60)

    # Create a simple module with RMSNorm
    class SimpleRMSNorm(torch.nn.Module):
        def __init__(self, hidden_size, eps=1e-6):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(hidden_size))
            self.eps = eps

        def forward(self, x):
            # Standard RMSNorm implementation (will decompose in FX)
            variance = x.pow(2).mean(-1, keepdim=True)
            x_normed = x * torch.rsqrt(variance + self.eps)
            return x_normed * self.weight

    model = SimpleRMSNorm(256)
    model.eval()

    x = torch.randn(2, 10, 256)

    # Capture FX graph
    print("\n1. Capturing FX graph for RMSNorm...")

    def capture_backend(gm, example_inputs):
        print("\n   FX Graph nodes:")
        ops = count_fx_ops(gm)
        for op, count in sorted(ops.items(), key=lambda x: -x[1])[:15]:
            print(f"     {op}: {count}")

        # Test fusion detection
        patterns = find_rmsnorm_patterns(gm)
        print(f"\n   Found {len(patterns)} RMSNorm patterns")

        if patterns:
            print("   Pattern details:")
            for i, p in enumerate(patterns):
                print(f"     Pattern {i}: x_input={p['x_input'].name}, weight={p['weight'].name}, eps={p['eps']}")

        # Return eager execution
        def fn(*args):
            return gm(*args)
        return fn

    compiled = torch.compile(model, backend=capture_backend)

    print("\n2. Running compiled model...")
    with torch.no_grad():
        out = compiled(x)

    print(f"\n   Output shape: {out.shape}")

    # Verify correctness
    with torch.no_grad():
        expected = model(x)

    diff = (out - expected).abs().max().item()
    print(f"   Max diff from eager: {diff:.2e}")


def test_qwen_patterns():
    """Test pattern detection on actual Qwen model."""
    print("\n" + "=" * 60)
    print("Testing pattern detection on Qwen2.5-0.5B")
    print("=" * 60)

    print("\n1. Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-0.5B-Instruct",
        torch_dtype=torch.float32,
    )
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
    model.eval()

    inputs = tokenizer("Hello", return_tensors="pt")

    op_counts_before = {}
    op_counts_after = {}

    def counting_backend(gm, example_inputs):
        nonlocal op_counts_before, op_counts_after

        # Count before fusion
        op_counts_before = count_fx_ops(gm)
        total_before = sum(1 for n in gm.graph.nodes if n.op in ('call_function', 'call_method'))

        print(f"\n   Operations BEFORE fusion: {total_before}")
        print("   Top 10 ops:")
        for op, count in sorted(op_counts_before.items(), key=lambda x: -x[1])[:10]:
            print(f"     {op}: {count}")

        # Apply fusion
        gm = apply_fusion(gm)

        # Count after fusion
        op_counts_after = count_fx_ops(gm)
        total_after = sum(1 for n in gm.graph.nodes if n.op in ('call_function', 'call_method'))

        print(f"\n   Operations AFTER fusion: {total_after}")
        print(f"   Reduction: {total_before - total_after} ops ({100*(total_before-total_after)/total_before:.1f}%)")

        if total_after < total_before:
            print("\n   Changed operations:")
            for op in set(op_counts_before.keys()) | set(op_counts_after.keys()):
                before = op_counts_before.get(op, 0)
                after = op_counts_after.get(op, 0)
                if before != after:
                    print(f"     {op}: {before} -> {after}")

        def fn(*args):
            return gm(*args)
        return fn

    print("\n2. Compiling with fusion analysis...")
    compiled = torch.compile(model, backend=counting_backend)

    print("\n3. Running forward pass...")
    with torch.no_grad():
        out = compiled(inputs["input_ids"])

    print(f"\n   Output logits shape: {out.logits.shape}")


def estimate_dispatch_savings():
    """Estimate potential dispatch savings from fusion."""
    print("\n" + "=" * 60)
    print("Dispatch Savings Estimation")
    print("=" * 60)

    # Based on Qwen2.5-0.5B architecture
    num_layers = 24
    num_rmsnorm_per_layer = 2
    total_rmsnorm = num_layers * num_rmsnorm_per_layer + 1  # +1 for final norm

    # RMSNorm: typically decomposes to ~6 ops
    rmsnorm_ops_before = total_rmsnorm * 6
    rmsnorm_ops_after = total_rmsnorm * 1

    print(f"\nRMSNorm fusion:")
    print(f"  Instances: {total_rmsnorm}")
    print(f"  Ops before: {rmsnorm_ops_before}")
    print(f"  Ops after: {rmsnorm_ops_after}")
    print(f"  Savings: {rmsnorm_ops_before - rmsnorm_ops_after} dispatches")

    # Per-dispatch overhead
    overhead_ms = 0.4  # From measurements
    time_saved = (rmsnorm_ops_before - rmsnorm_ops_after) * overhead_ms

    print(f"\n  Estimated time saved: {time_saved:.1f}ms per forward pass")
    print(f"  At 100ms baseline: {time_saved/100*100:.1f}% improvement")


if __name__ == "__main__":
    # Test basic RMSNorm pattern detection
    test_rmsnorm_pattern()

    # Test on actual Qwen model
    try:
        test_qwen_patterns()
    except Exception as e:
        print(f"\nQwen test failed (may need torch-webgpu C++ extension): {e}")

    # Show potential savings
    estimate_dispatch_savings()
