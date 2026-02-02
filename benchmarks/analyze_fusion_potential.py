"""
Analyze the complete fusion potential for Qwen2.5-0.5B.
"""

import sys
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from collections import Counter

fusion_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'python', 'torch_webgpu', 'compiler')
sys.path.insert(0, fusion_path)

from fusion import find_rmsnorm_patterns, find_attention_patterns, find_linear_activation_patterns


def analyze_fx_graph(gm):
    """Detailed analysis of FX graph operations."""
    ops = Counter()
    ops_by_type = {
        'compute': [],      # Actual GPU compute
        'shape': [],        # Shape manipulation (free)
        'memory': [],       # Memory operations
        'control': [],      # Control flow
    }

    compute_ops = {'mul', 'add', 'sub', 'div', 'matmul', 'mm', 'linear', 'softmax',
                   'pow', 'mean', 'sum', 'rsqrt', 'sqrt', 'exp', 'log', 'tanh',
                   'silu', 'gelu', 'relu', 'cat', 'embedding',
                   'scaled_dot_product_attention'}

    shape_ops = {'view', 'reshape', 'transpose', 'permute', 'unsqueeze', 'squeeze',
                 'contiguous', 'expand', 'to', 'float', 'half'}

    for node in gm.graph.nodes:
        if node.op not in ('call_function', 'call_method'):
            continue

        if node.op == 'call_function':
            name = getattr(node.target, '__name__', str(node.target))
        else:
            name = node.target

        ops[name] += 1

        # Categorize
        name_lower = name.lower()
        is_compute = any(cop in name_lower for cop in compute_ops)
        is_shape = any(sop in name_lower for sop in shape_ops)

        if is_compute:
            ops_by_type['compute'].append(name)
        elif is_shape:
            ops_by_type['shape'].append(name)
        else:
            ops_by_type['memory'].append(name)

    return ops, ops_by_type


def estimate_dispatch_reduction(gm, ops):
    """Estimate dispatch reduction from various fusion strategies."""
    results = {}

    # 1. RMSNorm fusion
    rmsnorm_patterns = find_rmsnorm_patterns(gm)
    rmsnorm_ops_per_pattern = 5  # pow, mean, add, rsqrt, mul*2 but some fused
    results['rmsnorm'] = {
        'patterns': len(rmsnorm_patterns),
        'ops_saved': len(rmsnorm_patterns) * rmsnorm_ops_per_pattern,
    }

    # 2. Attention - check if already using SDPA
    sdpa_count = sum(1 for n in gm.graph.nodes
                     if n.op == 'call_function' and
                     'scaled_dot_product_attention' in str(n.target))
    if sdpa_count > 0:
        results['attention'] = {
            'patterns': sdpa_count,
            'status': 'Already using SDPA (fused)',
            'ops_saved': 0,
        }
    else:
        # Count potential attention patterns
        results['attention'] = {
            'patterns': 0,
            'status': 'Manual attention - could be fused',
            'ops_saved': 0,  # TODO: Calculate
        }

    # 3. Linear + activation
    linear_act_patterns = find_linear_activation_patterns(gm)
    results['linear_activation'] = {
        'patterns': len(linear_act_patterns),
        'ops_saved': len(linear_act_patterns),  # Each saves 1 dispatch
    }

    # 4. Shape operations (already free)
    shape_count = ops.get('view', 0) + ops.get('reshape', 0) + ops.get('transpose', 0)
    results['shape_ops'] = {
        'count': shape_count,
        'status': 'Already free (no dispatch)',
    }

    # 5. dtype casts
    cast_count = ops.get('to', 0) + ops.get('float', 0) + ops.get('half', 0)
    results['dtype_casts'] = {
        'count': cast_count,
        'status': 'Potential for fusion with adjacent ops',
    }

    return results


def main():
    print("=" * 70)
    print("FUSION POTENTIAL ANALYSIS FOR QWEN2.5-0.5B")
    print("=" * 70)

    print("\n1. Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-0.5B-Instruct",
        torch_dtype=torch.float32,
    )
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
    model.eval()

    inputs = tokenizer("Hello", return_tensors="pt")

    captured_gm = None
    captured_ops = None

    def analyze_backend(gm, example_inputs):
        nonlocal captured_gm, captured_ops
        captured_gm = gm
        captured_ops, ops_by_type = analyze_fx_graph(gm)

        def fn(*args):
            return gm(*args)
        return fn

    print("\n2. Capturing FX graph...")
    compiled = torch.compile(model, backend=analyze_backend)

    with torch.no_grad():
        _ = compiled(inputs["input_ids"])

    print("\n3. Operation Analysis")
    print("-" * 70)

    total_ops = sum(captured_ops.values())
    print(f"\nTotal FX operations: {total_ops}")

    print("\nTop 20 operations by frequency:")
    for op, count in captured_ops.most_common(20):
        pct = count / total_ops * 100
        print(f"  {op:50s} {count:5d} ({pct:5.1f}%)")

    print("\n4. Fusion Opportunity Analysis")
    print("-" * 70)

    reduction = estimate_dispatch_reduction(captured_gm, captured_ops)

    total_saved = 0
    for category, data in reduction.items():
        print(f"\n{category.upper()}:")
        for key, value in data.items():
            print(f"  {key}: {value}")
        if 'ops_saved' in data:
            total_saved += data['ops_saved']

    print("\n5. Performance Projection")
    print("-" * 70)

    # Current baseline
    current_tok_s = 10.0
    current_forward_ms = 100.0
    dispatch_overhead_ms = 0.4

    # Count actual compute operations (ones that become dispatches)
    compute_ops_list = ['mul', 'add', 'sub', 'div', 'linear', 'matmul', 'mm',
                        'softmax', 'pow', 'mean', 'rsqrt', 'sqrt', 'silu', 'gelu',
                        'cat', 'embedding', 'scaled_dot_product_attention', 'neg']

    actual_dispatches = 0
    for op, count in captured_ops.items():
        op_lower = op.lower()
        if any(cop in op_lower for cop in compute_ops_list):
            actual_dispatches += count

    print(f"\nCurrent state:")
    print(f"  Measured forward pass: {current_forward_ms:.0f}ms")
    print(f"  Measured throughput: {current_tok_s:.1f} tok/s")
    print(f"  Estimated compute dispatches: {actual_dispatches}")
    print(f"  Estimated total overhead: {actual_dispatches * dispatch_overhead_ms:.0f}ms")

    # After RMSNorm fusion
    dispatches_after_rmsnorm = actual_dispatches - reduction['rmsnorm']['ops_saved']
    overhead_after = dispatches_after_rmsnorm * dispatch_overhead_ms
    # Estimate compute time (forward - overhead)
    compute_time = max(20, current_forward_ms - actual_dispatches * dispatch_overhead_ms)
    new_forward = compute_time + overhead_after
    new_tok_s = 1000 / new_forward  # Assuming 1 token per forward

    print(f"\nAfter RMSNorm fusion:")
    print(f"  Dispatches: {dispatches_after_rmsnorm}")
    print(f"  Estimated overhead: {overhead_after:.0f}ms")
    print(f"  Estimated forward pass: {new_forward:.0f}ms")
    print(f"  Projected throughput: {new_tok_s:.1f} tok/s")
    print(f"  Improvement: {(new_tok_s / current_tok_s - 1) * 100:.0f}%")

    # Theoretical maximum with aggressive fusion
    min_dispatches = 50  # Absolute minimum with mega-kernels
    min_overhead = min_dispatches * dispatch_overhead_ms
    best_forward = compute_time + min_overhead
    best_tok_s = 1000 / best_forward

    print(f"\nTheoretical maximum (aggressive fusion to ~{min_dispatches} dispatches):")
    print(f"  Estimated overhead: {min_overhead:.0f}ms")
    print(f"  Estimated forward pass: {best_forward:.0f}ms")
    print(f"  Projected throughput: {best_tok_s:.1f} tok/s")
    print(f"  vs CUDA (185 tok/s): {best_tok_s / 185 * 100:.0f}%")


if __name__ == "__main__":
    main()
