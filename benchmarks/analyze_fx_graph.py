"""Analyze the FX graph for Qwen2.5-0.5B to count operations."""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from collections import Counter

def main():
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-0.5B-Instruct",
        torch_dtype=torch.float32,
    )
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
    model.eval()

    print("\nCapturing FX graph...")
    prompt = "The capital of France is"
    inputs = tokenizer(prompt, return_tensors="pt")

    # Use torch.compile with a counting backend
    op_counts = Counter()
    total_nodes = 0

    def counting_backend(gm: torch.fx.GraphModule, example_inputs):
        nonlocal op_counts, total_nodes

        for node in gm.graph.nodes:
            op_counts[node.op] += 1
            total_nodes += 1

            # Also count specific call_function targets
            if node.op == "call_function":
                target_name = str(node.target).split(".")[-1] if hasattr(node.target, "__name__") else str(node.target)
                op_counts[f"call_function:{target_name}"] += 1
            elif node.op == "call_method":
                op_counts[f"call_method:{node.target}"] += 1

        print(f"\nFX Graph captured with {total_nodes} nodes")

        # Return a simple function that does eager execution
        def eager_fn(*args):
            return gm(*args)
        return eager_fn

    compiled_model = torch.compile(model, backend=counting_backend)

    print("\nRunning forward pass to capture graph...")
    with torch.no_grad():
        outputs = compiled_model(inputs["input_ids"])

    print(f"\n{'='*60}")
    print(f"FX GRAPH ANALYSIS RESULTS")
    print(f"{'='*60}")
    print(f"\nTotal FX nodes: {total_nodes}")

    print(f"\nNode types:")
    for op, count in sorted(op_counts.items()):
        if not op.startswith("call_"):
            print(f"  {op}: {count}")

    print(f"\nTop 20 call_function operations:")
    call_funcs = [(k, v) for k, v in op_counts.items() if k.startswith("call_function:")]
    for op, count in sorted(call_funcs, key=lambda x: -x[1])[:20]:
        print(f"  {op}: {count}")

    print(f"\nTop 20 call_method operations:")
    call_methods = [(k, v) for k, v in op_counts.items() if k.startswith("call_method:")]
    for op, count in sorted(call_methods, key=lambda x: -x[1])[:20]:
        print(f"  {op}: {count}")

    # Estimate WebGPU dispatches
    # Most tensor operations become 1 dispatch
    # Some (like RMSNorm) decompose into multiple
    dispatch_ops = [
        "call_function:linear", "call_function:matmul", "call_function:mm", "call_function:bmm",
        "call_function:softmax", "call_function:_softmax",
        "call_function:add", "call_function:mul", "call_function:div", "call_function:sub",
        "call_function:pow", "call_function:mean", "call_function:sum",
        "call_function:rsqrt", "call_function:sqrt",
        "call_function:silu", "call_function:gelu", "call_function:relu",
        "call_function:embedding",
        "call_function:layer_norm", "call_function:rms_norm",
        "call_function:transpose", "call_function:view", "call_function:reshape",
        "call_function:scaled_dot_product_attention",
        "call_method:to", "call_method:view", "call_method:transpose", "call_method:contiguous",
        "call_method:pow", "call_method:mean", "call_method:rsqrt", "call_method:mul",
    ]

    estimated_dispatches = 0
    print(f"\n{'='*60}")
    print(f"ESTIMATED WEBGPU DISPATCHES")
    print(f"{'='*60}")

    for op in dispatch_ops:
        if op in op_counts:
            count = op_counts[op]
            # Most ops are 1 dispatch, but some shapes/transposes are free
            if "view" in op or "reshape" in op or "transpose" in op or "contiguous" in op:
                print(f"  {op}: {count} (0 dispatches - shape only)")
            elif "to" in op:
                print(f"  {op}: {count} (0 dispatches - dtype cast, fused)")
            else:
                estimated_dispatches += count
                print(f"  {op}: {count} ({count} dispatches)")

    print(f"\n{'='*60}")
    print(f"TOTAL ESTIMATED DISPATCHES: ~{estimated_dispatches}")
    print(f"{'='*60}")

    # Model info
    print(f"\nModel architecture reference:")
    print(f"  Layers: {model.config.num_hidden_layers}")
    print(f"  Hidden: {model.config.hidden_size}")
    print(f"  Heads: {model.config.num_attention_heads}")
    print(f"  Vocab: {model.config.vocab_size}")


if __name__ == "__main__":
    main()
