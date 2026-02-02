"""Count the number of shader dispatches in a Qwen2.5-0.5B forward pass."""

import torch
import sys
import os

# Add parent to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformers import AutoModelForCausalLM, AutoTokenizer

# Patch the compiler to count operations
dispatch_count = 0
high_ir_count = 0

def count_dispatches():
    """Count dispatches by examining the compiled IR."""
    global dispatch_count, high_ir_count

    from torch_webgpu.compiler.low_ir import LowIROp
    from torch_webgpu.compiler import webgpu_compiler

    # Store original lowering function
    original_lowering = webgpu_compiler.lowering

    def counting_lowering(low_ir, placeholder_names):
        global dispatch_count
        # Count RUN_SHADER operations
        run_shader_count = sum(1 for node in low_ir if node.ir_op == LowIROp.RUN_SHADER)
        dispatch_count += run_shader_count
        print(f"  Low IR RUN_SHADER count: {run_shader_count}")
        return original_lowering(low_ir, placeholder_names)

    webgpu_compiler.lowering = counting_lowering

    # Also count High IR ops
    original_fx_to_high_ir = webgpu_compiler.fx_to_high_ir

    def counting_fx_to_high_ir(gm):
        global high_ir_count
        high_ir = original_fx_to_high_ir(gm)
        high_ir_count += len(high_ir)
        print(f"  High IR node count: {len(high_ir)}")
        return high_ir

    webgpu_compiler.fx_to_high_ir = counting_fx_to_high_ir


def main():
    global dispatch_count, high_ir_count

    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-0.5B-Instruct",
        torch_dtype=torch.float32,
    )
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
    model.eval()

    print("\nSetting up dispatch counting...")
    count_dispatches()

    print("\nCompiling model with torch-webgpu backend...")
    from torch_webgpu import webgpu_backend

    compiled_model = torch.compile(model, backend=webgpu_backend)

    print("\nRunning forward pass to trigger compilation...")
    prompt = "The capital of France is"
    inputs = tokenizer(prompt, return_tensors="pt")

    dispatch_count = 0
    high_ir_count = 0

    with torch.no_grad():
        # Single forward pass
        outputs = compiled_model(inputs["input_ids"])

    print(f"\n{'='*50}")
    print(f"RESULTS FOR SINGLE FORWARD PASS:")
    print(f"{'='*50}")
    print(f"High IR nodes (total):     {high_ir_count}")
    print(f"Low IR RUN_SHADER (dispatches): {dispatch_count}")
    print(f"{'='*50}")

    # Also count model layers for reference
    num_layers = model.config.num_hidden_layers
    hidden_size = model.config.hidden_size
    vocab_size = model.config.vocab_size

    print(f"\nModel architecture:")
    print(f"  Layers: {num_layers}")
    print(f"  Hidden size: {hidden_size}")
    print(f"  Vocabulary: {vocab_size}")
    print(f"\nExpected operations per layer (approx):")
    print(f"  Attention: Q, K, V proj (3) + scores matmul (1) + softmax (1) + output proj (1) = ~6")
    print(f"  MLP: gate_proj (1) + up_proj (1) + silu (1) + mul (1) + down_proj (1) = ~5")
    print(f"  Norms: 2 RMSNorm × ~6 ops each = ~12")
    print(f"  Total per layer estimate: ~23")
    print(f"  24 layers × 23 = ~552 operations (High IR)")
    print(f"  Plus embedding, final norm, lm_head")


if __name__ == "__main__":
    main()
