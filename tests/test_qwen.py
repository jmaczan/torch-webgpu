"""
Test script to run Qwen/Qwen2.5-0.5B-Instruct on WebGPU backend.
"""

import torch
import torch_webgpu
from transformers import AutoModelForCausalLM, AutoTokenizer


def test_qwen_forward():
    """Test a simple forward pass of Qwen model on WebGPU."""

    print("Loading Qwen2.5-0.5B-Instruct model...")
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"

    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float32, trust_remote_code=True
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
    attention_mask = inputs["attention_mask"].to(device)

    print(f"Input shape: {input_ids.shape}")

    # Run forward pass
    print("Running forward pass...")
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)

    # Get logits back to CPU
    logits = outputs.logits.to("cpu")
    print(f"Output logits shape: {logits.shape}")
    print(f"Output logits (first 5): {logits[0, -1, :5]}")

    # Simple generation test
    print("\nAttempting simple generation...")
    next_token_logits = logits[0, -1, :]
    next_token = torch.argmax(next_token_logits).item()
    next_token_str = tokenizer.decode([next_token])
    print(f"Next token: {next_token} -> '{next_token_str}'")

    print("\nForward pass completed successfully!")
    return True


def test_qwen_generate():
    """Test text generation with Qwen model on WebGPU."""

    print("Loading Qwen2.5-0.5B-Instruct model...")
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float32, trust_remote_code=True
    )
    model.eval()

    # Move model to WebGPU
    print("Moving model to WebGPU device...")
    device = torch.device("webgpu")
    model = model.to(device)

    # Prepare input
    prompt = "The capital of France is"
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)

    print(f"Prompt: '{prompt}'")
    print("Generating...")

    # Generate
    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=30,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )

    # Decode
    output_ids = output_ids.to("cpu")
    generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    print(f"Generated: '{generated_text}'")

    return True


if __name__ == "__main__":
    print("=" * 60)
    print("Testing Qwen on WebGPU Backend")
    print("=" * 60)

    try:
        test_qwen_forward()
    except Exception as e:
        print(f"\nError during forward pass test: {e}")
        import traceback

        traceback.print_exc()

    print("\n" + "=" * 60)

    try:
        test_qwen_generate()
    except Exception as e:
        print(f"\nError during generation test: {e}")
        import traceback

        traceback.print_exc()
