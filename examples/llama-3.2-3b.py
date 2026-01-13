"""
Interactive CLI chat with Meta's Llama-3.2-3B running on WebGPU backend.

Llama-3.2-3B specs:
- 3.21B parameters
- GQA (Grouped Query Attention)
- 128k context length
- ~6GB VRAM in BF16, ~12GB in FP32

Usage:
    python examples/llama-3.2-3b.py

Requirements:
    - Access to meta-llama/Llama-3.2-3B on HuggingFace (gated model)
    - Run: huggingface-cli login
"""

import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from torch_webgpu.compiler.webgpu_compiler import webgpu_backend


MODEL_NAME = "meta-llama/Llama-3.2-3B"
MAX_NEW_TOKENS = 256
TEMPERATURE = 0.7


def print_banner():
    print("\n" + "=" * 60)
    print("  Llama 3.2 3B - WebGPU Backend")
    print("=" * 60)
    print("  Type your message and press Enter to chat.")
    print("  Commands:")
    print("    /clear  - Clear conversation history")
    print("    /quit   - Exit the chat")
    print("=" * 60 + "\n")


def generate_response(
    model, tokenizer, prompt: str, max_tokens: int = MAX_NEW_TOKENS
) -> str:
    """Generate a response token by token with streaming output."""
    inputs = tokenizer(prompt, return_tensors="pt")
    generated_ids = inputs["input_ids"].clone()

    # Track what we've already printed
    prompt_length = len(prompt)

    print("\nLlama: ", end="", flush=True)

    generated_text = ""
    with torch.no_grad():
        for i in range(max_tokens):
            outputs = model(generated_ids)

            # Get next token logits
            next_token_logits = outputs.logits[0, -1, :]

            # Apply temperature
            if TEMPERATURE > 0:
                next_token_logits = next_token_logits / TEMPERATURE

            # Sample or greedy decode
            if TEMPERATURE > 0:
                probs = torch.softmax(next_token_logits, dim=-1)
                next_token_id = torch.multinomial(probs, num_samples=1)
            else:
                next_token_id = next_token_logits.argmax().unsqueeze(0)

            # Append to generated sequence
            generated_ids = torch.cat(
                [generated_ids, next_token_id.unsqueeze(0)], dim=1
            )

            # Decode and print the new token
            new_token = tokenizer.decode(next_token_id, skip_special_tokens=True)
            print(new_token, end="", flush=True)
            generated_text += new_token

            # Check for EOS token
            if next_token_id.item() == tokenizer.eos_token_id:
                break

            # Also stop on common end patterns
            if generated_text.endswith("<|eot_id|>"):
                generated_text = generated_text[:-10]  # Remove the tag
                break

    print()  # New line after response
    return generated_text.strip()


def format_chat_prompt(tokenizer, messages: list) -> str:
    """Format messages into Llama 3.2 chat format."""
    # Try to use the tokenizer's chat template if available
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            return tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            pass

    # Fallback: Manual Llama 3 format
    prompt = "<|begin_of_text|>"
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        prompt += f"<|start_header_id|>{role}<|end_header_id|>\n\n{content}<|eot_id|>"
    prompt += "<|start_header_id|>assistant<|end_header_id|>\n\n"
    return prompt


def main():
    print_banner()

    print(f"Loading {MODEL_NAME}...")
    print("(This may take a minute on first run)\n")

    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.float32,
            device_map="cpu",
        )
        model.eval()
    except Exception as e:
        print(f"Error loading model: {e}")
        print("\nMake sure you have access to the gated model:")
        print(
            "  1. Accept the license at https://huggingface.co/meta-llama/Llama-3.2-3B"
        )
        print("  2. Run: huggingface-cli login")
        sys.exit(1)

    print("Compiling model with WebGPU backend...")
    compiled_model = torch.compile(model, backend=webgpu_backend, dynamic=False)

    # Warm up the model with a simple prompt
    print("Warming up...")
    warmup_messages = [{"role": "user", "content": "Hi"}]
    warmup_prompt = format_chat_prompt(tokenizer, warmup_messages)
    with torch.no_grad():
        warmup_inputs = tokenizer(warmup_prompt, return_tensors="pt")
        _ = compiled_model(warmup_inputs["input_ids"])

    print("\nReady! Start chatting.\n")

    # Conversation history
    messages = []
    system_prompt = "You are a helpful, harmless, and honest AI assistant."
    messages.append({"role": "system", "content": system_prompt})

    while True:
        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n\nGoodbye!")
            break

        if not user_input:
            continue

        # Handle commands
        if user_input.lower() == "/quit" or user_input.lower() == "/exit":
            print("\nGoodbye!")
            break

        if user_input.lower() == "/clear":
            messages = [{"role": "system", "content": system_prompt}]
            print("\n[Conversation cleared]\n")
            continue

        # Add user message to history
        messages.append({"role": "user", "content": user_input})

        # Format the full conversation
        prompt = format_chat_prompt(tokenizer, messages)

        # Generate response
        try:
            response = generate_response(compiled_model, tokenizer, prompt)

            # Add assistant response to history
            messages.append({"role": "assistant", "content": response})

        except Exception as e:
            print(f"\nError generating response: {e}")
            # Remove the failed user message
            messages.pop()

        print()  # Extra spacing


if __name__ == "__main__":
    main()
