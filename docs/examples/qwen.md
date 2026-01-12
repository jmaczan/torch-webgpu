# Running Qwen LLM on WebGPU

This example shows how to compile and run Qwen/Qwen2.5-0.5B-Instruct on WebGPU.

## Prerequisites

```bash
pip install torch-webgpu transformers
```

## Full Example

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch_webgpu.compiler.webgpu_compiler import webgpu_backend

# Load model and tokenizer
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-0.5B-Instruct",
    torch_dtype=torch.float32
)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
model.eval()

# Compile for WebGPU
compiled_model = torch.compile(model, backend=webgpu_backend)

# Generate text
with torch.no_grad():
    # Tokenize input
    inputs = tokenizer("Hello, how are you?", return_tensors="pt")
    input_ids = inputs["input_ids"]
    generated_ids = input_ids.clone()

    # Generate tokens one by one
    for _ in range(10):
        outputs = compiled_model(generated_ids)
        next_token = outputs.logits[0, -1].argmax().unsqueeze(0).unsqueeze(0)
        generated_ids = torch.cat([generated_ids, next_token], dim=1)

    # Decode output
    generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    print(generated_text)
```

## Step-by-Step Explanation

### 1. Load the Model

```python
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-0.5B-Instruct",
    torch_dtype=torch.float32  # Use float32 for WebGPU
)
model.eval()  # Set to evaluation mode
```

!!! note
    Use `torch.float32` - float16 is not fully supported yet.

### 2. Compile with WebGPU Backend

```python
from torch_webgpu.compiler.webgpu_compiler import webgpu_backend

compiled_model = torch.compile(model, backend=webgpu_backend)
```

The first forward pass will trigger compilation. Subsequent calls are faster.

### 3. Generate Text

```python
with torch.no_grad():
    outputs = compiled_model(input_ids)
    next_token = outputs.logits[0, -1].argmax()
```

The model outputs logits. Take argmax of the last position to get the next token.

## Running the Tests

torch-webgpu includes tests for Qwen compilation:

```bash
pytest tests/test_qwen_compile.py -v
```

This tests:

- Model compilation
- Forward pass
- Output shape
- Output matches CPU reference
- Token generation

## Performance Notes

- **First run is slow** - Compilation happens on first forward pass
- **Subsequent runs are faster** - Compiled graph is cached
- **Memory usage** - 0.5B model needs ~2GB memory
- **Generation speed** - Currently not optimized for speed

## Troubleshooting

### Out of Memory

Try reducing batch size or use a smaller model.

### Compilation Error

If you see "Unsupported op", please [open an issue](https://github.com/jmaczan/torch-webgpu/issues) with the full error message.

### Wrong Output

Verify outputs match CPU:

```python
# CPU reference
ref_output = model(input_ids)

# WebGPU compiled
compiled_output = compiled_model(input_ids)

# Compare
diff = (ref_output.logits - compiled_output.logits).abs().max()
print(f"Max difference: {diff}")  # Should be < 1e-3
```
