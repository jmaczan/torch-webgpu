# Running Tests

torch-webgpu has both Python and C++ tests.

## Python Tests

### Run All Tests

```bash
pytest tests/
```

### Run Specific Test File

```bash
pytest tests/ops/test_cos.py
```

### Run with Verbose Output

```bash
pytest tests/ -v
```

### Run with Print Statements

```bash
pytest tests/ -s
```

### Run Qwen Compilation Tests

```bash
pytest tests/test_qwen_compile.py -v -s
```

## C++ Tests

### Build Tests

```bash
# First, rebuild the main library
./build.sh

# Build C++ tests
./build-ctests.sh
```

### Run Tests

```bash
./run-ctests.sh
```

### Individual Test Files

C++ tests are in `ctests/`:

- `test_add.cpp` - Addition operations
- `test_mm.cpp` - Matrix multiplication
- `test_cos.cpp` - Cosine function
- etc.

## Writing Tests

### Python Test Example

```python
# tests/ops/test_my_op.py
import pytest
import torch
import torch_webgpu
from torch_webgpu.compiler.webgpu_compiler import webgpu_backend


def test_my_op_basic():
    """Test basic functionality."""
    x = torch.randn(3, 4)
    expected = torch.my_op(x)

    @torch.compile(backend=webgpu_backend)
    def fn(x):
        return torch.my_op(x)

    result = fn(x)
    torch.testing.assert_close(result, expected, rtol=1e-3, atol=1e-3)


def test_my_op_shapes():
    """Test with different shapes."""
    for shape in [(1,), (10,), (3, 4), (2, 3, 4)]:
        x = torch.randn(shape)
        expected = torch.my_op(x)

        @torch.compile(backend=webgpu_backend)
        def fn(x):
            return torch.my_op(x)

        result = fn(x)
        torch.testing.assert_close(result, expected, rtol=1e-3, atol=1e-3)
```

### C++ Test Example

```cpp
// ctests/test_my_op.cpp
#include <torch/torch.h>
#include <cassert>
#include <iostream>

int main() {
    // Create input
    auto x = torch::randn({3, 4});
    auto x_webgpu = x.to("webgpu");

    // Run operation
    auto result = torch::my_op(x_webgpu);

    // Move back and compare
    auto result_cpu = result.to("cpu");
    auto expected = torch::my_op(x);

    auto diff = (result_cpu - expected).abs().max().item<float>();
    assert(diff < 1e-3);

    std::cout << "test_my_op passed!" << std::endl;
    return 0;
}
```

## Test Coverage

Key test areas:

| Area | Test Files |
|------|------------|
| Basic ops | `tests/ops/test_*.py` |
| Compilation | `tests/compile/test_ir.py` |
| Qwen model | `tests/test_qwen_compile.py` |
| C++ ops | `ctests/test_*.cpp` |

## Debugging Test Failures

### Enable Debug Output

In `python/torch_webgpu/compiler/lowering.py`:

```python
DEBUG_LOWERING = True
```

This prints shape information for each operation.

### Compare with CPU

```python
# Get CPU reference
cpu_result = some_op(cpu_tensor)

# Get WebGPU result
webgpu_result = some_op(webgpu_tensor).to("cpu")

# Print difference
print(f"Max diff: {(cpu_result - webgpu_result).abs().max()}")
print(f"Mean diff: {(cpu_result - webgpu_result).abs().mean()}")
```
