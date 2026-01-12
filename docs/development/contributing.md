# Contributing

Thank you for your interest in contributing to torch-webgpu!

## Before You Start

Please read the contributor policy:

- Contributions should be **well thought out**
- Code must be **covered with unit tests**
- You should **understand everything you wrote**
- Keep PRs **as concise as possible**

The maintainer builds this project after hours and needs to minimize review overhead.

## Setting Up Development Environment

```bash
# Clone the repo
git clone https://github.com/jmaczan/torch-webgpu.git
cd torch-webgpu

# Build Dawn
./scripts/build-dawn.sh

# Install in development mode
pip install -e .

# Run tests to verify setup
pytest tests/
```

## Making Changes

### 1. Create a Branch

```bash
git checkout -b feature/my-feature
```

### 2. Make Your Changes

- Follow existing code style
- Add tests for new functionality
- Update documentation if needed

### 3. Run Tests

```bash
# Python tests
pytest tests/

# C++ tests (if you modified C++ code)
./build-ctests.sh
./run-ctests.sh
```

### 4. Submit PR

- Write a clear description of what you changed and why
- Link any related issues
- Be prepared to explain your implementation choices

## Adding a New Op

To add support for a new PyTorch operation:

### 1. Add to High IR

In `python/torch_webgpu/compiler/high_ir.py`:

```python
class HighIROp(StrEnum):
    # ... existing ops ...
    MY_NEW_OP = auto()
```

Add the node class and mapping.

### 2. Add to Low IR

In `python/torch_webgpu/compiler/low_ir.py`:

```python
high_ir_op_to_low_ir_op = {
    # ... existing mappings ...
    HighIROp.MY_NEW_OP: [LowIROp.RUN_SHADER],
}
```

### 3. Add to Lowering

In `python/torch_webgpu/compiler/lowering.py`:

```python
SHADER_TO_FUNC = {
    # ... existing mappings ...
    "my_new_op": torch.my_new_op,
}
```

### 4. Add Tests

Create `tests/ops/test_my_new_op.py`:

```python
import torch
import torch_webgpu

def test_my_new_op():
    x = torch.randn(3, 4)
    expected = torch.my_new_op(x)

    # Test via compile
    @torch.compile(backend=webgpu_backend)
    def fn(x):
        return torch.my_new_op(x)

    result = fn(x)
    torch.testing.assert_close(result, expected)
```

## Code Style

- Python: Follow PEP 8
- C++: Follow existing style in `csrc/`
- Use meaningful variable names
- Add comments for non-obvious logic

## Questions?

Open an issue on GitHub if you have questions about contributing.
