# Contributing to torch-webgpu

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

# Build Dawn (WebGPU implementation)
./scripts/build-dawn.sh

# Install in development mode
DAWN_PREFIX="$PWD/dawn/install/Release" pip install -e .

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

See [docs/development/contributing.md](docs/development/contributing.md) for detailed instructions on adding new operations.

## Code Style

- **Python**: Follow PEP 8, use `ruff` for linting
- **C++**: Follow existing style in `csrc/`
- Use meaningful variable names
- Add comments for non-obvious logic

## Questions?

Open an issue on GitHub if you have questions about contributing.
