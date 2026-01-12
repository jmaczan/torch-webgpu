# Supported Operations

torch-webgpu supports many PyTorch operations. This page lists what's available.

## Tensor Creation

| Operation | Status | Notes |
|-----------|--------|-------|
| `torch.tensor()` | ✅ | |
| `torch.zeros()` | ✅ | |
| `torch.ones()` | ✅ | |
| `torch.randn()` | ✅ | |
| `torch.arange()` | ✅ | |
| `torch.full()` | ✅ | |
| `torch.empty()` | ✅ | |

## Arithmetic Operations

| Operation | Status | Notes |
|-----------|--------|-------|
| `+` / `torch.add()` | ✅ | |
| `-` / `torch.sub()` | ✅ | |
| `*` / `torch.mul()` | ✅ | |
| `/` / `torch.div()` | ✅ | |
| `-x` / `torch.neg()` | ✅ | |
| `torch.pow()` | ✅ | |

## Matrix Operations

| Operation | Status | Notes |
|-----------|--------|-------|
| `@` / `torch.matmul()` | ✅ | |
| `torch.mm()` | ✅ | |
| `F.linear()` | ✅ | |

## Activation Functions

| Operation | Status | Notes |
|-----------|--------|-------|
| `torch.relu()` | ✅ | |
| `F.silu()` | ✅ | |
| `F.gelu()` | ✅ | |
| `torch.tanh()` | ✅ | |
| `F.softmax()` | ✅ | |
| `torch.sigmoid()` | 🚧 | Coming soon |

## Math Functions

| Operation | Status | Notes |
|-----------|--------|-------|
| `torch.exp()` | ✅ | |
| `torch.log()` | 🚧 | |
| `torch.sqrt()` | ✅ | |
| `torch.rsqrt()` | ✅ | |
| `torch.cos()` | ✅ | |
| `torch.sin()` | ✅ | |
| `torch.abs()` | 🚧 | |

## Reduction Operations

| Operation | Status | Notes |
|-----------|--------|-------|
| `torch.sum()` | ✅ | |
| `torch.mean()` | ✅ | |
| `torch.max()` | ✅ | |
| `torch.min()` | ✅ | |
| `torch.argmax()` | ✅ | |
| `torch.cumsum()` | ✅ | |

## Shape Operations

| Operation | Status | Notes |
|-----------|--------|-------|
| `view()` | ✅ | |
| `reshape()` | ✅ | |
| `transpose()` | ✅ | |
| `permute()` | ✅ | |
| `squeeze()` | ✅ | |
| `unsqueeze()` | ✅ | |
| `expand()` | ✅ | |
| `contiguous()` | ✅ | |
| `clone()` | ✅ | |

## Indexing & Slicing

| Operation | Status | Notes |
|-----------|--------|-------|
| `tensor[idx]` | ✅ | |
| `tensor[start:end]` | ✅ | |
| `torch.cat()` | ✅ | |
| `torch.index_select()` | ✅ | |

## Comparison Operations

| Operation | Status | Notes |
|-----------|--------|-------|
| `==` / `torch.eq()` | ✅ | |
| `!=` / `torch.ne()` | ✅ | |
| `<` / `torch.lt()` | ✅ | |
| `<=` / `torch.le()` | ✅ | |
| `>` / `torch.gt()` | ✅ | |
| `>=` / `torch.ge()` | ✅ | |
| `torch.where()` | ✅ | |

## Neural Network Layers

| Operation | Status | Notes |
|-----------|--------|-------|
| `F.linear()` | ✅ | |
| `F.embedding()` | ✅ | |
| `F.layer_norm()` | ✅ | |
| `F.scaled_dot_product_attention()` | ✅ | |
| `F.dropout()` | ✅ | Identity in eval mode |

## Type Casting

| Operation | Status | Notes |
|-----------|--------|-------|
| `.float()` | ✅ | |
| `.half()` | 🚧 | |
| `.int()` | ✅ | |
| `.long()` | ✅ | |
| `.bool()` | ✅ | |
| `.to(dtype)` | ✅ | |

## Legend

- ✅ Supported
- 🚧 Coming soon / Partial support
- ❌ Not supported

## Missing an Op?

If you need an operation that's not listed:

1. Check if it works anyway (many ops just work)
2. [Open an issue](https://github.com/jmaczan/torch-webgpu/issues)
3. [Submit a PR](../development/contributing.md)
