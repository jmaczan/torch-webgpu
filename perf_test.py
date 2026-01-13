"""Test WebGPU buffer creation overhead."""
import torch
import time

device = torch.device("webgpu")

# Create tensors on CPU then move to device
print("Testing overhead sources...")

# Test 1: Empty tensor creation (should be fast - just allocation)
print("\n1. Empty tensor creation:")
for size in [100, 10000, 1000000]:
    start = time.time()
    for _ in range(100):
        t = torch.empty(size, device=device)
    elapsed = time.time() - start
    print(f"   torch.empty({size}): {elapsed/100*1000:.3f}ms per call")

# Test 2: Add operation (binary kernel)
print("\n2. Add operation:")
a = torch.randn(1000).to(device)
b = torch.randn(1000).to(device)
# warmup
_ = a + b
start = time.time()
for _ in range(100):
    c = a + b
elapsed = time.time() - start
print(f"   a + b (1000 elements): {elapsed/100*1000:.3f}ms per call")

# Test 3: Multiple adds in sequence (test async batching)
print("\n3. Multiple adds in sequence:")
start = time.time()
for _ in range(100):
    c = a + b
    d = c + a
    e = d + b
elapsed = time.time() - start
print(f"   3 adds per iter: {elapsed/100*1000:.3f}ms per 3 ops = {elapsed/300*1000:.3f}ms per op")

# Test 4: Copy to CPU (forces sync)
print("\n4. Copy to CPU (forces sync):")
# warmup
_ = a.to('cpu')
start = time.time()
for _ in range(20):
    _ = a.to('cpu')
elapsed = time.time() - start
print(f"   to CPU (1000 elements): {elapsed/20*1000:.3f}ms per call")

# Test 5: MM operations
print("\n5. MM operations:")
A = torch.randn(100, 100).to(device)
B = torch.randn(100, 100).to(device)
# warmup
_ = torch.mm(A, B)
start = time.time()
for _ in range(100):
    C = torch.mm(A, B)
elapsed = time.time() - start
print(f"   mm (100x100): {elapsed/100*1000:.3f}ms per call")

# Test 6: Multiple MMs without sync
print("\n6. Multiple MMs (test async):")
start = time.time()
for _ in range(10):
    C1 = torch.mm(A, B)
    C2 = torch.mm(B, A)
    C3 = torch.mm(C1, C2)
elapsed = time.time() - start
print(f"   3 MMs per iter: {elapsed/10*1000:.3f}ms per 3 ops = {elapsed/30*1000:.3f}ms per op")

# Test 7: MMs with sync after each batch
print("\n7. MMs with sync after batch:")
start = time.time()
for _ in range(10):
    C1 = torch.mm(A, B)
    C2 = torch.mm(B, A)
    C3 = torch.mm(C1, C2)
    _ = C3.to('cpu')  # Force sync
elapsed = time.time() - start
print(f"   3 MMs + sync: {elapsed/10*1000:.3f}ms per batch")
