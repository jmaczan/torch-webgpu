#pragma once
// Caching device allocator.
//
// Every tensor storage is backed by one wgpu::Buffer. Freed buffers go back to
// a size-bucketed pool and are reused by later allocations. This is safe
// without any fences because all GPU work is executed in queue order: commands
// encoded after a reuse are ordered after all earlier commands that used the
// buffer.

#include <ATen/ATen.h>
#include <c10/core/Allocator.h>
#include <webgpu/webgpu_cpp.h>

#include <cstdint>

namespace torch_webgpu::core
{
    // Object stored as the DataPtr of every WebGPU storage.
    // Device memory is exposed to PyTorch through *virtual* addresses: each
    // allocation owns a unique, never-reused range [va, va + capacity), so
    // any pointer into a tensor (data_ptr(), from_blob, DLPack) resolves to
    // its buffer and byte offset. The addresses are never dereferenced.
    struct Allocation
    {
        wgpu::Buffer buffer;
        size_t size;     // requested size in bytes
        size_t capacity; // actual buffer size in bytes (>= size, multiple of 4)
        uint64_t va = 0; // virtual base address
    };

    struct MemoryStats
    {
        int64_t allocated_bytes = 0;      // bytes in live tensors (capacity)
        int64_t reserved_bytes = 0;       // live + cached bytes
        int64_t peak_allocated_bytes = 0; // high-water mark of allocated_bytes
        int64_t num_allocs = 0;           // allocate() calls that hit the device
        int64_t num_cache_hits = 0;       // allocate() calls served from the pool
        int64_t num_ooms = 0;
    };

    // The allocator registered for c10::DeviceType::PrivateUse1.
    c10::Allocator *get_allocator();

    MemoryStats memory_stats();
    void reset_peak_memory_stats();
    // Releases all cached (unused) buffers back to the driver.
    void empty_cache();

    // Live allocation containing virtual address `ptr`, or nullptr.
    Allocation *find_allocation(const void *ptr);
    bool owns(const c10::DataPtr &p);

    // Helpers to reach the backing buffer of a tensor/storage. Storages made
    // by our allocator carry their Allocation as context; foreign wrappers of
    // our memory (at::from_blob) are resolved through the address registry.
    inline Allocation *allocation_of(const c10::Storage &storage)
    {
        const auto &dp = storage.data_ptr();
        if (owns(dp))
            return static_cast<Allocation *>(dp.get_context());
        return dp.get() ? find_allocation(dp.get()) : nullptr;
    }
    inline Allocation *allocation_of(const at::TensorBase &t)
    {
        return allocation_of(t.storage());
    }
    // Byte offset of a storage's data pointer inside its allocation (0 unless
    // the storage wraps an interior pointer).
    inline uint64_t storage_delta(const c10::Storage &storage, const Allocation *a)
    {
        return a ? static_cast<uint64_t>(reinterpret_cast<uintptr_t>(storage.data_ptr().get()) - a->va) : 0;
    }
    // A small storage buffer bound in place of zero-sized tensors (which have
    // no allocation). Kernels never access it because they have no elements.
    const wgpu::Buffer &dummy_buffer();

    inline const wgpu::Buffer &buffer_of(const at::TensorBase &t)
    {
        auto *a = allocation_of(t);
        if (a == nullptr)
        {
            TORCH_INTERNAL_ASSERT(t.numel() == 0 || t.storage().nbytes() == 0,
                                  "torch_webgpu: tensor has no device buffer");
            return dummy_buffer();
        }
        return a->buffer;
    }
    // Byte offset of the tensor's first element inside its buffer.
    inline uint64_t byte_offset_of(const at::TensorBase &t)
    {
        return storage_delta(t.storage(), allocation_of(t)) + static_cast<uint64_t>(t.storage_offset()) * t.element_size();
    }
    // Same, in elements (the buffer offset kernels index with).
    inline uint64_t element_offset_of(const at::TensorBase &t)
    {
        const uint64_t b = byte_offset_of(t);
        TORCH_INTERNAL_ASSERT(b % t.element_size() == 0, "torch_webgpu: misaligned tensor data pointer");
        return b / t.element_size();
    }
}
