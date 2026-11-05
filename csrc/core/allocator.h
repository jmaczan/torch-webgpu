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
    struct Allocation
    {
        wgpu::Buffer buffer;
        size_t size;     // requested size in bytes
        size_t capacity; // actual buffer size in bytes (>= size, multiple of 4)
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

    // Helpers to reach the backing buffer of a tensor/storage.
    inline Allocation *allocation_of(const c10::Storage &storage)
    {
        return static_cast<Allocation *>(storage.data_ptr().get_context());
    }
    inline Allocation *allocation_of(const at::TensorBase &t)
    {
        return allocation_of(t.storage());
    }
    inline const wgpu::Buffer &buffer_of(const at::TensorBase &t)
    {
        auto *a = allocation_of(t);
        TORCH_INTERNAL_ASSERT(a != nullptr, "torch_webgpu: tensor has no device buffer");
        return a->buffer;
    }
    // Byte offset of the tensor's first element inside its buffer.
    inline uint64_t byte_offset_of(const at::TensorBase &t)
    {
        return static_cast<uint64_t>(t.storage_offset()) * t.element_size();
    }
}
