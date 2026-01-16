#pragma once
#include <webgpu/webgpu_cpp.h>
#include <vector>
#include <unordered_map>
#include <cstdint>

namespace torch_webgpu
{
    namespace core
    {
        /**
         * BufferPool manages a pool of reusable WebGPU uniform buffers.
         *
         * Instead of creating a new buffer for each operation's parameters,
         * we reuse buffers from a pool. This reduces the overhead of
         * CreateBuffer() calls (~0.06ms each, 200 ops = 12ms per forward).
         *
         * Usage:
         *   wgpu::Buffer buf = getBufferPool().acquire(size);
         *   queue.WriteBuffer(buf, 0, data, size);
         *   // ... use buffer in dispatch ...
         *   // Buffer automatically returned to pool on batch flush
         */
        class BufferPool
        {
        public:
            // Size classes for buffers (rounded up to nearest class)
            // Common param struct sizes: 64, 128, 256, 512 bytes
            static constexpr size_t SIZE_CLASSES[] = {64, 128, 256, 512, 1024, 2048, 4096};
            static constexpr size_t NUM_SIZE_CLASSES = 7;

            BufferPool(wgpu::Device device);
            ~BufferPool() = default;

            // Acquire a buffer of at least `size` bytes
            // Buffer is marked as in-use until releaseAll() is called
            wgpu::Buffer acquire(size_t size);

            // Release all in-use buffers back to the pool
            // Call this after command batch is submitted
            void releaseAll();

            // Get statistics
            size_t getTotalBuffers() const;
            size_t getInUseBuffers() const;
            size_t getAvailableBuffers() const;

            // Pre-allocate buffers for each size class
            void preallocate(size_t buffersPerClass = 32);

        private:
            size_t getSizeClass(size_t size) const;
            wgpu::Buffer createBuffer(size_t size);

            wgpu::Device device_;

            // Pool organized by size class
            // available_[sizeClass] = list of available buffers
            std::vector<std::vector<wgpu::Buffer>> available_;

            // Buffers currently in use (will be returned on releaseAll)
            std::vector<wgpu::Buffer> in_use_;

            // Stats
            size_t total_created_ = 0;
            size_t total_acquired_ = 0;
            size_t total_released_ = 0;
        };

        // Global buffer pool (lazily initialized)
        BufferPool& getBufferPool();

    } // namespace core
} // namespace torch_webgpu
