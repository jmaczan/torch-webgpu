#pragma once
#include <string>
#include <vector>
// Command stream: batches GPU work into one command encoder.
//
// Kernels are recorded into a single open compute pass and only submitted
// when (a) the batch reaches TORCH_WEBGPU_MAX_BATCH dispatches, (b) the host
// needs data back (read_buffer / synchronize), or (c) flush() is called.
// This removes the per-op Submit() overhead of the original implementation.
//
// Ordering guarantees:
//  * all recorded commands execute in recording order;
//  * host->device uploads (write_buffer) are recorded into the same encoder
//    via staging buffers, so they are ordered with respect to kernels;
//  * kernel parameters live in a per-batch uniform arena that is uploaded
//    right before the batch is submitted.

#include <webgpu/webgpu_cpp.h>

#include <cstdint>
#include <vector>

namespace torch_webgpu::core
{
    // Error bits of the device-side error flag.
    constexpr uint32_t kErrorIndexOutOfBounds = 1u;
    constexpr uint32_t kErrorInvalidDistribution = 2u;

    struct UniformSlice
    {
        wgpu::Buffer buffer;
        uint64_t offset = 0;
        uint64_t size = 0;
    };

    class Stream
    {
    public:
        static Stream &get();

        // Copies `size` bytes of kernel parameters into the uniform arena.
        UniformSlice alloc_uniform(const void *data, size_t size);

        // Records a compute dispatch.
        void dispatch(const wgpu::ComputePipeline &pipeline, const wgpu::BindGroup &bind_group,
                      uint32_t x, uint32_t y = 1, uint32_t z = 1, const std::string *label = nullptr);

        // Records a buffer->buffer copy. Offsets/size must be multiples of 4.
        void copy_buffer(const wgpu::Buffer &src, uint64_t src_offset, const wgpu::Buffer &dst,
                         uint64_t dst_offset, uint64_t size);

        // Host -> device. dst_offset/size must be multiples of 4.
        void write_buffer(const wgpu::Buffer &dst, uint64_t dst_offset, const void *data, uint64_t size);

        // Device -> host (blocking). Any offset/size is accepted.
        void read_buffer(const wgpu::Buffer &src, uint64_t src_offset, void *dst, uint64_t size);

        // Submits recorded work without waiting.
        void flush();
        // Submits recorded work and waits for the GPU to finish it.
        void synchronize();

        // Device-side error flag for indexing kernels (bind as
        // array<atomic<u32>>; kernels atomicOr a non-zero code on
        // out-of-bounds indices). Calling this marks the flag as "maybe set";
        // it is checked at the next host synchronization and raised as an
        // IndexError there (like CUDA device-side asserts).
        // Device-side error flag: kernels atomicOr an error bit into word 0
        // (see kError*); it is checked and raised at the next readback.
        const wgpu::Buffer &index_error_flag();

        size_t pending_dispatches() const { return pending_dispatches_; }
        uint64_t total_dispatches() const { return total_dispatches_; }
        uint64_t total_submits() const { return total_submits_; }

        // GPU timing of every dispatch with timestamp queries (needs the
        // timestamp-query feature). While profiling, each dispatch gets its
        // own compute pass, which adds a little overhead of its own.
        struct ProfileRecord
        {
            std::string label;
            uint32_t x = 0, y = 0, z = 0; // workgroups
            double gpu_us = 0.0;
        };
        void start_profiling();
        // Synchronizes and returns the records since start_profiling().
        std::vector<ProfileRecord> stop_profiling();
        bool profiling() const { return profiling_; }

    private:
        Stream();
        wgpu::CommandEncoder &encoder();
        wgpu::ComputePassEncoder &pass();
        void end_pass();
        void flush_locked();
        void raise_index_error_if_set(uint32_t flag);

        wgpu::CommandEncoder encoder_;
        wgpu::ComputePassEncoder pass_;
        bool pass_open_ = false;
        size_t pending_dispatches_ = 0;
        size_t pending_commands_ = 0;
        size_t max_batch_;
        uint64_t total_dispatches_ = 0;
        uint64_t total_submits_ = 0;

        wgpu::Buffer error_flag_;
        bool error_flag_dirty_ = false;

        // Uniform arena
        wgpu::Buffer arena_;
        std::vector<uint8_t> arena_host_;
        size_t arena_used_ = 0;
        size_t arena_capacity_;
        size_t uniform_align_;

        bool profiling_ = false;
        wgpu::QuerySet query_set_;
        uint32_t query_capacity_ = 0;
        uint32_t queries_used_ = 0;              // in the open encoder
        std::vector<ProfileRecord> open_records_; // dispatches in the open encoder
        struct ProfileBatch
        {
            wgpu::Buffer readback;
            std::vector<ProfileRecord> records;
        };
        std::vector<ProfileBatch> profile_batches_;
        void resolve_queries_locked();
    };
}
