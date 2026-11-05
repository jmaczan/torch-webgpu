#include "core/stream.h"

#include "core/context.h"

#include <c10/util/Exception.h>

#include <cstdlib>
#include <cstring>
#include <mutex>

namespace torch_webgpu::core
{
    namespace
    {
        size_t env_size(const char *name, size_t def)
        {
            const char *v = std::getenv(name);
            if (!v || !*v)
                return def;
            long long n = std::atoll(v);
            return n > 0 ? static_cast<size_t>(n) : def;
        }

        uint64_t align_up(uint64_t x, uint64_t a) { return (x + a - 1) / a * a; }
    }

    Stream &Stream::get()
    {
        static Stream *s = new Stream();
        return *s;
    }

    Stream::Stream()
    {
        auto &ctx = Context::get();
        max_batch_ = env_size("TORCH_WEBGPU_MAX_BATCH", 128);
        uniform_align_ = std::max<size_t>(ctx.limits().minUniformBufferOffsetAlignment, 16);
        arena_capacity_ = align_up(env_size("TORCH_WEBGPU_UNIFORM_ARENA", 1 << 20), uniform_align_);
        arena_host_.resize(arena_capacity_);

        wgpu::BufferDescriptor desc{};
        desc.label = "torch_webgpu uniform arena";
        desc.usage = wgpu::BufferUsage::Uniform | wgpu::BufferUsage::CopyDst;
        desc.size = arena_capacity_;
        arena_ = ctx.device().CreateBuffer(&desc);
    }

    wgpu::CommandEncoder &Stream::encoder()
    {
        if (!encoder_)
        {
            wgpu::CommandEncoderDescriptor desc{};
            encoder_ = Context::get().device().CreateCommandEncoder(&desc);
        }
        return encoder_;
    }

    wgpu::ComputePassEncoder &Stream::pass()
    {
        if (!pass_open_)
        {
            wgpu::ComputePassDescriptor desc{};
            pass_ = encoder().BeginComputePass(&desc);
            pass_open_ = true;
        }
        return pass_;
    }

    void Stream::end_pass()
    {
        if (pass_open_)
        {
            pass_.End();
            pass_ = nullptr;
            pass_open_ = false;
        }
    }

    UniformSlice Stream::alloc_uniform(const void *data, size_t size)
    {
        auto &ctx = Context::get();
        std::lock_guard<std::recursive_mutex> g(ctx.mutex());
        TORCH_CHECK(size <= arena_capacity_, "torch_webgpu: kernel parameters too large (", size, " bytes)");
        // Uniform bindings must be at least 16 bytes and sized in 16-byte units.
        size_t bind_size = align_up(std::max<size_t>(size, 16), 16);
        if (arena_used_ + bind_size > arena_capacity_)
            flush_locked();
        UniformSlice s{arena_, arena_used_, bind_size};
        std::memcpy(arena_host_.data() + arena_used_, data, size);
        if (bind_size > size)
            std::memset(arena_host_.data() + arena_used_ + size, 0, bind_size - size);
        arena_used_ = align_up(arena_used_ + bind_size, uniform_align_);
        return s;
    }

    void Stream::dispatch(const wgpu::ComputePipeline &pipeline, const wgpu::BindGroup &bind_group,
                          uint32_t x, uint32_t y, uint32_t z)
    {
        if (x == 0 || y == 0 || z == 0)
            return;
        auto &ctx = Context::get();
        std::lock_guard<std::recursive_mutex> g(ctx.mutex());
        auto &p = pass();
        p.SetPipeline(pipeline);
        p.SetBindGroup(0, bind_group);
        p.DispatchWorkgroups(x, y, z);
        ++pending_dispatches_;
        ++pending_commands_;
        ++total_dispatches_;
        if (pending_dispatches_ >= max_batch_)
            flush_locked();
    }

    void Stream::copy_buffer(const wgpu::Buffer &src, uint64_t src_offset, const wgpu::Buffer &dst,
                             uint64_t dst_offset, uint64_t size)
    {
        if (size == 0)
            return;
        TORCH_INTERNAL_ASSERT(src_offset % 4 == 0 && dst_offset % 4 == 0 && size % 4 == 0,
                              "copy_buffer requires 4-byte aligned offsets and size");
        auto &ctx = Context::get();
        std::lock_guard<std::recursive_mutex> g(ctx.mutex());
        end_pass();
        encoder().CopyBufferToBuffer(src, src_offset, dst, dst_offset, size);
        ++pending_commands_;
    }

    void Stream::write_buffer(const wgpu::Buffer &dst, uint64_t dst_offset, const void *data, uint64_t size)
    {
        if (size == 0)
            return;
        TORCH_INTERNAL_ASSERT(dst_offset % 4 == 0 && size % 4 == 0,
                              "write_buffer requires 4-byte aligned offset and size");
        auto &ctx = Context::get();
        std::lock_guard<std::recursive_mutex> g(ctx.mutex());
        // A staging buffer filled at creation time and copied inside the
        // encoder keeps the upload ordered with the recorded kernels.
        wgpu::BufferDescriptor desc{};
        desc.label = "torch_webgpu upload";
        desc.usage = wgpu::BufferUsage::CopySrc | wgpu::BufferUsage::MapWrite;
        desc.size = size;
        desc.mappedAtCreation = true;
        wgpu::Buffer staging = ctx.device().CreateBuffer(&desc);
        TORCH_CHECK(staging, "torch_webgpu: failed to allocate an upload buffer of ", size, " bytes");
        void *mapped = staging.GetMappedRange(0, size);
        TORCH_CHECK(mapped != nullptr, "torch_webgpu: failed to map upload buffer of ", size, " bytes");
        std::memcpy(mapped, data, size);
        staging.Unmap();
        end_pass();
        encoder().CopyBufferToBuffer(staging, 0, dst, dst_offset, size);
        ++pending_commands_;
    }

    void Stream::read_buffer(const wgpu::Buffer &src, uint64_t src_offset, void *dst, uint64_t size)
    {
        if (size == 0)
            return;
        auto &ctx = Context::get();
        std::lock_guard<std::recursive_mutex> g(ctx.mutex());
        uint64_t begin = src_offset / 4 * 4;
        uint64_t end = align_up(src_offset + size, 4);
        TORCH_INTERNAL_ASSERT(end <= src.GetSize(), "read_buffer out of bounds");
        uint64_t n = end - begin;

        wgpu::BufferDescriptor desc{};
        desc.label = "torch_webgpu readback";
        desc.usage = wgpu::BufferUsage::CopyDst | wgpu::BufferUsage::MapRead;
        desc.size = n;
        wgpu::Buffer staging = ctx.device().CreateBuffer(&desc);
        TORCH_CHECK(staging, "torch_webgpu: failed to allocate a readback buffer of ", n, " bytes");

        end_pass();
        encoder().CopyBufferToBuffer(src, begin, staging, 0, n);
        ++pending_commands_;
        flush_locked();

        bool ok = false;
        std::string err;
        ctx.wait(staging.MapAsync(wgpu::MapMode::Read, 0, n, wgpu::CallbackMode::WaitAnyOnly,
                                  [&](wgpu::MapAsyncStatus status, wgpu::StringView msg)
                                  {
                                      ok = status == wgpu::MapAsyncStatus::Success;
                                      if (!ok && msg.data)
                                          err.assign(msg.data, msg.length == WGPU_STRLEN ? std::strlen(msg.data) : msg.length);
                                  }));
        ctx.check_errors();
        TORCH_CHECK(ok, "torch_webgpu: failed to map readback buffer: ", err);
        const uint8_t *mapped = static_cast<const uint8_t *>(staging.GetConstMappedRange(0, n));
        std::memcpy(dst, mapped + (src_offset - begin), size);
        staging.Unmap();
    }

    void Stream::flush_locked()
    {
        auto &ctx = Context::get();
        end_pass();
        if (pending_commands_ == 0 && !encoder_)
        {
            ctx.check_errors();
            return;
        }
        if (arena_used_ > 0)
        {
            // Executes before the command buffer below, after all earlier submits.
            ctx.queue().WriteBuffer(arena_, 0, arena_host_.data(), arena_used_);
            arena_used_ = 0;
        }
        wgpu::CommandBuffer cb = encoder().Finish();
        encoder_ = nullptr;
        ctx.queue().Submit(1, &cb);
        pending_dispatches_ = 0;
        pending_commands_ = 0;
        ++total_submits_;
        ctx.check_errors();
    }

    void Stream::flush()
    {
        auto &ctx = Context::get();
        std::lock_guard<std::recursive_mutex> g(ctx.mutex());
        flush_locked();
    }

    void Stream::synchronize()
    {
        auto &ctx = Context::get();
        std::lock_guard<std::recursive_mutex> g(ctx.mutex());
        flush_locked();
        wgpu::QueueWorkDoneStatus result = wgpu::QueueWorkDoneStatus::Success;
        ctx.wait(ctx.queue().OnSubmittedWorkDone(wgpu::CallbackMode::WaitAnyOnly,
                                                 [&](wgpu::QueueWorkDoneStatus status, wgpu::StringView)
                                                 { result = status; }));
        ctx.check_errors();
        TORCH_CHECK(result == wgpu::QueueWorkDoneStatus::Success, "torch_webgpu: queue synchronization failed");
    }
}
