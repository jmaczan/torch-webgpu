#include "core/allocator.h"

#include "core/context.h"
#include "core/stream.h"

#include <c10/util/Exception.h>

#include <map>
#include <mutex>

namespace torch_webgpu::core
{
    namespace
    {
        constexpr size_t kSmallLimit = 1 << 20; // 1 MiB
        constexpr size_t kSmallRound = 512;
        constexpr size_t kLargeRound = 2 << 20; // 2 MiB

        size_t round_size(size_t size)
        {
            if (size <= kSmallLimit)
                return std::max<size_t>(kSmallRound, (size + kSmallRound - 1) / kSmallRound * kSmallRound);
            return (size + kLargeRound - 1) / kLargeRound * kLargeRound;
        }

        class CachingAllocator final : public c10::Allocator
        {
        public:
            at::DataPtr allocate(size_t size) override
            {
                auto device = c10::Device(c10::DeviceType::PrivateUse1, 0);
                if (size == 0)
                    return {nullptr, nullptr, &noop_delete, device};
                Allocation *a = acquire(size);
                return {a, a, &release_raw, device};
            }

            at::DeleterFnPtr raw_deleter() const override { return &release_raw; }

            void copy_data(void *dest, const void *src, std::size_t count) const override
            {
                TORCH_CHECK(false, "torch_webgpu: copy_data on raw device pointers is not supported");
            }

            MemoryStats stats()
            {
                std::lock_guard<std::mutex> g(mutex_);
                return stats_;
            }

            void reset_peak()
            {
                std::lock_guard<std::mutex> g(mutex_);
                stats_.peak_allocated_bytes = stats_.allocated_bytes;
            }

            void empty_cache()
            {
                std::lock_guard<std::mutex> g(mutex_);
                for (auto &[cap, buf] : pool_)
                    stats_.reserved_bytes -= static_cast<int64_t>(cap);
                // Dropping the last reference is enough: Dawn keeps the
                // memory alive until in-flight GPU work referencing it ends.
                pool_.clear();
            }

            static CachingAllocator &instance()
            {
                static CachingAllocator *a = new CachingAllocator();
                return *a;
            }

        private:
            static void noop_delete(void *) {}

            static void release_raw(void *ptr)
            {
                instance().release(static_cast<Allocation *>(ptr));
            }

            Allocation *acquire(size_t size)
            {
                const size_t cap = round_size(size);
                {
                    std::lock_guard<std::mutex> g(mutex_);
                    // Best fit, but don't waste more than 25% (large) / 2x (small).
                    auto it = pool_.lower_bound(cap);
                    size_t limit = cap <= kSmallLimit ? cap * 2 : cap + cap / 4;
                    if (it != pool_.end() && it->first <= limit)
                    {
                        auto *a = new Allocation{std::move(it->second), size, it->first};
                        pool_.erase(it);
                        stats_.num_cache_hits++;
                        on_alloc(a->capacity);
                        return a;
                    }
                }

                wgpu::Buffer buffer = create(cap);
                if (!buffer)
                {
                    // Free the cache and retry once before giving up.
                    Stream::get().synchronize();
                    empty_cache();
                    buffer = create(cap);
                }
                if (!buffer)
                {
                    auto s = stats();
                    {
                        std::lock_guard<std::mutex> g(mutex_);
                        stats_.num_ooms++;
                    }
                    TORCH_CHECK_WITH(OutOfMemoryError, false,
                                     "torch_webgpu: out of memory trying to allocate ", size, " bytes (",
                                     s.allocated_bytes, " bytes allocated, ", s.reserved_bytes, " bytes reserved)");
                }
                std::lock_guard<std::mutex> g(mutex_);
                stats_.num_allocs++;
                stats_.reserved_bytes += static_cast<int64_t>(cap);
                auto *a = new Allocation{std::move(buffer), size, cap};
                on_alloc(cap);
                return a;
            }

            wgpu::Buffer create(size_t cap)
            {
                auto &ctx = Context::get();
                if (cap > ctx.limits().maxBufferSize)
                    return nullptr;
                std::lock_guard<std::recursive_mutex> g(ctx.mutex());
                wgpu::BufferDescriptor desc{};
                desc.label = "torch_webgpu storage";
                desc.usage = wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopySrc | wgpu::BufferUsage::CopyDst;
                desc.size = cap;
                ctx.device().PushErrorScope(wgpu::ErrorFilter::OutOfMemory);
                wgpu::Buffer buffer = ctx.device().CreateBuffer(&desc);
                bool oom = false;
                ctx.wait(ctx.device().PopErrorScope(
                    wgpu::CallbackMode::WaitAnyOnly,
                    [&](wgpu::PopErrorScopeStatus status, wgpu::ErrorType type, wgpu::StringView)
                    {
                        oom = status == wgpu::PopErrorScopeStatus::Success && type == wgpu::ErrorType::OutOfMemory;
                    }));
                return oom ? nullptr : buffer;
            }

            void on_alloc(size_t cap)
            {
                stats_.allocated_bytes += static_cast<int64_t>(cap);
                stats_.peak_allocated_bytes = std::max(stats_.peak_allocated_bytes, stats_.allocated_bytes);
            }

            void release(Allocation *a)
            {
                if (!a)
                    return;
                std::lock_guard<std::mutex> g(mutex_);
                stats_.allocated_bytes -= static_cast<int64_t>(a->capacity);
                pool_.emplace(a->capacity, std::move(a->buffer));
                delete a;
            }

            std::mutex mutex_;
            std::multimap<size_t, wgpu::Buffer> pool_;
            MemoryStats stats_;
        };
    }

    c10::Allocator *get_allocator() { return &CachingAllocator::instance(); }
    MemoryStats memory_stats() { return CachingAllocator::instance().stats(); }
    void reset_peak_memory_stats() { CachingAllocator::instance().reset_peak(); }
    void empty_cache()
    {
        Stream::get().flush();
        CachingAllocator::instance().empty_cache();
    }

    REGISTER_ALLOCATOR(c10::DeviceType::PrivateUse1, &CachingAllocator::instance());
}
