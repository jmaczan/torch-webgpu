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
                return {reinterpret_cast<void *>(a->va), a, &release_raw, device};
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

            Allocation *find(const void *ptr)
            {
                const auto p = reinterpret_cast<uintptr_t>(ptr);
                std::lock_guard<std::mutex> g(mutex_);
                auto it = live_.upper_bound(p);
                if (it == live_.begin())
                    return nullptr;
                --it;
                Allocation *a = it->second;
                return p < a->va + a->capacity ? a : nullptr;
            }

            static bool is_release(at::DeleterFnPtr f) { return f == &release_raw; }

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
                const uint64_t max_buffer = Context::get().limits().maxBufferSize / 4 * 4;
                TORCH_CHECK(size <= max_buffer, "torch_webgpu: a tensor of ", size,
                            " bytes exceeds this device's maximum buffer size (maxBufferSize = ", max_buffer,
                            " bytes). Use a smaller tensor or dtype, or a device with larger limits.");
                // Rounding up for reuse must not cross the device limit.
                const size_t cap = std::max<size_t>(std::min<size_t>(round_size(size), max_buffer), (size + 3) / 4 * 4);
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
                        register_va(a);
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
                register_va(a);
                return a;
            }

            wgpu::Buffer create(size_t cap)
            {
                auto &ctx = Context::get();
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

            // Caller holds mutex_. Ranges are 256-byte aligned with a gap so
            // one-past-the-end pointers never alias the next allocation.
            void register_va(Allocation *a)
            {
                a->va = next_va_;
                next_va_ += (a->capacity + 2 * 256 - 1) / 256 * 256;
                live_.emplace(a->va, a);
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
                live_.erase(a->va);
                pool_.emplace(a->capacity, std::move(a->buffer));
                delete a;
            }

            std::mutex mutex_;
            std::multimap<size_t, wgpu::Buffer> pool_;
            std::map<uintptr_t, Allocation *> live_; // va -> live allocation
            uintptr_t next_va_ = uintptr_t(1) << 40;   // well clear of 0 / small integers
            MemoryStats stats_;
        };
    }

    c10::Allocator *get_allocator() { return &CachingAllocator::instance(); }
    Allocation *find_allocation(const void *ptr) { return CachingAllocator::instance().find(ptr); }
    bool owns(const c10::DataPtr &p) { return CachingAllocator::is_release(p.get_deleter()); }

    const wgpu::Buffer &dummy_buffer()
    {
        static wgpu::Buffer *b = []
        {
            wgpu::BufferDescriptor desc{};
            desc.label = "torch_webgpu dummy";
            desc.usage = wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopySrc | wgpu::BufferUsage::CopyDst;
            desc.size = 256;
            return new wgpu::Buffer(Context::get().device().CreateBuffer(&desc));
        }();
        return *b;
    }
    MemoryStats memory_stats() { return CachingAllocator::instance().stats(); }
    void reset_peak_memory_stats() { CachingAllocator::instance().reset_peak(); }
    void empty_cache()
    {
        Stream::get().flush();
        CachingAllocator::instance().empty_cache();
    }

    REGISTER_ALLOCATOR(c10::DeviceType::PrivateUse1, &CachingAllocator::instance());
}
