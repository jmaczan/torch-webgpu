#pragma once
#include <webgpu/webgpu_cpp.h>
#include "webgpu_context.h"

namespace torch_webgpu
{
    namespace core
    {
        struct WebGPUAllocation
        {
            wgpu::Buffer buffer;
            explicit WebGPUAllocation(wgpu::Buffer &&b) : buffer(std::move(b)) {}
        };

        struct WebGPUAllocator
        {
            void allocate(void **ptr, size_t size);
        };

        WebGPUAllocator &getWebGPUAllocator();

        void WebGPUCachingHostDeleter(void *ptr);

        // not really caching, yet
        struct WebGPUCachingAllocator final : public c10::Allocator
        {
            at::DataPtr allocate(size_t size) override;

            at::DeleterFnPtr raw_deleter() const override;

            void copy_data(void *dest, const void *src, std::size_t count) const;
        };

        at::Allocator *getWebGPUCachingAllocator();
    }
}