#pragma once
#include <webgpu/webgpu_cpp.h>
#include "webgpu_context.h"

namespace torch_webgpu
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
}