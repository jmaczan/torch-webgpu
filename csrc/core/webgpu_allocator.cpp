#include <webgpu/webgpu_cpp.h>
#include "webgpu_context.h"
#include "webgpu_allocator.h"

namespace torch_webgpu
{
    namespace core
    {
        void WebGPUAllocator::allocate(void **ptr, size_t size)
        {
            wgpu::BufferDescriptor buffer_desc;
            buffer_desc.label = "WebGPU buffer";
            buffer_desc.usage = wgpu::BufferUsage::CopyDst | wgpu::BufferUsage::CopySrc | wgpu::BufferUsage::Storage;
            buffer_desc.size = size;
            buffer_desc.mappedAtCreation = false;
            *ptr = new WebGPUAllocation(std::move(torch_webgpu::core::getWebGPUContext().getDevice().CreateBuffer(&buffer_desc)));
        }

        WebGPUAllocator &getWebGPUAllocator()
        {
            static WebGPUAllocator webgpu_allocator;
            return webgpu_allocator;
        }

        void WebGPUCachingHostDeleter(void *ptr)
        {
            delete static_cast<WebGPUAllocation *>(ptr);
        }
    }
}