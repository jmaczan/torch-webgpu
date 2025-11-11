#include <torch/library.h>
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

        at::DataPtr WebGPUCachingAllocator::allocate(size_t size)
        {
            void *ptr = nullptr;
            if (size > 0)
            {
                core::getWebGPUAllocator().allocate(&ptr, size);
            }
            return {ptr, ptr, &core::WebGPUCachingHostDeleter, at::DeviceType::PrivateUse1};
        }

        at::DeleterFnPtr WebGPUCachingAllocator::raw_deleter() const
        {
            return &core::WebGPUCachingHostDeleter;
        }

        void WebGPUCachingAllocator::copy_data(void *dest, const void *src, std::size_t count) const
        {
            TORCH_CHECK_NOT_IMPLEMENTED(false, "copy_data not implemented in WebGPUCachingAllocator");
        }

        at::Allocator *getWebGPUCachingAllocator()
        {
            static WebGPUCachingAllocator webgpu_caching_allocator;
            return &webgpu_caching_allocator;
        }
    }
}