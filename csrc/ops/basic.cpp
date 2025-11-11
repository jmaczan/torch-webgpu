#include <ATen/ATen.h>
#include <torch/library.h>
#include "core/webgpu_allocator.h"

namespace torch_webgpu
{
    namespace ops
    {
        at::Tensor empty_memory_format(
            c10::IntArrayRef size,
            c10::optional<at::ScalarType> dtype_opt,
            c10::optional<at::Layout> layout_opt,
            c10::optional<at::Device> device_opt,
            c10::optional<bool> pin_memory_opt,
            c10::optional<c10::MemoryFormat> memory_format_opt)
        {
            auto allocator = core::getWebGPUCachingAllocator();
            constexpr c10::DispatchKeySet privateuse1_ks(c10::DispatchKey::PrivateUse1);
            return at::detail::empty_generic(size, allocator, privateuse1_ks, dtype_or_default(dtype_opt), memory_format_opt);
        }

        at::Tensor empty_strided(
            c10::IntArrayRef size,
            c10::IntArrayRef stride,
            c10::optional<at::ScalarType> dtype_opt,
            c10::optional<at::Layout> layout_opt,
            c10::optional<at::Device> device_opt,
            c10::optional<bool> pin_memory_opt)
        {
            auto allocator = core::getWebGPUCachingAllocator();
            constexpr c10::DispatchKeySet privateuse1_ks(c10::DispatchKey::PrivateUse1);
            return at::detail::empty_strided_generic(size, stride, allocator, privateuse1_ks, dtype_or_default(dtype_opt));
        }

        at::Tensor view(at::Tensor const &self, at::IntArrayRef size)
        {
            return at::native::view(self, size);
        }

        const at::Tensor &resize_(at::Tensor const &self, at::IntArrayRef size, c10::optional<c10::MemoryFormat> format)
        {
            TORCH_CHECK(self.is_contiguous());
            auto result = at::native::resize_(self, size, format);
            return result;
        }
    }

    TORCH_LIBRARY_IMPL(aten, PrivateUse1, m)
    {
        m.impl("view", TORCH_FN(ops::view));
        m.impl("resize_", TORCH_FN(ops::resize_));
        m.impl("empty.memory_format", TORCH_FN(torch_webgpu::ops::empty_memory_format));
        m.impl("empty_strided", TORCH_FN(torch_webgpu::ops::empty_strided));
    }
}
