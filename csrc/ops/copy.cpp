#include <ATen/ATen.h>
#include <ATen/RedispatchFunctions.h>
#include <webgpu/webgpu_cpp.h>
#include <iostream>
#include <vector>
#include <cstring>
#include "core/webgpu_context.h"
#include "core/webgpu_allocator.h"
#include "unary.h"

namespace torch_webgpu
{
    namespace ops
    {
        at::Tensor &cpu_copy_with_webgpu(
            at::Tensor &self, at::Tensor const &src, bool non_blocking = false)
        {
            // TODO: take non_blocking into consideration
            if (src.device().is_privateuseone() && self.device().is_cpu())
            {
                TORCH_CHECK(src.dtype() == self.dtype());
                TORCH_CHECK(src.numel() == self.numel());
                TORCH_CHECK(self.is_contiguous());

                // Make source contiguous if needed (on WebGPU)
                at::Tensor src_contig = src.is_contiguous() ? src : src.contiguous();

                auto src_size = src_contig.numel() * at::elementSize(src_contig.dtype().toScalarType());
                auto self_size = self.numel() * at::elementSize(self.dtype().toScalarType());

                TORCH_CHECK(self_size == src_size);
                auto src_data = static_cast<core::WebGPUAllocation *>(src_contig.storage().data_ptr().get());
                auto src_storage_offset = src_contig.storage_offset();

                auto self_data = self.data_ptr();

                // WebGPU requires copy/map sizes to be a multiple of 4
                constexpr uint64_t WGPU_BUFFER_ALIGNMENT = 4;
                uint64_t aligned_size = ((static_cast<uint64_t>(src_size) + WGPU_BUFFER_ALIGNMENT - 1) / WGPU_BUFFER_ALIGNMENT) * WGPU_BUFFER_ALIGNMENT;
                uint64_t src_buffer_offset = static_cast<uint64_t>(src_storage_offset) * static_cast<uint64_t>(at::elementSize(src_contig.scalar_type()));

                TORCH_CHECK(src_data->buffer.GetSize() >= src_buffer_offset + aligned_size);

                wgpu::BufferDescriptor buffer_desc;
                buffer_desc.label = "WebGPU temp buffer";
                buffer_desc.usage = wgpu::BufferUsage::CopyDst | wgpu::BufferUsage::MapRead;
                buffer_desc.size = aligned_size;
                buffer_desc.mappedAtCreation = false;

                core::WebGPUContext &ctx = core::getWebGPUContext();
                wgpu::Buffer tmp = ctx.getDevice().CreateBuffer(&buffer_desc);

                wgpu::CommandEncoder encoder = ctx.getDevice().CreateCommandEncoder();
                encoder.CopyBufferToBuffer(src_data->buffer, src_buffer_offset, tmp, 0, aligned_size);
                wgpu::CommandBuffer command = encoder.Finish();

                ctx.getQueue().Submit(1, &command);

                auto noop = [](wgpu::MapAsyncStatus, wgpu::StringView) {};

                wgpu::Future map_async_future = tmp.MapAsync(wgpu::MapMode::Read, 0, aligned_size, wgpu::CallbackMode::WaitAnyOnly, noop);

                ctx.getInstance().WaitAny(map_async_future, UINT64_MAX);

                // Map and copy only the actual data size (not the aligned size)
                const void *mapped = tmp.GetConstMappedRange(0, aligned_size);
                std::memcpy(self_data, mapped, src_size);
                tmp.Unmap();
                return self;
            }
            else
            {
                return at::native::copy_(self, src, non_blocking);
            }
        }

        void copy_kernel_webgpu(at::TensorIteratorBase &iter)
        {
            unary_kernel<UnaryOp::Copy>(iter);
        }

        at::Tensor &copy_(
            at::Tensor &self, at::Tensor const &src, bool non_blocking = false)
        {
            // TODO: take non_blocking into consideration

            if (src.device().is_cpu() && self.device().is_privateuseone())
            {
                TORCH_CHECK(src.dtype() == self.dtype(),
                    "copy_ CPU->WebGPU: src.dtype()=", src.dtype(), " self.dtype()=", self.dtype());
                TORCH_CHECK(src.numel() == self.numel());
                TORCH_CHECK(self.is_contiguous(), "WebGPU doesn't support copying from CPU to non-contiguous WebGPU tensor, yet");

                at::Tensor src_contiguous = src.is_contiguous() ? src : src.contiguous();
                uint64_t write_nbytes = static_cast<uint64_t>(src_contiguous.numel()) * static_cast<uint64_t>(at::elementSize(src_contiguous.scalar_type()));

                auto self_data = static_cast<core::WebGPUAllocation *>(self.storage().data_ptr().get());
                auto self_storage_offset = self.storage_offset();
                TORCH_CHECK(self_storage_offset >= 0, "WebGPU doesn't support negative offset yet");
                uint64_t buffer_offset = static_cast<uint64_t>(self_storage_offset) * static_cast<uint64_t>(at::elementSize(self.scalar_type()));

                TORCH_CHECK(self_data->buffer.GetSize() >= buffer_offset + write_nbytes);

                // WebGPU requires write size to be a multiple of 4
                constexpr uint64_t WGPU_BUFFER_ALIGNMENT = 4;
                uint64_t aligned_nbytes = ((write_nbytes + WGPU_BUFFER_ALIGNMENT - 1) / WGPU_BUFFER_ALIGNMENT) * WGPU_BUFFER_ALIGNMENT;

                if (aligned_nbytes == write_nbytes)
                {
                    // Already aligned, write directly
                    core::getWebGPUContext().getQueue().WriteBuffer(self_data->buffer, buffer_offset, src_contiguous.data_ptr(), write_nbytes);
                }
                else
                {
                    // Need to pad - create temporary aligned buffer
                    std::vector<char> aligned_buffer(aligned_nbytes, 0);
                    std::memcpy(aligned_buffer.data(), src_contiguous.data_ptr(), write_nbytes);
                    core::getWebGPUContext().getQueue().WriteBuffer(self_data->buffer, buffer_offset, aligned_buffer.data(), aligned_nbytes);
                }
                return self;
            }
            else if (src.device().is_privateuseone() && self.device().is_privateuseone())
            {
                if (src.is_contiguous() and self.is_contiguous())
                {
                    // TODO: handle a scenario when src and self share the storage and their memory ranges overlap

                    TORCH_CHECK(src.dtype() == self.dtype(),
                        "copy_ WebGPU->WebGPU: src.dtype()=", src.dtype(), " self.dtype()=", self.dtype());
                    TORCH_CHECK(src.numel() == self.numel());

                    auto src_data = static_cast<core::WebGPUAllocation *>(src.storage().data_ptr().get());
                    auto src_storage_offset = src.storage_offset();
                    TORCH_CHECK(src_storage_offset >= 0, "WebGPU doesn't support negative offset yet");
                    uint64_t src_buffer_offset = static_cast<uint64_t>(src_storage_offset) * static_cast<uint64_t>(at::elementSize(src.scalar_type()));

                    auto self_data = static_cast<core::WebGPUAllocation *>(self.storage().data_ptr().get());
                    auto self_storage_offset = self.storage_offset();
                    TORCH_CHECK(self_storage_offset >= 0, "WebGPU doesn't support negative offset yet");
                    uint64_t self_buffer_offset = static_cast<uint64_t>(self_storage_offset) * static_cast<uint64_t>(at::elementSize(self.scalar_type()));

                    uint64_t write_nbytes = static_cast<uint64_t>(src.numel()) * at::elementSize(src.dtype().toScalarType());

                    // WebGPU requires copy size to be a multiple of 4
                    constexpr uint64_t WGPU_BUFFER_ALIGNMENT = 4;
                    uint64_t aligned_nbytes = ((write_nbytes + WGPU_BUFFER_ALIGNMENT - 1) / WGPU_BUFFER_ALIGNMENT) * WGPU_BUFFER_ALIGNMENT;

                    TORCH_CHECK(src_data->buffer.GetSize() >= src_buffer_offset + aligned_nbytes);
                    TORCH_CHECK(self_data->buffer.GetSize() >= self_buffer_offset + aligned_nbytes);

                    wgpu::CommandEncoder encoder = core::getWebGPUContext().getDevice().CreateCommandEncoder();
                    encoder.CopyBufferToBuffer(src_data->buffer, src_buffer_offset, self_data->buffer, self_buffer_offset, aligned_nbytes);
                    wgpu::CommandBuffer command = encoder.Finish();

                    core::getWebGPUContext().getQueue().Submit(1, &command); // TODO: Submit is async, handle it correctly
                    return self;
                }
                else
                {
                    at::TensorIteratorConfig config;
                    config.set_check_mem_overlap(true);
                    config.add_output(self);
                    config.add_input(src);
                    config.check_all_same_dtype(true);
                    auto iter = config.build();

                    copy_kernel_webgpu(iter);
                    return self;
                }
            }
            else
            {
                return at::native::copy_(self, src, non_blocking);
            }
        }

        at::Tensor _copy_from(at::Tensor const &self, at::Tensor const &dst, bool non_blocking = false)
        {
            auto &dst_non_const = const_cast<at::Tensor &>(dst);

            c10::DispatchKeySet ks;

            if (dst.device().is_privateuseone())
            {
                ks = c10::DispatchKeySet(c10::DispatchKey::PrivateUse1);
            }
            else if (dst.device().is_cpu())
            {
                ks = c10::DispatchKeySet(c10::DispatchKey::CPU);
            }
            else
            {
                TORCH_CHECK(false, "Unsupported destination device in _copy_from");
            }

            at::redispatch::copy_(ks, dst_non_const, self, non_blocking);
            return dst;
        }

        at::Tensor _copy_from_and_resize(at::Tensor const &self, at::Tensor const &dst)
        {
            TORCH_CHECK(self.is_contiguous());
            dst.resize_(self.sizes());
            return dst.copy_(self);
        }

        at::Tensor to_device(
            const at::Tensor &self,
            at::Device device,
            at::ScalarType dtype,
            bool non_blocking = false,
            bool copy = false,
            std::optional<c10::MemoryFormat> memory_format = std::nullopt)
        {
            // Handle dtype conversion by first converting on source device, then moving
            at::Tensor src = self;
            if (dtype != self.scalar_type()) {
                // Convert dtype on source device first
                if (self.device().is_privateuseone()) {
                    // WebGPU: move to CPU, convert dtype, then proceed
                    src = self.to(at::kCPU).to(dtype);
                } else {
                    // CPU: convert dtype directly
                    src = self.to(dtype);
                }
            }

            // Same device case
            if (src.device() == device) {
                if (copy) {
                    return src.clone();
                }
                return src;
            }

            // WebGPU → CPU
            if (src.device().is_privateuseone() && device.is_cpu()) {
                auto mem_fmt = memory_format.value_or(c10::MemoryFormat::Contiguous);
                auto result = at::empty(src.sizes(), src.options().device(at::kCPU).memory_format(mem_fmt));
                result.copy_(src, non_blocking);
                return result;
            }

            // CPU → WebGPU
            if (src.device().is_cpu() && device.is_privateuseone()) {
                auto mem_fmt = memory_format.value_or(c10::MemoryFormat::Contiguous);
                auto result = at::empty(src.sizes(), src.options().device(device).memory_format(mem_fmt));
                result.copy_(src, non_blocking);
                return result;
            }

            // Fallback for other cases
            return at::native::to(src, device, dtype, non_blocking, copy, memory_format);
        }

        TORCH_LIBRARY_IMPL(aten, PrivateUse1, m)
        {
            m.impl("copy_", TORCH_FN(ops::copy_));
            m.impl("_copy_from", TORCH_FN(ops::_copy_from));
            m.impl("_copy_from_and_resize", TORCH_FN(ops::_copy_from_and_resize));
            m.impl("to.device", TORCH_FN(ops::to_device));
        }

        at::Tensor cpu_to_device_with_webgpu(
            const at::Tensor &self,
            at::Device device,
            at::ScalarType dtype,
            bool non_blocking = false,
            bool copy = false,
            std::optional<c10::MemoryFormat> memory_format = std::nullopt)
        {
            // Handle CPU → WebGPU
            if (self.device().is_cpu() && device.is_privateuseone()) {
                // Use self's dtype, ignoring the passed dtype parameter
                // (dtype parameter is often defaulted to Float by PyTorch dispatch)
                auto mem_fmt = memory_format.value_or(c10::MemoryFormat::Contiguous);
                auto result = at::empty(self.sizes(), self.options().device(device).memory_format(mem_fmt));
                result.copy_(self, non_blocking);
                return result;
            }

            // Fallback for other cases (including CPU → CPU)
            return at::native::to(self, device, dtype, non_blocking, copy, memory_format);
        }

        TORCH_LIBRARY_IMPL(aten, CPU, m)
        {
            m.impl("copy_", TORCH_FN(ops::cpu_copy_with_webgpu));
            m.impl("to.device", TORCH_FN(ops::cpu_to_device_with_webgpu));
        }
    }
}