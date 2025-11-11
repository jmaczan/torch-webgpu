#include <ATen/ATen.h>
#include <ATen/RedispatchFunctions.h>
#include <webgpu/webgpu_cpp.h>
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
                auto src_size = src.numel() * at::elementSize(src.dtype().toScalarType()); // TODO: probably some size check here?
                auto self_size = self.numel() * at::elementSize(self.dtype().toScalarType());

                TORCH_CHECK(src.dtype() == self.dtype());
                TORCH_CHECK(src.dtype() == at::ScalarType::Float);
                TORCH_CHECK(src.numel() == self.numel());
                TORCH_CHECK(src.is_contiguous());
                TORCH_CHECK(self.is_contiguous());
                TORCH_CHECK(self_size == src_size);
                auto src_data = static_cast<core::WebGPUAllocation *>(src.storage().data_ptr().get());

                auto self_data = self.data_ptr();
                TORCH_CHECK(src_data->buffer.GetSize() >= src_size);

                wgpu::BufferDescriptor buffer_desc;
                buffer_desc.label = "WebGPU temp buffer";
                buffer_desc.usage = wgpu::BufferUsage::CopyDst | wgpu::BufferUsage::MapRead;
                buffer_desc.size = src_size;
                buffer_desc.mappedAtCreation = false;

                core::WebGPUContext &ctx = core::getWebGPUContext();
                wgpu::Buffer tmp = ctx.getDevice().CreateBuffer(&buffer_desc);

                wgpu::CommandEncoder encoder = ctx.getDevice().CreateCommandEncoder();
                encoder.CopyBufferToBuffer(src_data->buffer, 0, tmp, 0, src_size);
                wgpu::CommandBuffer command = encoder.Finish();

                ctx.getQueue().Submit(1, &command);

                auto noop = [](wgpu::MapAsyncStatus, wgpu::StringView) {};

                wgpu::Future map_async_future = tmp.MapAsync(wgpu::MapMode::Read, 0, src_size, wgpu::CallbackMode::WaitAnyOnly, noop);

                ctx.getInstance().WaitAny(map_async_future, UINT64_MAX);

                const void *mapped = tmp.GetConstMappedRange(0, src_size);
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
            TORCH_CHECK(iter.ntensors() == 2);
            TORCH_CHECK(iter.common_dtype() == at::ScalarType::Float);
            TORCH_CHECK(iter.device_type() == c10::DeviceType::PrivateUse1);

            UnaryKernel &kernel = get_unary_kernel(UnaryOp::Copy);
            auto out = iter.tensor(0);
            auto self = iter.tensor(1);
            auto ndim = static_cast<uint32_t>(iter.ndim());
            auto shape = iter.shape();
            auto out_strides_bytes = iter.strides(0);
            auto self_strides_bytes = iter.strides(1);

            auto element_size = iter.element_size(0);

            std::vector<int64_t> out_strides(ndim);
            std::vector<int64_t> self_strides(ndim);

            for (int64_t i = 0; i < ndim; ++i)
            {
                int64_t out_bytes = out_strides_bytes[i];
                if (out_bytes == 0)
                {
                    out_strides[i] = 0;
                }
                else
                {
                    TORCH_CHECK(out_bytes % element_size == 0);
                    out_strides[i] = out_bytes / element_size;
                }

                int64_t self_bytes = self_strides_bytes[i];
                if (self_bytes == 0)
                {
                    self_strides[i] = 0;
                }
                else
                {
                    TORCH_CHECK(self_bytes % element_size == 0);
                    self_strides[i] = self_bytes / element_size;
                }
            }

            auto length = iter.numel();

            core::WebGPUAllocation *out_allocation = static_cast<core::WebGPUAllocation *>(out.storage().data_ptr().get());
            core::WebGPUAllocation *self_allocation = static_cast<core::WebGPUAllocation *>(self.storage().data_ptr().get());

            wgpu::Buffer self_buffer = self_allocation->buffer;
            wgpu::Buffer out_buffer = out_allocation->buffer;

            auto out_offset = out.storage_offset();
            auto self_offset = self.storage_offset();

            constexpr uint32_t MAX_DIMS = 8;
            TORCH_CHECK(ndim <= MAX_DIMS);

            struct Params
            {
                uint32_t length;
                uint32_t ndim;
                uint32_t _pad; // allegedly, it's a padding we need for webgpu

                uint32_t out_offset;
                uint32_t self_offset;
                uint32_t _pad2;

                uint32_t out_strides[MAX_DIMS];
                uint32_t self_strides[MAX_DIMS];
                uint32_t shape[MAX_DIMS];
            };

            Params params{};
            params.length = static_cast<uint32_t>(iter.numel());
            params.ndim = ndim;
            params._pad = 0;

            params.out_offset = static_cast<uint32_t>(out_offset);
            params.self_offset = static_cast<uint32_t>(self_offset);
            params._pad2 = 0;

            for (uint32_t d = 0; d < MAX_DIMS; ++d)
            {
                params.out_strides[d] = 0;
                params.self_strides[d] = 0;
                params.shape[d] = 1;
            }

            for (int64_t i = 0; i < ndim; ++i)
            {
                auto dim_size = shape[i];
                TORCH_CHECK(dim_size >= 0 && dim_size <= std::numeric_limits<uint32_t>::max());
                params.shape[i] = static_cast<uint32_t>(dim_size);

                auto out_stride = out_strides[i];
                auto self_stride = self_strides[i];

                TORCH_CHECK(out_stride >= 0 && out_stride <= std::numeric_limits<uint32_t>::max());
                TORCH_CHECK(self_stride >= 0 && self_stride <= std::numeric_limits<uint32_t>::max());

                params.out_strides[i] = static_cast<uint32_t>(out_stride);
                params.self_strides[i] = static_cast<uint32_t>(self_stride);
            }

            wgpu::BufferDescriptor uniform_descriptor{};
            uniform_descriptor.label = "Params";
            uniform_descriptor.usage = wgpu::BufferUsage::Uniform | wgpu::BufferUsage::CopyDst;
            uniform_descriptor.size = sizeof(Params);
            uniform_descriptor.mappedAtCreation = false;

            core::WebGPUContext &ctx = core::getWebGPUContext();
            wgpu::Buffer params_buffer = ctx.getDevice().CreateBuffer(&uniform_descriptor);
            ctx.getQueue().WriteBuffer(params_buffer, 0, &params, sizeof(Params));

            wgpu::BindGroupEntry bind_group_entries[3]{};
            bind_group_entries[0].binding = 0;
            bind_group_entries[0].buffer = self_buffer;
            bind_group_entries[0].offset = 0;
            bind_group_entries[0].size = self_buffer.GetSize();

            bind_group_entries[1].binding = 1;
            bind_group_entries[1].buffer = out_buffer;
            bind_group_entries[1].offset = 0;
            bind_group_entries[1].size = out_buffer.GetSize();

            bind_group_entries[2].binding = 2;
            bind_group_entries[2].buffer = params_buffer;
            bind_group_entries[2].offset = 0;
            bind_group_entries[2].size = sizeof(Params);

            wgpu::BindGroupDescriptor bind_group_descriptor{};
            bind_group_descriptor.layout = kernel.bind_group_layout;
            bind_group_descriptor.entryCount = 3;
            bind_group_descriptor.entries = bind_group_entries;

            wgpu::BindGroup bind_group = ctx.getDevice().CreateBindGroup(&bind_group_descriptor);

            wgpu::CommandEncoder encoder = ctx.getDevice().CreateCommandEncoder();
            wgpu::ComputePassDescriptor pass_descriptor;
            wgpu::ComputePassEncoder pass_encoder = encoder.BeginComputePass(&pass_descriptor);
            pass_encoder.SetPipeline(kernel.pipeline);
            pass_encoder.SetBindGroup(0, bind_group);

            const uint32_t workgroup_size = 64;
            uint32_t num_workgroups = (length + workgroup_size - 1) / workgroup_size;

            pass_encoder.DispatchWorkgroups(num_workgroups);
            pass_encoder.End();

            wgpu::CommandBuffer command_buffer = encoder.Finish();
            ctx.getQueue().Submit(1, &command_buffer);
        }

        at::Tensor &copy_(
            at::Tensor &self, at::Tensor const &src, bool non_blocking = false)
        {
            // TODO: take non_blocking into consideration

            if (src.device().is_cpu() && self.device().is_privateuseone())
            {
                TORCH_CHECK(src.dtype() == self.dtype());
                TORCH_CHECK(src.numel() == self.numel());
                TORCH_CHECK(self.is_contiguous(), "WebGPU doesn't support copying from CPU to non-contiguous WebGPU tensor, yet");

                at::Tensor src_contiguous = src.is_contiguous() ? src : src.contiguous();
                uint64_t write_nbytes = static_cast<u_int64_t>(src_contiguous.numel()) * static_cast<uint64_t>(at::elementSize(src_contiguous.scalar_type()));

                auto self_data = static_cast<core::WebGPUAllocation *>(self.storage().data_ptr().get());
                auto self_storage_offset = self.storage_offset();
                TORCH_CHECK(self_storage_offset >= 0, "WebGPU doesn't support negative offset yet");
                uint64_t buffer_offset = static_cast<uint64_t>(self_storage_offset) * static_cast<uint64_t>(at::elementSize(self.scalar_type()));

                TORCH_CHECK(self_data->buffer.GetSize() >= buffer_offset + write_nbytes);

                core::getWebGPUContext().getQueue().WriteBuffer(self_data->buffer, buffer_offset, src_contiguous.data_ptr(), write_nbytes);
                return self;
            }
            else if (src.device().is_privateuseone() && self.device().is_privateuseone())
            {
                if (src.is_contiguous() and self.is_contiguous())
                {
                    // TODO: handle a scenario when src and self share the storage and their memory ranges overlap

                    TORCH_CHECK(src.dtype() == self.dtype());
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
                    TORCH_CHECK(src_data->buffer.GetSize() >= src_buffer_offset + write_nbytes);
                    TORCH_CHECK(self_data->buffer.GetSize() >= self_buffer_offset + write_nbytes);

                    wgpu::CommandEncoder encoder = core::getWebGPUContext().getDevice().CreateCommandEncoder();
                    encoder.CopyBufferToBuffer(src_data->buffer, src_buffer_offset, self_data->buffer, self_buffer_offset, write_nbytes);
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
    }

    TORCH_LIBRARY_IMPL(aten, PrivateUse1, m)
    {
        m.impl("copy_", TORCH_FN(ops::copy_));
        m.impl("_copy_from", TORCH_FN(ops::_copy_from));
        m.impl("_copy_from_and_resize", TORCH_FN(ops::_copy_from_and_resize));
    }

    TORCH_LIBRARY_IMPL(aten, CPU, m)
    {
        m.impl("copy_", TORCH_FN(ops::cpu_copy_with_webgpu));
    }
}