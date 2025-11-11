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
        void relu_kernel_webgpu(at::TensorIteratorBase &iter)
        {
            TORCH_CHECK(iter.ntensors() == 2);
            TORCH_CHECK(iter.common_dtype() == at::ScalarType::Float);
            TORCH_CHECK(iter.device_type() == c10::DeviceType::PrivateUse1);
            UnaryKernel &kernel = get_unary_kernel(UnaryOp::ReLU);
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

            auto length = static_cast<uint32_t>(iter.numel());

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
            params.length = length;
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

        at::Tensor relu(at::Tensor const &self)
        {
            at::Tensor out = at::empty_like(self, self.options().device(at::DeviceType::PrivateUse1));
            at::TensorIteratorConfig config;
            config.add_output(out);
            config.add_input(self);
            config.check_all_same_dtype(true);
            auto iter = config.build();

            relu_kernel_webgpu(iter);

            return out;
        }

        at::Tensor &relu_out(
            at::Tensor const &self,
            at::Tensor &out)
        {
            at::TensorIteratorConfig config;
            config.add_output(out);
            config.add_input(self);
            config.check_all_same_dtype(true);
            auto iter = config.build();

            relu_kernel_webgpu(iter);

            return out;
        }
    }
    TORCH_LIBRARY_IMPL(aten, PrivateUse1, m)
    {
        m.impl("relu", TORCH_FN(ops::relu));
        m.impl("relu.out", TORCH_FN(ops::relu_out));
    }
}