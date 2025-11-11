#include <ATen/ATen.h>
#include <ATen/native/BinaryOps.h>
#include <ATen/RedispatchFunctions.h>
#include <ATen/native/DispatchStub.h>
#include <webgpu/webgpu_cpp.h>
#include "core/webgpu_context.h"
#include "core/webgpu_allocator.h"
#include "unary.h"

namespace torch_webgpu
{
    namespace ops
    {
        void add_kernel_webgpu(::at::TensorIteratorBase &iter, const ::at::Scalar &alpha)
        {
            TORCH_CHECK(iter.ntensors() == 3);
            TORCH_CHECK(iter.common_dtype() == at::ScalarType::Float);
            TORCH_CHECK(iter.device_type() == c10::DeviceType::PrivateUse1);
            TORCH_CHECK(iter.dtype(0) == iter.dtype(1));
            TORCH_CHECK(iter.dtype(1) == iter.dtype(2));

            core::WebGPUContext &ctx = core::getWebGPUContext();
            constexpr const char *addWGSL = R"wgsl(
const MAX_DIMS: u32 = 8u;

struct Params {
    length: u32,
    ndim: u32,
    alpha: f32,
    _pad: u32,

    out_offset: u32,
    self_offset: u32,
    other_offset: u32,
    _pad2: u32,

    out_strides: array<u32, MAX_DIMS>,
    self_strides: array<u32, MAX_DIMS>,
    other_strides: array<u32, MAX_DIMS>,
    shape: array<u32, MAX_DIMS>,
};

@group(0) @binding(0)
var<storage, read> selfBuffer: array<f32>;

@group(0) @binding(1)
var<storage, read> otherBuffer: array<f32>;

@group(0) @binding(2)
var<storage, read_write> outBuffer: array<f32>;

@group(0) @binding(3)
var<uniform> params: Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.length) { return; }

    var remaining = i;
    var coord: array<u32, MAX_DIMS>;

    for (var d: i32 = i32(params.ndim) - 1; d >= 0; d--) {
        let ud = u32(d);
        let s = params.shape[ud];
        coord[ud] = remaining % s;
        remaining = remaining / s;
    }

    var idx_out: u32 = 0u;
    var idx_self: u32 = 0u;
    var idx_other: u32 = 0u;

    for (var d: u32 = 0u; d < params.ndim; d++) {
        let c = coord[d];
        idx_out += c * params.out_strides[d];
        idx_self += c * params.self_strides[d];
        idx_other += c * params.other_strides[d];
    }

    idx_out += params.out_offset;
    idx_self += params.self_offset;
    idx_other += params.other_offset;

    outBuffer[idx_out] = selfBuffer[idx_self] + params.alpha * otherBuffer[idx_other];
}
)wgsl";

            wgpu::ShaderSourceWGSL shader_source{
                wgpu::ShaderSourceWGSL::Init{
                    nullptr,
                    wgpu::StringView{addWGSL, std::strlen(addWGSL)},
                }};

            wgpu::ShaderModuleDescriptor shader_descriptor{};
            shader_descriptor.nextInChain = &shader_source;
            shader_descriptor.label = "at::Tensor add shader";
            wgpu::ShaderModule shader_module = ctx.getDevice().CreateShaderModule(&shader_descriptor);

            wgpu::BindGroupLayoutEntry bindings[4]{};

            bindings[0].binding = 0;
            bindings[0].visibility = wgpu::ShaderStage::Compute;
            bindings[0].buffer.type = wgpu::BufferBindingType::ReadOnlyStorage;
            bindings[0].buffer.hasDynamicOffset = false;
            bindings[0].buffer.minBindingSize = 0;

            bindings[1].binding = 1;
            bindings[1].visibility = wgpu::ShaderStage::Compute;
            bindings[1].buffer.type = wgpu::BufferBindingType::ReadOnlyStorage;
            bindings[1].buffer.hasDynamicOffset = false;
            bindings[1].buffer.minBindingSize = 0;

            bindings[2].binding = 2;
            bindings[2].visibility = wgpu::ShaderStage::Compute;
            bindings[2].buffer.type = wgpu::BufferBindingType::Storage;
            bindings[2].buffer.hasDynamicOffset = false;
            bindings[2].buffer.minBindingSize = 0;

            bindings[3].binding = 3;
            bindings[3].visibility = wgpu::ShaderStage::Compute;
            bindings[3].buffer.type = wgpu::BufferBindingType::Uniform;
            bindings[3].buffer.hasDynamicOffset = false;
            bindings[3].buffer.minBindingSize = 0;

            wgpu::BindGroupLayoutDescriptor layout_descriptor{};
            layout_descriptor.entryCount = 4;
            layout_descriptor.entries = bindings;

            wgpu::BindGroupLayout bind_group_layout = ctx.getDevice().CreateBindGroupLayout(&layout_descriptor);

            wgpu::PipelineLayoutDescriptor pipeline_layout_descriptor{};
            pipeline_layout_descriptor.bindGroupLayoutCount = 1;
            pipeline_layout_descriptor.bindGroupLayouts = &bind_group_layout;

            wgpu::PipelineLayout pipeline_layout = ctx.getDevice().CreatePipelineLayout(&pipeline_layout_descriptor);

            wgpu::ComputePipelineDescriptor pipeline_descriptor{};
            pipeline_descriptor.layout = pipeline_layout;
            pipeline_descriptor.compute.module = shader_module;
            pipeline_descriptor.compute.entryPoint = wgpu::StringView{"main", 4};

            wgpu::ComputePipeline pipeline = ctx.getDevice().CreateComputePipeline(&pipeline_descriptor);
            // everything above should be cached, uniform (params) perhaps too?
            auto out = iter.tensor(0);
            auto self = iter.tensor(1);
            auto other = iter.tensor(2);
            auto ndim = static_cast<uint32_t>(iter.ndim());
            auto shape = iter.shape();
            auto out_strides_bytes = iter.strides(0);
            auto self_strides_bytes = iter.strides(1);
            auto other_strides_bytes = iter.strides(2);

            auto element_size = iter.element_size(0);

            std::vector<int64_t> out_strides(ndim);
            std::vector<int64_t> self_strides(ndim);
            std::vector<int64_t> other_strides(ndim);

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

                int64_t other_bytes = other_strides_bytes[i];
                if (other_bytes == 0)
                {
                    other_strides[i] = 0;
                }
                else
                {
                    TORCH_CHECK(other_bytes % element_size == 0);
                    other_strides[i] = other_bytes / element_size;
                }
            }

            auto length = iter.numel();

            core::WebGPUAllocation *out_allocation = static_cast<core::WebGPUAllocation *>(out.storage().data_ptr().get());
            core::WebGPUAllocation *self_allocation = static_cast<core::WebGPUAllocation *>(self.storage().data_ptr().get());
            core::WebGPUAllocation *other_allocation = static_cast<core::WebGPUAllocation *>(other.storage().data_ptr().get());

            wgpu::Buffer self_buffer = self_allocation->buffer;
            wgpu::Buffer other_buffer = other_allocation->buffer;
            wgpu::Buffer out_buffer = out_allocation->buffer;

            auto out_offset = out.storage_offset();
            auto self_offset = self.storage_offset();
            auto other_offset = other.storage_offset();

            constexpr uint32_t MAX_DIMS = 8;
            TORCH_CHECK(ndim <= MAX_DIMS);

            struct Params
            {
                uint32_t length;
                uint32_t ndim;
                float alpha;
                uint32_t _pad; // allegedly, it's a padding we need for webgpu

                uint32_t out_offset;
                uint32_t self_offset;
                uint32_t other_offset;
                uint32_t _pad2;

                uint32_t out_strides[MAX_DIMS];
                uint32_t self_strides[MAX_DIMS];
                uint32_t other_strides[MAX_DIMS];
                uint32_t shape[MAX_DIMS];
            };

            Params params{};
            params.length = static_cast<uint32_t>(iter.numel());
            params.ndim = ndim;
            params.alpha = alpha.to<float>();
            params._pad = 0;

            params.out_offset = static_cast<uint32_t>(out_offset);
            params.self_offset = static_cast<uint32_t>(self_offset);
            params.other_offset = static_cast<uint32_t>(other_offset);
            params._pad2 = 0;

            for (uint32_t d = 0; d < MAX_DIMS; ++d)
            {
                params.out_strides[d] = 0;
                params.self_strides[d] = 0;
                params.other_strides[d] = 0;
                params.shape[d] = 1;
            }

            for (int64_t i = 0; i < ndim; ++i)
            {
                auto dim_size = shape[i];
                TORCH_CHECK(dim_size >= 0 && dim_size <= std::numeric_limits<uint32_t>::max());
                params.shape[i] = static_cast<uint32_t>(dim_size);

                auto out_stride = out_strides[i];
                auto self_stride = self_strides[i];
                auto other_stride = other_strides[i];

                TORCH_CHECK(out_stride >= 0 && out_stride <= std::numeric_limits<uint32_t>::max());
                TORCH_CHECK(self_stride >= 0 && self_stride <= std::numeric_limits<uint32_t>::max());
                TORCH_CHECK(other_stride >= 0 && other_stride <= std::numeric_limits<uint32_t>::max());

                params.out_strides[i] = static_cast<uint32_t>(out_stride);
                params.self_strides[i] = static_cast<uint32_t>(self_stride);
                params.other_strides[i] = static_cast<uint32_t>(other_stride);
            }

            wgpu::BufferDescriptor uniform_descriptor{};
            uniform_descriptor.label = "Params";
            uniform_descriptor.usage = wgpu::BufferUsage::Uniform | wgpu::BufferUsage::CopyDst;
            uniform_descriptor.size = sizeof(Params);
            uniform_descriptor.mappedAtCreation = false;

            wgpu::Buffer params_buffer = ctx.getDevice().CreateBuffer(&uniform_descriptor);
            ctx.getQueue().WriteBuffer(params_buffer, 0, &params, sizeof(Params));

            wgpu::BindGroupEntry bind_group_entries[4]{};
            bind_group_entries[0].binding = 0;
            bind_group_entries[0].buffer = self_buffer;
            bind_group_entries[0].offset = 0;
            bind_group_entries[0].size = self_buffer.GetSize();

            bind_group_entries[1].binding = 1;
            bind_group_entries[1].buffer = other_buffer;
            bind_group_entries[1].offset = 0;
            bind_group_entries[1].size = other_buffer.GetSize();

            bind_group_entries[2].binding = 2;
            bind_group_entries[2].buffer = out_buffer;
            bind_group_entries[2].offset = 0;
            bind_group_entries[2].size = out_buffer.GetSize();

            bind_group_entries[3].binding = 3;
            bind_group_entries[3].buffer = params_buffer;
            bind_group_entries[3].offset = 0;
            bind_group_entries[3].size = sizeof(Params);

            wgpu::BindGroupDescriptor bind_group_descriptor{};
            bind_group_descriptor.layout = bind_group_layout;
            bind_group_descriptor.entryCount = 4;
            bind_group_descriptor.entries = bind_group_entries;

            wgpu::BindGroup bind_group = ctx.getDevice().CreateBindGroup(&bind_group_descriptor);

            wgpu::CommandEncoder encoder = ctx.getDevice().CreateCommandEncoder();
            wgpu::ComputePassDescriptor pass_descriptor;
            wgpu::ComputePassEncoder pass_encoder = encoder.BeginComputePass(&pass_descriptor);
            pass_encoder.SetPipeline(pipeline);
            pass_encoder.SetBindGroup(0, bind_group);

            const uint32_t workgroup_size = 64;
            uint32_t num_workgroups = (length + workgroup_size - 1) / workgroup_size;

            pass_encoder.DispatchWorkgroups(num_workgroups);
            pass_encoder.End();

            wgpu::CommandBuffer command_buffer = encoder.Finish();
            ctx.getQueue().Submit(1, &command_buffer);
        }

        at::Tensor &add_out_webgpu(
            const at::Tensor &self,
            const at::Tensor &other,
            const at::Scalar &alpha,
            at::Tensor &out)
        {
            at::TensorIteratorConfig config;
            config.set_check_mem_overlap(true);
            config.add_output(out);
            config.add_input(self);
            config.add_input(other);
            config.promote_inputs_to_common_dtype(true);
            config.cast_common_dtype_to_outputs(true);
            config.check_all_same_device(false);
            auto iter = config.build();

            add_kernel_webgpu(iter, alpha);

            return out;
        }
    }
    TORCH_LIBRARY_IMPL(aten, PrivateUse1, m)
    {
        m.impl("add.out", TORCH_FN(ops::add_out_webgpu));
    }
}

// it needs to be like this because of what REGISTER_PRIVATEUSE1_DISPATCH expects with at::native
namespace at
{
    namespace native
    {
        void add_kernel_webgpu(TensorIteratorBase &iter, const Scalar &alpha)
        {
            torch_webgpu::ops::add_kernel_webgpu(iter, alpha);
        }
        REGISTER_PRIVATEUSE1_DISPATCH(add_stub, &add_kernel_webgpu);
    }
}