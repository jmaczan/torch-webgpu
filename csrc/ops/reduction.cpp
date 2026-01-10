#include <ATen/ATen.h>
#include <ATen/native/ReduceOps.h>
#include <torch/library.h>
#include <webgpu/webgpu_cpp.h>
#include "utils/string.h"
#include "core/webgpu_context.h"
#include "core/webgpu_allocator.h"

namespace torch_webgpu
{
    namespace ops
    {
        namespace
        {
            // Simple mean implementation - reduces all dimensions
            // For dim-specific reduction, we'd need a more complex kernel
            const std::string mean_all_shader = R"wgsl(
struct Params {
    length: u32,
    out_offset: u32,
    self_offset: u32,
    _pad: u32,
};

@group(0) @binding(0)
var<storage, read> selfBuffer: array<f32>;

@group(0) @binding(1)
var<storage, read_write> outBuffer: array<f32>;

@group(0) @binding(2)
var<uniform> params: Params;

var<workgroup> partial_sums: array<f32, 256>;

@compute @workgroup_size(256)
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>
) {
    let local_id = lid.x;
    let global_id = gid.x;

    // Each thread loads and sums multiple elements
    var sum: f32 = 0.0;
    var idx = global_id;
    while (idx < params.length) {
        sum += selfBuffer[params.self_offset + idx];
        idx += 256u * 256u; // stride by total threads
    }

    partial_sums[local_id] = sum;
    workgroupBarrier();

    // Parallel reduction in workgroup
    for (var stride: u32 = 128u; stride > 0u; stride = stride >> 1u) {
        if (local_id < stride) {
            partial_sums[local_id] += partial_sums[local_id + stride];
        }
        workgroupBarrier();
    }

    // First thread in workgroup writes result
    if (local_id == 0u) {
        // Atomically add to output (simplified - single workgroup for now)
        outBuffer[params.out_offset] = partial_sums[0] / f32(params.length);
    }
}
)wgsl";

            const std::string sum_all_shader = R"wgsl(
struct Params {
    length: u32,
    out_offset: u32,
    self_offset: u32,
    _pad: u32,
};

@group(0) @binding(0)
var<storage, read> selfBuffer: array<f32>;

@group(0) @binding(1)
var<storage, read_write> outBuffer: array<f32>;

@group(0) @binding(2)
var<uniform> params: Params;

var<workgroup> partial_sums: array<f32, 256>;

@compute @workgroup_size(256)
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>
) {
    let local_id = lid.x;
    let global_id = gid.x;

    // Each thread loads and sums multiple elements
    var sum: f32 = 0.0;
    var idx = global_id;
    while (idx < params.length) {
        sum += selfBuffer[params.self_offset + idx];
        idx += 256u * 256u; // stride by total threads
    }

    partial_sums[local_id] = sum;
    workgroupBarrier();

    // Parallel reduction in workgroup
    for (var stride: u32 = 128u; stride > 0u; stride = stride >> 1u) {
        if (local_id < stride) {
            partial_sums[local_id] += partial_sums[local_id + stride];
        }
        workgroupBarrier();
    }

    // First thread in workgroup writes result
    if (local_id == 0u) {
        outBuffer[params.out_offset] = partial_sums[0];
    }
}
)wgsl";

            struct ReductionKernel
            {
                wgpu::BindGroupLayout bind_group_layout;
                wgpu::ComputePipeline pipeline;
            };

            ReductionKernel create_reduction_kernel(const std::string &shader_code, const char *label)
            {
                wgpu::ShaderSourceWGSL shader_source{
                    wgpu::ShaderSourceWGSL::Init{
                        nullptr,
                        wgpu::StringView{shader_code.c_str(), shader_code.size()},
                    }};

                wgpu::ShaderModuleDescriptor shader_descriptor{};
                shader_descriptor.nextInChain = &shader_source;
                shader_descriptor.label = label;
                core::WebGPUContext &ctx = core::getWebGPUContext();
                wgpu::ShaderModule shader_module = ctx.getDevice().CreateShaderModule(&shader_descriptor);

                wgpu::BindGroupLayoutEntry bindings[3]{};

                bindings[0].binding = 0;
                bindings[0].visibility = wgpu::ShaderStage::Compute;
                bindings[0].buffer.type = wgpu::BufferBindingType::ReadOnlyStorage;
                bindings[0].buffer.hasDynamicOffset = false;
                bindings[0].buffer.minBindingSize = 0;

                bindings[1].binding = 1;
                bindings[1].visibility = wgpu::ShaderStage::Compute;
                bindings[1].buffer.type = wgpu::BufferBindingType::Storage;
                bindings[1].buffer.hasDynamicOffset = false;
                bindings[1].buffer.minBindingSize = 0;

                bindings[2].binding = 2;
                bindings[2].visibility = wgpu::ShaderStage::Compute;
                bindings[2].buffer.type = wgpu::BufferBindingType::Uniform;
                bindings[2].buffer.hasDynamicOffset = false;
                bindings[2].buffer.minBindingSize = 0;

                wgpu::BindGroupLayoutDescriptor layout_descriptor{};
                layout_descriptor.entryCount = 3;
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

                return ReductionKernel{bind_group_layout, pipeline};
            }

            void run_reduction_kernel(ReductionKernel &kernel, const at::Tensor &self, at::Tensor &out, bool divide_by_count = false)
            {
                core::WebGPUAllocation *self_allocation = static_cast<core::WebGPUAllocation *>(self.storage().data_ptr().get());
                core::WebGPUAllocation *out_allocation = static_cast<core::WebGPUAllocation *>(out.storage().data_ptr().get());

                wgpu::Buffer self_buffer = self_allocation->buffer;
                wgpu::Buffer out_buffer = out_allocation->buffer;

                struct Params
                {
                    uint32_t length;
                    uint32_t out_offset;
                    uint32_t self_offset;
                    uint32_t _pad;
                };

                Params params{};
                params.length = static_cast<uint32_t>(self.numel());
                params.out_offset = static_cast<uint32_t>(out.storage_offset());
                params.self_offset = static_cast<uint32_t>(self.storage_offset());
                params._pad = 0;

                core::WebGPUContext &ctx = core::getWebGPUContext();

                wgpu::BufferDescriptor uniform_descriptor{};
                uniform_descriptor.label = "Params";
                uniform_descriptor.usage = wgpu::BufferUsage::Uniform | wgpu::BufferUsage::CopyDst;
                uniform_descriptor.size = sizeof(Params);
                uniform_descriptor.mappedAtCreation = false;
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

                // Use single workgroup for now (limited to 256*256 elements for full accuracy)
                pass_encoder.DispatchWorkgroups(1);
                pass_encoder.End();

                wgpu::CommandBuffer command_buffer = encoder.Finish();
                ctx.getQueue().Submit(1, &command_buffer);
            }

            static ReductionKernel *mean_kernel = nullptr;
            static ReductionKernel *sum_kernel = nullptr;

            ReductionKernel &get_mean_kernel()
            {
                if (mean_kernel == nullptr)
                {
                    static ReductionKernel k = create_reduction_kernel(mean_all_shader, "Mean kernel");
                    mean_kernel = &k;
                }
                return *mean_kernel;
            }

            ReductionKernel &get_sum_kernel()
            {
                if (sum_kernel == nullptr)
                {
                    static ReductionKernel k = create_reduction_kernel(sum_all_shader, "Sum kernel");
                    sum_kernel = &k;
                }
                return *sum_kernel;
            }
        }

        // Mean over all elements
        at::Tensor mean(const at::Tensor &self, c10::optional<at::ScalarType> dtype)
        {
            TORCH_CHECK(self.device().type() == c10::DeviceType::PrivateUse1);
            TORCH_CHECK(self.scalar_type() == c10::ScalarType::Float);
            TORCH_CHECK(self.is_contiguous(), "mean requires contiguous tensor for now");

            at::Tensor out = at::empty({}, self.options());
            ReductionKernel &kernel = get_mean_kernel();
            run_reduction_kernel(kernel, self, out, true);
            return out;
        }

        // Sum over all elements
        at::Tensor sum(const at::Tensor &self, c10::optional<at::ScalarType> dtype)
        {
            TORCH_CHECK(self.device().type() == c10::DeviceType::PrivateUse1);
            TORCH_CHECK(self.scalar_type() == c10::ScalarType::Float);
            TORCH_CHECK(self.is_contiguous(), "sum requires contiguous tensor for now");

            at::Tensor out = at::empty({}, self.options());
            ReductionKernel &kernel = get_sum_kernel();
            run_reduction_kernel(kernel, self, out, false);
            return out;
        }

        // Sum with dims
        at::Tensor sum_dim(const at::Tensor &self, at::OptionalIntArrayRef dim, bool keepdim, c10::optional<at::ScalarType> dtype)
        {
            // For now, if no dims specified or all dims, reduce to scalar
            if (!dim.has_value() || dim->empty())
            {
                auto result = torch_webgpu::ops::sum(self, dtype);
                if (keepdim)
                {
                    std::vector<int64_t> new_shape(self.dim(), 1);
                    return result.reshape(new_shape);
                }
                return result;
            }

            // TODO: Implement dim-specific reduction
            // For now, fall back to moving to CPU, doing the operation, and moving back
            TORCH_CHECK(false, "sum with specific dims not yet implemented on WebGPU - use contiguous reduction");
            return self;
        }

        // Mean with dims
        at::Tensor mean_dim(const at::Tensor &self, at::OptionalIntArrayRef dim, bool keepdim, c10::optional<at::ScalarType> dtype)
        {
            // For now, if no dims specified or all dims, reduce to scalar
            if (!dim.has_value() || dim->empty())
            {
                auto result = torch_webgpu::ops::mean(self, dtype);
                if (keepdim)
                {
                    std::vector<int64_t> new_shape(self.dim(), 1);
                    return result.reshape(new_shape);
                }
                return result;
            }

            // TODO: Implement dim-specific reduction
            TORCH_CHECK(false, "mean with specific dims not yet implemented on WebGPU - use contiguous reduction");
            return self;
        }
    }

    TORCH_LIBRARY_IMPL(aten, PrivateUse1, m)
    {
        m.impl("mean", TORCH_FN(ops::mean));
        m.impl("sum", TORCH_FN(ops::sum));
        m.impl("sum.dim_IntList", TORCH_FN(ops::sum_dim));
        m.impl("mean.dim", TORCH_FN(ops::mean_dim));
    }
}
