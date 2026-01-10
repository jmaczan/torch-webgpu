#include <ATen/ATen.h>
#include <torch/library.h>
#include <webgpu/webgpu_cpp.h>
#include "core/webgpu_context.h"
#include "core/webgpu_allocator.h"

namespace torch_webgpu
{
    namespace ops
    {
        namespace
        {
            // Softmax shader - operates on the last dimension
            // For simplicity, works with contiguous tensors on last dim
            const std::string softmax_shader = R"wgsl(
struct Params {
    batch_size: u32,
    dim_size: u32,
    self_offset: u32,
    out_offset: u32,
};

@group(0) @binding(0)
var<storage, read> selfBuffer: array<f32>;

@group(0) @binding(1)
var<storage, read_write> outBuffer: array<f32>;

@group(0) @binding(2)
var<uniform> params: Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let batch_idx = gid.x;
    if (batch_idx >= params.batch_size) { return; }

    let base_idx = params.self_offset + batch_idx * params.dim_size;
    let out_base_idx = params.out_offset + batch_idx * params.dim_size;

    // Find max for numerical stability
    var max_val: f32 = selfBuffer[base_idx];
    for (var i: u32 = 1u; i < params.dim_size; i++) {
        let val = selfBuffer[base_idx + i];
        max_val = max(max_val, val);
    }

    // Compute exp and sum
    var exp_sum: f32 = 0.0;
    for (var i: u32 = 0u; i < params.dim_size; i++) {
        let exp_val = exp(selfBuffer[base_idx + i] - max_val);
        outBuffer[out_base_idx + i] = exp_val;
        exp_sum += exp_val;
    }

    // Normalize
    let inv_sum = 1.0 / exp_sum;
    for (var i: u32 = 0u; i < params.dim_size; i++) {
        outBuffer[out_base_idx + i] *= inv_sum;
    }
}
)wgsl";

            struct SoftmaxKernel
            {
                wgpu::BindGroupLayout bind_group_layout;
                wgpu::ComputePipeline pipeline;
            };

            static SoftmaxKernel *softmax_kernel = nullptr;

            SoftmaxKernel &get_softmax_kernel()
            {
                if (softmax_kernel != nullptr)
                {
                    return *softmax_kernel;
                }

                wgpu::ShaderSourceWGSL shader_source{
                    wgpu::ShaderSourceWGSL::Init{
                        nullptr,
                        wgpu::StringView{softmax_shader.c_str(), softmax_shader.size()},
                    }};

                wgpu::ShaderModuleDescriptor shader_descriptor{};
                shader_descriptor.nextInChain = &shader_source;
                shader_descriptor.label = "Softmax kernel";
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

                static SoftmaxKernel k{bind_group_layout, pipeline};
                softmax_kernel = &k;
                return *softmax_kernel;
            }
        }

        at::Tensor softmax(const at::Tensor &self, int64_t dim, c10::optional<at::ScalarType> dtype)
        {
            TORCH_CHECK(self.device().type() == c10::DeviceType::PrivateUse1);
            TORCH_CHECK(self.scalar_type() == c10::ScalarType::Float);

            auto ndim = self.dim();
            if (dim < 0)
                dim += ndim;

            TORCH_CHECK(dim >= 0 && dim < ndim, "softmax: dimension out of range");

            // For now, only support softmax on last dimension
            if (dim != ndim - 1)
            {
                // Move dim to last position, apply softmax, move back
                std::vector<int64_t> perm;
                for (int64_t i = 0; i < ndim; ++i)
                {
                    if (i != dim)
                        perm.push_back(i);
                }
                perm.push_back(dim);

                auto permuted = self.permute(perm).contiguous();
                auto result = torch_webgpu::ops::softmax(permuted, -1, dtype);

                // Inverse permutation
                std::vector<int64_t> inv_perm(ndim);
                for (int64_t i = 0; i < ndim; ++i)
                {
                    inv_perm[perm[i]] = i;
                }

                return result.permute(inv_perm).contiguous();
            }

            // Flatten to 2D: [batch, dim_size]
            auto input_contig = self.contiguous();
            auto dim_size = input_contig.size(-1);
            int64_t batch_size = input_contig.numel() / dim_size;

            at::Tensor out = at::empty_like(input_contig);

            if (batch_size == 0)
            {
                return out;
            }

            SoftmaxKernel &kernel = get_softmax_kernel();

            core::WebGPUAllocation *self_allocation = static_cast<core::WebGPUAllocation *>(input_contig.storage().data_ptr().get());
            core::WebGPUAllocation *out_allocation = static_cast<core::WebGPUAllocation *>(out.storage().data_ptr().get());

            wgpu::Buffer self_buffer = self_allocation->buffer;
            wgpu::Buffer out_buffer = out_allocation->buffer;

            struct Params
            {
                uint32_t batch_size;
                uint32_t dim_size;
                uint32_t self_offset;
                uint32_t out_offset;
            };

            Params params{};
            params.batch_size = static_cast<uint32_t>(batch_size);
            params.dim_size = static_cast<uint32_t>(dim_size);
            params.self_offset = static_cast<uint32_t>(input_contig.storage_offset());
            params.out_offset = static_cast<uint32_t>(out.storage_offset());

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

            const uint32_t workgroup_size = 64;
            uint32_t num_workgroups = (params.batch_size + workgroup_size - 1) / workgroup_size;

            pass_encoder.DispatchWorkgroups(num_workgroups);
            pass_encoder.End();

            wgpu::CommandBuffer command_buffer = encoder.Finish();
            ctx.getQueue().Submit(1, &command_buffer);

            return out;
        }

        at::Tensor log_softmax(const at::Tensor &self, int64_t dim, c10::optional<at::ScalarType> dtype)
        {
            return at::log(torch_webgpu::ops::softmax(self, dim, dtype));
        }

        at::Tensor &softmax_out(const at::Tensor &self, int64_t dim, c10::optional<at::ScalarType> dtype, at::Tensor &out)
        {
            auto result = torch_webgpu::ops::softmax(self, dim, dtype);
            out.copy_(result);
            return out;
        }
    }

    TORCH_LIBRARY_IMPL(aten, PrivateUse1, m)
    {
        m.impl("softmax.int", TORCH_FN(ops::softmax));
        m.impl("softmax.int_out", TORCH_FN(ops::softmax_out));
        m.impl("log_softmax.int", TORCH_FN(ops::log_softmax));
    }
}
