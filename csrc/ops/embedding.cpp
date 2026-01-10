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
            const std::string embedding_shader = R"wgsl(
struct Params {
    num_indices: u32,
    embedding_dim: u32,
    weight_offset: u32,
    indices_offset: u32,
    out_offset: u32,
    weight_stride0: u32,
    weight_stride1: u32,
    _pad: u32,
};

@group(0) @binding(0)
var<storage, read> weightBuffer: array<f32>;

@group(0) @binding(1)
var<storage, read> indicesBuffer: array<i32>;

@group(0) @binding(2)
var<storage, read_write> outBuffer: array<f32>;

@group(0) @binding(3)
var<uniform> params: Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    let total_elements = params.num_indices * params.embedding_dim;
    if (i >= total_elements) { return; }

    // Calculate which index and which element within embedding
    let idx_pos = i / params.embedding_dim;
    let emb_pos = i % params.embedding_dim;

    // Get the vocabulary index
    let vocab_idx = u32(indicesBuffer[params.indices_offset + idx_pos]);

    // Read from weight matrix
    let weight_idx = params.weight_offset + vocab_idx * params.weight_stride0 + emb_pos * params.weight_stride1;
    let value = weightBuffer[weight_idx];

    // Write to output
    outBuffer[params.out_offset + i] = value;
}
)wgsl";

            struct EmbeddingKernel
            {
                wgpu::BindGroupLayout bind_group_layout;
                wgpu::ComputePipeline pipeline;
            };

            static EmbeddingKernel *embedding_kernel = nullptr;

            EmbeddingKernel &get_embedding_kernel()
            {
                if (embedding_kernel != nullptr)
                {
                    return *embedding_kernel;
                }

                wgpu::ShaderSourceWGSL shader_source{
                    wgpu::ShaderSourceWGSL::Init{
                        nullptr,
                        wgpu::StringView{embedding_shader.c_str(), embedding_shader.size()},
                    }};

                wgpu::ShaderModuleDescriptor shader_descriptor{};
                shader_descriptor.nextInChain = &shader_source;
                shader_descriptor.label = "Embedding kernel";
                core::WebGPUContext &ctx = core::getWebGPUContext();
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

                static EmbeddingKernel k{bind_group_layout, pipeline};
                embedding_kernel = &k;
                return *embedding_kernel;
            }
        }

        at::Tensor embedding(
            const at::Tensor &weight,
            const at::Tensor &indices,
            c10::SymInt padding_idx,
            bool scale_grad_by_freq,
            bool sparse)
        {
            TORCH_CHECK(weight.device().type() == c10::DeviceType::PrivateUse1);
            TORCH_CHECK(indices.device().type() == c10::DeviceType::PrivateUse1);
            TORCH_CHECK(weight.dim() == 2, "embedding weight must be 2D");
            TORCH_CHECK(weight.scalar_type() == c10::ScalarType::Float, "embedding weight must be float32");
            TORCH_CHECK(indices.scalar_type() == c10::ScalarType::Int || indices.scalar_type() == c10::ScalarType::Long,
                        "indices must be int32 or int64");

            auto num_embeddings = weight.size(0);
            auto embedding_dim = weight.size(1);
            auto num_indices = indices.numel();

            // Output shape: indices.shape + [embedding_dim]
            std::vector<int64_t> out_shape(indices.sizes().begin(), indices.sizes().end());
            out_shape.push_back(embedding_dim);

            at::Tensor out = at::empty(out_shape, weight.options());

            if (num_indices == 0)
            {
                return out;
            }

            // Convert indices to int32 if needed
            at::Tensor indices_int32 = indices;
            if (indices.scalar_type() == c10::ScalarType::Long)
            {
                indices_int32 = indices.to(c10::ScalarType::Int);
            }
            indices_int32 = indices_int32.contiguous();

            EmbeddingKernel &kernel = get_embedding_kernel();

            core::WebGPUAllocation *weight_allocation = static_cast<core::WebGPUAllocation *>(weight.storage().data_ptr().get());
            core::WebGPUAllocation *indices_allocation = static_cast<core::WebGPUAllocation *>(indices_int32.storage().data_ptr().get());
            core::WebGPUAllocation *out_allocation = static_cast<core::WebGPUAllocation *>(out.storage().data_ptr().get());

            wgpu::Buffer weight_buffer = weight_allocation->buffer;
            wgpu::Buffer indices_buffer = indices_allocation->buffer;
            wgpu::Buffer out_buffer = out_allocation->buffer;

            struct Params
            {
                uint32_t num_indices;
                uint32_t embedding_dim;
                uint32_t weight_offset;
                uint32_t indices_offset;
                uint32_t out_offset;
                uint32_t weight_stride0;
                uint32_t weight_stride1;
                uint32_t _pad;
            };

            Params params{};
            params.num_indices = static_cast<uint32_t>(num_indices);
            params.embedding_dim = static_cast<uint32_t>(embedding_dim);
            params.weight_offset = static_cast<uint32_t>(weight.storage_offset());
            params.indices_offset = static_cast<uint32_t>(indices_int32.storage_offset());
            params.out_offset = static_cast<uint32_t>(out.storage_offset());
            params.weight_stride0 = static_cast<uint32_t>(weight.stride(0));
            params.weight_stride1 = static_cast<uint32_t>(weight.stride(1));
            params._pad = 0;

            core::WebGPUContext &ctx = core::getWebGPUContext();

            wgpu::BufferDescriptor uniform_descriptor{};
            uniform_descriptor.label = "Params";
            uniform_descriptor.usage = wgpu::BufferUsage::Uniform | wgpu::BufferUsage::CopyDst;
            uniform_descriptor.size = sizeof(Params);
            uniform_descriptor.mappedAtCreation = false;
            wgpu::Buffer params_buffer = ctx.getDevice().CreateBuffer(&uniform_descriptor);
            ctx.getQueue().WriteBuffer(params_buffer, 0, &params, sizeof(Params));

            wgpu::BindGroupEntry bind_group_entries[4]{};
            bind_group_entries[0].binding = 0;
            bind_group_entries[0].buffer = weight_buffer;
            bind_group_entries[0].offset = 0;
            bind_group_entries[0].size = weight_buffer.GetSize();

            bind_group_entries[1].binding = 1;
            bind_group_entries[1].buffer = indices_buffer;
            bind_group_entries[1].offset = 0;
            bind_group_entries[1].size = indices_buffer.GetSize();

            bind_group_entries[2].binding = 2;
            bind_group_entries[2].buffer = out_buffer;
            bind_group_entries[2].offset = 0;
            bind_group_entries[2].size = out_buffer.GetSize();

            bind_group_entries[3].binding = 3;
            bind_group_entries[3].buffer = params_buffer;
            bind_group_entries[3].offset = 0;
            bind_group_entries[3].size = sizeof(Params);

            wgpu::BindGroupDescriptor bind_group_descriptor{};
            bind_group_descriptor.layout = kernel.bind_group_layout;
            bind_group_descriptor.entryCount = 4;
            bind_group_descriptor.entries = bind_group_entries;

            wgpu::BindGroup bind_group = ctx.getDevice().CreateBindGroup(&bind_group_descriptor);

            wgpu::CommandEncoder encoder = ctx.getDevice().CreateCommandEncoder();
            wgpu::ComputePassDescriptor pass_descriptor;
            wgpu::ComputePassEncoder pass_encoder = encoder.BeginComputePass(&pass_descriptor);
            pass_encoder.SetPipeline(kernel.pipeline);
            pass_encoder.SetBindGroup(0, bind_group);

            const uint32_t workgroup_size = 64;
            uint32_t total_elements = params.num_indices * params.embedding_dim;
            uint32_t num_workgroups = (total_elements + workgroup_size - 1) / workgroup_size;

            pass_encoder.DispatchWorkgroups(num_workgroups);
            pass_encoder.End();

            wgpu::CommandBuffer command_buffer = encoder.Finish();
            ctx.getQueue().Submit(1, &command_buffer);

            return out;
        }
    }

    TORCH_LIBRARY_IMPL(aten, PrivateUse1, m)
    {
        m.impl("embedding", TORCH_FN(ops::embedding));
    }
}
