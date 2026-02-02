/**
 * Fused Q, K, V Projection for WebGPU
 *
 * Fuses three linear projections into single dispatch:
 *   Q = x @ W_q.T, K = x @ W_k.T, V = x @ W_v.T
 */

#include <ATen/ATen.h>
#include <torch/library.h>
#include <webgpu/webgpu_cpp.h>
#include "core/webgpu_context.h"
#include "core/webgpu_allocator.h"
#include "core/command_batcher.h"
#include "core/bind_group_cache.h"

namespace torch_webgpu
{
    namespace ops
    {
        namespace
        {
            const std::string fused_qkv_shader = R"wgsl(
struct Params {
    M: u32,
    K: u32,
    N: u32,
    x_offset: u32,
    q_weight_offset: u32,
    k_weight_offset: u32,
    v_weight_offset: u32,
    q_out_offset: u32,
    k_out_offset: u32,
    v_out_offset: u32,
    _pad1: u32,
    _pad2: u32,
};

@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> q_weight: array<f32>;
@group(0) @binding(2) var<storage, read> k_weight: array<f32>;
@group(0) @binding(3) var<storage, read> v_weight: array<f32>;
@group(0) @binding(4) var<storage, read_write> q_out: array<f32>;
@group(0) @binding(5) var<storage, read_write> k_out: array<f32>;
@group(0) @binding(6) var<storage, read_write> v_out: array<f32>;
@group(0) @binding(7) var<uniform> params: Params;

const TILE_SIZE: u32 = 16u;

var<workgroup> tile_x: array<f32, 256>;
var<workgroup> tile_q_w: array<f32, 256>;
var<workgroup> tile_k_w: array<f32, 256>;
var<workgroup> tile_v_w: array<f32, 256>;

@compute @workgroup_size(16, 16)
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>
) {
    let row = gid.y;
    let col = gid.x;

    if (row >= params.M || col >= params.N) {
        return;
    }

    var q_acc: f32 = 0.0;
    var k_acc: f32 = 0.0;
    var v_acc: f32 = 0.0;

    let num_tiles = (params.K + TILE_SIZE - 1u) / TILE_SIZE;

    for (var t = 0u; t < num_tiles; t++) {
        let tile_k_idx = t * TILE_SIZE + lid.x;

        if (row < params.M && tile_k_idx < params.K) {
            tile_x[lid.y * TILE_SIZE + lid.x] = x[params.x_offset + row * params.K + tile_k_idx];
        } else {
            tile_x[lid.y * TILE_SIZE + lid.x] = 0.0;
        }

        let weight_k = t * TILE_SIZE + lid.y;
        if (weight_k < params.K && col < params.N) {
            tile_q_w[lid.y * TILE_SIZE + lid.x] = q_weight[params.q_weight_offset + weight_k * params.N + col];
            tile_k_w[lid.y * TILE_SIZE + lid.x] = k_weight[params.k_weight_offset + weight_k * params.N + col];
            tile_v_w[lid.y * TILE_SIZE + lid.x] = v_weight[params.v_weight_offset + weight_k * params.N + col];
        } else {
            tile_q_w[lid.y * TILE_SIZE + lid.x] = 0.0;
            tile_k_w[lid.y * TILE_SIZE + lid.x] = 0.0;
            tile_v_w[lid.y * TILE_SIZE + lid.x] = 0.0;
        }

        workgroupBarrier();

        for (var k = 0u; k < TILE_SIZE; k++) {
            let x_val = tile_x[lid.y * TILE_SIZE + k];
            q_acc += x_val * tile_q_w[k * TILE_SIZE + lid.x];
            k_acc += x_val * tile_k_w[k * TILE_SIZE + lid.x];
            v_acc += x_val * tile_v_w[k * TILE_SIZE + lid.x];
        }

        workgroupBarrier();
    }

    if (row < params.M && col < params.N) {
        q_out[params.q_out_offset + row * params.N + col] = q_acc;
        k_out[params.k_out_offset + row * params.N + col] = k_acc;
        v_out[params.v_out_offset + row * params.N + col] = v_acc;
    }
}
)wgsl";

            struct FusedQKVKernel
            {
                wgpu::BindGroupLayout bind_group_layout;
                wgpu::ComputePipeline pipeline;
            };

            static FusedQKVKernel *fused_qkv_kernel = nullptr;

            FusedQKVKernel create_fused_qkv_kernel()
            {
                wgpu::ShaderSourceWGSL shader_source{
                    wgpu::ShaderSourceWGSL::Init{
                        nullptr,
                        wgpu::StringView{fused_qkv_shader.c_str(), fused_qkv_shader.size()},
                    }};

                wgpu::ShaderModuleDescriptor shader_descriptor{};
                shader_descriptor.nextInChain = &shader_source;
                shader_descriptor.label = "fused_qkv";

                core::WebGPUContext &ctx = core::getWebGPUContext();
                wgpu::ShaderModule shader_module = ctx.getDevice().CreateShaderModule(&shader_descriptor);

                wgpu::BindGroupLayoutEntry bindings[8]{};

                for (int i = 0; i < 4; i++)
                {
                    bindings[i].binding = i;
                    bindings[i].visibility = wgpu::ShaderStage::Compute;
                    bindings[i].buffer.type = wgpu::BufferBindingType::ReadOnlyStorage;
                    bindings[i].buffer.hasDynamicOffset = false;
                    bindings[i].buffer.minBindingSize = 0;
                }

                for (int i = 4; i < 7; i++)
                {
                    bindings[i].binding = i;
                    bindings[i].visibility = wgpu::ShaderStage::Compute;
                    bindings[i].buffer.type = wgpu::BufferBindingType::Storage;
                    bindings[i].buffer.hasDynamicOffset = false;
                    bindings[i].buffer.minBindingSize = 0;
                }

                bindings[7].binding = 7;
                bindings[7].visibility = wgpu::ShaderStage::Compute;
                bindings[7].buffer.type = wgpu::BufferBindingType::Uniform;
                bindings[7].buffer.hasDynamicOffset = false;
                bindings[7].buffer.minBindingSize = 0;

                wgpu::BindGroupLayoutDescriptor layout_descriptor{};
                layout_descriptor.entryCount = 8;
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

                return FusedQKVKernel{bind_group_layout, pipeline};
            }

            FusedQKVKernel &get_fused_qkv_kernel()
            {
                if (fused_qkv_kernel != nullptr)
                {
                    return *fused_qkv_kernel;
                }
                static FusedQKVKernel k = create_fused_qkv_kernel();
                fused_qkv_kernel = &k;
                return *fused_qkv_kernel;
            }
        }

        std::tuple<at::Tensor, at::Tensor, at::Tensor> fused_qkv_proj(
            const at::Tensor &x,
            const at::Tensor &q_weight,
            const at::Tensor &k_weight,
            const at::Tensor &v_weight)
        {
            TORCH_CHECK(x.device().type() == c10::DeviceType::PrivateUse1);
            TORCH_CHECK(q_weight.device().type() == c10::DeviceType::PrivateUse1);
            TORCH_CHECK(k_weight.device().type() == c10::DeviceType::PrivateUse1);
            TORCH_CHECK(v_weight.device().type() == c10::DeviceType::PrivateUse1);

            auto x_contig = x.contiguous();
            auto q_contig = q_weight.contiguous();
            auto k_contig = k_weight.contiguous();
            auto v_contig = v_weight.contiguous();

            auto x_2d = x_contig.dim() == 3 ? x_contig.view({-1, x_contig.size(-1)}) : x_contig;
            auto M = x_2d.size(0);
            auto K = x_2d.size(1);
            auto N = q_contig.size(0);

            TORCH_CHECK(q_contig.size(1) == K);
            TORCH_CHECK(k_contig.size(0) == N && k_contig.size(1) == K);
            TORCH_CHECK(v_contig.size(0) == N && v_contig.size(1) == K);

            at::Tensor q_out = at::empty({M, N}, x.options());
            at::Tensor k_out = at::empty({M, N}, x.options());
            at::Tensor v_out = at::empty({M, N}, x.options());

            if (M == 0 || N == 0)
            {
                if (x.dim() == 3)
                {
                    return std::make_tuple(
                        q_out.view({x.size(0), x.size(1), N}),
                        k_out.view({x.size(0), x.size(1), N}),
                        v_out.view({x.size(0), x.size(1), N}));
                }
                return std::make_tuple(q_out, k_out, v_out);
            }

            FusedQKVKernel &kernel = get_fused_qkv_kernel();

            core::WebGPUAllocation *x_alloc = static_cast<core::WebGPUAllocation *>(x_2d.storage().data_ptr().get());
            core::WebGPUAllocation *q_w_alloc = static_cast<core::WebGPUAllocation *>(q_contig.storage().data_ptr().get());
            core::WebGPUAllocation *k_w_alloc = static_cast<core::WebGPUAllocation *>(k_contig.storage().data_ptr().get());
            core::WebGPUAllocation *v_w_alloc = static_cast<core::WebGPUAllocation *>(v_contig.storage().data_ptr().get());
            core::WebGPUAllocation *q_out_alloc = static_cast<core::WebGPUAllocation *>(q_out.storage().data_ptr().get());
            core::WebGPUAllocation *k_out_alloc = static_cast<core::WebGPUAllocation *>(k_out.storage().data_ptr().get());
            core::WebGPUAllocation *v_out_alloc = static_cast<core::WebGPUAllocation *>(v_out.storage().data_ptr().get());

            struct Params
            {
                uint32_t M, K, N;
                uint32_t x_offset, q_weight_offset, k_weight_offset, v_weight_offset;
                uint32_t q_out_offset, k_out_offset, v_out_offset;
                uint32_t _pad1, _pad2;
            };

            Params params{};
            params.M = static_cast<uint32_t>(M);
            params.K = static_cast<uint32_t>(K);
            params.N = static_cast<uint32_t>(N);
            params.x_offset = static_cast<uint32_t>(x_2d.storage_offset());
            params.q_weight_offset = static_cast<uint32_t>(q_contig.storage_offset());
            params.k_weight_offset = static_cast<uint32_t>(k_contig.storage_offset());
            params.v_weight_offset = static_cast<uint32_t>(v_contig.storage_offset());
            params.q_out_offset = static_cast<uint32_t>(q_out.storage_offset());
            params.k_out_offset = static_cast<uint32_t>(k_out.storage_offset());
            params.v_out_offset = static_cast<uint32_t>(v_out.storage_offset());
            params._pad1 = 0;
            params._pad2 = 0;

            core::WebGPUContext &ctx = core::getWebGPUContext();
            wgpu::Buffer params_buffer = core::acquireUniformBuffer(&params, sizeof(Params));

            wgpu::BindGroupEntry entries[8]{};
            entries[0] = {nullptr, 0, x_alloc->buffer, 0, x_alloc->buffer.GetSize()};
            entries[1] = {nullptr, 1, q_w_alloc->buffer, 0, q_w_alloc->buffer.GetSize()};
            entries[2] = {nullptr, 2, k_w_alloc->buffer, 0, k_w_alloc->buffer.GetSize()};
            entries[3] = {nullptr, 3, v_w_alloc->buffer, 0, v_w_alloc->buffer.GetSize()};
            entries[4] = {nullptr, 4, q_out_alloc->buffer, 0, q_out_alloc->buffer.GetSize()};
            entries[5] = {nullptr, 5, k_out_alloc->buffer, 0, k_out_alloc->buffer.GetSize()};
            entries[6] = {nullptr, 6, v_out_alloc->buffer, 0, v_out_alloc->buffer.GetSize()};
            entries[7] = {nullptr, 7, params_buffer, 0, sizeof(Params)};

            wgpu::BindGroupDescriptor bind_group_descriptor{};
            bind_group_descriptor.layout = kernel.bind_group_layout;
            bind_group_descriptor.entryCount = 8;
            bind_group_descriptor.entries = entries;

            wgpu::BindGroup bind_group = ctx.getDevice().CreateBindGroup(&bind_group_descriptor);

            const uint32_t TILE_SIZE = 16;
            uint32_t num_wg_x = (static_cast<uint32_t>(N) + TILE_SIZE - 1) / TILE_SIZE;
            uint32_t num_wg_y = (static_cast<uint32_t>(M) + TILE_SIZE - 1) / TILE_SIZE;

            core::dispatchCompute(kernel.pipeline, bind_group, num_wg_x, num_wg_y, 1);

            if (x.dim() == 3)
            {
                return std::make_tuple(
                    q_out.view({x.size(0), x.size(1), N}),
                    k_out.view({x.size(0), x.size(1), N}),
                    v_out.view({x.size(0), x.size(1), N}));
            }
            return std::make_tuple(q_out, k_out, v_out);
        }
    }

    TORCH_LIBRARY_FRAGMENT(webgpu, m)
    {
        m.def("fused_qkv_proj(Tensor x, Tensor q_weight, Tensor k_weight, Tensor v_weight) -> (Tensor, Tensor, Tensor)");
    }

    TORCH_LIBRARY_IMPL(webgpu, PrivateUse1, m)
    {
        m.impl("fused_qkv_proj", TORCH_FN(ops::fused_qkv_proj));
    }
}
