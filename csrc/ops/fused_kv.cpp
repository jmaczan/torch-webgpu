/**
 * Fused K, V Projection for WebGPU
 *
 * For Grouped Query Attention where K and V have the same dimensions.
 * Fuses two linear projections into single dispatch:
 *   K = x @ W_k.T, V = x @ W_v.T
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
            const std::string fused_kv_shader = R"wgsl(
struct Params {
    M: u32,           // batch * seq_len
    K: u32,           // hidden_size (input dim)
    N: u32,           // kv_hidden_size (output dim)
    x_offset: u32,
    k_weight_offset: u32,
    v_weight_offset: u32,
    k_out_offset: u32,
    v_out_offset: u32,
};

@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> k_weight: array<f32>;
@group(0) @binding(2) var<storage, read> v_weight: array<f32>;
@group(0) @binding(3) var<storage, read_write> k_out: array<f32>;
@group(0) @binding(4) var<storage, read_write> v_out: array<f32>;
@group(0) @binding(5) var<uniform> params: Params;

const TILE_SIZE: u32 = 16u;

var<workgroup> tile_x: array<f32, 256>;
var<workgroup> tile_k_w: array<f32, 256>;
var<workgroup> tile_v_w: array<f32, 256>;

@compute @workgroup_size(16, 16)
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>
) {
    let row = gid.y;
    let col = gid.x;

    // Don't early return - all threads must participate in barriers
    let valid = row < params.M && col < params.N;

    var k_acc: f32 = 0.0;
    var v_acc: f32 = 0.0;

    let num_tiles = (params.K + TILE_SIZE - 1u) / TILE_SIZE;

    for (var t = 0u; t < num_tiles; t++) {
        let tile_k_idx = t * TILE_SIZE + lid.x;

        // Load x tile - use safe bounds
        let x_row = min(row, params.M - 1u);
        if (tile_k_idx < params.K) {
            tile_x[lid.y * TILE_SIZE + lid.x] = x[params.x_offset + x_row * params.K + tile_k_idx];
        } else {
            tile_x[lid.y * TILE_SIZE + lid.x] = 0.0;
        }

        // Load weight tiles - weights are [N, K], we need W[n, k] for W.T
        let weight_k = t * TILE_SIZE + lid.y;
        let safe_col = min(col, params.N - 1u);
        if (weight_k < params.K) {
            tile_k_w[lid.y * TILE_SIZE + lid.x] = k_weight[params.k_weight_offset + safe_col * params.K + weight_k];
            tile_v_w[lid.y * TILE_SIZE + lid.x] = v_weight[params.v_weight_offset + safe_col * params.K + weight_k];
        } else {
            tile_k_w[lid.y * TILE_SIZE + lid.x] = 0.0;
            tile_v_w[lid.y * TILE_SIZE + lid.x] = 0.0;
        }

        workgroupBarrier();

        // Only accumulate if this thread is valid
        if (valid) {
            for (var k = 0u; k < TILE_SIZE; k++) {
                let x_val = tile_x[lid.y * TILE_SIZE + k];
                k_acc += x_val * tile_k_w[k * TILE_SIZE + lid.x];
                v_acc += x_val * tile_v_w[k * TILE_SIZE + lid.x];
            }
        }

        workgroupBarrier();
    }

    if (valid) {
        k_out[params.k_out_offset + row * params.N + col] = k_acc;
        v_out[params.v_out_offset + row * params.N + col] = v_acc;
    }
}
)wgsl";

            struct FusedKVKernel
            {
                wgpu::BindGroupLayout bind_group_layout;
                wgpu::ComputePipeline pipeline;
            };

            static FusedKVKernel *fused_kv_kernel = nullptr;

            FusedKVKernel create_fused_kv_kernel()
            {
                wgpu::ShaderSourceWGSL shader_source{
                    wgpu::ShaderSourceWGSL::Init{
                        nullptr,
                        wgpu::StringView{fused_kv_shader.c_str(), fused_kv_shader.size()},
                    }};

                wgpu::ShaderModuleDescriptor shader_descriptor{};
                shader_descriptor.nextInChain = &shader_source;
                shader_descriptor.label = "fused_kv";

                core::WebGPUContext &ctx = core::getWebGPUContext();
                wgpu::ShaderModule shader_module = ctx.getDevice().CreateShaderModule(&shader_descriptor);

                wgpu::BindGroupLayoutEntry bindings[6]{};

                // x buffer (read-only)
                bindings[0].binding = 0;
                bindings[0].visibility = wgpu::ShaderStage::Compute;
                bindings[0].buffer.type = wgpu::BufferBindingType::ReadOnlyStorage;
                bindings[0].buffer.hasDynamicOffset = false;
                bindings[0].buffer.minBindingSize = 0;

                // k_weight buffer (read-only)
                bindings[1].binding = 1;
                bindings[1].visibility = wgpu::ShaderStage::Compute;
                bindings[1].buffer.type = wgpu::BufferBindingType::ReadOnlyStorage;
                bindings[1].buffer.hasDynamicOffset = false;
                bindings[1].buffer.minBindingSize = 0;

                // v_weight buffer (read-only)
                bindings[2].binding = 2;
                bindings[2].visibility = wgpu::ShaderStage::Compute;
                bindings[2].buffer.type = wgpu::BufferBindingType::ReadOnlyStorage;
                bindings[2].buffer.hasDynamicOffset = false;
                bindings[2].buffer.minBindingSize = 0;

                // k_out buffer (read-write)
                bindings[3].binding = 3;
                bindings[3].visibility = wgpu::ShaderStage::Compute;
                bindings[3].buffer.type = wgpu::BufferBindingType::Storage;
                bindings[3].buffer.hasDynamicOffset = false;
                bindings[3].buffer.minBindingSize = 0;

                // v_out buffer (read-write)
                bindings[4].binding = 4;
                bindings[4].visibility = wgpu::ShaderStage::Compute;
                bindings[4].buffer.type = wgpu::BufferBindingType::Storage;
                bindings[4].buffer.hasDynamicOffset = false;
                bindings[4].buffer.minBindingSize = 0;

                // params buffer (uniform)
                bindings[5].binding = 5;
                bindings[5].visibility = wgpu::ShaderStage::Compute;
                bindings[5].buffer.type = wgpu::BufferBindingType::Uniform;
                bindings[5].buffer.hasDynamicOffset = false;
                bindings[5].buffer.minBindingSize = 0;

                wgpu::BindGroupLayoutDescriptor layout_descriptor{};
                layout_descriptor.entryCount = 6;
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

                return FusedKVKernel{bind_group_layout, pipeline};
            }

            FusedKVKernel &get_fused_kv_kernel()
            {
                if (fused_kv_kernel != nullptr)
                {
                    return *fused_kv_kernel;
                }
                static FusedKVKernel k = create_fused_kv_kernel();
                fused_kv_kernel = &k;
                return *fused_kv_kernel;
            }
        }

        std::tuple<at::Tensor, at::Tensor> fused_kv_proj(
            const at::Tensor &x,
            const at::Tensor &k_weight,
            const at::Tensor &v_weight)
        {
            TORCH_CHECK(x.device().type() == c10::DeviceType::PrivateUse1);
            TORCH_CHECK(k_weight.device().type() == c10::DeviceType::PrivateUse1);
            TORCH_CHECK(v_weight.device().type() == c10::DeviceType::PrivateUse1);
            TORCH_CHECK(x.scalar_type() == c10::ScalarType::Float);

            auto x_contig = x.contiguous();
            auto k_contig = k_weight.contiguous();
            auto v_contig = v_weight.contiguous();

            // x: [batch, seq_len, hidden_size] or [batch * seq_len, hidden_size]
            auto x_2d = x_contig.dim() == 3 ? x_contig.view({-1, x_contig.size(-1)}) : x_contig;
            auto M = x_2d.size(0);
            auto K = x_2d.size(1);
            auto N = k_contig.size(0);

            TORCH_CHECK(k_contig.size(1) == K, "k_weight K dimension mismatch");
            TORCH_CHECK(v_contig.size(0) == N && v_contig.size(1) == K, "v_weight dimension mismatch");

            at::Tensor k_output = at::empty({M, N}, x.options());
            at::Tensor v_output = at::empty({M, N}, x.options());

            if (M == 0 || N == 0)
            {
                if (x.dim() == 3)
                {
                    return std::make_tuple(
                        k_output.view({x.size(0), x.size(1), N}),
                        v_output.view({x.size(0), x.size(1), N}));
                }
                return std::make_tuple(k_output, v_output);
            }

            FusedKVKernel &kernel = get_fused_kv_kernel();

            core::WebGPUAllocation *x_alloc = static_cast<core::WebGPUAllocation *>(x_2d.storage().data_ptr().get());
            core::WebGPUAllocation *k_w_alloc = static_cast<core::WebGPUAllocation *>(k_contig.storage().data_ptr().get());
            core::WebGPUAllocation *v_w_alloc = static_cast<core::WebGPUAllocation *>(v_contig.storage().data_ptr().get());
            core::WebGPUAllocation *k_out_alloc = static_cast<core::WebGPUAllocation *>(k_output.storage().data_ptr().get());
            core::WebGPUAllocation *v_out_alloc = static_cast<core::WebGPUAllocation *>(v_output.storage().data_ptr().get());

            wgpu::Buffer x_buffer = x_alloc->buffer;
            wgpu::Buffer k_w_buffer = k_w_alloc->buffer;
            wgpu::Buffer v_w_buffer = v_w_alloc->buffer;
            wgpu::Buffer k_out_buffer = k_out_alloc->buffer;
            wgpu::Buffer v_out_buffer = v_out_alloc->buffer;

            struct Params
            {
                uint32_t M, K, N;
                uint32_t x_offset, k_weight_offset, v_weight_offset, k_out_offset, v_out_offset;
            };

            Params params{};
            params.M = static_cast<uint32_t>(M);
            params.K = static_cast<uint32_t>(K);
            params.N = static_cast<uint32_t>(N);
            params.x_offset = static_cast<uint32_t>(x_2d.storage_offset());
            params.k_weight_offset = static_cast<uint32_t>(k_contig.storage_offset());
            params.v_weight_offset = static_cast<uint32_t>(v_contig.storage_offset());
            params.k_out_offset = static_cast<uint32_t>(k_output.storage_offset());
            params.v_out_offset = static_cast<uint32_t>(v_output.storage_offset());

            core::WebGPUContext &ctx = core::getWebGPUContext();
            wgpu::Buffer params_buffer = core::acquireUniformBuffer(&params, sizeof(Params));

            wgpu::BindGroupEntry bind_group_entries[6]{};
            bind_group_entries[0].binding = 0;
            bind_group_entries[0].buffer = x_buffer;
            bind_group_entries[0].offset = 0;
            bind_group_entries[0].size = x_buffer.GetSize();

            bind_group_entries[1].binding = 1;
            bind_group_entries[1].buffer = k_w_buffer;
            bind_group_entries[1].offset = 0;
            bind_group_entries[1].size = k_w_buffer.GetSize();

            bind_group_entries[2].binding = 2;
            bind_group_entries[2].buffer = v_w_buffer;
            bind_group_entries[2].offset = 0;
            bind_group_entries[2].size = v_w_buffer.GetSize();

            bind_group_entries[3].binding = 3;
            bind_group_entries[3].buffer = k_out_buffer;
            bind_group_entries[3].offset = 0;
            bind_group_entries[3].size = k_out_buffer.GetSize();

            bind_group_entries[4].binding = 4;
            bind_group_entries[4].buffer = v_out_buffer;
            bind_group_entries[4].offset = 0;
            bind_group_entries[4].size = v_out_buffer.GetSize();

            bind_group_entries[5].binding = 5;
            bind_group_entries[5].buffer = params_buffer;
            bind_group_entries[5].offset = 0;
            bind_group_entries[5].size = sizeof(Params);

            wgpu::BindGroupDescriptor bind_group_descriptor{};
            bind_group_descriptor.layout = kernel.bind_group_layout;
            bind_group_descriptor.entryCount = 6;
            bind_group_descriptor.entries = bind_group_entries;

            wgpu::BindGroup bind_group = ctx.getDevice().CreateBindGroup(&bind_group_descriptor);

            const uint32_t TILE_SIZE = 16;
            uint32_t num_wg_x = (static_cast<uint32_t>(N) + TILE_SIZE - 1) / TILE_SIZE;
            uint32_t num_wg_y = (static_cast<uint32_t>(M) + TILE_SIZE - 1) / TILE_SIZE;

            core::dispatchCompute(kernel.pipeline, bind_group, num_wg_x, num_wg_y, 1);

            if (x.dim() == 3)
            {
                return std::make_tuple(
                    k_output.view({x.size(0), x.size(1), N}),
                    v_output.view({x.size(0), x.size(1), N}));
            }
            return std::make_tuple(k_output, v_output);
        }
    }

    TORCH_LIBRARY_FRAGMENT(webgpu, m)
    {
        m.def("fused_kv_proj(Tensor x, Tensor k_weight, Tensor v_weight) -> (Tensor, Tensor)");
    }

    TORCH_LIBRARY_IMPL(webgpu, PrivateUse1, m)
    {
        m.impl("fused_kv_proj", TORCH_FN(ops::fused_kv_proj));
    }
}
