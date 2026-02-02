/**
 * Fused MLP Gate+Up+SiLU for WebGPU
 *
 * Fuses: gate = linear(x, W_gate), up = linear(x, W_up), out = silu(gate) * up
 * Into single dispatch: out = silu(x @ W_gate.T) * (x @ W_up.T)
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
            // Fused gate+up+silu shader
            // Computes: silu(x @ W_gate.T) * (x @ W_up.T)
            const std::string fused_gate_up_silu_shader = R"wgsl(
struct Params {
    M: u32,           // batch * seq_len
    K: u32,           // hidden_size (input dim)
    N: u32,           // intermediate_size (output dim)
    x_offset: u32,
    gate_offset: u32,
    up_offset: u32,
    out_offset: u32,
    _pad: u32,
};

@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> gate_weight: array<f32>;
@group(0) @binding(2) var<storage, read> up_weight: array<f32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;
@group(0) @binding(4) var<uniform> params: Params;

const TILE_SIZE: u32 = 16u;

var<workgroup> tile_x: array<f32, 256>;
var<workgroup> tile_gate: array<f32, 256>;
var<workgroup> tile_up: array<f32, 256>;

fn silu(val: f32) -> f32 {
    return val / (1.0 + exp(-val));
}

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

    var gate_acc: f32 = 0.0;
    var up_acc: f32 = 0.0;

    let num_tiles = (params.K + TILE_SIZE - 1u) / TILE_SIZE;

    for (var t = 0u; t < num_tiles; t++) {
        let tile_k = t * TILE_SIZE + lid.x;

        // Load x tile
        if (row < params.M && tile_k < params.K) {
            tile_x[lid.y * TILE_SIZE + lid.x] = x[params.x_offset + row * params.K + tile_k];
        } else {
            tile_x[lid.y * TILE_SIZE + lid.x] = 0.0;
        }

        // Load weight tiles (transposed access)
        let weight_k = t * TILE_SIZE + lid.y;
        if (weight_k < params.K && col < params.N) {
            tile_gate[lid.y * TILE_SIZE + lid.x] = gate_weight[params.gate_offset + weight_k * params.N + col];
            tile_up[lid.y * TILE_SIZE + lid.x] = up_weight[params.up_offset + weight_k * params.N + col];
        } else {
            tile_gate[lid.y * TILE_SIZE + lid.x] = 0.0;
            tile_up[lid.y * TILE_SIZE + lid.x] = 0.0;
        }

        workgroupBarrier();

        for (var k = 0u; k < TILE_SIZE; k++) {
            let x_val = tile_x[lid.y * TILE_SIZE + k];
            gate_acc += x_val * tile_gate[k * TILE_SIZE + lid.x];
            up_acc += x_val * tile_up[k * TILE_SIZE + lid.x];
        }

        workgroupBarrier();
    }

    if (row < params.M && col < params.N) {
        output[params.out_offset + row * params.N + col] = silu(gate_acc) * up_acc;
    }
}
)wgsl";

            struct FusedGateUpSiluKernel
            {
                wgpu::BindGroupLayout bind_group_layout;
                wgpu::ComputePipeline pipeline;
            };

            static FusedGateUpSiluKernel *fused_gate_up_silu_kernel = nullptr;

            FusedGateUpSiluKernel create_fused_gate_up_silu_kernel()
            {
                wgpu::ShaderSourceWGSL shader_source{
                    wgpu::ShaderSourceWGSL::Init{
                        nullptr,
                        wgpu::StringView{fused_gate_up_silu_shader.c_str(), fused_gate_up_silu_shader.size()},
                    }};

                wgpu::ShaderModuleDescriptor shader_descriptor{};
                shader_descriptor.nextInChain = &shader_source;
                shader_descriptor.label = "fused_gate_up_silu";

                core::WebGPUContext &ctx = core::getWebGPUContext();
                wgpu::ShaderModule shader_module = ctx.getDevice().CreateShaderModule(&shader_descriptor);

                wgpu::BindGroupLayoutEntry bindings[5]{};

                // x buffer (read-only)
                bindings[0].binding = 0;
                bindings[0].visibility = wgpu::ShaderStage::Compute;
                bindings[0].buffer.type = wgpu::BufferBindingType::ReadOnlyStorage;
                bindings[0].buffer.hasDynamicOffset = false;
                bindings[0].buffer.minBindingSize = 0;

                // gate_weight buffer (read-only)
                bindings[1].binding = 1;
                bindings[1].visibility = wgpu::ShaderStage::Compute;
                bindings[1].buffer.type = wgpu::BufferBindingType::ReadOnlyStorage;
                bindings[1].buffer.hasDynamicOffset = false;
                bindings[1].buffer.minBindingSize = 0;

                // up_weight buffer (read-only)
                bindings[2].binding = 2;
                bindings[2].visibility = wgpu::ShaderStage::Compute;
                bindings[2].buffer.type = wgpu::BufferBindingType::ReadOnlyStorage;
                bindings[2].buffer.hasDynamicOffset = false;
                bindings[2].buffer.minBindingSize = 0;

                // output buffer (read-write)
                bindings[3].binding = 3;
                bindings[3].visibility = wgpu::ShaderStage::Compute;
                bindings[3].buffer.type = wgpu::BufferBindingType::Storage;
                bindings[3].buffer.hasDynamicOffset = false;
                bindings[3].buffer.minBindingSize = 0;

                // params buffer (uniform)
                bindings[4].binding = 4;
                bindings[4].visibility = wgpu::ShaderStage::Compute;
                bindings[4].buffer.type = wgpu::BufferBindingType::Uniform;
                bindings[4].buffer.hasDynamicOffset = false;
                bindings[4].buffer.minBindingSize = 0;

                wgpu::BindGroupLayoutDescriptor layout_descriptor{};
                layout_descriptor.entryCount = 5;
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

                return FusedGateUpSiluKernel{bind_group_layout, pipeline};
            }

            FusedGateUpSiluKernel &get_fused_gate_up_silu_kernel()
            {
                if (fused_gate_up_silu_kernel != nullptr)
                {
                    return *fused_gate_up_silu_kernel;
                }
                static FusedGateUpSiluKernel k = create_fused_gate_up_silu_kernel();
                fused_gate_up_silu_kernel = &k;
                return *fused_gate_up_silu_kernel;
            }
        }

        at::Tensor fused_gate_up_silu(
            const at::Tensor &x,
            const at::Tensor &gate_weight,
            const at::Tensor &up_weight)
        {
            TORCH_CHECK(x.device().type() == c10::DeviceType::PrivateUse1);
            TORCH_CHECK(gate_weight.device().type() == c10::DeviceType::PrivateUse1);
            TORCH_CHECK(up_weight.device().type() == c10::DeviceType::PrivateUse1);
            TORCH_CHECK(x.scalar_type() == c10::ScalarType::Float);

            auto x_contig = x.contiguous();
            auto gate_contig = gate_weight.contiguous();
            auto up_contig = up_weight.contiguous();

            // x: [batch, seq_len, hidden_size] or [batch * seq_len, hidden_size]
            auto x_2d = x_contig.dim() == 3 ? x_contig.view({-1, x_contig.size(-1)}) : x_contig;
            auto M = x_2d.size(0);
            auto K = x_2d.size(1);
            auto N = gate_contig.size(0);

            TORCH_CHECK(gate_contig.size(1) == K, "gate_weight K dimension mismatch");
            TORCH_CHECK(up_contig.size(0) == N && up_contig.size(1) == K, "up_weight dimension mismatch");

            at::Tensor output = at::empty({M, N}, x.options());

            if (M == 0 || N == 0)
            {
                return x.dim() == 3 ? output.view({x.size(0), x.size(1), N}) : output;
            }

            FusedGateUpSiluKernel &kernel = get_fused_gate_up_silu_kernel();

            core::WebGPUAllocation *x_alloc = static_cast<core::WebGPUAllocation *>(x_2d.storage().data_ptr().get());
            core::WebGPUAllocation *gate_alloc = static_cast<core::WebGPUAllocation *>(gate_contig.storage().data_ptr().get());
            core::WebGPUAllocation *up_alloc = static_cast<core::WebGPUAllocation *>(up_contig.storage().data_ptr().get());
            core::WebGPUAllocation *out_alloc = static_cast<core::WebGPUAllocation *>(output.storage().data_ptr().get());

            wgpu::Buffer x_buffer = x_alloc->buffer;
            wgpu::Buffer gate_buffer = gate_alloc->buffer;
            wgpu::Buffer up_buffer = up_alloc->buffer;
            wgpu::Buffer out_buffer = out_alloc->buffer;

            struct Params
            {
                uint32_t M, K, N;
                uint32_t x_offset, gate_offset, up_offset, out_offset;
                uint32_t _pad;
            };

            Params params{};
            params.M = static_cast<uint32_t>(M);
            params.K = static_cast<uint32_t>(K);
            params.N = static_cast<uint32_t>(N);
            params.x_offset = static_cast<uint32_t>(x_2d.storage_offset());
            params.gate_offset = static_cast<uint32_t>(gate_contig.storage_offset());
            params.up_offset = static_cast<uint32_t>(up_contig.storage_offset());
            params.out_offset = static_cast<uint32_t>(output.storage_offset());
            params._pad = 0;

            core::WebGPUContext &ctx = core::getWebGPUContext();
            wgpu::Buffer params_buffer = core::acquireUniformBuffer(&params, sizeof(Params));

            wgpu::BindGroupEntry bind_group_entries[5]{};
            bind_group_entries[0].binding = 0;
            bind_group_entries[0].buffer = x_buffer;
            bind_group_entries[0].offset = 0;
            bind_group_entries[0].size = x_buffer.GetSize();

            bind_group_entries[1].binding = 1;
            bind_group_entries[1].buffer = gate_buffer;
            bind_group_entries[1].offset = 0;
            bind_group_entries[1].size = gate_buffer.GetSize();

            bind_group_entries[2].binding = 2;
            bind_group_entries[2].buffer = up_buffer;
            bind_group_entries[2].offset = 0;
            bind_group_entries[2].size = up_buffer.GetSize();

            bind_group_entries[3].binding = 3;
            bind_group_entries[3].buffer = out_buffer;
            bind_group_entries[3].offset = 0;
            bind_group_entries[3].size = out_buffer.GetSize();

            bind_group_entries[4].binding = 4;
            bind_group_entries[4].buffer = params_buffer;
            bind_group_entries[4].offset = 0;
            bind_group_entries[4].size = sizeof(Params);

            wgpu::BindGroupDescriptor bind_group_descriptor{};
            bind_group_descriptor.layout = kernel.bind_group_layout;
            bind_group_descriptor.entryCount = 5;
            bind_group_descriptor.entries = bind_group_entries;

            wgpu::BindGroup bind_group = ctx.getDevice().CreateBindGroup(&bind_group_descriptor);

            const uint32_t TILE_SIZE = 16;
            uint32_t num_wg_x = (static_cast<uint32_t>(N) + TILE_SIZE - 1) / TILE_SIZE;
            uint32_t num_wg_y = (static_cast<uint32_t>(M) + TILE_SIZE - 1) / TILE_SIZE;

            core::dispatchCompute(kernel.pipeline, bind_group, num_wg_x, num_wg_y, 1);

            if (x.dim() == 3)
            {
                return output.view({x.size(0), x.size(1), N});
            }
            return output;
        }
    }

    TORCH_LIBRARY_FRAGMENT(webgpu, m)
    {
        m.def("fused_gate_up_silu(Tensor x, Tensor gate_weight, Tensor up_weight) -> Tensor");
    }

    TORCH_LIBRARY_IMPL(webgpu, PrivateUse1, m)
    {
        m.impl("fused_gate_up_silu", TORCH_FN(ops::fused_gate_up_silu));
    }
}
