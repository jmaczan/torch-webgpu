/**
 * Mega MLP Kernel for Qwen2.5-0.5B
 *
 * Fuses the ENTIRE MLP block into a single dispatch:
 * - RMSNorm
 * - Gate projection
 * - Up projection
 * - SiLU activation
 * - Element-wise multiply
 * - Down projection
 * - Residual add
 *
 * Hardcoded for Qwen2.5-0.5B: hidden=896, intermediate=4864
 */

#include <ATen/ATen.h>
#include <torch/library.h>
#include <webgpu/webgpu_cpp.h>
#include "core/webgpu_context.h"
#include "core/webgpu_allocator.h"
#include "core/command_batcher.h"

namespace torch_webgpu
{
    namespace ops
    {
        namespace
        {
            const std::string mega_mlp_shader = R"wgsl(
// Qwen2.5-0.5B constants
const HIDDEN: u32 = 896u;
const INTERMEDIATE: u32 = 4864u;

struct Params {
    eps: f32,
    _pad1: u32,
    _pad2: u32,
    _pad3: u32,
};

@group(0) @binding(0) var<storage, read> input: array<f32>;           // [HIDDEN] - after attention
@group(0) @binding(1) var<storage, read> residual: array<f32>;        // [HIDDEN] - skip connection
@group(0) @binding(2) var<storage, read> norm_weight: array<f32>;     // [HIDDEN]
@group(0) @binding(3) var<storage, read> gate_weight: array<f32>;     // [INTERMEDIATE, HIDDEN]
@group(0) @binding(4) var<storage, read> up_weight: array<f32>;       // [INTERMEDIATE, HIDDEN]
@group(0) @binding(5) var<storage, read> down_weight: array<f32>;     // [HIDDEN, INTERMEDIATE]
@group(0) @binding(6) var<storage, read_write> output: array<f32>;    // [HIDDEN]
@group(0) @binding(7) var<uniform> params: Params;

var<workgroup> normalized: array<f32, 896>;
var<workgroup> gate_up: array<f32, 4864>;  // SiLU(gate) * up
var<workgroup> variance: f32;

fn silu(x: f32) -> f32 {
    return x / (1.0 + exp(-x));
}

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3<u32>) {
    let thread_id = lid.x;

    // Phase 1: RMSNorm
    // Compute variance (sum of squares)
    var local_sum: f32 = 0.0;
    for (var i = thread_id; i < HIDDEN; i += 256u) {
        let val = input[i];
        local_sum += val * val;
    }

    // Reduce within workgroup
    var wg_sum: array<f32, 256>;
    wg_sum[thread_id] = local_sum;
    workgroupBarrier();

    // Tree reduction
    for (var stride = 128u; stride > 0u; stride /= 2u) {
        if (thread_id < stride) {
            wg_sum[thread_id] += wg_sum[thread_id + stride];
        }
        workgroupBarrier();
    }

    if (thread_id == 0u) {
        variance = wg_sum[0] / f32(HIDDEN);
    }
    workgroupBarrier();

    let rsqrt_var = 1.0 / sqrt(variance + params.eps);

    // Apply normalization
    for (var i = thread_id; i < HIDDEN; i += 256u) {
        normalized[i] = input[i] * rsqrt_var * norm_weight[i];
    }
    workgroupBarrier();

    // Phase 2: Gate and Up projections with SiLU
    // Each thread handles multiple intermediate dimensions
    for (var i = thread_id; i < INTERMEDIATE; i += 256u) {
        var gate_sum: f32 = 0.0;
        var up_sum: f32 = 0.0;

        for (var k = 0u; k < HIDDEN; k++) {
            let n = normalized[k];
            gate_sum += n * gate_weight[i * HIDDEN + k];
            up_sum += n * up_weight[i * HIDDEN + k];
        }

        // SiLU(gate) * up
        gate_up[i] = silu(gate_sum) * up_sum;
    }
    workgroupBarrier();

    // Phase 3: Down projection + residual add
    for (var i = thread_id; i < HIDDEN; i += 256u) {
        var down_sum: f32 = 0.0;
        for (var k = 0u; k < INTERMEDIATE; k++) {
            down_sum += gate_up[k] * down_weight[i * INTERMEDIATE + k];
        }
        // Add residual
        output[i] = down_sum + residual[i];
    }
}
)wgsl";

            struct MegaMLPKernel
            {
                wgpu::BindGroupLayout bind_group_layout;
                wgpu::ComputePipeline pipeline;
            };

            static MegaMLPKernel *mega_mlp_kernel = nullptr;

            MegaMLPKernel create_mega_mlp_kernel()
            {
                wgpu::ShaderSourceWGSL shader_source{
                    wgpu::ShaderSourceWGSL::Init{
                        nullptr,
                        wgpu::StringView{mega_mlp_shader.c_str(), mega_mlp_shader.size()},
                    }};

                wgpu::ShaderModuleDescriptor shader_descriptor{};
                shader_descriptor.nextInChain = &shader_source;
                shader_descriptor.label = "mega_mlp";

                core::WebGPUContext &ctx = core::getWebGPUContext();
                wgpu::ShaderModule shader_module = ctx.getDevice().CreateShaderModule(&shader_descriptor);

                wgpu::BindGroupLayoutEntry bindings[8]{};

                bindings[0].binding = 0;
                bindings[0].visibility = wgpu::ShaderStage::Compute;
                bindings[0].buffer.type = wgpu::BufferBindingType::ReadOnlyStorage;

                bindings[1].binding = 1;
                bindings[1].visibility = wgpu::ShaderStage::Compute;
                bindings[1].buffer.type = wgpu::BufferBindingType::ReadOnlyStorage;

                bindings[2].binding = 2;
                bindings[2].visibility = wgpu::ShaderStage::Compute;
                bindings[2].buffer.type = wgpu::BufferBindingType::ReadOnlyStorage;

                bindings[3].binding = 3;
                bindings[3].visibility = wgpu::ShaderStage::Compute;
                bindings[3].buffer.type = wgpu::BufferBindingType::ReadOnlyStorage;

                bindings[4].binding = 4;
                bindings[4].visibility = wgpu::ShaderStage::Compute;
                bindings[4].buffer.type = wgpu::BufferBindingType::ReadOnlyStorage;

                bindings[5].binding = 5;
                bindings[5].visibility = wgpu::ShaderStage::Compute;
                bindings[5].buffer.type = wgpu::BufferBindingType::ReadOnlyStorage;

                bindings[6].binding = 6;
                bindings[6].visibility = wgpu::ShaderStage::Compute;
                bindings[6].buffer.type = wgpu::BufferBindingType::Storage;

                bindings[7].binding = 7;
                bindings[7].visibility = wgpu::ShaderStage::Compute;
                bindings[7].buffer.type = wgpu::BufferBindingType::Uniform;

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

                return MegaMLPKernel{bind_group_layout, pipeline};
            }

            MegaMLPKernel &get_mega_mlp_kernel()
            {
                if (mega_mlp_kernel != nullptr)
                {
                    return *mega_mlp_kernel;
                }
                static MegaMLPKernel k = create_mega_mlp_kernel();
                mega_mlp_kernel = &k;
                return *mega_mlp_kernel;
            }
        }

        // Mega MLP: entire MLP block (norm + gate + up + silu + down + residual) in one dispatch
        at::Tensor mega_mlp(
            const at::Tensor &input,
            const at::Tensor &residual,
            const at::Tensor &norm_weight,
            const at::Tensor &gate_weight,
            const at::Tensor &up_weight,
            const at::Tensor &down_weight,
            double eps)
        {
            TORCH_CHECK(input.device().type() == c10::DeviceType::PrivateUse1);
            TORCH_CHECK(input.scalar_type() == c10::ScalarType::Float);

            const int64_t hidden_size = 896;

            auto input_contig = input.contiguous();
            auto residual_contig = residual.contiguous();
            auto norm_w_contig = norm_weight.contiguous();
            auto gate_w_contig = gate_weight.contiguous();
            auto up_w_contig = up_weight.contiguous();
            auto down_w_contig = down_weight.contiguous();

            at::Tensor output = at::empty({1, hidden_size}, input.options());

            MegaMLPKernel &kernel = get_mega_mlp_kernel();

            core::WebGPUAllocation *input_alloc = static_cast<core::WebGPUAllocation *>(input_contig.storage().data_ptr().get());
            core::WebGPUAllocation *residual_alloc = static_cast<core::WebGPUAllocation *>(residual_contig.storage().data_ptr().get());
            core::WebGPUAllocation *norm_w_alloc = static_cast<core::WebGPUAllocation *>(norm_w_contig.storage().data_ptr().get());
            core::WebGPUAllocation *gate_w_alloc = static_cast<core::WebGPUAllocation *>(gate_w_contig.storage().data_ptr().get());
            core::WebGPUAllocation *up_w_alloc = static_cast<core::WebGPUAllocation *>(up_w_contig.storage().data_ptr().get());
            core::WebGPUAllocation *down_w_alloc = static_cast<core::WebGPUAllocation *>(down_w_contig.storage().data_ptr().get());
            core::WebGPUAllocation *output_alloc = static_cast<core::WebGPUAllocation *>(output.storage().data_ptr().get());

            struct Params
            {
                float eps;
                uint32_t _pad1, _pad2, _pad3;
            };

            Params params{};
            params.eps = static_cast<float>(eps);

            core::WebGPUContext &ctx = core::getWebGPUContext();
            wgpu::Buffer params_buffer = core::acquireUniformBuffer(&params, sizeof(Params));

            wgpu::BindGroupEntry entries[8]{};
            entries[0] = {nullptr, 0, input_alloc->buffer, 0, input_alloc->buffer.GetSize()};
            entries[1] = {nullptr, 1, residual_alloc->buffer, 0, residual_alloc->buffer.GetSize()};
            entries[2] = {nullptr, 2, norm_w_alloc->buffer, 0, norm_w_alloc->buffer.GetSize()};
            entries[3] = {nullptr, 3, gate_w_alloc->buffer, 0, gate_w_alloc->buffer.GetSize()};
            entries[4] = {nullptr, 4, up_w_alloc->buffer, 0, up_w_alloc->buffer.GetSize()};
            entries[5] = {nullptr, 5, down_w_alloc->buffer, 0, down_w_alloc->buffer.GetSize()};
            entries[6] = {nullptr, 6, output_alloc->buffer, 0, output_alloc->buffer.GetSize()};
            entries[7] = {nullptr, 7, params_buffer, 0, sizeof(Params)};

            wgpu::BindGroupDescriptor bind_group_descriptor{};
            bind_group_descriptor.layout = kernel.bind_group_layout;
            bind_group_descriptor.entryCount = 8;
            bind_group_descriptor.entries = entries;

            wgpu::BindGroup bind_group = ctx.getDevice().CreateBindGroup(&bind_group_descriptor);

            // Single workgroup of 256 threads
            core::dispatchCompute(kernel.pipeline, bind_group, 1, 1, 1);

            return output;
        }
    }

    TORCH_LIBRARY_FRAGMENT(webgpu, m)
    {
        m.def("mega_mlp(Tensor input, Tensor residual, Tensor norm_weight, Tensor gate_weight, Tensor up_weight, Tensor down_weight, float eps) -> Tensor");
    }

    TORCH_LIBRARY_IMPL(webgpu, PrivateUse1, m)
    {
        m.impl("mega_mlp", TORCH_FN(ops::mega_mlp));
    }
}
