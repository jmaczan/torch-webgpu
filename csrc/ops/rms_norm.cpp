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
            // Simple RMSNorm shader for small dimensions (< 1024 elements)
            // Each thread handles one batch row
            // RMSNorm: y = x * rsqrt(mean(x^2) + eps) * weight
            const std::string rms_norm_shader_simple = R"wgsl(
struct Params {
    batch_size: u32,
    hidden_size: u32,
    eps: f32,
    self_offset: u32,
    weight_offset: u32,
    out_offset: u32,
    _pad0: u32,
    _pad1: u32,
};

@group(0) @binding(0)
var<storage, read> selfBuffer: array<f32>;

@group(0) @binding(1)
var<storage, read> weightBuffer: array<f32>;

@group(0) @binding(2)
var<storage, read_write> outBuffer: array<f32>;

@group(0) @binding(3)
var<uniform> params: Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let batch_idx = gid.x;
    if (batch_idx >= params.batch_size) { return; }

    let base_idx = params.self_offset + batch_idx * params.hidden_size;
    let out_base_idx = params.out_offset + batch_idx * params.hidden_size;

    // Compute sum of squares
    var sum_sq: f32 = 0.0;
    for (var i: u32 = 0u; i < params.hidden_size; i++) {
        let val = selfBuffer[base_idx + i];
        sum_sq += val * val;
    }

    // Compute rsqrt(mean(x^2) + eps)
    let mean_sq = sum_sq / f32(params.hidden_size);
    let inv_rms = inverseSqrt(mean_sq + params.eps);

    // Normalize and scale by weight
    for (var i: u32 = 0u; i < params.hidden_size; i++) {
        let val = selfBuffer[base_idx + i];
        let weight = weightBuffer[params.weight_offset + i];
        outBuffer[out_base_idx + i] = val * inv_rms * weight;
    }
}
)wgsl";

            // Optimized RMSNorm shader for large dimensions using parallel reduction
            // One workgroup per batch row, multiple threads collaborate on reduction
            const std::string rms_norm_shader_parallel = R"wgsl(
struct Params {
    batch_size: u32,
    hidden_size: u32,
    eps: f32,
    self_offset: u32,
    weight_offset: u32,
    out_offset: u32,
    _pad0: u32,
    _pad1: u32,
};

@group(0) @binding(0)
var<storage, read> selfBuffer: array<f32>;

@group(0) @binding(1)
var<storage, read> weightBuffer: array<f32>;

@group(0) @binding(2)
var<storage, read_write> outBuffer: array<f32>;

@group(0) @binding(3)
var<uniform> params: Params;

const WORKGROUP_SIZE: u32 = 256u;

var<workgroup> shared_sum: array<f32, WORKGROUP_SIZE>;

@compute @workgroup_size(256)
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>
) {
    let batch_idx = wid.x;
    let thread_idx = lid.x;

    if (batch_idx >= params.batch_size) { return; }

    let base_idx = params.self_offset + batch_idx * params.hidden_size;
    let out_base_idx = params.out_offset + batch_idx * params.hidden_size;

    // Each thread processes multiple elements (strided access for coalescing)
    let elements_per_thread = (params.hidden_size + WORKGROUP_SIZE - 1u) / WORKGROUP_SIZE;

    // Phase 1: Compute local sum of squares
    var local_sum_sq: f32 = 0.0;
    for (var i: u32 = 0u; i < elements_per_thread; i++) {
        let idx = thread_idx + i * WORKGROUP_SIZE;
        if (idx < params.hidden_size) {
            let val = selfBuffer[base_idx + idx];
            local_sum_sq += val * val;
        }
    }
    shared_sum[thread_idx] = local_sum_sq;
    workgroupBarrier();

    // Parallel reduction for sum
    for (var stride: u32 = WORKGROUP_SIZE / 2u; stride > 0u; stride = stride / 2u) {
        if (thread_idx < stride) {
            shared_sum[thread_idx] = shared_sum[thread_idx] + shared_sum[thread_idx + stride];
        }
        workgroupBarrier();
    }

    // Compute inverse RMS
    let sum_sq = shared_sum[0];
    let mean_sq = sum_sq / f32(params.hidden_size);
    let inv_rms = inverseSqrt(mean_sq + params.eps);
    workgroupBarrier();

    // Phase 2: Normalize and scale by weight
    for (var i: u32 = 0u; i < elements_per_thread; i++) {
        let idx = thread_idx + i * WORKGROUP_SIZE;
        if (idx < params.hidden_size) {
            let val = selfBuffer[base_idx + idx];
            let weight = weightBuffer[params.weight_offset + idx];
            outBuffer[out_base_idx + idx] = val * inv_rms * weight;
        }
    }
}
)wgsl";

            // Threshold for using parallel kernel
            constexpr uint32_t PARALLEL_THRESHOLD = 1024;
            constexpr uint32_t PARALLEL_WORKGROUP_SIZE = 256;

            struct RmsNormKernel
            {
                wgpu::BindGroupLayout bind_group_layout;
                wgpu::ComputePipeline pipeline;
            };

            static RmsNormKernel *rms_norm_kernel_simple = nullptr;
            static RmsNormKernel *rms_norm_kernel_parallel = nullptr;

            RmsNormKernel create_kernel(const std::string &shader_code, const char *label)
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

                wgpu::BindGroupLayoutEntry bindings[4]{};

                // Input buffer (read-only)
                bindings[0].binding = 0;
                bindings[0].visibility = wgpu::ShaderStage::Compute;
                bindings[0].buffer.type = wgpu::BufferBindingType::ReadOnlyStorage;
                bindings[0].buffer.hasDynamicOffset = false;
                bindings[0].buffer.minBindingSize = 0;

                // Weight buffer (read-only)
                bindings[1].binding = 1;
                bindings[1].visibility = wgpu::ShaderStage::Compute;
                bindings[1].buffer.type = wgpu::BufferBindingType::ReadOnlyStorage;
                bindings[1].buffer.hasDynamicOffset = false;
                bindings[1].buffer.minBindingSize = 0;

                // Output buffer (read-write)
                bindings[2].binding = 2;
                bindings[2].visibility = wgpu::ShaderStage::Compute;
                bindings[2].buffer.type = wgpu::BufferBindingType::Storage;
                bindings[2].buffer.hasDynamicOffset = false;
                bindings[2].buffer.minBindingSize = 0;

                // Params buffer (uniform)
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

                return RmsNormKernel{bind_group_layout, pipeline};
            }

            RmsNormKernel &get_rms_norm_kernel_simple()
            {
                if (rms_norm_kernel_simple != nullptr)
                {
                    return *rms_norm_kernel_simple;
                }
                static RmsNormKernel k = create_kernel(rms_norm_shader_simple, "RMSNorm simple kernel");
                rms_norm_kernel_simple = &k;
                return *rms_norm_kernel_simple;
            }

            RmsNormKernel &get_rms_norm_kernel_parallel()
            {
                if (rms_norm_kernel_parallel != nullptr)
                {
                    return *rms_norm_kernel_parallel;
                }
                static RmsNormKernel k = create_kernel(rms_norm_shader_parallel, "RMSNorm parallel kernel");
                rms_norm_kernel_parallel = &k;
                return *rms_norm_kernel_parallel;
            }
        }

        at::Tensor rms_norm(const at::Tensor &self, const at::Tensor &weight, double eps)
        {
            TORCH_CHECK(self.device().type() == c10::DeviceType::PrivateUse1);
            TORCH_CHECK(weight.device().type() == c10::DeviceType::PrivateUse1);
            TORCH_CHECK(self.scalar_type() == c10::ScalarType::Float);
            TORCH_CHECK(weight.scalar_type() == c10::ScalarType::Float);

            auto ndim = self.dim();
            TORCH_CHECK(ndim >= 1, "rms_norm requires at least 1D input");

            // Get hidden size (last dimension)
            auto hidden_size = self.size(-1);
            TORCH_CHECK(weight.numel() == hidden_size, "weight must have same size as hidden dimension");

            // Make inputs contiguous
            auto input_contig = self.contiguous();
            auto weight_contig = weight.contiguous();

            // Compute batch size (product of all dims except last)
            int64_t batch_size = input_contig.numel() / hidden_size;

            at::Tensor out = at::empty_like(input_contig);

            if (batch_size == 0)
            {
                return out;
            }

            // Choose kernel based on hidden size
            bool use_parallel = (hidden_size > PARALLEL_THRESHOLD);
            RmsNormKernel &kernel = use_parallel ? get_rms_norm_kernel_parallel() : get_rms_norm_kernel_simple();

            core::WebGPUAllocation *self_allocation = static_cast<core::WebGPUAllocation *>(input_contig.storage().data_ptr().get());
            core::WebGPUAllocation *weight_allocation = static_cast<core::WebGPUAllocation *>(weight_contig.storage().data_ptr().get());
            core::WebGPUAllocation *out_allocation = static_cast<core::WebGPUAllocation *>(out.storage().data_ptr().get());

            wgpu::Buffer self_buffer = self_allocation->buffer;
            wgpu::Buffer weight_buffer = weight_allocation->buffer;
            wgpu::Buffer out_buffer = out_allocation->buffer;

            struct Params
            {
                uint32_t batch_size;
                uint32_t hidden_size;
                float eps;
                uint32_t self_offset;
                uint32_t weight_offset;
                uint32_t out_offset;
                uint32_t _pad0;
                uint32_t _pad1;
            };

            Params params{};
            params.batch_size = static_cast<uint32_t>(batch_size);
            params.hidden_size = static_cast<uint32_t>(hidden_size);
            params.eps = static_cast<float>(eps);
            params.self_offset = static_cast<uint32_t>(input_contig.storage_offset());
            params.weight_offset = static_cast<uint32_t>(weight_contig.storage_offset());
            params.out_offset = static_cast<uint32_t>(out.storage_offset());
            params._pad0 = 0;
            params._pad1 = 0;

            core::WebGPUContext &ctx = core::getWebGPUContext();

            // Use buffer pool for reduced allocation overhead
            wgpu::Buffer params_buffer = core::acquireUniformBuffer(&params, sizeof(Params));

            // Try bind group cache first
            std::vector<wgpu::Buffer> buffers = {self_buffer, weight_buffer, out_buffer, params_buffer};
            std::vector<uint64_t> sizes = {self_buffer.GetSize(), weight_buffer.GetSize(), out_buffer.GetSize(), sizeof(Params)};
            core::BindGroupKey cache_key = core::makeBindGroupKey(kernel.pipeline, buffers, sizes);

            wgpu::BindGroup bind_group;
            auto cached = core::getBindGroupCache().get(cache_key);
            if (cached.has_value())
            {
                bind_group = cached.value();
            }
            else
            {
                wgpu::BindGroupEntry bind_group_entries[4]{};
                bind_group_entries[0].binding = 0;
                bind_group_entries[0].buffer = self_buffer;
                bind_group_entries[0].offset = 0;
                bind_group_entries[0].size = self_buffer.GetSize();

                bind_group_entries[1].binding = 1;
                bind_group_entries[1].buffer = weight_buffer;
                bind_group_entries[1].offset = 0;
                bind_group_entries[1].size = weight_buffer.GetSize();

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

                bind_group = ctx.getDevice().CreateBindGroup(&bind_group_descriptor);
                core::getBindGroupCache().put(cache_key, bind_group);
            }

            uint32_t num_workgroups;
            if (use_parallel)
            {
                // Parallel kernel: one workgroup per batch row
                num_workgroups = params.batch_size;
            }
            else
            {
                // Simple kernel: one thread per batch row
                const uint32_t workgroup_size = 64;
                num_workgroups = (params.batch_size + workgroup_size - 1) / workgroup_size;
            }

            // Use batched dispatch for reduced submission overhead
            core::dispatchCompute(kernel.pipeline, bind_group, num_workgroups, 1, 1);

            return out;
        }
    }

    // Register as custom webgpu op (not aten override)
    // This will be called via torch.ops.webgpu.rms_norm
    TORCH_LIBRARY_FRAGMENT(webgpu, m)
    {
        m.def("rms_norm(Tensor self, Tensor weight, float eps=1e-6) -> Tensor");
    }

    TORCH_LIBRARY_IMPL(webgpu, PrivateUse1, m)
    {
        m.impl("rms_norm", TORCH_FN(ops::rms_norm));
    }
}
