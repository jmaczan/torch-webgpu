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
            // Simple softmax shader for small dimensions (< 1024 elements)
            // Each thread handles one batch row
            const std::string softmax_shader_simple = R"wgsl(
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

            // Optimized softmax shader for large dimensions using parallel reduction
            // One workgroup per batch row, multiple threads collaborate on reduction
            const std::string softmax_shader_parallel = R"wgsl(
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

const WORKGROUP_SIZE: u32 = 256u;

var<workgroup> shared_max: array<f32, WORKGROUP_SIZE>;
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

    let base_idx = params.self_offset + batch_idx * params.dim_size;
    let out_base_idx = params.out_offset + batch_idx * params.dim_size;

    // Each thread processes multiple elements (strided access for coalescing)
    let elements_per_thread = (params.dim_size + WORKGROUP_SIZE - 1u) / WORKGROUP_SIZE;

    // Phase 1: Find local max
    var local_max: f32 = -3.402823e+38; // -FLT_MAX
    for (var i: u32 = 0u; i < elements_per_thread; i++) {
        let idx = thread_idx + i * WORKGROUP_SIZE;
        if (idx < params.dim_size) {
            let val = selfBuffer[base_idx + idx];
            local_max = max(local_max, val);
        }
    }
    shared_max[thread_idx] = local_max;
    workgroupBarrier();

    // Parallel reduction for max
    for (var stride: u32 = WORKGROUP_SIZE / 2u; stride > 0u; stride = stride / 2u) {
        if (thread_idx < stride) {
            shared_max[thread_idx] = max(shared_max[thread_idx], shared_max[thread_idx + stride]);
        }
        workgroupBarrier();
    }
    let global_max = shared_max[0];
    workgroupBarrier();

    // Phase 2: Compute exp and write to output, compute local sum
    var local_sum: f32 = 0.0;
    for (var i: u32 = 0u; i < elements_per_thread; i++) {
        let idx = thread_idx + i * WORKGROUP_SIZE;
        if (idx < params.dim_size) {
            let exp_val = exp(selfBuffer[base_idx + idx] - global_max);
            outBuffer[out_base_idx + idx] = exp_val;
            local_sum += exp_val;
        }
    }
    shared_sum[thread_idx] = local_sum;
    workgroupBarrier();

    // Parallel reduction for sum
    for (var stride: u32 = WORKGROUP_SIZE / 2u; stride > 0u; stride = stride / 2u) {
        if (thread_idx < stride) {
            shared_sum[thread_idx] = shared_sum[thread_idx] + shared_sum[thread_idx + stride];
        }
        workgroupBarrier();
    }
    let global_sum = shared_sum[0];
    let inv_sum = 1.0 / global_sum;
    workgroupBarrier();

    // Phase 3: Normalize
    for (var i: u32 = 0u; i < elements_per_thread; i++) {
        let idx = thread_idx + i * WORKGROUP_SIZE;
        if (idx < params.dim_size) {
            outBuffer[out_base_idx + idx] *= inv_sum;
        }
    }
}
)wgsl";

            // Threshold for using parallel kernel (vocab size 151936 >> 1024)
            constexpr uint32_t PARALLEL_THRESHOLD = 1024;
            constexpr uint32_t PARALLEL_WORKGROUP_SIZE = 256;

            struct SoftmaxKernel
            {
                wgpu::BindGroupLayout bind_group_layout;
                wgpu::ComputePipeline pipeline;
            };

            static SoftmaxKernel *softmax_kernel_simple = nullptr;
            static SoftmaxKernel *softmax_kernel_parallel = nullptr;

            SoftmaxKernel create_kernel(const std::string& shader_code, const char* label)
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

                return SoftmaxKernel{bind_group_layout, pipeline};
            }

            SoftmaxKernel &get_softmax_kernel_simple()
            {
                if (softmax_kernel_simple != nullptr)
                {
                    return *softmax_kernel_simple;
                }
                static SoftmaxKernel k = create_kernel(softmax_shader_simple, "Softmax simple kernel");
                softmax_kernel_simple = &k;
                return *softmax_kernel_simple;
            }

            SoftmaxKernel &get_softmax_kernel_parallel()
            {
                if (softmax_kernel_parallel != nullptr)
                {
                    return *softmax_kernel_parallel;
                }
                static SoftmaxKernel k = create_kernel(softmax_shader_parallel, "Softmax parallel kernel");
                softmax_kernel_parallel = &k;
                return *softmax_kernel_parallel;
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

            // Choose kernel based on dimension size
            bool use_parallel = (dim_size > PARALLEL_THRESHOLD);
            SoftmaxKernel &kernel = use_parallel ? get_softmax_kernel_parallel() : get_softmax_kernel_simple();

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

            uint32_t num_workgroups;
            if (use_parallel) {
                // Parallel kernel: one workgroup per batch row
                num_workgroups = params.batch_size;
            } else {
                // Simple kernel: one thread per batch row
                const uint32_t workgroup_size = 64;
                num_workgroups = (params.batch_size + workgroup_size - 1) / workgroup_size;
            }

            // Use batched dispatch for reduced submission overhead
            core::dispatchCompute(kernel.pipeline, bind_group, num_workgroups, 1, 1);

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
