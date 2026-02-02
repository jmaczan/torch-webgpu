/**
 * Mega Attention Kernel for Qwen2.5-0.5B
 *
 * Fuses the ENTIRE attention block into a single dispatch:
 * - Q, K, V projections
 * - Rotary position embeddings
 * - Scaled dot-product attention with causal mask
 * - Output projection
 *
 * This is a proof-of-concept mega-kernel demonstrating peak WebGPU performance.
 * Hardcoded for Qwen2.5-0.5B: hidden=896, heads=14, kv_heads=2, head_dim=64
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
            // Mega attention kernel for single-token generation (seq_len=1)
            // Optimized for Qwen2.5-0.5B dimensions
            const std::string mega_attention_shader = R"wgsl(
// Qwen2.5-0.5B constants
const HIDDEN: u32 = 896u;
const NUM_HEADS: u32 = 14u;
const NUM_KV_HEADS: u32 = 2u;
const HEAD_DIM: u32 = 64u;
const KV_DIM: u32 = 128u;  // NUM_KV_HEADS * HEAD_DIM
const HEADS_PER_KV: u32 = 7u;  // NUM_HEADS / NUM_KV_HEADS

struct Params {
    seq_pos: u32,      // Current position in sequence for RoPE
    _pad1: u32,
    _pad2: u32,
    _pad3: u32,
};

@group(0) @binding(0) var<storage, read> input: array<f32>;        // [1, HIDDEN]
@group(0) @binding(1) var<storage, read> q_weight: array<f32>;     // [HIDDEN, HIDDEN]
@group(0) @binding(2) var<storage, read> k_weight: array<f32>;     // [KV_DIM, HIDDEN]
@group(0) @binding(3) var<storage, read> v_weight: array<f32>;     // [KV_DIM, HIDDEN]
@group(0) @binding(4) var<storage, read> o_weight: array<f32>;     // [HIDDEN, HIDDEN]
@group(0) @binding(5) var<storage, read> k_cache: array<f32>;      // [max_seq, KV_DIM]
@group(0) @binding(6) var<storage, read> v_cache: array<f32>;      // [max_seq, KV_DIM]
@group(0) @binding(7) var<storage, read_write> output: array<f32>; // [1, HIDDEN]
@group(0) @binding(8) var<storage, read_write> k_cache_out: array<f32>;
@group(0) @binding(9) var<storage, read_write> v_cache_out: array<f32>;
@group(0) @binding(10) var<uniform> params: Params;

// Shared memory for intermediate results
var<workgroup> q_local: array<f32, 896>;   // Q projection result
var<workgroup> k_local: array<f32, 128>;   // K projection result
var<workgroup> v_local: array<f32, 128>;   // V projection result
var<workgroup> attn_out: array<f32, 896>;  // Attention output before O projection

// RoPE frequencies (precomputed for position)
fn get_rope_freq(pos: u32, dim_idx: u32) -> vec2<f32> {
    let freq = 1.0 / pow(10000.0, f32(dim_idx * 2u) / f32(HEAD_DIM));
    let angle = f32(pos) * freq;
    return vec2<f32>(cos(angle), sin(angle));
}

fn apply_rope(x: vec2<f32>, freq: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(
        x.x * freq.x - x.y * freq.y,
        x.x * freq.y + x.y * freq.x
    );
}

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>) {
    let thread_id = lid.x;
    let seq_pos = params.seq_pos;

    // Phase 1: Compute Q projection (896 outputs, 4 per thread for 256 threads)
    for (var i = 0u; i < 4u; i++) {
        let out_idx = thread_id * 4u + i;
        if (out_idx < HIDDEN) {
            var sum: f32 = 0.0;
            for (var k = 0u; k < HIDDEN; k++) {
                sum += input[k] * q_weight[out_idx * HIDDEN + k];
            }
            q_local[out_idx] = sum;
        }
    }

    // Compute K projection (128 outputs)
    if (thread_id < KV_DIM) {
        var sum: f32 = 0.0;
        for (var k = 0u; k < HIDDEN; k++) {
            sum += input[k] * k_weight[thread_id * HIDDEN + k];
        }
        k_local[thread_id] = sum;
    }

    // Compute V projection (128 outputs)
    if (thread_id < KV_DIM) {
        var sum: f32 = 0.0;
        for (var k = 0u; k < HIDDEN; k++) {
            sum += input[k] * v_weight[thread_id * HIDDEN + k];
        }
        v_local[thread_id] = sum;
    }

    workgroupBarrier();

    // Phase 2: Apply RoPE to Q and K
    // Q: apply to all 14 heads
    for (var i = 0u; i < 4u; i++) {
        let out_idx = thread_id * 4u + i;
        if (out_idx < HIDDEN) {
            let head = out_idx / HEAD_DIM;
            let dim_in_head = out_idx % HEAD_DIM;
            let pair_idx = dim_in_head / 2u;

            if (dim_in_head % 2u == 0u && dim_in_head + 1u < HEAD_DIM) {
                let freq = get_rope_freq(seq_pos, pair_idx);
                let x = vec2<f32>(q_local[out_idx], q_local[out_idx + 1u]);
                let rotated = apply_rope(x, freq);
                q_local[out_idx] = rotated.x;
                q_local[out_idx + 1u] = rotated.y;
            }
        }
    }

    // K: apply to 2 heads
    if (thread_id < KV_DIM) {
        let dim_in_head = thread_id % HEAD_DIM;
        let pair_idx = dim_in_head / 2u;

        if (dim_in_head % 2u == 0u && dim_in_head + 1u < HEAD_DIM) {
            let freq = get_rope_freq(seq_pos, pair_idx);
            let x = vec2<f32>(k_local[thread_id], k_local[thread_id + 1u]);
            let rotated = apply_rope(x, freq);
            k_local[thread_id] = rotated.x;
            k_local[thread_id + 1u] = rotated.y;
        }
    }

    workgroupBarrier();

    // Update KV cache
    if (thread_id < KV_DIM) {
        k_cache_out[seq_pos * KV_DIM + thread_id] = k_local[thread_id];
        v_cache_out[seq_pos * KV_DIM + thread_id] = v_local[thread_id];
    }

    workgroupBarrier();

    // Phase 3: Compute attention for each Q head
    // Each thread handles multiple heads
    for (var i = 0u; i < 4u; i++) {
        let head_idx = thread_id * 4u + i;
        if (head_idx < NUM_HEADS) {
            let kv_head = head_idx / HEADS_PER_KV;
            let q_offset = head_idx * HEAD_DIM;
            let kv_offset = kv_head * HEAD_DIM;

            // Compute attention scores for all positions up to seq_pos
            var max_score: f32 = -1e10;
            var scores: array<f32, 128>;  // Max sequence length we support
            let scale = 1.0 / sqrt(f32(HEAD_DIM));

            for (var pos = 0u; pos <= seq_pos; pos++) {
                var score: f32 = 0.0;
                for (var d = 0u; d < HEAD_DIM; d++) {
                    let q_val = q_local[q_offset + d];
                    let k_val = k_cache[pos * KV_DIM + kv_offset + d];
                    score += q_val * k_val;
                }
                score *= scale;
                scores[pos] = score;
                max_score = max(max_score, score);
            }

            // Softmax
            var sum_exp: f32 = 0.0;
            for (var pos = 0u; pos <= seq_pos; pos++) {
                scores[pos] = exp(scores[pos] - max_score);
                sum_exp += scores[pos];
            }

            // Weighted sum of V
            for (var d = 0u; d < HEAD_DIM; d++) {
                var weighted_sum: f32 = 0.0;
                for (var pos = 0u; pos <= seq_pos; pos++) {
                    let attn_weight = scores[pos] / sum_exp;
                    let v_val = v_cache[pos * KV_DIM + kv_offset + d];
                    weighted_sum += attn_weight * v_val;
                }
                attn_out[q_offset + d] = weighted_sum;
            }
        }
    }

    workgroupBarrier();

    // Phase 4: O projection
    for (var i = 0u; i < 4u; i++) {
        let out_idx = thread_id * 4u + i;
        if (out_idx < HIDDEN) {
            var sum: f32 = 0.0;
            for (var k = 0u; k < HIDDEN; k++) {
                sum += attn_out[k] * o_weight[out_idx * HIDDEN + k];
            }
            output[out_idx] = sum;
        }
    }
}
)wgsl";

            struct MegaAttentionKernel
            {
                wgpu::BindGroupLayout bind_group_layout;
                wgpu::ComputePipeline pipeline;
            };

            static MegaAttentionKernel *mega_attention_kernel = nullptr;

            MegaAttentionKernel create_mega_attention_kernel()
            {
                wgpu::ShaderSourceWGSL shader_source{
                    wgpu::ShaderSourceWGSL::Init{
                        nullptr,
                        wgpu::StringView{mega_attention_shader.c_str(), mega_attention_shader.size()},
                    }};

                wgpu::ShaderModuleDescriptor shader_descriptor{};
                shader_descriptor.nextInChain = &shader_source;
                shader_descriptor.label = "mega_attention";

                core::WebGPUContext &ctx = core::getWebGPUContext();
                wgpu::ShaderModule shader_module = ctx.getDevice().CreateShaderModule(&shader_descriptor);

                wgpu::BindGroupLayoutEntry bindings[11]{};

                // input buffer
                bindings[0].binding = 0;
                bindings[0].visibility = wgpu::ShaderStage::Compute;
                bindings[0].buffer.type = wgpu::BufferBindingType::ReadOnlyStorage;

                // q_weight
                bindings[1].binding = 1;
                bindings[1].visibility = wgpu::ShaderStage::Compute;
                bindings[1].buffer.type = wgpu::BufferBindingType::ReadOnlyStorage;

                // k_weight
                bindings[2].binding = 2;
                bindings[2].visibility = wgpu::ShaderStage::Compute;
                bindings[2].buffer.type = wgpu::BufferBindingType::ReadOnlyStorage;

                // v_weight
                bindings[3].binding = 3;
                bindings[3].visibility = wgpu::ShaderStage::Compute;
                bindings[3].buffer.type = wgpu::BufferBindingType::ReadOnlyStorage;

                // o_weight
                bindings[4].binding = 4;
                bindings[4].visibility = wgpu::ShaderStage::Compute;
                bindings[4].buffer.type = wgpu::BufferBindingType::ReadOnlyStorage;

                // k_cache (read)
                bindings[5].binding = 5;
                bindings[5].visibility = wgpu::ShaderStage::Compute;
                bindings[5].buffer.type = wgpu::BufferBindingType::ReadOnlyStorage;

                // v_cache (read)
                bindings[6].binding = 6;
                bindings[6].visibility = wgpu::ShaderStage::Compute;
                bindings[6].buffer.type = wgpu::BufferBindingType::ReadOnlyStorage;

                // output
                bindings[7].binding = 7;
                bindings[7].visibility = wgpu::ShaderStage::Compute;
                bindings[7].buffer.type = wgpu::BufferBindingType::Storage;

                // k_cache_out
                bindings[8].binding = 8;
                bindings[8].visibility = wgpu::ShaderStage::Compute;
                bindings[8].buffer.type = wgpu::BufferBindingType::Storage;

                // v_cache_out
                bindings[9].binding = 9;
                bindings[9].visibility = wgpu::ShaderStage::Compute;
                bindings[9].buffer.type = wgpu::BufferBindingType::Storage;

                // params
                bindings[10].binding = 10;
                bindings[10].visibility = wgpu::ShaderStage::Compute;
                bindings[10].buffer.type = wgpu::BufferBindingType::Uniform;

                wgpu::BindGroupLayoutDescriptor layout_descriptor{};
                layout_descriptor.entryCount = 11;
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

                return MegaAttentionKernel{bind_group_layout, pipeline};
            }

            MegaAttentionKernel &get_mega_attention_kernel()
            {
                if (mega_attention_kernel != nullptr)
                {
                    return *mega_attention_kernel;
                }
                static MegaAttentionKernel k = create_mega_attention_kernel();
                mega_attention_kernel = &k;
                return *mega_attention_kernel;
            }
        }

        // Mega attention: entire attention block in one dispatch
        std::tuple<at::Tensor, at::Tensor, at::Tensor> mega_attention(
            const at::Tensor &input,
            const at::Tensor &q_weight,
            const at::Tensor &k_weight,
            const at::Tensor &v_weight,
            const at::Tensor &o_weight,
            const at::Tensor &k_cache,
            const at::Tensor &v_cache,
            int64_t seq_pos)
        {
            TORCH_CHECK(input.device().type() == c10::DeviceType::PrivateUse1);
            TORCH_CHECK(input.scalar_type() == c10::ScalarType::Float);

            const int64_t hidden_size = 896;
            const int64_t kv_dim = 128;

            auto input_contig = input.contiguous();
            auto q_w_contig = q_weight.contiguous();
            auto k_w_contig = k_weight.contiguous();
            auto v_w_contig = v_weight.contiguous();
            auto o_w_contig = o_weight.contiguous();

            at::Tensor output = at::empty({1, hidden_size}, input.options());
            at::Tensor k_cache_out = k_cache.clone();
            at::Tensor v_cache_out = v_cache.clone();

            MegaAttentionKernel &kernel = get_mega_attention_kernel();

            core::WebGPUAllocation *input_alloc = static_cast<core::WebGPUAllocation *>(input_contig.storage().data_ptr().get());
            core::WebGPUAllocation *q_w_alloc = static_cast<core::WebGPUAllocation *>(q_w_contig.storage().data_ptr().get());
            core::WebGPUAllocation *k_w_alloc = static_cast<core::WebGPUAllocation *>(k_w_contig.storage().data_ptr().get());
            core::WebGPUAllocation *v_w_alloc = static_cast<core::WebGPUAllocation *>(v_w_contig.storage().data_ptr().get());
            core::WebGPUAllocation *o_w_alloc = static_cast<core::WebGPUAllocation *>(o_w_contig.storage().data_ptr().get());
            core::WebGPUAllocation *k_cache_alloc = static_cast<core::WebGPUAllocation *>(k_cache.storage().data_ptr().get());
            core::WebGPUAllocation *v_cache_alloc = static_cast<core::WebGPUAllocation *>(v_cache.storage().data_ptr().get());
            core::WebGPUAllocation *output_alloc = static_cast<core::WebGPUAllocation *>(output.storage().data_ptr().get());
            core::WebGPUAllocation *k_cache_out_alloc = static_cast<core::WebGPUAllocation *>(k_cache_out.storage().data_ptr().get());
            core::WebGPUAllocation *v_cache_out_alloc = static_cast<core::WebGPUAllocation *>(v_cache_out.storage().data_ptr().get());

            struct Params
            {
                uint32_t seq_pos;
                uint32_t _pad1, _pad2, _pad3;
            };

            Params params{};
            params.seq_pos = static_cast<uint32_t>(seq_pos);

            core::WebGPUContext &ctx = core::getWebGPUContext();
            wgpu::Buffer params_buffer = core::acquireUniformBuffer(&params, sizeof(Params));

            wgpu::BindGroupEntry entries[11]{};
            entries[0] = {nullptr, 0, input_alloc->buffer, 0, input_alloc->buffer.GetSize()};
            entries[1] = {nullptr, 1, q_w_alloc->buffer, 0, q_w_alloc->buffer.GetSize()};
            entries[2] = {nullptr, 2, k_w_alloc->buffer, 0, k_w_alloc->buffer.GetSize()};
            entries[3] = {nullptr, 3, v_w_alloc->buffer, 0, v_w_alloc->buffer.GetSize()};
            entries[4] = {nullptr, 4, o_w_alloc->buffer, 0, o_w_alloc->buffer.GetSize()};
            entries[5] = {nullptr, 5, k_cache_alloc->buffer, 0, k_cache_alloc->buffer.GetSize()};
            entries[6] = {nullptr, 6, v_cache_alloc->buffer, 0, v_cache_alloc->buffer.GetSize()};
            entries[7] = {nullptr, 7, output_alloc->buffer, 0, output_alloc->buffer.GetSize()};
            entries[8] = {nullptr, 8, k_cache_out_alloc->buffer, 0, k_cache_out_alloc->buffer.GetSize()};
            entries[9] = {nullptr, 9, v_cache_out_alloc->buffer, 0, v_cache_out_alloc->buffer.GetSize()};
            entries[10] = {nullptr, 10, params_buffer, 0, sizeof(Params)};

            wgpu::BindGroupDescriptor bind_group_descriptor{};
            bind_group_descriptor.layout = kernel.bind_group_layout;
            bind_group_descriptor.entryCount = 11;
            bind_group_descriptor.entries = entries;

            wgpu::BindGroup bind_group = ctx.getDevice().CreateBindGroup(&bind_group_descriptor);

            // Single workgroup of 256 threads
            core::dispatchCompute(kernel.pipeline, bind_group, 1, 1, 1);

            return std::make_tuple(output, k_cache_out, v_cache_out);
        }
    }

    TORCH_LIBRARY_FRAGMENT(webgpu, m)
    {
        m.def("mega_attention(Tensor input, Tensor q_weight, Tensor k_weight, Tensor v_weight, Tensor o_weight, Tensor k_cache, Tensor v_cache, int seq_pos) -> (Tensor, Tensor, Tensor)");
    }

    TORCH_LIBRARY_IMPL(webgpu, PrivateUse1, m)
    {
        m.impl("mega_attention", TORCH_FN(ops::mega_attention));
    }
}
