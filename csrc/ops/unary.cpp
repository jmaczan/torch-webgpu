#include <ATen/ATen.h>
#include <torch/library.h>
#include <webgpu/webgpu_cpp.h>
#include "utils/string.h"
#include "core/webgpu_context.h"
#include "unary.h"

namespace torch_webgpu
{
    namespace ops
    {
        namespace
        {
            struct CacheHash
            {
                template <typename T>
                std::size_t operator()(T t) const noexcept
                {
                    return static_cast<std::size_t>(t);
                }
            };

            const std::string unary_shader_template = R"wgsl(
const MAX_DIMS: u32 = 8u;

struct Params {
    length: u32,
    ndim: u32,
    _pad: u32,

    out_offset: u32,
    self_offset: u32,
    _pad2: u32,

    out_strides: array<u32, MAX_DIMS>,
    self_strides: array<u32, MAX_DIMS>,
    shape: array<u32, MAX_DIMS>,
};

@group(0) @binding(0)
var<storage, read> selfBuffer: array<f32>;

@group(0) @binding(1)
var<storage, read_write> outBuffer: array<f32>;

@group(0) @binding(2)
var<uniform> params: Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.length) { return; }

    var remaining = i;
    var coord: array<u32, MAX_DIMS>;

    for (var d: i32 = i32(params.ndim) - 1; d >= 0; d--) {
        let ud = u32(d);
        let s = params.shape[ud];
        coord[ud] = remaining % s;
        remaining = remaining / s;
    }

    var idx_out: u32 = 0u;
    var idx_self: u32 = 0u;

    for (var d: u32 = 0u; d < params.ndim; d++) {
        let c = coord[d];
        idx_out += c * params.out_strides[d];
        idx_self += c * params.self_strides[d];
    }

    idx_out += params.out_offset;
    idx_self += params.self_offset;

    outBuffer[idx_out] = __UNARY_OP__;
}
)wgsl";
        }

        std::string get_unary_shader(UnaryOp unary_op)
        {
            std::string shader = unary_shader_template;
            std::string op_impl;
            switch (unary_op)
            {
            case UnaryOp::Copy:
                op_impl = "selfBuffer[idx_self]";
                break;
            case UnaryOp::ReLU:
                op_impl = "max(0.0, selfBuffer[idx_self])";
                break;
            case UnaryOp::GeLU:
                op_impl = "0.5*selfBuffer[idx_self]*(1 + tanh(sqrt(2/radians(180)) * (selfBuffer[idx_self] + 0.044715 * pow(selfBuffer[idx_self], 3))))";
                break;
            case UnaryOp::SiLU:
                op_impl = "selfBuffer[idx_self] * ( 1 / (1 + exp(-1 * selfBuffer[idx_self])))";
                break;
            default:
                TORCH_CHECK(false, "Unsupported unary op, can't produce a WGSL shader");
            }

            replace_string(shader, "__UNARY_OP__", op_impl);

            return shader;
        }

        UnaryKernel &get_unary_kernel(UnaryOp unary_op)
        {
            static std::unordered_map<UnaryOp, UnaryKernel, CacheHash> kernel_cache;
            auto cached_kernel = kernel_cache.find(unary_op);
            if (cached_kernel != kernel_cache.end())
            {
                return cached_kernel->second;
            }

            std::string shader = get_unary_shader(unary_op);

            wgpu::ShaderSourceWGSL shader_source{
                wgpu::ShaderSourceWGSL::Init{
                    nullptr,
                    wgpu::StringView{shader.c_str(), shader.size()},
                }};

            wgpu::ShaderModuleDescriptor shader_descriptor{};
            shader_descriptor.nextInChain = &shader_source;
            shader_descriptor.label = "Unary kernel";
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
            auto [iter, inserted] = kernel_cache.emplace(unary_op, UnaryKernel{bind_group_layout, pipeline});
            TORCH_CHECK(inserted, "Failed to insert a kernel to the cache");
            return iter->second;
        }

    }
}