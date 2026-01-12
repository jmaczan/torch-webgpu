#include <ATen/ATen.h>
#include <torch/library.h>
#include <webgpu/webgpu_cpp.h>
#include "utils/string.h"
#include "core/webgpu_context.h"
#include "core/webgpu_allocator.h"

namespace torch_webgpu
{
    namespace ops
    {
        namespace
        {
            enum class CompareOp
            {
                Eq,
                Ne,
                Lt,
                Le,
                Gt,
                Ge
            };

            struct CacheHash
            {
                template <typename T>
                std::size_t operator()(T t) const noexcept
                {
                    return static_cast<std::size_t>(t);
                }
            };

            // Comparison shader template - outputs 1.0 for true, 0.0 for false
            const std::string compare_shader_template = R"wgsl(
const MAX_DIMS: u32 = 8u;

struct Params {
    length: u32,
    ndim: u32,
    scalar_val: f32,
    use_scalar: u32,

    out_offset: u32,
    self_offset: u32,
    other_offset: u32,
    _pad: u32,

    out_strides: array<u32, MAX_DIMS>,
    self_strides: array<u32, MAX_DIMS>,
    other_strides: array<u32, MAX_DIMS>,
    shape: array<u32, MAX_DIMS>,
};

@group(0) @binding(0)
var<storage, read> selfBuffer: array<f32>;

@group(0) @binding(1)
var<storage, read> otherBuffer: array<f32>;

@group(0) @binding(2)
var<storage, read_write> outBuffer: array<f32>;

@group(0) @binding(3)
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
    var idx_other: u32 = 0u;

    for (var d: u32 = 0u; d < params.ndim; d++) {
        let c = coord[d];
        idx_out += c * params.out_strides[d];
        idx_self += c * params.self_strides[d];
        idx_other += c * params.other_strides[d];
    }

    idx_out += params.out_offset;
    idx_self += params.self_offset;
    idx_other += params.other_offset;

    let self_val = selfBuffer[idx_self];
    var other_val: f32;
    if (params.use_scalar == 1u) {
        other_val = params.scalar_val;
    } else {
        other_val = otherBuffer[idx_other];
    }

    var result: f32 = 0.0;
    __COMPARE_OP__
    outBuffer[idx_out] = result;
}
)wgsl";

            struct CompareKernel
            {
                wgpu::BindGroupLayout bind_group_layout;
                wgpu::ComputePipeline pipeline;
            };

            std::string get_compare_shader(CompareOp op)
            {
                std::string shader = compare_shader_template;
                std::string op_impl;

                switch (op)
                {
                case CompareOp::Eq:
                    op_impl = "if (self_val == other_val) { result = 1.0; }";
                    break;
                case CompareOp::Ne:
                    op_impl = "if (self_val != other_val) { result = 1.0; }";
                    break;
                case CompareOp::Lt:
                    op_impl = "if (self_val < other_val) { result = 1.0; }";
                    break;
                case CompareOp::Le:
                    op_impl = "if (self_val <= other_val) { result = 1.0; }";
                    break;
                case CompareOp::Gt:
                    op_impl = "if (self_val > other_val) { result = 1.0; }";
                    break;
                case CompareOp::Ge:
                    op_impl = "if (self_val >= other_val) { result = 1.0; }";
                    break;
                }

                replace_string(shader, "__COMPARE_OP__", op_impl);
                return shader;
            }

            CompareKernel &get_compare_kernel(CompareOp op)
            {
                static std::unordered_map<CompareOp, CompareKernel, CacheHash> kernel_cache;
                auto cached_kernel = kernel_cache.find(op);
                if (cached_kernel != kernel_cache.end())
                {
                    return cached_kernel->second;
                }

                std::string shader = get_compare_shader(op);

                wgpu::ShaderSourceWGSL shader_source{
                    wgpu::ShaderSourceWGSL::Init{
                        nullptr,
                        wgpu::StringView{shader.c_str(), shader.size()},
                    }};

                wgpu::ShaderModuleDescriptor shader_descriptor{};
                shader_descriptor.nextInChain = &shader_source;
                shader_descriptor.label = "Compare shader";
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
                auto [iter, inserted] = kernel_cache.emplace(op, CompareKernel{bind_group_layout, pipeline});
                TORCH_CHECK(inserted, "Failed to insert compare kernel to cache");
                return iter->second;
            }

            template <CompareOp Op>
            at::Tensor compare_tensor_tensor(const at::Tensor &self, const at::Tensor &other)
            {
                TORCH_CHECK(self.device().type() == c10::DeviceType::PrivateUse1);
                TORCH_CHECK(other.device().type() == c10::DeviceType::PrivateUse1);

                // Convert to float32 if needed for the shader
                at::Tensor self_f = self.scalar_type() == c10::ScalarType::Float ? self : self.to(c10::ScalarType::Float);
                at::Tensor other_f = other.scalar_type() == c10::ScalarType::Float ? other : other.to(c10::ScalarType::Float);

                // Broadcast shapes
                auto out_shape = at::infer_size(self_f.sizes(), other_f.sizes());
                at::Tensor out = at::empty(out_shape, self.options().dtype(at::kBool));

                if (out.numel() == 0)
                    return out;

                // Expand tensors to output shape
                auto self_expanded = self_f.expand(out_shape).contiguous();
                auto other_expanded = other_f.expand(out_shape).contiguous();

                // Create float output tensor for shader
                at::Tensor out_float = at::empty(out_shape, self_f.options());

                CompareKernel &kernel = get_compare_kernel(Op);

                core::WebGPUAllocation *self_allocation = static_cast<core::WebGPUAllocation *>(self_expanded.storage().data_ptr().get());
                core::WebGPUAllocation *other_allocation = static_cast<core::WebGPUAllocation *>(other_expanded.storage().data_ptr().get());
                core::WebGPUAllocation *out_allocation = static_cast<core::WebGPUAllocation *>(out_float.storage().data_ptr().get());

                wgpu::Buffer self_buffer = self_allocation->buffer;
                wgpu::Buffer other_buffer = other_allocation->buffer;
                wgpu::Buffer out_buffer = out_allocation->buffer;

                constexpr uint32_t MAX_DIMS = 8;
                auto ndim = static_cast<uint32_t>(out_float.dim());
                TORCH_CHECK(ndim <= MAX_DIMS);

                struct Params
                {
                    uint32_t length;
                    uint32_t ndim;
                    float scalar_val;
                    uint32_t use_scalar;

                    uint32_t out_offset;
                    uint32_t self_offset;
                    uint32_t other_offset;
                    uint32_t _pad;

                    uint32_t out_strides[MAX_DIMS];
                    uint32_t self_strides[MAX_DIMS];
                    uint32_t other_strides[MAX_DIMS];
                    uint32_t shape[MAX_DIMS];
                };

                Params params{};
                params.length = static_cast<uint32_t>(out_float.numel());
                params.ndim = ndim;
                params.scalar_val = 0.0f;
                params.use_scalar = 0;
                params.out_offset = static_cast<uint32_t>(out_float.storage_offset());
                params.self_offset = static_cast<uint32_t>(self_expanded.storage_offset());
                params.other_offset = static_cast<uint32_t>(other_expanded.storage_offset());
                params._pad = 0;

                for (uint32_t d = 0; d < MAX_DIMS; ++d)
                {
                    params.out_strides[d] = 0;
                    params.self_strides[d] = 0;
                    params.other_strides[d] = 0;
                    params.shape[d] = 1;
                }

                for (uint32_t i = 0; i < ndim; ++i)
                {
                    params.shape[i] = static_cast<uint32_t>(out_float.size(i));
                    params.out_strides[i] = static_cast<uint32_t>(out_float.stride(i));
                    params.self_strides[i] = static_cast<uint32_t>(self_expanded.stride(i));
                    params.other_strides[i] = static_cast<uint32_t>(other_expanded.stride(i));
                }

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
                bind_group_entries[0].buffer = self_buffer;
                bind_group_entries[0].offset = 0;
                bind_group_entries[0].size = self_buffer.GetSize();

                bind_group_entries[1].binding = 1;
                bind_group_entries[1].buffer = other_buffer;
                bind_group_entries[1].offset = 0;
                bind_group_entries[1].size = other_buffer.GetSize();

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
                uint32_t num_workgroups = (params.length + workgroup_size - 1) / workgroup_size;

                pass_encoder.DispatchWorkgroups(num_workgroups);
                pass_encoder.End();

                wgpu::CommandBuffer command_buffer = encoder.Finish();
                ctx.getQueue().Submit(1, &command_buffer);

                // Convert float result to bool
                return out_float.to(at::kBool);
            }

            template <CompareOp Op>
            at::Tensor compare_tensor_scalar(const at::Tensor &self, const at::Scalar &other)
            {
                // Create a scalar tensor and use tensor-tensor comparison
                at::Tensor other_tensor = at::full({}, other.toFloat(), self.options().dtype(at::kFloat));
                at::Tensor self_f = self.scalar_type() == c10::ScalarType::Float ? self : self.to(c10::ScalarType::Float);
                return compare_tensor_tensor<Op>(self_f, other_tensor);
            }
        }

        // Eq
        at::Tensor eq_tensor(const at::Tensor &self, const at::Tensor &other)
        {
            return compare_tensor_tensor<CompareOp::Eq>(self, other);
        }
        at::Tensor eq_scalar(const at::Tensor &self, const at::Scalar &other)
        {
            return compare_tensor_scalar<CompareOp::Eq>(self, other);
        }

        // Ne
        at::Tensor ne_tensor(const at::Tensor &self, const at::Tensor &other)
        {
            return compare_tensor_tensor<CompareOp::Ne>(self, other);
        }
        at::Tensor ne_scalar(const at::Tensor &self, const at::Scalar &other)
        {
            return compare_tensor_scalar<CompareOp::Ne>(self, other);
        }

        // Lt
        at::Tensor lt_tensor(const at::Tensor &self, const at::Tensor &other)
        {
            return compare_tensor_tensor<CompareOp::Lt>(self, other);
        }
        at::Tensor lt_scalar(const at::Tensor &self, const at::Scalar &other)
        {
            return compare_tensor_scalar<CompareOp::Lt>(self, other);
        }

        // Le
        at::Tensor le_tensor(const at::Tensor &self, const at::Tensor &other)
        {
            return compare_tensor_tensor<CompareOp::Le>(self, other);
        }
        at::Tensor le_scalar(const at::Tensor &self, const at::Scalar &other)
        {
            return compare_tensor_scalar<CompareOp::Le>(self, other);
        }

        // Gt
        at::Tensor gt_tensor(const at::Tensor &self, const at::Tensor &other)
        {
            return compare_tensor_tensor<CompareOp::Gt>(self, other);
        }
        at::Tensor gt_scalar(const at::Tensor &self, const at::Scalar &other)
        {
            return compare_tensor_scalar<CompareOp::Gt>(self, other);
        }

        // Ge
        at::Tensor ge_tensor(const at::Tensor &self, const at::Tensor &other)
        {
            return compare_tensor_tensor<CompareOp::Ge>(self, other);
        }
        at::Tensor ge_scalar(const at::Tensor &self, const at::Scalar &other)
        {
            return compare_tensor_scalar<CompareOp::Ge>(self, other);
        }

        // All - check if all elements are true
        at::Tensor all(const at::Tensor &self)
        {
            // Move to CPU, perform all(), return result
            // This is simple but works; can optimize with GPU kernel later
            auto cpu_tensor = self.to(at::kCPU);
            auto result = at::all(cpu_tensor);
            return result.to(self.device());
        }

        at::Tensor &all_out(const at::Tensor &self, int64_t dim, bool keepdim, at::Tensor &out)
        {
            // CPU fallback with dim support
            auto cpu_tensor = self.to(at::kCPU);
            auto cpu_out = out.to(at::kCPU);
            at::all_out(cpu_out, cpu_tensor, dim, keepdim);
            out.copy_(cpu_out);
            return out;
        }

        // Any - check if any element is true
        at::Tensor any(const at::Tensor &self)
        {
            auto cpu_tensor = self.to(at::kCPU);
            auto result = at::any(cpu_tensor);
            return result.to(self.device());
        }

        at::Tensor &any_out(const at::Tensor &self, int64_t dim, bool keepdim, at::Tensor &out)
        {
            // CPU fallback with dim support
            auto cpu_tensor = self.to(at::kCPU);
            auto cpu_out = out.to(at::kCPU);
            at::any_out(cpu_out, cpu_tensor, dim, keepdim);
            out.copy_(cpu_out);
            return out;
        }

        // isin - check if elements of first tensor are in second tensor
        at::Tensor isin_tensor_tensor(const at::Tensor &elements, const at::Tensor &test_elements, bool assume_unique, bool invert)
        {
            // For small tensors (like token IDs), use CPU fallback
            auto elements_cpu = elements.to(at::kCPU);
            auto test_elements_cpu = test_elements.to(at::kCPU);
            auto result = at::isin(elements_cpu, test_elements_cpu, assume_unique, invert);
            return result.to(elements.device());
        }

        at::Tensor &isin_tensor_tensor_out(const at::Tensor &elements, const at::Tensor &test_elements, bool assume_unique, bool invert, at::Tensor &out)
        {
            auto result = isin_tensor_tensor(elements, test_elements, assume_unique, invert);
            out.copy_(result);
            return out;
        }

        // bitwise_not - used for boolean tensor negation (~)
        at::Tensor bitwise_not_impl(const at::Tensor &self)
        {
            // For small tensors (like boolean masks), use CPU fallback
            auto self_cpu = self.to(at::kCPU);
            auto result = at::bitwise_not(self_cpu);
            return result.to(self.device());
        }

        at::Tensor &bitwise_not_out_impl(const at::Tensor &self, at::Tensor &out)
        {
            auto result = bitwise_not_impl(self);
            out.copy_(result);
            return out;
        }

        // bitwise_and - used for boolean tensor AND (&)
        at::Tensor bitwise_and_tensor(const at::Tensor &self, const at::Tensor &other)
        {
            auto self_cpu = self.to(at::kCPU);
            auto other_cpu = other.to(at::kCPU);
            auto result = at::bitwise_and(self_cpu, other_cpu);
            return result.to(self.device());
        }

        at::Tensor &bitwise_and_tensor_out(const at::Tensor &self, const at::Tensor &other, at::Tensor &out)
        {
            auto result = bitwise_and_tensor(self, other);
            out.copy_(result);
            return out;
        }

        // bitwise_or - used for boolean tensor OR (|)
        at::Tensor bitwise_or_tensor(const at::Tensor &self, const at::Tensor &other)
        {
            auto self_cpu = self.to(at::kCPU);
            auto other_cpu = other.to(at::kCPU);
            auto result = at::bitwise_or(self_cpu, other_cpu);
            return result.to(self.device());
        }

        at::Tensor &bitwise_or_tensor_out(const at::Tensor &self, const at::Tensor &other, at::Tensor &out)
        {
            auto result = bitwise_or_tensor(self, other);
            out.copy_(result);
            return out;
        }
    }

    TORCH_LIBRARY_IMPL(aten, PrivateUse1, m)
    {
        m.impl("eq.Tensor", TORCH_FN(ops::eq_tensor));
        m.impl("eq.Scalar", TORCH_FN(ops::eq_scalar));
        m.impl("ne.Tensor", TORCH_FN(ops::ne_tensor));
        m.impl("ne.Scalar", TORCH_FN(ops::ne_scalar));
        m.impl("lt.Tensor", TORCH_FN(ops::lt_tensor));
        m.impl("lt.Scalar", TORCH_FN(ops::lt_scalar));
        m.impl("le.Tensor", TORCH_FN(ops::le_tensor));
        m.impl("le.Scalar", TORCH_FN(ops::le_scalar));
        m.impl("gt.Tensor", TORCH_FN(ops::gt_tensor));
        m.impl("gt.Scalar", TORCH_FN(ops::gt_scalar));
        m.impl("ge.Tensor", TORCH_FN(ops::ge_tensor));
        m.impl("ge.Scalar", TORCH_FN(ops::ge_scalar));
        m.impl("all", TORCH_FN(ops::all));
        m.impl("all.out", TORCH_FN(ops::all_out));
        m.impl("any", TORCH_FN(ops::any));
        m.impl("any.out", TORCH_FN(ops::any_out));
        m.impl("isin.Tensor_Tensor", TORCH_FN(ops::isin_tensor_tensor));
        m.impl("isin.Tensor_Tensor_out", TORCH_FN(ops::isin_tensor_tensor_out));
        m.impl("bitwise_not", TORCH_FN(ops::bitwise_not_impl));
        m.impl("bitwise_not.out", TORCH_FN(ops::bitwise_not_out_impl));
        m.impl("bitwise_and.Tensor", TORCH_FN(ops::bitwise_and_tensor));
        m.impl("bitwise_and.Tensor_out", TORCH_FN(ops::bitwise_and_tensor_out));
        m.impl("bitwise_or.Tensor", TORCH_FN(ops::bitwise_or_tensor));
        m.impl("bitwise_or.Tensor_out", TORCH_FN(ops::bitwise_or_tensor_out));
    }
}
