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
            // Shader for Long to Int conversion
            const std::string long_to_int_shader = R"wgsl(
struct Params {
    length: u32,
    src_offset: u32,
    dst_offset: u32,
    _pad: u32,
};

@group(0) @binding(0)
var<storage, read> srcBuffer: array<i32>;

@group(0) @binding(1)
var<storage, read_write> dstBuffer: array<i32>;

@group(0) @binding(2)
var<uniform> params: Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.length) { return; }

    // Long is stored as two i32 values (low, high) in little-endian
    // We just take the low part since indices are typically small
    let src_idx = params.src_offset + idx * 2u;
    let value = srcBuffer[src_idx]; // low 32 bits
    dstBuffer[params.dst_offset + idx] = value;
}
)wgsl";

            // Shader for Float to Int conversion
            const std::string float_to_int_shader = R"wgsl(
struct Params {
    length: u32,
    src_offset: u32,
    dst_offset: u32,
    _pad: u32,
};

@group(0) @binding(0)
var<storage, read> srcBuffer: array<f32>;

@group(0) @binding(1)
var<storage, read_write> dstBuffer: array<i32>;

@group(0) @binding(2)
var<uniform> params: Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.length) { return; }

    let value = srcBuffer[params.src_offset + idx];
    dstBuffer[params.dst_offset + idx] = i32(value);
}
)wgsl";

            // Shader for Int to Float conversion
            const std::string int_to_float_shader = R"wgsl(
struct Params {
    length: u32,
    src_offset: u32,
    dst_offset: u32,
    _pad: u32,
};

@group(0) @binding(0)
var<storage, read> srcBuffer: array<i32>;

@group(0) @binding(1)
var<storage, read_write> dstBuffer: array<f32>;

@group(0) @binding(2)
var<uniform> params: Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.length) { return; }

    let value = srcBuffer[params.src_offset + idx];
    dstBuffer[params.dst_offset + idx] = f32(value);
}
)wgsl";

            // Shader for Long to Float conversion
            const std::string long_to_float_shader = R"wgsl(
struct Params {
    length: u32,
    src_offset: u32,
    dst_offset: u32,
    _pad: u32,
};

@group(0) @binding(0)
var<storage, read> srcBuffer: array<i32>;

@group(0) @binding(1)
var<storage, read_write> dstBuffer: array<f32>;

@group(0) @binding(2)
var<uniform> params: Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.length) { return; }

    // Long is stored as two i32 values (low, high) in little-endian
    let src_idx = params.src_offset + idx * 2u;
    let low = srcBuffer[src_idx];
    let high = srcBuffer[src_idx + 1u];

    // Reconstruct as float (only works for small values)
    let value = f32(low) + f32(high) * 4294967296.0;
    dstBuffer[params.dst_offset + idx] = value;
}
)wgsl";

            struct TypeConversionKernel
            {
                wgpu::BindGroupLayout bind_group_layout;
                wgpu::ComputePipeline pipeline;
            };

            TypeConversionKernel create_type_conversion_kernel(const std::string &shader_code, const char *label)
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

                return TypeConversionKernel{bind_group_layout, pipeline};
            }

            static TypeConversionKernel *long_to_int_kernel = nullptr;
            static TypeConversionKernel *float_to_int_kernel = nullptr;
            static TypeConversionKernel *int_to_float_kernel = nullptr;
            static TypeConversionKernel *long_to_float_kernel = nullptr;

            TypeConversionKernel &get_long_to_int_kernel()
            {
                if (long_to_int_kernel == nullptr)
                {
                    static TypeConversionKernel k = create_type_conversion_kernel(long_to_int_shader, "Long to Int kernel");
                    long_to_int_kernel = &k;
                }
                return *long_to_int_kernel;
            }

            TypeConversionKernel &get_float_to_int_kernel()
            {
                if (float_to_int_kernel == nullptr)
                {
                    static TypeConversionKernel k = create_type_conversion_kernel(float_to_int_shader, "Float to Int kernel");
                    float_to_int_kernel = &k;
                }
                return *float_to_int_kernel;
            }

            TypeConversionKernel &get_int_to_float_kernel()
            {
                if (int_to_float_kernel == nullptr)
                {
                    static TypeConversionKernel k = create_type_conversion_kernel(int_to_float_shader, "Int to Float kernel");
                    int_to_float_kernel = &k;
                }
                return *int_to_float_kernel;
            }

            TypeConversionKernel &get_long_to_float_kernel()
            {
                if (long_to_float_kernel == nullptr)
                {
                    static TypeConversionKernel k = create_type_conversion_kernel(long_to_float_shader, "Long to Float kernel");
                    long_to_float_kernel = &k;
                }
                return *long_to_float_kernel;
            }

            void run_type_conversion_kernel(TypeConversionKernel &kernel, const at::Tensor &src, at::Tensor &dst, bool src_is_long = false)
            {
                core::WebGPUAllocation *src_allocation = static_cast<core::WebGPUAllocation *>(src.storage().data_ptr().get());
                core::WebGPUAllocation *dst_allocation = static_cast<core::WebGPUAllocation *>(dst.storage().data_ptr().get());

                wgpu::Buffer src_buffer = src_allocation->buffer;
                wgpu::Buffer dst_buffer = dst_allocation->buffer;

                struct Params
                {
                    uint32_t length;
                    uint32_t src_offset;
                    uint32_t dst_offset;
                    uint32_t _pad;
                };

                Params params{};
                params.length = static_cast<uint32_t>(dst.numel());
                // For long (i64), the storage offset is in elements of 8 bytes, but we read as i32 pairs
                if (src_is_long)
                {
                    params.src_offset = static_cast<uint32_t>(src.storage_offset() * 2); // Each i64 is 2 i32s
                }
                else
                {
                    params.src_offset = static_cast<uint32_t>(src.storage_offset());
                }
                params.dst_offset = static_cast<uint32_t>(dst.storage_offset());
                params._pad = 0;

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
                bind_group_entries[0].buffer = src_buffer;
                bind_group_entries[0].offset = 0;
                bind_group_entries[0].size = src_buffer.GetSize();

                bind_group_entries[1].binding = 1;
                bind_group_entries[1].buffer = dst_buffer;
                bind_group_entries[1].offset = 0;
                bind_group_entries[1].size = dst_buffer.GetSize();

                bind_group_entries[2].binding = 2;
                bind_group_entries[2].buffer = params_buffer;
                bind_group_entries[2].offset = 0;
                bind_group_entries[2].size = sizeof(Params);

                wgpu::BindGroupDescriptor bind_group_descriptor{};
                bind_group_descriptor.layout = kernel.bind_group_layout;
                bind_group_descriptor.entryCount = 3;
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
            }
        }

        at::Tensor _to_copy(
            const at::Tensor &self,
            c10::optional<at::ScalarType> dtype,
            c10::optional<at::Layout> layout,
            c10::optional<at::Device> device,
            c10::optional<bool> pin_memory,
            bool non_blocking,
            c10::optional<c10::MemoryFormat> memory_format)
        {
            // If device is specified and different, handle device transfer
            if (device.has_value() && device->type() != c10::DeviceType::PrivateUse1)
            {
                // Transfer to CPU first, then convert
                at::Tensor cpu_tensor = self.to(at::kCPU, self.scalar_type(), non_blocking, false);
                return at::native::_to_copy(cpu_tensor, dtype, layout, device, pin_memory, non_blocking, memory_format);
            }

            // Handle dtype conversion on WebGPU
            at::ScalarType target_dtype = dtype.value_or(self.scalar_type());
            at::ScalarType src_dtype = self.scalar_type();

            if (src_dtype == target_dtype)
            {
                // Same dtype - just clone
                return self.clone();
            }

            // Make contiguous for type conversion
            at::Tensor src_contig = self.contiguous();

            // Create output tensor with target dtype
            at::Tensor out = at::empty(self.sizes(), self.options().dtype(target_dtype));

            // Handle specific conversions
            if (src_dtype == c10::ScalarType::Long && target_dtype == c10::ScalarType::Int)
            {
                TypeConversionKernel &kernel = get_long_to_int_kernel();
                run_type_conversion_kernel(kernel, src_contig, out, true);
            }
            else if (src_dtype == c10::ScalarType::Float && target_dtype == c10::ScalarType::Int)
            {
                TypeConversionKernel &kernel = get_float_to_int_kernel();
                run_type_conversion_kernel(kernel, src_contig, out, false);
            }
            else if (src_dtype == c10::ScalarType::Int && target_dtype == c10::ScalarType::Float)
            {
                TypeConversionKernel &kernel = get_int_to_float_kernel();
                run_type_conversion_kernel(kernel, src_contig, out, false);
            }
            else if (src_dtype == c10::ScalarType::Long && target_dtype == c10::ScalarType::Float)
            {
                TypeConversionKernel &kernel = get_long_to_float_kernel();
                run_type_conversion_kernel(kernel, src_contig, out, true);
            }
            else
            {
                // Fallback: convert through CPU
                at::Tensor cpu_src = src_contig.to(at::kCPU);
                at::Tensor cpu_converted = cpu_src.to(target_dtype);
                out.copy_(cpu_converted.to(out.device()));
            }

            return out;
        }
    }

    TORCH_LIBRARY_IMPL(aten, PrivateUse1, m)
    {
        m.impl("_to_copy", TORCH_FN(ops::_to_copy));
    }
}
