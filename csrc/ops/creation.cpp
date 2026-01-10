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
            const std::string arange_shader = R"wgsl(
struct Params {
    length: u32,
    start: f32,
    step: f32,
    out_offset: u32,
};

@group(0) @binding(0)
var<storage, read_write> outBuffer: array<f32>;

@group(0) @binding(1)
var<uniform> params: Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.length) { return; }

    outBuffer[params.out_offset + i] = params.start + f32(i) * params.step;
}
)wgsl";

            const std::string fill_shader = R"wgsl(
struct Params {
    length: u32,
    value: f32,
    out_offset: u32,
    _pad: u32,
};

@group(0) @binding(0)
var<storage, read_write> outBuffer: array<f32>;

@group(0) @binding(1)
var<uniform> params: Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.length) { return; }

    outBuffer[params.out_offset + i] = params.value;
}
)wgsl";

            struct CreationKernel
            {
                wgpu::BindGroupLayout bind_group_layout;
                wgpu::ComputePipeline pipeline;
            };

            CreationKernel create_kernel(const std::string &shader_code, const char *label)
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

                wgpu::BindGroupLayoutEntry bindings[2]{};

                bindings[0].binding = 0;
                bindings[0].visibility = wgpu::ShaderStage::Compute;
                bindings[0].buffer.type = wgpu::BufferBindingType::Storage;
                bindings[0].buffer.hasDynamicOffset = false;
                bindings[0].buffer.minBindingSize = 0;

                bindings[1].binding = 1;
                bindings[1].visibility = wgpu::ShaderStage::Compute;
                bindings[1].buffer.type = wgpu::BufferBindingType::Uniform;
                bindings[1].buffer.hasDynamicOffset = false;
                bindings[1].buffer.minBindingSize = 0;

                wgpu::BindGroupLayoutDescriptor layout_descriptor{};
                layout_descriptor.entryCount = 2;
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

                return CreationKernel{bind_group_layout, pipeline};
            }

            static CreationKernel *arange_kernel = nullptr;
            static CreationKernel *fill_kernel = nullptr;

            CreationKernel &get_arange_kernel()
            {
                if (arange_kernel == nullptr)
                {
                    static CreationKernel k = create_kernel(arange_shader, "Arange kernel");
                    arange_kernel = &k;
                }
                return *arange_kernel;
            }

            CreationKernel &get_fill_kernel()
            {
                if (fill_kernel == nullptr)
                {
                    static CreationKernel k = create_kernel(fill_shader, "Fill kernel");
                    fill_kernel = &k;
                }
                return *fill_kernel;
            }
        }

        at::Tensor arange(
            const at::Scalar &start,
            const at::Scalar &end,
            const at::Scalar &step,
            c10::optional<at::ScalarType> dtype,
            c10::optional<at::Layout> layout,
            c10::optional<at::Device> device,
            c10::optional<bool> pin_memory)
        {
            float start_val = start.to<float>();
            float end_val = end.to<float>();
            float step_val = step.to<float>();

            TORCH_CHECK(step_val != 0, "arange: step cannot be zero");
            TORCH_CHECK((step_val > 0 && start_val < end_val) || (step_val < 0 && start_val > end_val),
                        "arange: invalid range");

            int64_t length = static_cast<int64_t>(std::ceil((end_val - start_val) / step_val));
            if (length < 0)
                length = 0;

            auto allocator = core::getWebGPUCachingAllocator();
            constexpr c10::DispatchKeySet privateuse1_ks(c10::DispatchKey::PrivateUse1);
            at::Tensor out = at::detail::empty_generic({length}, allocator, privateuse1_ks, at::ScalarType::Float, c10::nullopt);

            if (length == 0)
            {
                return out;
            }

            CreationKernel &kernel = get_arange_kernel();

            core::WebGPUAllocation *out_allocation = static_cast<core::WebGPUAllocation *>(out.storage().data_ptr().get());
            wgpu::Buffer out_buffer = out_allocation->buffer;

            struct Params
            {
                uint32_t length;
                float start;
                float step;
                uint32_t out_offset;
            };

            Params params{};
            params.length = static_cast<uint32_t>(length);
            params.start = start_val;
            params.step = step_val;
            params.out_offset = static_cast<uint32_t>(out.storage_offset());

            core::WebGPUContext &ctx = core::getWebGPUContext();

            wgpu::BufferDescriptor uniform_descriptor{};
            uniform_descriptor.label = "Params";
            uniform_descriptor.usage = wgpu::BufferUsage::Uniform | wgpu::BufferUsage::CopyDst;
            uniform_descriptor.size = sizeof(Params);
            uniform_descriptor.mappedAtCreation = false;
            wgpu::Buffer params_buffer = ctx.getDevice().CreateBuffer(&uniform_descriptor);
            ctx.getQueue().WriteBuffer(params_buffer, 0, &params, sizeof(Params));

            wgpu::BindGroupEntry bind_group_entries[2]{};
            bind_group_entries[0].binding = 0;
            bind_group_entries[0].buffer = out_buffer;
            bind_group_entries[0].offset = 0;
            bind_group_entries[0].size = out_buffer.GetSize();

            bind_group_entries[1].binding = 1;
            bind_group_entries[1].buffer = params_buffer;
            bind_group_entries[1].offset = 0;
            bind_group_entries[1].size = sizeof(Params);

            wgpu::BindGroupDescriptor bind_group_descriptor{};
            bind_group_descriptor.layout = kernel.bind_group_layout;
            bind_group_descriptor.entryCount = 2;
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

            return out;
        }

        at::Tensor arange_start(
            const at::Scalar &end,
            c10::optional<at::ScalarType> dtype,
            c10::optional<at::Layout> layout,
            c10::optional<at::Device> device,
            c10::optional<bool> pin_memory)
        {
            return arange(at::Scalar(0), end, at::Scalar(1), dtype, layout, device, pin_memory);
        }

        at::Tensor arange_start_step(
            const at::Scalar &start,
            const at::Scalar &end,
            c10::optional<at::ScalarType> dtype,
            c10::optional<at::Layout> layout,
            c10::optional<at::Device> device,
            c10::optional<bool> pin_memory)
        {
            return arange(start, end, at::Scalar(1), dtype, layout, device, pin_memory);
        }

        // Fill tensor with a value
        at::Tensor &fill_(at::Tensor &self, const at::Scalar &value)
        {
            TORCH_CHECK(self.device().type() == c10::DeviceType::PrivateUse1);
            TORCH_CHECK(self.scalar_type() == c10::ScalarType::Float);
            TORCH_CHECK(self.is_contiguous(), "fill_ requires contiguous tensor");

            if (self.numel() == 0)
            {
                return self;
            }

            CreationKernel &kernel = get_fill_kernel();

            core::WebGPUAllocation *out_allocation = static_cast<core::WebGPUAllocation *>(self.storage().data_ptr().get());
            wgpu::Buffer out_buffer = out_allocation->buffer;

            struct Params
            {
                uint32_t length;
                float value;
                uint32_t out_offset;
                uint32_t _pad;
            };

            Params params{};
            params.length = static_cast<uint32_t>(self.numel());
            params.value = value.to<float>();
            params.out_offset = static_cast<uint32_t>(self.storage_offset());
            params._pad = 0;

            core::WebGPUContext &ctx = core::getWebGPUContext();

            wgpu::BufferDescriptor uniform_descriptor{};
            uniform_descriptor.label = "Params";
            uniform_descriptor.usage = wgpu::BufferUsage::Uniform | wgpu::BufferUsage::CopyDst;
            uniform_descriptor.size = sizeof(Params);
            uniform_descriptor.mappedAtCreation = false;
            wgpu::Buffer params_buffer = ctx.getDevice().CreateBuffer(&uniform_descriptor);
            ctx.getQueue().WriteBuffer(params_buffer, 0, &params, sizeof(Params));

            wgpu::BindGroupEntry bind_group_entries[2]{};
            bind_group_entries[0].binding = 0;
            bind_group_entries[0].buffer = out_buffer;
            bind_group_entries[0].offset = 0;
            bind_group_entries[0].size = out_buffer.GetSize();

            bind_group_entries[1].binding = 1;
            bind_group_entries[1].buffer = params_buffer;
            bind_group_entries[1].offset = 0;
            bind_group_entries[1].size = sizeof(Params);

            wgpu::BindGroupDescriptor bind_group_descriptor{};
            bind_group_descriptor.layout = kernel.bind_group_layout;
            bind_group_descriptor.entryCount = 2;
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

            return self;
        }

        at::Tensor &fill_tensor_(at::Tensor &self, const at::Tensor &value)
        {
            TORCH_CHECK(value.numel() == 1, "fill_: source tensor must have single element");
            // Move value to CPU to get scalar
            auto cpu_val = value.to(at::kCPU);
            return torch_webgpu::ops::fill_(self, cpu_val.item());
        }

        at::Tensor &zero_(at::Tensor &self)
        {
            return torch_webgpu::ops::fill_(self, 0.0);
        }

        at::Tensor zeros(
            c10::IntArrayRef size,
            c10::optional<at::ScalarType> dtype,
            c10::optional<at::Layout> layout,
            c10::optional<at::Device> device,
            c10::optional<bool> pin_memory)
        {
            auto allocator = core::getWebGPUCachingAllocator();
            constexpr c10::DispatchKeySet privateuse1_ks(c10::DispatchKey::PrivateUse1);
            at::Tensor out = at::detail::empty_generic(size, allocator, privateuse1_ks, at::ScalarType::Float, c10::nullopt);
            return torch_webgpu::ops::fill_(out, 0.0);
        }

        at::Tensor ones(
            c10::IntArrayRef size,
            c10::optional<at::ScalarType> dtype,
            c10::optional<at::Layout> layout,
            c10::optional<at::Device> device,
            c10::optional<bool> pin_memory)
        {
            auto allocator = core::getWebGPUCachingAllocator();
            constexpr c10::DispatchKeySet privateuse1_ks(c10::DispatchKey::PrivateUse1);
            at::Tensor out = at::detail::empty_generic(size, allocator, privateuse1_ks, at::ScalarType::Float, c10::nullopt);
            return torch_webgpu::ops::fill_(out, 1.0);
        }

        at::Tensor full(
            c10::IntArrayRef size,
            const at::Scalar &fill_value,
            c10::optional<at::ScalarType> dtype,
            c10::optional<at::Layout> layout,
            c10::optional<at::Device> device,
            c10::optional<bool> pin_memory)
        {
            auto allocator = core::getWebGPUCachingAllocator();
            constexpr c10::DispatchKeySet privateuse1_ks(c10::DispatchKey::PrivateUse1);
            at::Tensor out = at::detail::empty_generic(size, allocator, privateuse1_ks, at::ScalarType::Float, c10::nullopt);
            return torch_webgpu::ops::fill_(out, fill_value);
        }

        // new_ones - create tensor of ones with same dtype/device as input
        at::Tensor new_ones(
            const at::Tensor &self,
            c10::IntArrayRef size,
            c10::optional<at::ScalarType> dtype,
            c10::optional<at::Layout> layout,
            c10::optional<at::Device> device,
            c10::optional<bool> pin_memory)
        {
            return ones(size, dtype.value_or(self.scalar_type()), layout, device.value_or(self.device()), pin_memory);
        }

        // new_zeros
        at::Tensor new_zeros(
            const at::Tensor &self,
            c10::IntArrayRef size,
            c10::optional<at::ScalarType> dtype,
            c10::optional<at::Layout> layout,
            c10::optional<at::Device> device,
            c10::optional<bool> pin_memory)
        {
            return zeros(size, dtype.value_or(self.scalar_type()), layout, device.value_or(self.device()), pin_memory);
        }
    }

    TORCH_LIBRARY_IMPL(aten, PrivateUse1, m)
    {
        m.impl("arange", TORCH_FN(ops::arange_start));
        m.impl("arange.start", TORCH_FN(ops::arange_start_step));
        m.impl("arange.start_step", TORCH_FN(ops::arange));
        m.impl("fill_.Scalar", TORCH_FN(ops::fill_));
        m.impl("fill_.Tensor", TORCH_FN(ops::fill_tensor_));
        m.impl("zero_", TORCH_FN(ops::zero_));
        m.impl("zeros", TORCH_FN(ops::zeros));
        m.impl("ones", TORCH_FN(ops::ones));
        m.impl("full", TORCH_FN(ops::full));
        m.impl("new_ones", TORCH_FN(ops::new_ones));
        m.impl("new_zeros", TORCH_FN(ops::new_zeros));
    }
}
