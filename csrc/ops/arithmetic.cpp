#include <ATen/ATen.h>
#include <ATen/native/BinaryOps.h>
#include <ATen/RedispatchFunctions.h>
#include <ATen/native/DispatchStub.h>
#include <webgpu/webgpu_cpp.h>
#include "core/webgpu_context.h"
#include "core/webgpu_allocator.h"
#include "binary.h"
#include "utils/math.h"

namespace torch_webgpu
{
    namespace ops
    {
        namespace
        {
            // TODO: no idea if optimal, just to start with something
            static constexpr uint32_t TILE_X = 16;
            static constexpr uint32_t TILE_Y = 16;

            struct CacheHash
            {
                template <typename T>
                std::size_t operator()(T t) const noexcept
                {
                    return static_cast<std::size_t>(t);
                }
            };

            // TODO: improve performance, currently it's a native impl
            // based on my first CUDA matrix multiplication kernel
            const std::string mm_shader = R"wgsl(
// A(M,N) x B(N,K) = C(M,K), where M,N,K are dims
const MAX_DIMS: u32 = 8u;

struct Params {
    M: u32,
    N: u32,
    K: u32,
    _pad: u32,

    self_offset: u32,
    mat2_offset: u32,
    out_offset: u32,
    _pad2: u32,

    self_strides: array<u32, MAX_DIMS>,
    mat2_strides: array<u32, MAX_DIMS>,
    out_strides: array<u32, MAX_DIMS>,
};

@group(0) @binding(0)
var<storage, read> A: array<f32>; // self

@group(0) @binding(1)
var<storage, read> B: array<f32>; // mat2

@group(0) @binding(2)
var<storage, read_write> C: array<f32>; // out

@group(0) @binding(3)
var<uniform> params: Params;

// one thread = one output element C[column, row]
@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let col = gid.x; // 0..K-1
    let row = gid.y; // 0..M-1
    if (col >= params.K || row >= params.M) { return; }

    var acc: f32 = 0.0;

    for (var k: u32 = 0u; k < params.N; k = k + 1u) {
        let a_idx = params.self_offset + row * params.N + k;
        let b_idx = params.mat2_offset + k * params.K + col;
        acc = acc + A[a_idx] * B[b_idx];
    }
    let c_idx = params.out_offset + row * params.K + col;
    C[c_idx] = acc;
}
)wgsl";
        }

        void add_kernel_webgpu(::at::TensorIteratorBase &iter, const ::at::Scalar &alpha)
        {
            run_binary_kernel<BinaryOp::Add>(iter, alpha);
        }

        void mul_kernel_webgpu(::at::TensorIteratorBase &iter)
        {
            run_binary_kernel<BinaryOp::Mul>(iter);
        }

        void mm_kernel_webgpu(::at::TensorIteratorBase &iter)
        {
            TORCH_CHECK(iter.ntensors() == 3);
            TORCH_CHECK(iter.common_dtype() == at::ScalarType::Float);
            TORCH_CHECK(iter.device_type() == c10::DeviceType::PrivateUse1);
            TORCH_CHECK(iter.dtype(0) == iter.dtype(1));
            TORCH_CHECK(iter.dtype(1) == iter.dtype(2));

            wgpu::ShaderSourceWGSL shader_source{
                wgpu::ShaderSourceWGSL::Init{
                    nullptr,
                    wgpu::StringView{mm_shader.c_str(), mm_shader.size()},
                }};

            wgpu::ShaderModuleDescriptor shader_descriptor{};
            shader_descriptor.nextInChain = &shader_source;
            shader_descriptor.label = "MM shader";
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
            BinaryKernel kernel{bind_group_layout, pipeline};

            auto out = iter.tensor(0);
            auto self = iter.tensor(1);
            auto mat2 = iter.tensor(2);
            auto ndim = static_cast<uint32_t>(iter.ndim());
            auto shape = iter.shape();
            auto self_strides_bytes = iter.strides(1);
            auto mat2_strides_bytes = iter.strides(2);
            auto out_strides_bytes = iter.strides(0);

            auto element_size = iter.element_size(0);

            std::vector<int64_t> self_strides(ndim);
            std::vector<int64_t> mat2_strides(ndim);
            std::vector<int64_t> out_strides(ndim);

            for (int64_t i = 0; i < ndim; ++i)
            {
                int64_t self_bytes = self_strides_bytes[i];
                if (self_bytes == 0)
                {
                    self_strides[i] = 0;
                }
                else
                {
                    TORCH_CHECK(self_bytes % element_size == 0);
                    self_strides[i] = self_bytes / element_size;
                }

                int64_t mat2_bytes = mat2_strides_bytes[i];
                if (mat2_bytes == 0)
                {
                    mat2_strides[i] = 0;
                }
                else
                {
                    TORCH_CHECK(mat2_bytes % element_size == 0);
                    mat2_strides[i] = mat2_bytes / element_size;
                }

                int64_t out_bytes = out_strides_bytes[i];
                if (out_bytes == 0)
                {
                    out_strides[i] = 0;
                }
                else
                {
                    TORCH_CHECK(out_bytes % element_size == 0);
                    out_strides[i] = out_bytes / element_size;
                }
            }

            auto length = iter.numel();

            core::WebGPUAllocation *self_allocation = static_cast<core::WebGPUAllocation *>(self.storage().data_ptr().get());
            core::WebGPUAllocation *mat2_allocation = static_cast<core::WebGPUAllocation *>(mat2.storage().data_ptr().get());
            core::WebGPUAllocation *out_allocation = static_cast<core::WebGPUAllocation *>(out.storage().data_ptr().get());

            wgpu::Buffer self_buffer = self_allocation->buffer;
            wgpu::Buffer mat2_buffer = mat2_allocation->buffer;
            wgpu::Buffer out_buffer = out_allocation->buffer;

            auto self_offset = self.storage_offset();
            auto mat2_offset = mat2.storage_offset();
            auto out_offset = out.storage_offset();

            constexpr uint32_t MAX_DIMS = 8;
            TORCH_CHECK(ndim <= MAX_DIMS);

            struct Params
            {
                uint32_t M;
                uint32_t N;
                uint32_t K;
                uint32_t _pad; // allegedly, it's a padding we need for webgpu

                uint32_t self_offset;
                uint32_t mat2_offset;
                uint32_t out_offset;
                uint32_t _pad2;

                uint32_t self_strides[MAX_DIMS];
                uint32_t mat2_strides[MAX_DIMS];
                uint32_t out_strides[MAX_DIMS];
                uint32_t shape[MAX_DIMS];
            };

            Params params{};
            params.M = static_cast<uint32_t>(self.size(0));
            params.N = static_cast<uint32_t>(self.size(1));
            TORCH_CHECK(self.size(1) == mat2.size(0));
            params.K = static_cast<uint32_t>(mat2.size(1));
            params._pad = 0;

            params.self_offset = static_cast<uint32_t>(self_offset);
            params.mat2_offset = static_cast<uint32_t>(mat2_offset);
            params.out_offset = static_cast<uint32_t>(out_offset);
            params._pad2 = 0;

            for (uint32_t d = 0; d < MAX_DIMS; ++d)
            {
                params.self_strides[d] = 0;
                params.mat2_strides[d] = 0;
                params.out_strides[d] = 0;
                params.shape[d] = 1;
            }

            for (int64_t i = 0; i < ndim; ++i)
            {
                auto dim_size = shape[i];
                TORCH_CHECK(dim_size >= 0 && dim_size <= std::numeric_limits<uint32_t>::max());
                params.shape[i] = static_cast<uint32_t>(dim_size);

                auto self_stride = self_strides[i];
                auto mat2_stride = mat2_strides[i];
                auto out_stride = out_strides[i];

                TORCH_CHECK(self_stride >= 0 && self_stride <= std::numeric_limits<uint32_t>::max());
                TORCH_CHECK(mat2_stride >= 0 && mat2_stride <= std::numeric_limits<uint32_t>::max());
                TORCH_CHECK(out_stride >= 0 && out_stride <= std::numeric_limits<uint32_t>::max());

                params.self_strides[i] = static_cast<uint32_t>(self_stride);
                params.mat2_strides[i] = static_cast<uint32_t>(mat2_stride);
                params.out_strides[i] = static_cast<uint32_t>(out_stride);
            }

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
            bind_group_entries[1].buffer = mat2_buffer;
            bind_group_entries[1].offset = 0;
            bind_group_entries[1].size = mat2_buffer.GetSize();

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

            const uint32_t x_group_size = ceil_div_u32(params.K, TILE_X);
            const uint32_t y_group_size = ceil_div_u32(params.M, TILE_Y);

            pass_encoder.DispatchWorkgroups(x_group_size, y_group_size, 1);
            pass_encoder.End();

            wgpu::CommandBuffer command_buffer = encoder.Finish();
            ctx.getQueue().Submit(1, &command_buffer);
        }

        at::Tensor &add_out_webgpu(
            const at::Tensor &self,
            const at::Tensor &other,
            const at::Scalar &alpha,
            at::Tensor &out)
        {
            at::TensorIteratorConfig config;
            config.set_check_mem_overlap(true);
            config.add_output(out);
            config.add_input(self);
            config.add_input(other);
            config.promote_inputs_to_common_dtype(true);
            config.cast_common_dtype_to_outputs(true);
            config.check_all_same_device(false);
            auto iter = config.build();

            add_kernel_webgpu(iter, alpha);

            return out;
        }

        at::Tensor &mul_out_webgpu(
            const at::Tensor &self,
            const at::Tensor &other,
            at::Tensor &out)
        {
            at::TensorIteratorConfig config;
            config.set_check_mem_overlap(true);
            config.add_output(out);
            config.add_input(self);
            config.add_input(other);
            config.promote_inputs_to_common_dtype(true);
            config.cast_common_dtype_to_outputs(true);
            config.check_all_same_device(false);
            auto iter = config.build();

            mul_kernel_webgpu(iter);

            return out;
        }

        at::Tensor &mm_out_webgpu(
            const at::Tensor &self,
            const at::Tensor &mat2,
            at::Tensor &out)
        {
            at::TensorIteratorConfig config;
            config.set_check_mem_overlap(true);
            config.add_output(out);
            config.add_input(self);
            config.add_input(mat2);
            config.promote_inputs_to_common_dtype(true);
            config.cast_common_dtype_to_outputs(true);
            config.check_all_same_device(false);
            auto iter = config.build();

            mm_kernel_webgpu(iter);

            return out;
        }
    }

    TORCH_LIBRARY_IMPL(aten, PrivateUse1, m)
    {
        m.impl("add.out", TORCH_FN(ops::add_out_webgpu));
        m.impl("mul.out", TORCH_FN(ops::mul_out_webgpu));
        m.impl("mm.out", TORCH_FN(ops::mm_out_webgpu));
    }
}

// it needs to be like this because of what REGISTER_PRIVATEUSE1_DISPATCH expects with at::native
namespace at
{
    namespace native
    {
        void add_kernel_webgpu(TensorIteratorBase &iter, const Scalar &alpha)
        {
            torch_webgpu::ops::add_kernel_webgpu(iter, alpha);
        }
        REGISTER_PRIVATEUSE1_DISPATCH(add_stub, &add_kernel_webgpu);

        void mul_kernel_webgpu(TensorIteratorBase &iter)
        {
            torch_webgpu::ops::mul_kernel_webgpu(iter);
        }
        REGISTER_PRIVATEUSE1_DISPATCH(mul_stub, &mul_kernel_webgpu);
    }
}