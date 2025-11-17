#include <ATen/ATen.h>
#include <ATen/native/BinaryOps.h>
#include <ATen/RedispatchFunctions.h>
#include <ATen/native/DispatchStub.h>
#include <webgpu/webgpu_cpp.h>
#include "core/webgpu_context.h"
#include "core/webgpu_allocator.h"
#include "binary.h"

namespace torch_webgpu
{
    namespace ops
    {
        void add_kernel_webgpu(::at::TensorIteratorBase &iter, const ::at::Scalar &alpha)
        {
            binary_kernel<BinaryOp::Add>(iter, alpha);
        }

        void mul_kernel_webgpu(::at::TensorIteratorBase &iter)
        {
            binary_kernel<BinaryOp::Mul>(iter);
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

        namespace
        {
            const std::string matmul_shader = R"wgsl(
const MAX_DIMS: u32 = 8u;

struct Params {
    length: u32,
    ndim: u32,
    _pad: u32,

    out_offset: u32,
    self_offset: u32,
    other_offset: u32,
    _pad2: u32,

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

    outBuffer[idx_out] = __BINARY_OP__;
}
)wgsl";
        }

        at::Tensor matmul(at::Tensor const &self, at::Tensor const &other)
        {
            // starting with 2D MxK @ KxN = MxN, contiguous, etc. all simplest
            TORCH_CHECK(self.is_contiguous() && other.is_contiguous());
            TORCH_CHECK(self.dtype() == other.dtype());
            TORCH_CHECK(self.dtype() == at::ScalarType::Float);
            TORCH_CHECK(self.device() == c10::DeviceType::PrivateUse1);
            TORCH_CHECK(other.device() == c10::DeviceType::PrivateUse1);
            TORCH_CHECK(self.ndimension() == other.ndimension());
            TORCH_CHECK(self.ndimension() == 2);
            TORCH_CHECK(self.sizes()[1] == other.sizes()[0]);
        }

        at::Tensor matmul_out(at::Tensor const &self, at::Tensor const &other, at::Tensor &out)
        {
            auto out = ops::matmul(self, other);

            return out;
        }
    }

    TORCH_LIBRARY_IMPL(aten, PrivateUse1, m)
    {
        m.impl("add.out", TORCH_FN(ops::add_out_webgpu));
        m.impl("mul.out", TORCH_FN(ops::mul_out_webgpu));
        m.impl("matmul", TORCH_FN(ops::matmul));
        m.impl("matmul.out", TORCH_FN(ops::matmul_out));
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