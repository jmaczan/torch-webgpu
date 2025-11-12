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
    }
    TORCH_LIBRARY_IMPL(aten, PrivateUse1, m)
    {
        m.impl("add.out", TORCH_FN(ops::add_out_webgpu));
        m.impl("mul.out", TORCH_FN(ops::mul_out_webgpu));
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