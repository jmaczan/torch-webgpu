#include <ATen/ATen.h>
#include <ATen/native/BinaryOps.h>
#include <ATen/RedispatchFunctions.h>
#include <ATen/native/DispatchStub.h>
#include <webgpu/webgpu_cpp.h>
#include "core/webgpu_context.h"
#include "core/webgpu_allocator.h"
#include "unary.h"
#include "utils/math.h"

namespace torch_webgpu
{
    namespace ops
    {
        void cos_kernel_webgpu(at::TensorIteratorBase &iter)
        {
            unary_kernel<UnaryOp::Cos>(iter);
        }

        at::Tensor cos(const at::Tensor &self)
        {
            at::Tensor out = at::empty_like(self, self.options().device(at::DeviceType::PrivateUse1));

            at::TensorIteratorConfig config;
            config.set_check_mem_overlap(true);
            config.add_output(out);
            config.add_input(self);
            config.promote_inputs_to_common_dtype(true);
            config.cast_common_dtype_to_outputs(true);
            config.check_all_same_device(true);
            auto iter = config.build();

            cos_kernel_webgpu(iter);

            return out;
        }

        at::Tensor &cos_out(
            const at::Tensor &self,
            at::Tensor &out)
        {
            at::TensorIteratorConfig config;
            config.set_check_mem_overlap(true);
            config.add_output(out);
            config.add_input(self);
            config.promote_inputs_to_common_dtype(true);
            config.cast_common_dtype_to_outputs(true);
            config.check_all_same_device(true);
            auto iter = config.build();

            cos_kernel_webgpu(iter);

            return out;
        }

        // Sin
        void sin_kernel_webgpu(at::TensorIteratorBase &iter)
        {
            unary_kernel<UnaryOp::Sin>(iter);
        }

        at::Tensor sin(const at::Tensor &self)
        {
            at::Tensor out = at::empty_like(self, self.options().device(at::DeviceType::PrivateUse1));

            at::TensorIteratorConfig config;
            config.set_check_mem_overlap(true);
            config.add_output(out);
            config.add_input(self);
            config.promote_inputs_to_common_dtype(true);
            config.cast_common_dtype_to_outputs(true);
            config.check_all_same_device(true);
            auto iter = config.build();

            sin_kernel_webgpu(iter);

            return out;
        }

        at::Tensor &sin_out(
            const at::Tensor &self,
            at::Tensor &out)
        {
            at::TensorIteratorConfig config;
            config.set_check_mem_overlap(true);
            config.add_output(out);
            config.add_input(self);
            config.promote_inputs_to_common_dtype(true);
            config.cast_common_dtype_to_outputs(true);
            config.check_all_same_device(true);
            auto iter = config.build();

            sin_kernel_webgpu(iter);

            return out;
        }

        // Tanh
        void tanh_kernel_webgpu(at::TensorIteratorBase &iter)
        {
            unary_kernel<UnaryOp::Tanh>(iter);
        }

        at::Tensor tanh(const at::Tensor &self)
        {
            at::Tensor out = at::empty_like(self, self.options().device(at::DeviceType::PrivateUse1));

            at::TensorIteratorConfig config;
            config.set_check_mem_overlap(true);
            config.add_output(out);
            config.add_input(self);
            config.promote_inputs_to_common_dtype(true);
            config.cast_common_dtype_to_outputs(true);
            config.check_all_same_device(true);
            auto iter = config.build();

            tanh_kernel_webgpu(iter);

            return out;
        }

        at::Tensor &tanh_out(
            const at::Tensor &self,
            at::Tensor &out)
        {
            at::TensorIteratorConfig config;
            config.set_check_mem_overlap(true);
            config.add_output(out);
            config.add_input(self);
            config.promote_inputs_to_common_dtype(true);
            config.cast_common_dtype_to_outputs(true);
            config.check_all_same_device(true);
            auto iter = config.build();

            tanh_kernel_webgpu(iter);

            return out;
        }

        // Exp
        void exp_kernel_webgpu(at::TensorIteratorBase &iter)
        {
            unary_kernel<UnaryOp::Exp>(iter);
        }

        at::Tensor exp(const at::Tensor &self)
        {
            at::Tensor out = at::empty_like(self, self.options().device(at::DeviceType::PrivateUse1));

            at::TensorIteratorConfig config;
            config.set_check_mem_overlap(true);
            config.add_output(out);
            config.add_input(self);
            config.promote_inputs_to_common_dtype(true);
            config.cast_common_dtype_to_outputs(true);
            config.check_all_same_device(true);
            auto iter = config.build();

            exp_kernel_webgpu(iter);

            return out;
        }

        at::Tensor &exp_out(
            const at::Tensor &self,
            at::Tensor &out)
        {
            at::TensorIteratorConfig config;
            config.set_check_mem_overlap(true);
            config.add_output(out);
            config.add_input(self);
            config.promote_inputs_to_common_dtype(true);
            config.cast_common_dtype_to_outputs(true);
            config.check_all_same_device(true);
            auto iter = config.build();

            exp_kernel_webgpu(iter);

            return out;
        }

        // Abs
        void abs_kernel_webgpu(at::TensorIteratorBase &iter)
        {
            unary_kernel<UnaryOp::Abs>(iter);
        }

        at::Tensor abs(const at::Tensor &self)
        {
            at::Tensor out = at::empty_like(self, self.options().device(at::DeviceType::PrivateUse1));

            at::TensorIteratorConfig config;
            config.set_check_mem_overlap(true);
            config.add_output(out);
            config.add_input(self);
            config.promote_inputs_to_common_dtype(true);
            config.cast_common_dtype_to_outputs(true);
            config.check_all_same_device(true);
            auto iter = config.build();

            abs_kernel_webgpu(iter);

            return out;
        }

        at::Tensor &abs_out(
            const at::Tensor &self,
            at::Tensor &out)
        {
            at::TensorIteratorConfig config;
            config.set_check_mem_overlap(true);
            config.add_output(out);
            config.add_input(self);
            config.promote_inputs_to_common_dtype(true);
            config.cast_common_dtype_to_outputs(true);
            config.check_all_same_device(true);
            auto iter = config.build();

            abs_kernel_webgpu(iter);

            return out;
        }

        // Rsqrt
        void rsqrt_kernel_webgpu(at::TensorIteratorBase &iter)
        {
            unary_kernel<UnaryOp::Rsqrt>(iter);
        }

        at::Tensor rsqrt(const at::Tensor &self)
        {
            at::Tensor out = at::empty_like(self, self.options().device(at::DeviceType::PrivateUse1));

            at::TensorIteratorConfig config;
            config.set_check_mem_overlap(true);
            config.add_output(out);
            config.add_input(self);
            config.promote_inputs_to_common_dtype(true);
            config.cast_common_dtype_to_outputs(true);
            config.check_all_same_device(true);
            auto iter = config.build();

            rsqrt_kernel_webgpu(iter);

            return out;
        }

        at::Tensor &rsqrt_out(
            const at::Tensor &self,
            at::Tensor &out)
        {
            at::TensorIteratorConfig config;
            config.set_check_mem_overlap(true);
            config.add_output(out);
            config.add_input(self);
            config.promote_inputs_to_common_dtype(true);
            config.cast_common_dtype_to_outputs(true);
            config.check_all_same_device(true);
            auto iter = config.build();

            rsqrt_kernel_webgpu(iter);

            return out;
        }

        // Neg
        void neg_kernel_webgpu(at::TensorIteratorBase &iter)
        {
            unary_kernel<UnaryOp::Neg>(iter);
        }

        at::Tensor neg(const at::Tensor &self)
        {
            at::Tensor out = at::empty_like(self, self.options().device(at::DeviceType::PrivateUse1));

            at::TensorIteratorConfig config;
            config.set_check_mem_overlap(true);
            config.add_output(out);
            config.add_input(self);
            config.promote_inputs_to_common_dtype(true);
            config.cast_common_dtype_to_outputs(true);
            config.check_all_same_device(true);
            auto iter = config.build();

            neg_kernel_webgpu(iter);

            return out;
        }

        at::Tensor &neg_out(
            const at::Tensor &self,
            at::Tensor &out)
        {
            at::TensorIteratorConfig config;
            config.set_check_mem_overlap(true);
            config.add_output(out);
            config.add_input(self);
            config.promote_inputs_to_common_dtype(true);
            config.cast_common_dtype_to_outputs(true);
            config.check_all_same_device(true);
            auto iter = config.build();

            neg_kernel_webgpu(iter);

            return out;
        }
    }

    TORCH_LIBRARY_IMPL(aten, PrivateUse1, m)
    {
        m.impl("cos", TORCH_FN(ops::cos));
        m.impl("cos.out", TORCH_FN(ops::cos_out));
        m.impl("sin", TORCH_FN(ops::sin));
        m.impl("sin.out", TORCH_FN(ops::sin_out));
        m.impl("tanh", TORCH_FN(ops::tanh));
        m.impl("tanh.out", TORCH_FN(ops::tanh_out));
        m.impl("exp", TORCH_FN(ops::exp));
        m.impl("exp.out", TORCH_FN(ops::exp_out));
        m.impl("abs", TORCH_FN(ops::abs));
        m.impl("abs.out", TORCH_FN(ops::abs_out));
        m.impl("rsqrt", TORCH_FN(ops::rsqrt));
        m.impl("rsqrt.out", TORCH_FN(ops::rsqrt_out));
        m.impl("neg", TORCH_FN(ops::neg));
        m.impl("neg.out", TORCH_FN(ops::neg_out));
    }
}
