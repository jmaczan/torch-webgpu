#include <ATen/ATen.h>
#include <ATen/RedispatchFunctions.h>
#include <webgpu/webgpu_cpp.h>
#include "core/webgpu_context.h"
#include "core/webgpu_allocator.h"
#include "unary.h"

namespace torch_webgpu
{
    namespace ops
    {
        void relu_kernel_webgpu(at::TensorIteratorBase &iter)
        {
            unary_kernel<UnaryOp::ReLU>(iter);
        }

        at::Tensor relu(at::Tensor const &self)
        {
            at::Tensor out = at::empty_like(self, self.options().device(at::DeviceType::PrivateUse1));
            at::TensorIteratorConfig config;
            config.add_output(out);
            config.add_input(self);
            config.check_all_same_dtype(true);
            auto iter = config.build();

            relu_kernel_webgpu(iter);

            return out;
        }

        at::Tensor &relu_out(
            at::Tensor const &self,
            at::Tensor &out)
        {
            at::TensorIteratorConfig config;
            config.add_output(out);
            config.add_input(self);
            config.check_all_same_dtype(true);
            auto iter = config.build();

            relu_kernel_webgpu(iter);

            return out;
        }
    }
    TORCH_LIBRARY_IMPL(aten, PrivateUse1, m)
    {
        m.impl("relu", TORCH_FN(ops::relu));
        m.impl("relu.out", TORCH_FN(ops::relu_out));
    }
}