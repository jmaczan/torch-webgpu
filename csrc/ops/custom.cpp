// torch.ops.webgpu.* custom operators used by the torch_webgpu compiler.

#include <ATen/ATen.h>
#include <ATen/TensorIterator.h>
#include <torch/library.h>

#include "kernels/elementwise.h"
#include "kernels/math.h"
#include "ops/common.h"

namespace torch_webgpu::ops
{
    namespace
    {
        at::Tensor create_buffer(c10::IntArrayRef size, c10::IntArrayRef stride, std::optional<at::ScalarType> dtype)
        {
            return at::empty_strided(size, stride,
                                     at::TensorOptions().dtype(dtype).device(c10::DeviceType::PrivateUse1));
        }

        at::Tensor &write_buffer(at::Tensor &self, const at::Tensor &src)
        {
            self.copy_(src);
            return self;
        }

        at::Tensor fused_add_relu(const at::Tensor &self, const at::Tensor &other)
        {
            at::Tensor out;
            auto iter = at::TensorIterator::binary_op(out, self, other);
            kernels::ElementwiseOp op;
            op.name = "fused_add_relu";
            op.in_ct = kernels::common_ct(iter);
            op.out_ct = op.in_ct;
            TORCH_CHECK(op.in_ct != kernels::CT::Bool, "fused_add_relu does not support bool tensors");
            op.code = op.in_ct == kernels::CT::F32 ? "select(max(x0 + x1, 0.0), x0 + x1, isnan_f(x0 + x1))"
                                                   : "max(x0 + x1, 0)";
            op.helpers = kernels::math_wgsl();
            kernels::run_elementwise(iter, op);
            return iter.output();
        }
    }

    TORCH_LIBRARY(webgpu, m)
    {
        m.def("create_buffer(int[] size, int[] stride, ScalarType? dtype=None) -> Tensor");
        m.def("write_buffer(Tensor(a!) self, Tensor src) -> Tensor(a!)");
        m.def("fused_add_relu(Tensor self, Tensor other) -> Tensor");
    }

    TORCH_LIBRARY_IMPL(webgpu, PrivateUse1, m)
    {
        m.impl("write_buffer", TORCH_FN(write_buffer));
        m.impl("fused_add_relu", TORCH_FN(fused_add_relu));
    }

    // create_buffer has no tensor arguments, so it can't dispatch on device.
    TORCH_LIBRARY_IMPL(webgpu, CompositeExplicitAutograd, m)
    {
        m.impl("create_buffer", TORCH_FN(create_buffer));
    }
}
