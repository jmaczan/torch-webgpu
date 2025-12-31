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
            config.check_all_same_device(false);
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
            config.check_all_same_device(false);
            auto iter = config.build();

            cos_kernel_webgpu(iter);

            return out;
        }
    }

    TORCH_LIBRARY_IMPL(aten, PrivateUse1, m)
    {
        m.impl("cos", TORCH_FN(ops::cos));
        m.impl("cos.out", TORCH_FN(ops::cos_out));
    }
}
