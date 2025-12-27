#pragma once
#include <ATen/ATen.h>
#include <ATen/RedispatchFunctions.h>
#include <webgpu/webgpu_cpp.h>
#include "core/webgpu_context.h"
#include "core/webgpu_allocator.h"
#include <torch/library.h>
#include "binary.h"

namespace torch_webgpu
{
    namespace ops
    {
        at::Tensor create_buffer(c10::IntArrayRef size, c10::IntArrayRef stride, c10::optional<at::ScalarType> dtype_opt)
        {
            // TODO: probably pass a device as an argument and replace below?
            return at::empty_strided(size, stride, at::TensorOptions().dtype(dtype_opt).device(c10::Device(c10::kPrivateUse1)));
        }

        at::Tensor write_buffer(at::Tensor &self, at::Tensor const &src)
        {
            self.copy_(src);
            return self;
        }

        at::Tensor fused_add_relu(
            const at::Tensor &self,
            const at::Tensor &other)
        {
            auto out = at::empty_like(self);
            at::TensorIteratorConfig config;
            config.set_check_mem_overlap(true);
            config.add_output(out);
            config.add_input(self);
            config.add_input(other);
            config.promote_inputs_to_common_dtype(true);
            config.cast_common_dtype_to_outputs(true);
            config.check_all_same_device(false);
            auto iter = config.build();

            binary_kernel<BinaryOp::FusedAddRelu>(iter, 1.0f);

            return out;
        }

    }
    TORCH_LIBRARY(webgpu, m)
    {
        m.def("create_buffer(int[] size, int[] stride, ScalarType? dtype=None) -> Tensor");
        m.def("write_buffer(Tensor self, Tensor src) -> Tensor");
        m.def("fused_add_relu(Tensor self, Tensor src) -> Tensor");
    }
    TORCH_LIBRARY_IMPL(webgpu, PrivateUse1, m)
    {
        m.impl("create_buffer", TORCH_FN(ops::create_buffer));
        m.impl("write_buffer", TORCH_FN(ops::write_buffer));
        m.impl("fused_add_relu", TORCH_FN(ops::fused_add_relu));
    }
    TORCH_LIBRARY_IMPL(webgpu, CatchAll, m)
    {
        m.impl("create_buffer", TORCH_FN(ops::create_buffer));
    }
    TORCH_LIBRARY_IMPL(_, AutogradPrivateUse1, m)
    {
        m.fallback(torch::CppFunction::makeFallthrough());
    }
}