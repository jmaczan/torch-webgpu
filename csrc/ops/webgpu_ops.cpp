#pragma once
#include <ATen/ATen.h>
#include <ATen/RedispatchFunctions.h>
#include <webgpu/webgpu_cpp.h>
#include "core/webgpu_context.h"
#include "core/webgpu_allocator.h"
#include <torch/library.h>

namespace torch_webgpu
{
    namespace ops
    {
        at::Tensor create_buffer(c10::IntArrayRef size, c10::IntArrayRef stride, c10::optional<at::ScalarType> dtype_opt)
        {
            return at::empty_strided(size, stride, at::TensorOptions().dtype(dtype_opt).device(c10::Device(c10::kPrivateUse1)));
        }

        at::Tensor write_buffer(at::Tensor &self, at::Tensor const &src)
        {
            self.copy_(src);
            return self;
        }

    }
    TORCH_LIBRARY(webgpu, m)
    {
        m.def("create_buffer(int[] size, int[] stride, ScalarType? dtype=None) -> Tensor");
        m.def("write_buffer(Tensor self, Tensor src) -> Tensor");
    }
    TORCH_LIBRARY_IMPL(webgpu, PrivateUse1, m)
    {
        m.impl("create_buffer", TORCH_FN(ops::create_buffer));
        m.impl("write_buffer", TORCH_FN(ops::write_buffer));
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