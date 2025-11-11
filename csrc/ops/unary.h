#pragma once
#include <ATen/ATen.h>
#include <torch/library.h>
#include <webgpu/webgpu_cpp.h>
#include "utils/string.h"
#include "core/webgpu_context.h"

namespace torch_webgpu
{
    namespace ops
    {
        struct UnaryKernel
        {
            wgpu::BindGroupLayout bind_group_layout;
            wgpu::ComputePipeline pipeline;
        };

        enum class UnaryOp
        {
            Copy,
            ReLU
        };

        std::string get_unary_shader(UnaryOp unary_op);
        UnaryKernel &get_unary_kernel(UnaryOp unary_op);
    }
}
