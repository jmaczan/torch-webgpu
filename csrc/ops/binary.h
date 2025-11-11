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
        struct BinaryKernel
        {
            wgpu::BindGroupLayout bind_group_layout;
            wgpu::ComputePipeline pipeline;
        };

        enum class BinaryOp
        {
            Add
        };

        std::string get_binary_shader(BinaryOp binary_op);
        BinaryKernel &get_binary_kernel(BinaryOp binary_op);
    }
}
