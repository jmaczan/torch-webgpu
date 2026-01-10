#include <ATen/ATen.h>
#include <torch/library.h>
#include <webgpu/webgpu_cpp.h>
#include "core/webgpu_context.h"
#include "core/webgpu_allocator.h"
#include "unary.h"

namespace torch_webgpu
{
    namespace ops
    {
        // Cat - concatenate tensors along a dimension
        at::Tensor cat(const at::ITensorListRef &tensors, int64_t dim)
        {
            std::vector<at::Tensor> tensor_list;
            for (const auto &t : tensors)
            {
                tensor_list.push_back(t);
            }

            TORCH_CHECK(!tensor_list.empty(), "cat: cannot concatenate empty tensor list");

            // Handle negative dim
            auto first = tensor_list[0];
            auto ndim = first.dim();
            if (dim < 0)
                dim += ndim;

            TORCH_CHECK(dim >= 0 && dim < ndim, "cat: dimension out of range");

            // Check all tensors have same shape except for cat dimension
            int64_t total_size = 0;
            for (size_t i = 0; i < tensor_list.size(); ++i)
            {
                auto &t = tensor_list[i];
                TORCH_CHECK(t.dim() == ndim, "cat: tensors must have same number of dimensions");
                TORCH_CHECK(t.device() == first.device(), "cat: tensors must be on same device");
                TORCH_CHECK(t.scalar_type() == first.scalar_type(), "cat: tensors must have same dtype");

                for (int64_t d = 0; d < ndim; ++d)
                {
                    if (d != dim)
                    {
                        TORCH_CHECK(t.size(d) == first.size(d), "cat: sizes must match except for cat dimension");
                    }
                }
                total_size += t.size(dim);
            }

            // Create output shape
            std::vector<int64_t> out_shape(first.sizes().begin(), first.sizes().end());
            out_shape[dim] = total_size;

            at::Tensor out = at::empty(out_shape, first.options());

            // Copy each tensor to appropriate position using slice
            int64_t offset = 0;
            for (const auto &t : tensor_list)
            {
                auto slice_size = t.size(dim);
                // Use narrow to get the slice and copy into it
                auto out_slice = out.narrow(dim, offset, slice_size);
                out_slice.copy_(t);
                offset += slice_size;
            }

            return out;
        }

        // Stack - stack tensors along a new dimension
        at::Tensor stack(at::TensorList tensors, int64_t dim)
        {
            TORCH_CHECK(!tensors.empty(), "stack: cannot stack empty tensor list");

            auto first = tensors[0];
            auto ndim = first.dim();

            // Handle negative dim (can be at ndim+1)
            if (dim < 0)
                dim += ndim + 1;

            TORCH_CHECK(dim >= 0 && dim <= ndim, "stack: dimension out of range");

            // Unsqueeze all tensors at dim and concatenate
            std::vector<at::Tensor> unsqueezed;
            for (const auto &t : tensors)
            {
                unsqueezed.push_back(t.unsqueeze(dim));
            }

            return torch_webgpu::ops::cat(unsqueezed, dim);
        }

        // Narrow - select a slice along a dimension
        at::Tensor narrow(const at::Tensor &self, int64_t dim, c10::SymInt start, c10::SymInt length)
        {
            auto ndim = self.dim();
            if (dim < 0)
                dim += ndim;

            TORCH_CHECK(dim >= 0 && dim < ndim, "narrow: dimension out of range");

            auto start_val = start.expect_int();
            auto length_val = length.expect_int();

            return at::slice(self, dim, start_val, start_val + length_val, 1);
        }
    }

    TORCH_LIBRARY_IMPL(aten, PrivateUse1, m)
    {
        m.impl("cat", TORCH_FN(ops::cat));
        m.impl("stack", TORCH_FN(ops::stack));
        m.impl("narrow", TORCH_FN(ops::narrow));
    }
}
