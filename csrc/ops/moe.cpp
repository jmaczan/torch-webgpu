/**
 * MoE (Mixture of Experts) operations for WebGPU backend.
 *
 * These ops are essential for running MoE models like Llama-4-Scout.
 * Currently implemented as CPU fallbacks with plans to add WebGPU shaders
 * for better performance.
 */

#include <ATen/ATen.h>
#include <torch/library.h>

namespace torch_webgpu
{
    namespace ops
    {
        // =====================================================
        // topk - Select top k values and indices
        // Essential for expert routing in MoE
        // =====================================================
        std::tuple<at::Tensor, at::Tensor> topk_impl(
            const at::Tensor &self,
            int64_t k,
            int64_t dim,
            bool largest,
            bool sorted)
        {
            // CPU fallback for now - can optimize with WebGPU shader later
            auto cpu_tensor = self.to(at::kCPU);
            auto [values_cpu, indices_cpu] = at::topk(cpu_tensor, k, dim, largest, sorted);
            return std::make_tuple(
                values_cpu.to(self.device()),
                indices_cpu.to(self.device()));
        }

        // gather - already implemented in basic.cpp

        // =====================================================
        // scatter - Scatter values along an axis
        // Used for routing tokens to experts
        // =====================================================
        at::Tensor scatter_value_impl(
            const at::Tensor &self,
            int64_t dim,
            const at::Tensor &index,
            const at::Scalar &value)
        {
            auto cpu_self = self.to(at::kCPU);
            auto cpu_index = index.to(at::kCPU);
            auto result_cpu = at::scatter(cpu_self, dim, cpu_index, value);
            return result_cpu.to(self.device());
        }

        at::Tensor scatter_src_impl(
            const at::Tensor &self,
            int64_t dim,
            const at::Tensor &index,
            const at::Tensor &src)
        {
            auto cpu_self = self.to(at::kCPU);
            auto cpu_index = index.to(at::kCPU);
            auto cpu_src = src.to(at::kCPU);
            auto result_cpu = at::scatter(cpu_self, dim, cpu_index, cpu_src);
            return result_cpu.to(self.device());
        }

        at::Tensor &scatter_value_out_impl(
            const at::Tensor &self,
            int64_t dim,
            const at::Tensor &index,
            const at::Scalar &value,
            at::Tensor &out)
        {
            auto result = scatter_value_impl(self, dim, index, value);
            out.copy_(result);
            return out;
        }

        at::Tensor &scatter_src_out_impl(
            const at::Tensor &self,
            int64_t dim,
            const at::Tensor &index,
            const at::Tensor &src,
            at::Tensor &out)
        {
            auto result = scatter_src_impl(self, dim, index, src);
            out.copy_(result);
            return out;
        }

        // In-place scatter
        at::Tensor &scatter_value_inplace_impl(
            at::Tensor &self,
            int64_t dim,
            const at::Tensor &index,
            const at::Scalar &value)
        {
            auto result = scatter_value_impl(self, dim, index, value);
            self.copy_(result);
            return self;
        }

        at::Tensor &scatter_src_inplace_impl(
            at::Tensor &self,
            int64_t dim,
            const at::Tensor &index,
            const at::Tensor &src)
        {
            auto result = scatter_src_impl(self, dim, index, src);
            self.copy_(result);
            return self;
        }

        // =====================================================
        // scatter_add - Scatter with addition (accumulate)
        // Used for combining expert outputs
        // =====================================================
        at::Tensor scatter_add_impl(
            const at::Tensor &self,
            int64_t dim,
            const at::Tensor &index,
            const at::Tensor &src)
        {
            auto cpu_self = self.to(at::kCPU);
            auto cpu_index = index.to(at::kCPU);
            auto cpu_src = src.to(at::kCPU);
            auto result_cpu = at::scatter_add(cpu_self, dim, cpu_index, cpu_src);
            return result_cpu.to(self.device());
        }

        at::Tensor &scatter_add_out_impl(
            const at::Tensor &self,
            int64_t dim,
            const at::Tensor &index,
            const at::Tensor &src,
            at::Tensor &out)
        {
            auto result = scatter_add_impl(self, dim, index, src);
            out.copy_(result);
            return out;
        }

        // =====================================================
        // any - Any reduction (boolean)
        // Used for mask operations in routing
        // =====================================================
        at::Tensor any_impl(const at::Tensor &self)
        {
            auto cpu_tensor = self.to(at::kCPU);
            auto result_cpu = at::any(cpu_tensor);
            return result_cpu.to(self.device());
        }

        at::Tensor any_dim_impl(const at::Tensor &self, int64_t dim, bool keepdim)
        {
            auto cpu_tensor = self.to(at::kCPU);
            auto result_cpu = at::any(cpu_tensor, dim, keepdim);
            return result_cpu.to(self.device());
        }

        at::Tensor &any_out_impl(const at::Tensor &self, int64_t dim, bool keepdim, at::Tensor &out)
        {
            auto result = any_dim_impl(self, dim, keepdim);
            out.copy_(result);
            return out;
        }

        // =====================================================
        // all - All reduction (boolean)
        // Also useful for mask operations
        // =====================================================
        at::Tensor all_impl(const at::Tensor &self)
        {
            auto cpu_tensor = self.to(at::kCPU);
            auto result_cpu = at::all(cpu_tensor);
            return result_cpu.to(self.device());
        }

        at::Tensor all_dim_impl(const at::Tensor &self, int64_t dim, bool keepdim)
        {
            auto cpu_tensor = self.to(at::kCPU);
            auto result_cpu = at::all(cpu_tensor, dim, keepdim);
            return result_cpu.to(self.device());
        }

        at::Tensor &all_out_impl(const at::Tensor &self, int64_t dim, bool keepdim, at::Tensor &out)
        {
            auto result = all_dim_impl(self, dim, keepdim);
            out.copy_(result);
            return out;
        }

        // =====================================================
        // nonzero - Find indices of non-zero elements
        // Used for sparse routing
        // =====================================================
        at::Tensor nonzero_impl(const at::Tensor &self)
        {
            auto cpu_tensor = self.to(at::kCPU);
            auto result_cpu = at::nonzero(cpu_tensor);
            return result_cpu.to(self.device());
        }

        // =====================================================
        // masked_select - Select elements by mask
        // Used for sparse token routing
        // =====================================================
        at::Tensor masked_select_impl(const at::Tensor &self, const at::Tensor &mask)
        {
            auto cpu_self = self.to(at::kCPU);
            auto cpu_mask = mask.to(at::kCPU);
            auto result_cpu = at::masked_select(cpu_self, cpu_mask);
            return result_cpu.to(self.device());
        }

        at::Tensor &masked_select_out_impl(const at::Tensor &self, const at::Tensor &mask, at::Tensor &out)
        {
            auto result = masked_select_impl(self, mask);
            out.resize_(result.sizes());
            out.copy_(result);
            return out;
        }

        // index_select - already implemented in basic.cpp
    }

    TORCH_LIBRARY_IMPL(aten, PrivateUse1, m)
    {
        // topk
        m.impl("topk", TORCH_FN(ops::topk_impl));

        // gather - already registered in basic.cpp

        // scatter
        m.impl("scatter.value", TORCH_FN(ops::scatter_value_impl));
        m.impl("scatter.src", TORCH_FN(ops::scatter_src_impl));
        m.impl("scatter.value_out", TORCH_FN(ops::scatter_value_out_impl));
        m.impl("scatter.src_out", TORCH_FN(ops::scatter_src_out_impl));
        m.impl("scatter_.value", TORCH_FN(ops::scatter_value_inplace_impl));
        m.impl("scatter_.src", TORCH_FN(ops::scatter_src_inplace_impl));

        // scatter_add
        m.impl("scatter_add", TORCH_FN(ops::scatter_add_impl));
        m.impl("scatter_add.out", TORCH_FN(ops::scatter_add_out_impl));

        // any / all
        m.impl("any", TORCH_FN(ops::any_impl));
        m.impl("any.dim", TORCH_FN(ops::any_dim_impl));
        m.impl("any.out", TORCH_FN(ops::any_out_impl));
        m.impl("all", TORCH_FN(ops::all_impl));
        m.impl("all.dim", TORCH_FN(ops::all_dim_impl));
        m.impl("all.out", TORCH_FN(ops::all_out_impl));

        // nonzero
        m.impl("nonzero", TORCH_FN(ops::nonzero_impl));

        // masked_select
        m.impl("masked_select", TORCH_FN(ops::masked_select_impl));
        m.impl("masked_select.out", TORCH_FN(ops::masked_select_out_impl));

        // index_select - already registered in basic.cpp
    }
}
