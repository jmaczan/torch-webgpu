#include <ATen/ATen.h>
#include <torch/library.h>
#include <webgpu/webgpu_cpp.h>
#include "core/webgpu_context.h"
#include "core/webgpu_allocator.h"

namespace torch_webgpu
{
    namespace ops
    {
        // Linear layer: output = input @ weight.T + bias
        // input: [..., in_features]
        // weight: [out_features, in_features]
        // bias: [out_features] or None
        // output: [..., out_features]
        at::Tensor linear(const at::Tensor &input, const at::Tensor &weight, const c10::optional<at::Tensor> &bias_opt)
        {
            TORCH_CHECK(input.device().type() == c10::DeviceType::PrivateUse1);
            TORCH_CHECK(weight.device().type() == c10::DeviceType::PrivateUse1);
            TORCH_CHECK(weight.dim() == 2, "linear weight must be 2D");

            auto in_features = weight.size(1);
            auto out_features = weight.size(0);

            TORCH_CHECK(input.size(-1) == in_features, "linear: input features must match weight");

            // Flatten input to 2D: [batch, in_features]
            auto input_shape = input.sizes();
            int64_t batch_size = 1;
            for (int64_t i = 0; i < input.dim() - 1; ++i)
            {
                batch_size *= input_shape[i];
            }

            auto input_2d = input.reshape({batch_size, in_features});

            // Transpose weight: [out_features, in_features] -> [in_features, out_features]
            auto weight_t = weight.t().contiguous();

            // Matrix multiply: [batch, in_features] @ [in_features, out_features] = [batch, out_features]
            auto output_2d = at::mm(input_2d, weight_t);

            // Add bias if present
            if (bias_opt.has_value() && bias_opt->defined())
            {
                auto bias = bias_opt.value();
                TORCH_CHECK(bias.device().type() == c10::DeviceType::PrivateUse1);
                TORCH_CHECK(bias.size(0) == out_features, "linear: bias size must match out_features");

                // Broadcast add: [batch, out_features] + [out_features]
                output_2d = output_2d + bias;
            }

            // Reshape back to [..., out_features]
            std::vector<int64_t> output_shape(input_shape.begin(), input_shape.end() - 1);
            output_shape.push_back(out_features);

            return output_2d.reshape(output_shape);
        }

        // Addmm: beta * input + alpha * (mat1 @ mat2)
        at::Tensor addmm(
            const at::Tensor &self,
            const at::Tensor &mat1,
            const at::Tensor &mat2,
            const at::Scalar &beta,
            const at::Scalar &alpha)
        {
            TORCH_CHECK(mat1.dim() == 2 && mat2.dim() == 2, "addmm: mat1 and mat2 must be 2D");
            TORCH_CHECK(mat1.size(1) == mat2.size(0), "addmm: mat1 and mat2 shapes incompatible for matmul");

            auto mm_result = at::mm(mat1, mat2);

            // Convert scalars to float to match tensor dtype
            float alpha_f = alpha.to<float>();
            float beta_f = beta.to<float>();

            // Create scalar tensors with matching dtype and device
            auto alpha_tensor = at::scalar_tensor(alpha_f, mm_result.options());
            auto beta_tensor = at::scalar_tensor(beta_f, self.options());

            if (alpha_f != 1.0f)
            {
                mm_result = mm_result * alpha_tensor;
            }

            if (beta_f == 0.0f)
            {
                return mm_result;
            }
            else if (beta_f == 1.0f)
            {
                return self + mm_result;
            }
            else
            {
                return self * beta_tensor + mm_result;
            }
        }

        at::Tensor &addmm_out(
            const at::Tensor &self,
            const at::Tensor &mat1,
            const at::Tensor &mat2,
            const at::Scalar &beta,
            const at::Scalar &alpha,
            at::Tensor &out)
        {
            auto result = torch_webgpu::ops::addmm(self, mat1, mat2, beta, alpha);
            out.copy_(result);
            return out;
        }

        // BMM: batched matrix multiply
        at::Tensor bmm(const at::Tensor &self, const at::Tensor &mat2)
        {
            TORCH_CHECK(self.dim() == 3 && mat2.dim() == 3, "bmm: inputs must be 3D");
            TORCH_CHECK(self.size(0) == mat2.size(0), "bmm: batch sizes must match");
            TORCH_CHECK(self.size(2) == mat2.size(1), "bmm: matrix dimensions incompatible");

            auto batch_size = self.size(0);
            auto M = self.size(1);
            auto K = self.size(2);
            auto N = mat2.size(2);

            at::Tensor out = at::empty({batch_size, M, N}, self.options());

            // For now, loop over batch (can be optimized with batched kernel later)
            for (int64_t b = 0; b < batch_size; ++b)
            {
                auto self_b = self.select(0, b);
                auto mat2_b = mat2.select(0, b);
                auto out_b = out.select(0, b);
                at::mm_out(out_b, self_b, mat2_b);
            }

            return out;
        }

        at::Tensor &bmm_out(const at::Tensor &self, const at::Tensor &mat2, at::Tensor &out)
        {
            auto result = torch_webgpu::ops::bmm(self, mat2);
            out.copy_(result);
            return out;
        }

        // Matmul - general matrix multiplication with broadcasting
        at::Tensor matmul(const at::Tensor &self, const at::Tensor &other)
        {
            auto self_dim = self.dim();
            auto other_dim = other.dim();

            // Handle scalar cases
            if (self_dim == 0 || other_dim == 0)
            {
                return self * other;
            }

            // Vector-vector: dot product
            if (self_dim == 1 && other_dim == 1)
            {
                return at::dot(self, other);
            }

            // Matrix-vector: [M, K] @ [K] = [M]
            if (self_dim == 2 && other_dim == 1)
            {
                return at::mv(self, other);
            }

            // Vector-matrix: [K] @ [K, N] = [N]
            if (self_dim == 1 && other_dim == 2)
            {
                return at::mm(self.unsqueeze(0), other).squeeze(0);
            }

            // Matrix-matrix: [M, K] @ [K, N] = [M, N]
            if (self_dim == 2 && other_dim == 2)
            {
                return at::mm(self, other);
            }

            // Batched cases
            if (self_dim == 3 && other_dim == 3)
            {
                return torch_webgpu::ops::bmm(self, other);
            }

            // General broadcasting case - flatten batch dims, do bmm, reshape
            // For simplicity, handle common transformer cases
            if (self_dim >= 3 || other_dim >= 3)
            {
                // Get batch dimensions
                auto self_batch_dims = self_dim - 2;
                auto other_batch_dims = other_dim - 2;

                // Compute broadcasted batch shape
                std::vector<int64_t> batch_shape;
                auto max_batch_dims = std::max(self_batch_dims, other_batch_dims);

                for (int64_t i = 0; i < max_batch_dims; ++i)
                {
                    auto self_idx = self_batch_dims - max_batch_dims + i;
                    auto other_idx = other_batch_dims - max_batch_dims + i;

                    int64_t self_size = (self_idx >= 0) ? self.size(self_idx) : 1;
                    int64_t other_size = (other_idx >= 0) ? other.size(other_idx) : 1;

                    if (self_size == other_size)
                    {
                        batch_shape.push_back(self_size);
                    }
                    else if (self_size == 1)
                    {
                        batch_shape.push_back(other_size);
                    }
                    else if (other_size == 1)
                    {
                        batch_shape.push_back(self_size);
                    }
                    else
                    {
                        TORCH_CHECK(false, "matmul: batch dimensions not compatible");
                    }
                }

                auto M = self.size(-2);
                auto K = self.size(-1);
                auto N = other.size(-1);

                TORCH_CHECK(other.size(-2) == K, "matmul: matrix dimensions not compatible");

                // Flatten batch dimensions
                int64_t batch_numel = 1;
                for (auto s : batch_shape)
                    batch_numel *= s;

                // Expand and reshape to 3D
                std::vector<int64_t> self_target_shape = batch_shape;
                self_target_shape.push_back(M);
                self_target_shape.push_back(K);

                std::vector<int64_t> other_target_shape = batch_shape;
                other_target_shape.push_back(K);
                other_target_shape.push_back(N);

                auto self_expanded = self.expand(self_target_shape).contiguous().reshape({batch_numel, M, K});
                auto other_expanded = other.expand(other_target_shape).contiguous().reshape({batch_numel, K, N});

                auto result_3d = torch_webgpu::ops::bmm(self_expanded, other_expanded);

                // Reshape back
                std::vector<int64_t> result_shape = batch_shape;
                result_shape.push_back(M);
                result_shape.push_back(N);

                return result_3d.reshape(result_shape);
            }

            TORCH_CHECK(false, "matmul: unsupported tensor dimensions");
            return self;
        }

        // Dot product: vector @ vector -> scalar
        at::Tensor dot(const at::Tensor &self, const at::Tensor &other)
        {
            TORCH_CHECK(self.dim() == 1, "dot: self must be 1D");
            TORCH_CHECK(other.dim() == 1, "dot: other must be 1D");
            TORCH_CHECK(self.size(0) == other.size(0), "dot: vectors must have same size");

            // Element-wise multiply then sum
            auto product = self * other;
            return at::sum(product);
        }

        // Matrix-vector multiply: mat @ vec -> vec
        at::Tensor mv(const at::Tensor &mat, const at::Tensor &vec)
        {
            TORCH_CHECK(mat.dim() == 2, "mv: mat must be 2D");
            TORCH_CHECK(vec.dim() == 1, "mv: vec must be 1D");
            TORCH_CHECK(mat.size(1) == vec.size(0), "mv: mat columns must match vec size");

            // Reshape vec to [K, 1], do mm, then squeeze back to 1D
            auto vec_2d = vec.reshape({vec.size(0), 1});
            auto result_2d = at::mm(mat, vec_2d);
            return result_2d.reshape({result_2d.size(0)});
        }
    }

    TORCH_LIBRARY_IMPL(aten, PrivateUse1, m)
    {
        m.impl("linear", TORCH_FN(ops::linear));
        m.impl("addmm", TORCH_FN(ops::addmm));
        m.impl("addmm.out", TORCH_FN(ops::addmm_out));
        m.impl("bmm", TORCH_FN(ops::bmm));
        m.impl("bmm.out", TORCH_FN(ops::bmm_out));
        m.impl("matmul", TORCH_FN(ops::matmul));
        m.impl("dot", TORCH_FN(ops::dot));
        m.impl("mv", TORCH_FN(ops::mv));
    }
}
