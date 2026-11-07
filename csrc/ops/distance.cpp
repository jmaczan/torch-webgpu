// Pairwise distances: cdist (_cdist_forward / _cdist_backward) and pdist
// (_pdist_forward / _pdist_backward), composed from native kernels.

#include <ATen/ATen.h>
#include <torch/library.h>

#include <cmath>
#include <limits>

#include "ops/common.h"

namespace torch_webgpu::ops
{
    using at::Tensor;

    namespace
    {
        // Exact p-norm distances between rows: [..., r1, m] x [..., r2, m] -> [..., r1, r2].
        Tensor pairwise_exact(const Tensor &x1, const Tensor &x2, double p)
        {
            Tensor diff = x1.unsqueeze(-2) - x2.unsqueeze(-3);
            return at::linalg_vector_norm(diff, p, {-1}, false, std::nullopt);
        }

        Tensor cdist_forward_fn(const Tensor &x1, const Tensor &x2, double p, std::optional<int64_t> compute_mode)
        {
            TORCH_CHECK(x1.dim() >= 2, "cdist only supports at least 2D tensors, X1 got: ", x1.dim(), "D");
            TORCH_CHECK(x2.dim() >= 2, "cdist only supports at least 2D tensors, X2 got: ", x2.dim(), "D");
            TORCH_CHECK(x1.size(-1) == x2.size(-1), "X1 and X2 must have the same number of columns. X1: ",
                        x1.size(-1), " X2: ", x2.size(-1));
            TORCH_CHECK(at::isFloatingType(x1.scalar_type()), "cdist only supports floating-point dtypes, X1 got: ",
                        x1.scalar_type());
            TORCH_CHECK(at::isFloatingType(x2.scalar_type()), "cdist only supports floating-point dtypes, X2 got: ",
                        x2.scalar_type());
            TORCH_CHECK(p >= 0, "cdist only supports non-negative p values");
            const int64_t mode = compute_mode.value_or(0);
            TORCH_CHECK(mode >= 0 && mode <= 2, "possible modes: 0, 1, 2, but was: ", mode);
            const int64_t r1 = x1.size(-2), r2 = x2.size(-2);
            // Broadcast batch dimensions.
            auto batch1 = x1.sizes().slice(0, x1.dim() - 2), batch2 = x2.sizes().slice(0, x2.dim() - 2);
            auto batch = at::infer_size(batch1, batch2);
            auto s1 = batch, s2 = batch;
            s1.insert(s1.end(), {r1, x1.size(-1)});
            s2.insert(s2.end(), {r2, x2.size(-1)});
            Tensor a = x1.expand(s1), b = x2.expand(s2);
            auto out_shape = batch;
            out_shape.insert(out_shape.end(), {r1, r2});
            if (x1.size(-1) == 0)
                return at::zeros(out_shape, x1.options());
            if (r1 == 0 || r2 == 0 || a.numel() == 0)
                return at::empty(out_shape, x1.options());
            if (p == 2.0 && (mode == 1 || (mode == 0 && (r1 > 25 || r2 > 25))))
            {
                // As CPU/CUDA: |a|^2 + |b|^2 - 2 a.b through one matmul.
                Tensor an = a.pow(2).sum(-1, true), bn = b.pow(2).sum(-1, true);
                Tensor a_ = at::cat({a.mul(-2), an, at::ones_like(an)}, -1);
                Tensor b_ = at::cat({b, at::ones_like(bn), bn}, -1);
                return a_.matmul(b_.mT()).clamp_min_(0).sqrt_();
            }
            return pairwise_exact(a, b, p);
        }

        // d dist(x1_i, x2_j) / d x1_i, weighted by grad and summed over j.
        Tensor cdist_backward_fn(const Tensor &grad, const Tensor &x1, const Tensor &x2, double p, const Tensor &cdist)
        {
            if (x1.size(-2) == 0 || x2.size(-2) == 0 || x1.size(-1) == 0 || x1.numel() == 0 || p == 0.0)
                return at::zeros_like(x1);
            Tensor diff = x1.unsqueeze(-2) - x2.unsqueeze(-3); // [..., r1, r2, m]
            Tensor d = cdist.unsqueeze(-1);
            Tensor g = grad.unsqueeze(-1);
            Tensor term;
            if (p == 1.0)
                term = diff.sign() * g;
            else if (p == 2.0)
                term = at::where(d == 0, at::zeros_like(diff), diff * g / d);
            else if (std::isinf(p))
                term = diff.sign() * (diff.abs() == d).to(diff.scalar_type()) * g;
            else if (p < 2.0)
                term = at::where((d == 0) | ((diff == 0) & (p < 1.0)), at::zeros_like(diff),
                                 diff.sign() * diff.abs().pow(p - 1) * g / d.pow(p - 1));
            else
                term = at::where(d == 0, at::zeros_like(diff), diff * diff.abs().pow(p - 2) * g / d.pow(p - 1));
            Tensor r = term.sum(-2);
            return r.sizes() == x1.sizes() ? r : r.sum_to_size(x1.sizes());
        }

        // pdist: upper triangle (i < j, row-major) of the exact distance matrix.
        Tensor pdist_forward_fn(const Tensor &self, double p)
        {
            TORCH_CHECK(self.is_contiguous(), "_pdist_forward requires contiguous input");
            TORCH_CHECK(self.dim() == 2, "pdist only supports 2D tensors, got: ", self.dim(), "D");
            TORCH_CHECK(p >= 0, "pdist only supports non-negative p values");
            const int64_t n = self.size(0);
            if (n <= 1)
                return at::empty({0}, self.options());
            if (self.size(1) == 0)
                return at::zeros({n * (n - 1) / 2}, self.options());
            Tensor full = pairwise_exact(self, self, p);
            Tensor tri = at::triu_indices(n, n, 1, self.options().dtype(at::kLong));
            return full.index({tri[0], tri[1]});
        }
        Tensor pdist_backward_fn(const Tensor &grad, const Tensor &self, double p, const Tensor &pdist)
        {
            const int64_t n = self.size(0);
            if (n <= 1 || self.size(1) == 0 || p == 0.0)
                return at::zeros_like(self);
            Tensor tri = at::triu_indices(n, n, 1, self.options().dtype(at::kLong));
            Tensor g = at::zeros({n, n}, grad.options());
            g.index_put_({tri[0], tri[1]}, grad);
            g = g + g.t();
            Tensor dist = at::zeros({n, n}, pdist.options());
            dist.index_put_({tri[0], tri[1]}, pdist);
            dist = dist + dist.t();
            return cdist_backward_fn(g, self, self, p, dist);
        }
    }

    TORCH_LIBRARY_IMPL(aten, PrivateUse1, m)
    {
        m.impl("_cdist_forward", TORCH_FN(cdist_forward_fn));
        m.impl("_cdist_backward", TORCH_FN(cdist_backward_fn));
        m.impl("_pdist_forward", TORCH_FN(pdist_forward_fn));
        m.impl("_pdist_backward", TORCH_FN(pdist_backward_fn));
    }
}
