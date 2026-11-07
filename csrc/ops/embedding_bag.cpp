// embedding_bag (sum / mean / max, per-sample weights, padding_idx) and its
// backward, plus the multi-class margin losses, composed from native
// gathers, segment reductions (index_add / scatter_reduce) and elementwise ops.
//
// Auxiliary outputs follow CPU's meaning (they only feed the backward):
//   offset2bag[i]       bag of index i
//   bag_size[b]         number of non-padding indices in bag b
//   max_indices[b, d]   weight row of the (first) maximum (max mode)

#include <ATen/ATen.h>
#include <ATen/native/Resize.h>
#include <ATen/ops/_embedding_bag_backward_native.h>
#include <torch/library.h>

#include <limits>

#include "ops/common.h"

namespace torch_webgpu::ops
{
    using at::Tensor;

    namespace
    {
        enum Mode
        {
            Sum = 0,
            Mean = 1,
            Max = 2
        };

        std::tuple<Tensor, Tensor, Tensor, Tensor> embedding_bag_impl(const Tensor &weight, const Tensor &indices_in,
                                                                      const Tensor &offsets_in, int64_t mode,
                                                                      const std::optional<Tensor> &psw_in,
                                                                      bool include_last_offset, int64_t padding_idx)
        {
            TORCH_CHECK(weight.dim() == 2, "weight has to be a 2D Tensor, but got Tensor of dimension ", weight.dim());
            TORCH_CHECK(indices_in.dim() == 1 && offsets_in.dim() == 1,
                        "embedding_bag: expected 1-D indices and offsets, got ", indices_in.dim(), "-D and ",
                        offsets_in.dim(), "-D");
            TORCH_CHECK(mode >= Sum && mode <= Max, "embedding_bag: unknown mode ", mode);
            const bool has_psw = psw_in.has_value() && psw_in->defined();
            TORCH_CHECK(!has_psw || mode == Sum,
                        "embedding_bag: per_sample_weights was not None. per_sample_weights is only supported for mode='sum' (got mode='",
                        mode == Mean ? "mean" : "max", "'). Please open a feature request on GitHub.");
            auto lopt = weight.options().dtype(at::kLong);
            Tensor indices = indices_in.to(at::kLong), offsets = offsets_in.to(at::kLong);
            const int64_t n = indices.numel(), D = weight.size(1);
            int64_t B = offsets.numel() - (include_last_offset ? 1 : 0);
            TORCH_CHECK(B >= 0, "embedding_bag: offsets has to be a 1D Tensor with at least one element when include_last_offset=True");
            if (n > 0 && B > 0)
                TORCH_CHECK(offsets.narrow(0, 0, 1).item<int64_t>() == 0,
                            "offsets[0] has to be 0, i.e., the first sequence in the mini-batch has to start from position 0. However, got ",
                            offsets.narrow(0, 0, 1).item<int64_t>());
            Tensor bag_offsets = offsets.narrow(0, 0, B);

            Tensor output = at::zeros({B, D}, weight.options());
            Tensor bag_size = at::zeros({B}, lopt);
            Tensor max_indices = mode == Max ? at::full({B, D}, -1, lopt) : at::zeros({B}, lopt);
            if (n == 0 || B == 0)
                return {output, at::zeros({n}, lopt), bag_size, max_indices};

            // Bag of each index (indices past the last offset belong to the last bag).
            Tensor offset2bag = at::searchsorted(bag_offsets, at::arange(n, lopt), false, /*right=*/true) - 1;
            if (include_last_offset)
            {
                // indices at or beyond offsets[B] are not part of any bag.
                Tensor end = offsets.narrow(0, B, 1);
                offset2bag = at::where(at::arange(n, lopt) >= end, at::full_like(offset2bag, -1), offset2bag);
            }
            Tensor keep = offset2bag >= 0;
            if (padding_idx >= 0)
                keep = keep & (indices != padding_idx);
            Tensor bag = offset2bag.clamp_min(0);
            Tensor counts = at::zeros({B}, weight.options().dtype(at::kInt));
            counts.index_add_(0, bag, keep.to(at::kInt));
            bag_size = counts.to(at::kLong);

            Tensor rows = weight.index_select(0, indices); // [n, D]
            Tensor keepw = keep.to(weight.scalar_type()).unsqueeze(1);
            if (mode == Max)
            {
                Tensor vals = at::where(keep.unsqueeze(1), rows, at::zeros_like(rows));
                Tensor bag2 = bag.unsqueeze(1).expand({n, D});
                Tensor lowest = at::full({B, D}, -std::numeric_limits<double>::infinity(), weight.options());
                // Excluded entries scatter -inf, which never wins.
                Tensor masked = at::where(keep.unsqueeze(1), rows, at::full_like(rows, -std::numeric_limits<double>::infinity()));
                Tensor mx = lowest.scatter_reduce(0, bag2, masked, "amax", /*include_self=*/true);
                Tensor nonempty = (bag_size > 0).unsqueeze(1);
                output = at::where(nonempty, mx, at::zeros_like(mx));
                // First position (in index order) attaining the max.
                // (int32 positions: 64-bit atomic min is unavailable on GPUs)
                auto iopt = weight.options().dtype(at::kInt);
                TORCH_CHECK(n < std::numeric_limits<int32_t>::max(), "embedding_bag: too many indices");
                Tensor pos = at::arange(n, iopt).unsqueeze(1).expand({n, D});
                Tensor is_max = keep.unsqueeze(1) & (masked == mx.gather(0, bag2));
                Tensor cand = at::where(is_max, pos, at::full_like(pos, n));
                Tensor first = at::full({B, D}, n, iopt).scatter_reduce(0, bag2, cand, "amin", true).to(at::kLong);
                Tensor safe_first = first.clamp_max(n - 1);
                max_indices = at::where(first < n, indices.index_select(0, safe_first.reshape({-1})).view({B, D}),
                                        at::full_like(first, -1));
                (void)vals;
            }
            else
            {
                Tensor contrib = rows * keepw;
                if (has_psw)
                    contrib = contrib * psw_in->to(weight.scalar_type()).unsqueeze(1);
                output.index_add_(0, bag, contrib);
                if (mode == Mean)
                    output = output / bag_size.clamp_min(1).unsqueeze(1).to(weight.scalar_type());
            }
            return {output, offset2bag.clamp_min(0), bag_size, max_indices};
        }

        std::tuple<Tensor, Tensor, Tensor, Tensor> embedding_bag_fn(const Tensor &weight, const Tensor &indices,
                                                                    const Tensor &offsets, bool, int64_t mode, bool,
                                                                    const std::optional<Tensor> &psw,
                                                                    bool include_last_offset, int64_t padding_idx)
        {
            // sparse=True only changes the gradient's layout, not the forward.
            return embedding_bag_impl(weight, indices, offsets, mode, psw, include_last_offset, padding_idx);
        }

        // Per-index gradient scale for scale_grad_by_freq: 1 / count(index).
        Tensor freq_scale(const Tensor &indices, int64_t num_weights, const at::TensorOptions &opt)
        {
            Tensor counts = at::zeros({num_weights}, opt.dtype(at::kInt));
            counts.index_add_(0, indices, at::ones({indices.numel()}, opt.dtype(at::kInt)));
            return counts.index_select(0, indices).to(opt.dtype());
        }

        Tensor embedding_bag_dense_backward_fn(const Tensor &grad, const Tensor &indices_in, const Tensor &offset2bag,
                                               const Tensor &bag_size, const Tensor &max_indices, c10::SymInt num_weights_s,
                                               bool scale_grad_by_freq, int64_t mode,
                                               const std::optional<Tensor> &psw, int64_t padding_idx)
        {
            const int64_t num_weights = num_weights_s.expect_int();
            const int64_t D = grad.size(1);
            Tensor indices = indices_in.to(at::kLong);
            Tensor gw = at::zeros({num_weights, D}, grad.options());
            if (indices.numel() == 0)
                return gw;
            if (mode == Max)
            {
                // grad_weight[max_indices[b, d], d] += grad[b, d]
                Tensor valid = max_indices >= 0;
                Tensor flat = (max_indices.clamp_min(0) * D + at::arange(D, max_indices.options()).unsqueeze(0)).reshape({-1});
                Tensor g = at::where(valid, grad, at::zeros_like(grad)).reshape({-1});
                gw.view({-1}).index_add_(0, flat, g);
                return gw;
            }
            Tensor keep = at::ones({indices.numel()}, grad.options());
            if (padding_idx >= 0)
                keep = (indices != padding_idx).to(grad.scalar_type());
            Tensor scale = keep;
            if (mode == Mean)
                scale = scale / bag_size.index_select(0, offset2bag).clamp_min(1).to(grad.scalar_type());
            if (psw.has_value() && psw->defined())
                scale = scale * psw->to(grad.scalar_type());
            if (scale_grad_by_freq)
                scale = scale / freq_scale(indices, num_weights, grad.options());
            gw.index_add_(0, indices, grad.index_select(0, offset2bag) * scale.unsqueeze(1));
            return gw;
        }

        Tensor embedding_bag_psw_backward_fn(const Tensor &grad, const Tensor &weight, const Tensor &indices_in,
                                             const Tensor &, const Tensor &offset2bag, int64_t mode, int64_t padding_idx)
        {
            TORCH_CHECK(mode == Sum, "embedding_bag_backward: per_sample_weights only supported for mode='sum'");
            Tensor indices = indices_in.to(at::kLong);
            if (indices.numel() == 0)
                return at::zeros({0}, grad.options());
            Tensor r = (grad.index_select(0, offset2bag) * weight.index_select(0, indices)).sum(1);
            if (padding_idx >= 0)
                r = at::where(indices != padding_idx, r, at::zeros_like(r));
            return r;
        }

        // ------------------------------------------------- margin losses
        Tensor reduce(const Tensor &l, int64_t reduction)
        {
            return reduction == at::Reduction::Mean ? l.mean() : reduction == at::Reduction::Sum ? l.sum() : l;
        }

        // loss_n = sum_{i != y} max(0, margin - x[y] + x[i])^p * w[y] / C
        struct MultiMargin
        {
            Tensor x, y, z, mask, wy;
            int64_t C;
        };
        MultiMargin multi_margin_terms(const Tensor &self, const Tensor &target, const at::Scalar &margin,
                                       const std::optional<Tensor> &weight)
        {
            MultiMargin m;
            m.x = self.dim() <= 1 ? self.reshape({1, -1}) : self;
            m.C = m.x.size(1);
            m.y = target.reshape({-1, 1}).to(at::kLong);
            TORCH_CHECK(m.y.size(0) == m.x.size(0), "inconsistent target size: ", target.sizes(), " for input of size: ",
                        self.sizes());
            Tensor xy = m.x.gather(1, m.y);
            m.z = margin.toDouble() - xy + m.x;
            Tensor not_target = at::arange(m.C, m.y.options()).unsqueeze(0) != m.y;
            m.mask = (m.z > 0) & not_target;
            m.wy = weight.has_value() && weight->defined() ? weight->index_select(0, m.y.view({-1})).unsqueeze(1)
                                                           : at::ones({m.x.size(0), 1}, m.x.options());
            return m;
        }
        Tensor multi_margin_loss_fn(const Tensor &self, const Tensor &target, const at::Scalar &p, const at::Scalar &margin,
                                    const std::optional<Tensor> &weight, int64_t reduction)
        {
            TORCH_CHECK(p.toLong() == 1 || p.toLong() == 2, "only p == 1 and p == 2 supported");
            TORCH_CHECK(self.numel() > 0 && self.dim() <= 2, "Expected non-empty vector or matrix with optional 0-dim batch size, but got: ",
                        self.sizes());
            auto m = multi_margin_terms(self, target, margin, weight);
            Tensor h = p.toLong() == 1 ? m.z : m.z * m.z;
            Tensor l = (at::where(m.mask, h, at::zeros_like(h)) * m.wy).sum(1) / static_cast<double>(m.C);
            if (self.dim() <= 1 && reduction == at::Reduction::None)
                l = l.squeeze(0);
            return reduce(l, reduction);
        }
        Tensor &multi_margin_loss_out_fn(const Tensor &self, const Tensor &target, const at::Scalar &p,
                                         const at::Scalar &margin, const std::optional<Tensor> &weight, int64_t reduction,
                                         Tensor &out)
        {
            Tensor r = multi_margin_loss_fn(self, target, p, margin, weight, reduction);
            at::native::resize_output(out, r.sizes());
            return out.copy_(r);
        }
        Tensor multi_margin_loss_backward_fn(const Tensor &grad_output, const Tensor &self, const Tensor &target,
                                             const at::Scalar &p, const at::Scalar &margin,
                                             const std::optional<Tensor> &weight, int64_t reduction)
        {
            auto m = multi_margin_terms(self, target, margin, weight);
            const int64_t N = m.x.size(0);
            double g = reduction == at::Reduction::Mean ? 1.0 / static_cast<double>(N * m.C) : 1.0 / static_cast<double>(m.C);
            Tensor h = (p.toLong() == 1 ? at::ones_like(m.z) : 2 * m.z) * g * m.wy;
            h = at::where(m.mask, h, at::zeros_like(h));
            Tensor gi = h.clone();
            gi.scatter_add_(1, m.y, -h.sum(1, true));
            Tensor go = reduction == at::Reduction::None ? grad_output.reshape({N, 1}) : grad_output.reshape({1, 1});
            gi = gi * go;
            return gi.view(self.sizes());
        }
        Tensor &multi_margin_loss_backward_out_fn(const Tensor &grad_output, const Tensor &self, const Tensor &target,
                                                  const at::Scalar &p, const at::Scalar &margin,
                                                  const std::optional<Tensor> &weight, int64_t reduction, Tensor &gi)
        {
            Tensor r = multi_margin_loss_backward_fn(grad_output, self, target, p, margin, weight, reduction);
            at::native::resize_output(gi, r.sizes());
            return gi.copy_(r);
        }

        // multilabel: targets per row are class indices up to the first -1.
        struct MultiLabel
        {
            Tensor x, tgt, valid, is_target, mask, z;
            int64_t C;
        };
        MultiLabel multilabel_terms(const Tensor &self, const Tensor &target)
        {
            MultiLabel m;
            m.x = self.dim() <= 1 ? self.reshape({1, -1}) : self;
            m.C = m.x.size(1);
            Tensor t = target.reshape(m.x.sizes()).to(at::kLong);
            m.valid = (t != -1).to(at::kLong).cumprod(1).to(at::kBool); // [N, J]
            m.tgt = at::where(m.valid, t, at::zeros_like(t));
            m.is_target = at::zeros_like(m.x);
            m.is_target.scatter_(1, m.tgt, m.valid.to(m.x.scalar_type()), "add");
            m.is_target.clamp_max_(1);
            Tensor xy = m.x.gather(1, m.tgt);                     // [N, J]
            m.z = 1 - xy.unsqueeze(2) + m.x.unsqueeze(1);         // [N, J, C]
            m.mask = m.valid.unsqueeze(2) & (m.is_target == 0).unsqueeze(1) & (m.z > 0);
            return m;
        }
        std::tuple<Tensor, Tensor> multilabel_margin_loss_forward_fn(const Tensor &self, const Tensor &target,
                                                                     int64_t reduction)
        {
            TORCH_CHECK(self.dim() <= 2, "Expected non-empty vector or matrix with optional 0-dim batch size, but got: ",
                        self.sizes());
            if (self.numel() == 0)
                return {reduce(at::zeros({self.dim() <= 1 ? 1 : self.size(0)}, self.options()), reduction),
                        at::zeros_like(self)};
            auto m = multilabel_terms(self, target);
            Tensor l = at::where(m.mask, m.z, at::zeros_like(m.z)).sum({1, 2}) / static_cast<double>(m.C);
            if (self.dim() <= 1 && reduction == at::Reduction::None)
                l = l.squeeze(0);
            return {reduce(l, reduction), m.is_target.view(self.sizes())};
        }
        Tensor multilabel_margin_loss_backward_fn(const Tensor &grad_output, const Tensor &self, const Tensor &target,
                                                  int64_t reduction, const Tensor &)
        {
            if (self.numel() == 0)
                return at::zeros_like(self);
            auto m = multilabel_terms(self, target);
            const int64_t N = m.x.size(0);
            Tensor M = m.mask.to(m.x.scalar_type());            // [N, J, C]
            Tensor gi = M.sum(1);                                // + for x[i]
            gi.scatter_add_(1, m.tgt, -(M.sum(2) * m.valid.to(m.x.scalar_type()))); // - for x[y_j]
            double g = reduction == at::Reduction::Mean ? 1.0 / static_cast<double>(N * m.C) : 1.0 / static_cast<double>(m.C);
            Tensor go = reduction == at::Reduction::None ? grad_output.reshape({N, 1}) : grad_output.reshape({1, 1});
            return (gi * g * go).view(self.sizes());
        }
    }

    TORCH_LIBRARY_IMPL(aten, PrivateUse1, m)
    {
        m.impl("_embedding_bag", TORCH_FN(embedding_bag_fn));
        m.impl("_embedding_bag_forward_only", TORCH_FN(embedding_bag_fn));
        m.impl("_embedding_bag_backward", TORCH_FN(at::native::_embedding_bag_backward_symint));
        m.impl("_embedding_bag_dense_backward", TORCH_FN(embedding_bag_dense_backward_fn));
        m.impl("_embedding_bag_per_sample_weights_backward", TORCH_FN(embedding_bag_psw_backward_fn));
        m.impl("multi_margin_loss", TORCH_FN(multi_margin_loss_fn));
        m.impl("multi_margin_loss.out", TORCH_FN(multi_margin_loss_out_fn));
        m.impl("multi_margin_loss_backward", TORCH_FN(multi_margin_loss_backward_fn));
        m.impl("multi_margin_loss_backward.grad_input", TORCH_FN(multi_margin_loss_backward_out_fn));
        m.impl("multilabel_margin_loss_forward", TORCH_FN(multilabel_margin_loss_forward_fn));
        m.impl("multilabel_margin_loss_backward", TORCH_FN(multilabel_margin_loss_backward_fn));
    }
}
