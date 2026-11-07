// scaled_dot_product_attention through PyTorch's "overrideable" SDPA backend:
// our _fused_sdp_choice picks the fused WGSL kernel (kernels/attention) when
// it applies (no dropout, head dims <= 128, enough shared memory) and the
// math path otherwise. The backward recomputes the probabilities (math form).

#include <ATen/ATen.h>
#include <ATen/SDPBackend.h>
#include <ATen/native/transformers/attention.h>
#include <torch/library.h>

#include <cmath>

#include "kernels/attention.h"
#include "ops/common.h"

namespace torch_webgpu::ops
{
    using at::Tensor;

    namespace
    {
        bool float_dtype(const Tensor &t)
        {
            return t.scalar_type() == at::kFloat || t.scalar_type() == at::kHalf || t.scalar_type() == at::kBFloat16;
        }

        int64_t sdp_choice(const Tensor &q, const Tensor &k, const Tensor &v, const std::optional<Tensor> &mask,
                           double dropout_p, bool, std::optional<double>, bool enable_gqa)
        {
            const auto math = static_cast<int64_t>(at::SDPBackend::math);
            if (q.dim() != 4 || k.dim() != 4 || v.dim() != 4 || dropout_p > 0.0)
                return math;
            if (!float_dtype(q) || k.scalar_type() != q.scalar_type() || v.scalar_type() != q.scalar_type())
                return math;
            if (q.size(0) != k.size(0) || k.size(0) != v.size(0) || k.size(2) != v.size(2) || q.size(3) != k.size(3) ||
                k.size(1) != v.size(1) || k.size(1) == 0 || q.size(1) % k.size(1) != 0 ||
                (!enable_gqa && q.size(1) != k.size(1)))
                return math;
            if (!kernels::attention_fits(q.size(3), v.size(3)))
                return math;
            // The fused kernel works on blocks of 16 query rows; for decode-like
            // shapes (few queries) the math path parallelizes better over keys,
            // unless its materialized scores would be large.
            const double score_bytes = 4.0 * q.size(0) * q.size(1) * q.size(2) * k.size(2);
            if (q.size(2) < 16 && score_bytes < 64.0 * (1 << 20))
                return math;
            if (mask.has_value() && mask->defined())
            {
                if (mask->dim() > 4 || !(float_dtype(*mask) || mask->scalar_type() == at::kBool))
                    return math;
                try
                {
                    (void)mask->expand({q.size(0), q.size(1), q.size(2), k.size(2)});
                }
                catch (const c10::Error &)
                {
                    return math;
                }
            }
            return static_cast<int64_t>(at::SDPBackend::overrideable);
        }

        double scale_of(const Tensor &q, std::optional<double> scale)
        {
            return scale.has_value() ? *scale : 1.0 / std::sqrt(static_cast<double>(q.size(-1)));
        }

        std::tuple<Tensor, Tensor, Tensor, Tensor, c10::SymInt, c10::SymInt, Tensor, Tensor, Tensor>
        sdpa_overrideable_fn(const Tensor &q, const Tensor &k, const Tensor &v, const std::optional<Tensor> &attn_bias,
                             double dropout_p, bool is_causal, bool, std::optional<double> scale)
        {
            TORCH_CHECK(dropout_p == 0.0, "torch_webgpu: fused attention does not support dropout");
            const int64_t B = q.size(0), H = q.size(1), Lq = q.size(2), Lk = k.size(2), Dv = v.size(3);
            Tensor out = at::empty({B, H, Lq, Dv}, q.options());
            Tensor lse = at::empty({B, H, Lq}, q.options().dtype(at::kFloat));
            Tensor bias;
            if (attn_bias.has_value() && attn_bias->defined())
            {
                bias = attn_bias->scalar_type() == at::kBool
                           ? at::where(*attn_bias, at::zeros({}, q.options()), at::full({}, -std::numeric_limits<double>::infinity(), q.options()))
                           : *attn_bias;
                bias = bias.expand({B, H, Lq, Lk});
            }
            kernels::run_attention(q, k, v, bias, is_causal, scale_of(q, scale), out, lse);
            Tensor empty = at::empty({0}, q.options().dtype(at::kLong));
            return {out, lse, empty, empty, Lq, Lk, empty, empty, Tensor()};
        }

        // Math-form backward with recomputed probabilities (GQA: kv heads are
        // shared by H / Hk query heads, whose gradients are summed).
        std::tuple<Tensor, Tensor, Tensor, Tensor> sdpa_overrideable_backward_fn(
            const Tensor &grad_out, const Tensor &q, const Tensor &k, const Tensor &v, const Tensor &attn_bias,
            std::array<bool, 4> mask, const Tensor &out, const Tensor &, const Tensor &, const Tensor &, c10::SymInt,
            c10::SymInt, double, bool is_causal, const Tensor &, const Tensor &, std::optional<double> scale)
        {
            const int64_t B = q.size(0), H = q.size(1), Lq = q.size(2), Hk = k.size(1), Lk = k.size(2);
            const int64_t G = H / Hk;
            const double sc = scale_of(q, scale);
            const auto acc = at::kFloat;
            Tensor qf = q.to(acc), kf = k.to(acc), vf = v.to(acc), go = grad_out.to(acc), of = out.to(acc);
            Tensor kk = G > 1 ? kf.repeat_interleave(G, 1) : kf, vv = G > 1 ? vf.repeat_interleave(G, 1) : vf;
            Tensor s = at::matmul(qf, kk.transpose(-2, -1)).mul_(sc);
            if (attn_bias.defined())
                s = s + (attn_bias.scalar_type() == at::kBool
                             ? at::where(attn_bias, at::zeros({}, s.options()), at::full({}, -std::numeric_limits<double>::infinity(), s.options()))
                             : attn_bias.to(acc));
            if (is_causal)
                s.masked_fill_(at::ones({Lq, Lk}, q.options().dtype(at::kBool)).triu(1), -std::numeric_limits<double>::infinity());
            Tensor p = at::_safe_softmax(s, -1);
            Tensor gq, gk, gv, gb;
            Tensor dp = at::matmul(go, vv.transpose(-2, -1));
            Tensor ds = p * (dp - (go * of).sum(-1, true));
            if (mask[0])
                gq = at::matmul(ds, kk).mul_(sc).to(q.scalar_type());
            if (mask[1])
            {
                Tensor t = at::matmul(ds.transpose(-2, -1), qf).mul_(sc);
                gk = (G > 1 ? t.view({B, Hk, G, Lk, -1}).sum(2) : t).to(k.scalar_type());
            }
            if (mask[2])
            {
                Tensor t = at::matmul(p.transpose(-2, -1), go);
                gv = (G > 1 ? t.view({B, Hk, G, Lk, -1}).sum(2) : t).to(v.scalar_type());
            }
            if (mask[3] && attn_bias.defined() && attn_bias.scalar_type() != at::kBool)
                gb = ds.sum_to_size(attn_bias.sizes()).to(attn_bias.scalar_type());
            return {gq, gk, gv, gb};
        }
    }

    TORCH_LIBRARY_IMPL(aten, PrivateUse1, m)
    {
        m.impl("_scaled_dot_product_fused_attention_overrideable", TORCH_FN(sdpa_overrideable_fn));
        m.impl("_scaled_dot_product_fused_attention_overrideable_backward", TORCH_FN(sdpa_overrideable_backward_fn));
    }
}

namespace at::native
{
    REGISTER_PRIVATEUSE1_DISPATCH(_fused_sdp_choice_stub, &torch_webgpu::ops::sdp_choice);
}
