// Neural-network layer ops: batch/group norm, pooling, upsampling, im2col /
// col2im and losses, with their backward functions.
//
// These are expressed with native WebGPU ops (reductions, GEMM, gather/
// scatter, elementwise). Separable resampling (adaptive average pooling,
// linear/bilinear/bicubic upsampling) is computed exactly as A_h @ X @ A_w^T
// with small interpolation matrices built on the host.

#include <ATen/ATen.h>
#include <ATen/ExpandUtils.h>
#include <ATen/OpMathType.h>
#include <ATen/core/Reduction.h>
#include <torch/library.h>

#include "kernels/spatial.h"
#include "ops/common.h"

#include <cmath>

namespace torch_webgpu::ops
{
    using at::Tensor;
    using Ints = std::vector<int64_t>;

    namespace
    {
        Ints vec(at::IntArrayRef a) { return a.vec(); }
        Ints vec(c10::SymIntArrayRef a) { return C10_AS_INTARRAYREF_SLOW(a).vec(); }

        Ints expand2(at::IntArrayRef p, size_t n, Ints def = {})
        {
            if (p.empty())
                return def;
            if (p.size() == n)
                return p.vec();
            TORCH_CHECK(p.size() == 1, "expected a single value or ", n, " values, got ", p.size());
            return Ints(n, p[0]);
        }

        Tensor reduce_loss(const Tensor &l, int64_t reduction)
        {
            if (reduction == at::Reduction::Mean)
                return l.mean();
            if (reduction == at::Reduction::Sum)
                return l.sum();
            return l;
        }
        double loss_norm(const Tensor &self, int64_t reduction)
        {
            return reduction == at::Reduction::Mean ? 1.0 / static_cast<double>(std::max<int64_t>(self.numel(), 1)) : 1.0;
        }

        // ============================================================ norms
        Ints channel_reduce_dims(const Tensor &x)
        {
            Ints dims{0};
            for (int64_t d = 2; d < x.dim(); ++d)
                dims.push_back(d);
            return dims;
        }
        Ints channel_shape(const Tensor &x)
        {
            Ints s(x.dim(), 1);
            s[1] = x.size(1);
            return s;
        }

        // Returns (out, save_mean, save_invstd); updates running stats in place.
        std::tuple<Tensor, Tensor, Tensor> batch_norm_impl(const Tensor &input, const std::optional<Tensor> &weight,
                                                           const std::optional<Tensor> &bias,
                                                           const std::optional<Tensor> &running_mean,
                                                           const std::optional<Tensor> &running_var, bool training,
                                                           double momentum, double eps)
        {
            TORCH_CHECK(input.dim() >= 2, "batch_norm: expected input with at least 2 dims");
            const auto acc = at::toOpMathType(input.scalar_type());
            Tensor x = input.to(acc);
            auto cs = channel_shape(x);
            Tensor mean, invstd;
            if (training)
            {
                auto dims = channel_reduce_dims(x);
                auto [var, m] = at::var_mean(x, dims, /*correction=*/0, /*keepdim=*/false);
                mean = m;
                invstd = (var + eps).rsqrt();
                const int64_t n = x.numel() / x.size(1);
                if (running_mean.has_value() && running_mean->defined())
                    running_mean->mul_(1 - momentum).add_(m.to(running_mean->scalar_type()), momentum);
                if (running_var.has_value() && running_var->defined())
                {
                    Tensor unbiased = n > 1 ? var * (static_cast<double>(n) / static_cast<double>(n - 1)) : var;
                    running_var->mul_(1 - momentum).add_(unbiased.to(running_var->scalar_type()), momentum);
                }
            }
            else
            {
                TORCH_CHECK(running_mean.has_value() && running_mean->defined() && running_var.has_value() &&
                                running_var->defined(),
                            "batch_norm: running_mean and running_var must be defined in evaluation mode");
                mean = running_mean->to(acc);
                invstd = (running_var->to(acc) + eps).rsqrt();
            }
            Tensor y = (x - mean.view(cs)) * invstd.view(cs);
            if (weight.has_value() && weight->defined())
                y = y * weight->to(acc).view(cs);
            if (bias.has_value() && bias->defined())
                y = y + bias->to(acc).view(cs);
            y = y.to(input.scalar_type());
            if (!training)
                return {y, at::empty({0}, x.options()), at::empty({0}, x.options())};
            return {y, mean, invstd};
        }

        std::tuple<Tensor, Tensor, Tensor> native_batch_norm_fn(const Tensor &input, const std::optional<Tensor> &weight,
                                                                const std::optional<Tensor> &bias,
                                                                const std::optional<Tensor> &running_mean,
                                                                const std::optional<Tensor> &running_var, bool training,
                                                                double momentum, double eps)
        {
            return batch_norm_impl(input, weight, bias, running_mean, running_var, training, momentum, eps);
        }

        std::tuple<Tensor, Tensor, Tensor> bn_legit_fn(const Tensor &input, const std::optional<Tensor> &weight,
                                                       const std::optional<Tensor> &bias, Tensor &running_mean,
                                                       Tensor &running_var, bool training, double momentum, double eps)
        {
            return batch_norm_impl(input, weight, bias, running_mean, running_var, training, momentum, eps);
        }
        std::tuple<Tensor, Tensor, Tensor> bn_legit_no_stats_fn(const Tensor &input, const std::optional<Tensor> &weight,
                                                                const std::optional<Tensor> &bias, bool training,
                                                                double momentum, double eps)
        {
            return batch_norm_impl(input, weight, bias, std::nullopt, std::nullopt, true, momentum, eps);
        }
        std::tuple<Tensor, Tensor, Tensor, Tensor> bn_with_update_fn(const Tensor &input, const std::optional<Tensor> &weight,
                                                                     const std::optional<Tensor> &bias, Tensor &running_mean,
                                                                     Tensor &running_var, double momentum, double eps)
        {
            auto [y, m, s] = batch_norm_impl(input, weight, bias, running_mean, running_var, true, momentum, eps);
            return {y, m, s, at::empty({0}, input.options().dtype(at::kByte))};
        }

        std::tuple<Tensor, Tensor, Tensor> native_batch_norm_backward_fn(
            const Tensor &grad_out, const Tensor &input, const std::optional<Tensor> &weight,
            const std::optional<Tensor> &running_mean, const std::optional<Tensor> &running_var,
            const std::optional<Tensor> &save_mean, const std::optional<Tensor> &save_invstd, bool train, double eps,
            std::array<bool, 3> mask)
        {
            const auto acc = at::toOpMathType(input.scalar_type());
            Tensor x = input.to(acc), g = grad_out.to(acc);
            auto cs = channel_shape(x);
            auto dims = channel_reduce_dims(x);
            Tensor mean, invstd;
            if (train)
            {
                mean = save_mean->to(acc);
                invstd = save_invstd->to(acc);
            }
            else
            {
                mean = running_mean->to(acc);
                invstd = (running_var->to(acc) + eps).rsqrt();
            }
            const bool has_w = weight.has_value() && weight->defined();
            Tensor w = has_w ? weight->to(acc) : at::ones({x.size(1)}, x.options());
            Tensor xhat = (x - mean.view(cs)) * invstd.view(cs);
            Tensor gsum = g.sum(dims);
            Tensor gxhat = (g * xhat).sum(dims);
            Tensor gi, gw, gb;
            if (mask[0])
            {
                if (train)
                {
                    const double n = static_cast<double>(x.numel() / x.size(1));
                    gi = (w * invstd).view(cs) * (g - (gsum / n).view(cs) - xhat * (gxhat / n).view(cs));
                }
                else
                {
                    gi = g * (w * invstd).view(cs);
                }
                gi = gi.to(input.scalar_type());
            }
            if (mask[1] && has_w)
                gw = gxhat.to(weight->scalar_type());
            if (mask[2] && has_w)
                gb = gsum.to(weight->scalar_type());
            return {gi, gw, gb};
        }

        std::tuple<Tensor, Tensor, Tensor> batch_norm_backward_fn(
            const Tensor &grad_out, const Tensor &input, const Tensor &weight, const std::optional<Tensor> &running_mean,
            const std::optional<Tensor> &running_var, const std::optional<Tensor> &save_mean,
            const std::optional<Tensor> &save_var, bool update, double eps, std::array<bool, 3> mask, const Tensor &reserve)
        {
            return native_batch_norm_backward_fn(grad_out, input, weight, running_mean, running_var, save_mean, save_var,
                                                 update, eps, mask);
        }

        std::tuple<Tensor, Tensor, Tensor> native_group_norm_fn(const Tensor &input, const std::optional<Tensor> &weight,
                                                                const std::optional<Tensor> &bias, c10::SymInt N_,
                                                                c10::SymInt C_, c10::SymInt HxW_, int64_t group, double eps)
        {
            const int64_t N = N_.expect_int(), C = C_.expect_int(), HxW = HxW_.expect_int();
            const auto acc = at::toOpMathType(input.scalar_type());
            Tensor x = input.to(acc).reshape({N, group, (C / group) * HxW});
            auto [var, mean] = at::var_mean(x, {-1}, 0, true);
            Tensor rstd = (var + eps).rsqrt();
            Tensor y = ((x - mean) * rstd).reshape({N, C, HxW});
            if (weight.has_value() && weight->defined())
                y = y * weight->to(acc).view({1, C, 1});
            if (bias.has_value() && bias->defined())
                y = y + bias->to(acc).view({1, C, 1});
            return {y.reshape(input.sizes()).to(input.scalar_type()), mean.view({N, group}), rstd.view({N, group})};
        }

        std::tuple<Tensor, Tensor, Tensor> native_group_norm_backward_fn(
            const Tensor &grad_out, const Tensor &input, const Tensor &mean, const Tensor &rstd,
            const std::optional<Tensor> &weight, c10::SymInt N_, c10::SymInt C_, c10::SymInt HxW_, int64_t group,
            std::array<bool, 3> mask)
        {
            const int64_t N = N_.expect_int(), C = C_.expect_int(), HxW = HxW_.expect_int();
            const auto acc = at::toOpMathType(input.scalar_type());
            Tensor x = input.to(acc).reshape({N, group, C / group, HxW});
            Tensor g = grad_out.to(acc).reshape({N, group, C / group, HxW});
            Tensor m = mean.to(acc).view({N, group, 1, 1}), r = rstd.to(acc).view({N, group, 1, 1});
            Tensor xhat = (x - m) * r;
            const bool has_w = weight.has_value() && weight->defined();
            Tensor gi, gw, gb;
            if (mask[0])
            {
                Tensor w = has_w ? weight->to(acc).view({1, group, C / group, 1}) : at::ones({1, group, C / group, 1}, x.options());
                Tensor gw_ = g * w;
                Tensor mg = gw_.mean({2, 3}, true);
                Tensor mgx = (gw_ * xhat).mean({2, 3}, true);
                gi = (r * (gw_ - mg - xhat * mgx)).reshape(input.sizes()).to(input.scalar_type());
            }
            if (mask[1] && has_w)
                gw = (g * xhat).sum({0, 3}).reshape({C}).to(weight->scalar_type());
            if (mask[2] && has_w)
                gb = g.sum({0, 3}).reshape({C}).to(weight->scalar_type());
            return {gi, gw, gb};
        }

        // ================================================= im2col / col2im
        Tensor im2col_impl(const Tensor &input, Ints k, Ints dil, Ints pad, Ints st)
        {
            const bool batched = input.dim() == 4;
            Tensor x = batched ? input : input.unsqueeze(0);
            const int64_t N = x.size(0), C = x.size(1), H = x.size(2), W = x.size(3);
            const int64_t oh = (H + 2 * pad[0] - dil[0] * (k[0] - 1) - 1) / st[0] + 1;
            const int64_t ow = (W + 2 * pad[1] - dil[1] * (k[1] - 1) - 1) / st[1] + 1;
            TORCH_CHECK(oh > 0 && ow > 0, "im2col: given input size is too small");
            Tensor xp = at::constant_pad_nd(x, {pad[1], pad[1], pad[0], pad[0]}, 0);
            Tensor p = xp.as_strided({N, C, k[0], k[1], oh, ow},
                                     {xp.stride(0), xp.stride(1), xp.stride(2) * dil[0], xp.stride(3) * dil[1],
                                      xp.stride(2) * st[0], xp.stride(3) * st[1]},
                                     xp.storage_offset());
            Tensor cols = p.reshape({N, C * k[0] * k[1], oh * ow});
            return batched ? cols : cols.squeeze(0);
        }
        Tensor im2col_fn(const Tensor &self, at::IntArrayRef k, at::IntArrayRef d, at::IntArrayRef p, at::IntArrayRef s)
        {
            return im2col_impl(self, expand2(k, 2), expand2(d, 2), expand2(p, 2), expand2(s, 2));
        }

        // Flat (padded) input position of every im2col element: [k0*k1, L].
        Tensor col_index_map(int64_t Hp, int64_t Wp, Ints k, Ints dil, Ints st, const at::TensorOptions &opts)
        {
            Tensor map = at::arange(Hp * Wp, opts.dtype(at::kLong)).view({1, 1, Hp, Wp});
            return im2col_impl(map, k, dil, {0, 0}, st).view({k[0] * k[1], -1});
        }

        Tensor col2im_impl(const Tensor &cols_in, Ints out_size, Ints k, Ints dil, Ints pad, Ints st)
        {
            const bool batched = cols_in.dim() == 3;
            Tensor cols = batched ? cols_in : cols_in.unsqueeze(0);
            const int64_t N = cols.size(0), KK = k[0] * k[1];
            const int64_t C = cols.size(1) / KK;
            const int64_t Hp = out_size[0] + 2 * pad[0], Wp = out_size[1] + 2 * pad[1];
            Tensor map = col_index_map(Hp, Wp, k, dil, st, cols.options()); // [KK, L]
            TORCH_CHECK(map.size(1) == cols.size(2), "col2im: expected ", map.size(1), " blocks, got ", cols.size(2));
            Tensor out = at::zeros({N, C, Hp * Wp}, cols.options());
            // Explicit sizes: -1 is ambiguous when N or C is 0.
            Tensor idx = map.reshape({1, 1, KK * map.size(1)}).expand({N, C, KK * map.size(1)});
            out.scatter_add_(2, idx, cols.reshape({N, C, KK * map.size(1)}));
            Tensor r = out.view({N, C, Hp, Wp}).slice(2, pad[0], pad[0] + out_size[0]).slice(3, pad[1], pad[1] + out_size[1]);
            r = r.contiguous();
            return batched ? r : r.squeeze(0);
        }
        Tensor col2im_fn(const Tensor &self, c10::SymIntArrayRef out_size, at::IntArrayRef k, at::IntArrayRef d,
                         at::IntArrayRef p, at::IntArrayRef s)
        {
            return col2im_impl(self, vec(out_size), expand2(k, 2), expand2(d, 2), expand2(p, 2), expand2(s, 2));
        }

        // =========================================================== pooling
        int64_t pool_out(int64_t in, int64_t k, int64_t pad, int64_t st, int64_t dil, bool ceil)
        {
            int64_t num = in + 2 * pad - dil * (k - 1) - 1 + (ceil ? st - 1 : 0);
            int64_t o = (num >= 0 ? num / st : -((-num + st - 1) / st)) + 1;
            if (ceil && (o - 1) * st >= in + pad)
                --o;
            return o;
        }

        struct Pool2d
        {
            Ints k, st, pad, dil;
            int64_t oh, ow, eh, ew; // extra bottom/right padding needed for ceil mode
        };
        Pool2d pool2d_geometry(const Tensor &x, at::IntArrayRef k, at::IntArrayRef st, at::IntArrayRef pad,
                               at::IntArrayRef dil, bool ceil)
        {
            Pool2d g;
            g.k = expand2(k, 2);
            g.st = expand2(st, 2, g.k);
            g.pad = expand2(pad, 2, {0, 0});
            g.dil = expand2(dil, 2, {1, 1});
            const int64_t H = x.size(-2), W = x.size(-1);
            TORCH_CHECK(g.pad[0] <= g.k[0] / 2 && g.pad[1] <= g.k[1] / 2,
                        "pad should be at most half of effective kernel size");
            g.oh = pool_out(H, g.k[0], g.pad[0], g.st[0], g.dil[0], ceil);
            g.ow = pool_out(W, g.k[1], g.pad[1], g.st[1], g.dil[1], ceil);
            TORCH_CHECK(g.oh > 0 && g.ow > 0, "Given input size ", x.sizes(), ", output size is too small");
            g.eh = std::max<int64_t>(0, (g.oh - 1) * g.st[0] + g.dil[0] * (g.k[0] - 1) + 1 - (H + 2 * g.pad[0]));
            g.ew = std::max<int64_t>(0, (g.ow - 1) * g.st[1] + g.dil[1] * (g.k[1] - 1) + 1 - (W + 2 * g.pad[1]));
            return g;
        }

        // [N, C, H, W] padded -> windows [N, C, oh, ow, k0*k1]
        Tensor pool_windows(const Tensor &xp, const Pool2d &g)
        {
            const int64_t N = xp.size(0), C = xp.size(1);
            return xp.as_strided({N, C, g.oh, g.ow, g.k[0], g.k[1]},
                                 {xp.stride(0), xp.stride(1), xp.stride(2) * g.st[0], xp.stride(3) * g.st[1],
                                  xp.stride(2) * g.dil[0], xp.stride(3) * g.dil[1]},
                                 xp.storage_offset())
                .reshape({N, C, g.oh, g.ow, g.k[0] * g.k[1]});
        }

        Tensor as4d(const Tensor &x) { return x.dim() == 3 ? x.unsqueeze(0) : x; }

        std::tuple<Tensor, Tensor> max_pool2d_with_indices_fn(const Tensor &self, at::IntArrayRef k, at::IntArrayRef st,
                                                              at::IntArrayRef pad, at::IntArrayRef dil, bool ceil)
        {
            TORCH_CHECK(self.dim() == 3 || self.dim() == 4, "max_pool2d: expected 3D or 4D input");
            Tensor x = as4d(self);
            auto g = pool2d_geometry(x, k, st, pad, dil, ceil);
            const int64_t W = x.size(3);
            if (at::isFloatingType(x.scalar_type()))
            {
                // Direct kernel (thread per output), ATen's window / NaN rules.
                Tensor vals = at::empty({x.size(0), x.size(1), g.oh, g.ow}, x.options());
                Tensor idx = at::empty({x.size(0), x.size(1), g.oh, g.ow}, x.options().dtype(at::kLong));
                kernels::run_max_pool2d(x, vals, idx, {g.k[0], g.k[1], g.st[0], g.st[1], g.pad[0], g.pad[1], g.dil[0], g.dil[1]});
                if (self.dim() == 3)
                    return {vals.squeeze(0), idx.squeeze(0)};
                return {vals, idx};
            }
            // Pad with the dtype's lowest value so padding never wins the max.
            at::Scalar lowest = -std::numeric_limits<double>::infinity();
            switch (x.scalar_type())
            {
            case at::kLong: lowest = std::numeric_limits<int64_t>::lowest(); break;
            case at::kInt: lowest = std::numeric_limits<int32_t>::lowest(); break;
            case at::kShort: lowest = std::numeric_limits<int16_t>::lowest(); break;
            case at::kChar: lowest = std::numeric_limits<int8_t>::lowest(); break;
            case at::kByte: lowest = 0; break;
            case at::kBool: lowest = false; break;
            default: break;
            }
            Tensor xp = at::constant_pad_nd(x, {g.pad[1], g.pad[1] + g.ew, g.pad[0], g.pad[0] + g.eh}, lowest);
            auto [vals, arg] = pool_windows(xp, g).max(-1);
            // local window index -> flat input index
            auto lopt = x.options().dtype(at::kLong);
            Tensor ky = at::floor_divide(arg, g.k[1]), kx = at::remainder(arg, g.k[1]);
            Tensor oy = at::arange(g.oh, lopt).view({g.oh, 1}), ox = at::arange(g.ow, lopt).view({1, g.ow});
            Tensor iy = oy * g.st[0] - g.pad[0] + ky * g.dil[0];
            Tensor ix = ox * g.st[1] - g.pad[1] + kx * g.dil[1];
            Tensor idx = iy * W + ix;
            if (self.dim() == 3)
                return {vals.squeeze(0), idx.squeeze(0)};
            return {vals, idx};
        }

        Tensor max_pool2d_with_indices_backward_fn(const Tensor &grad, const Tensor &self, at::IntArrayRef, at::IntArrayRef,
                                                   at::IntArrayRef, at::IntArrayRef, bool, const Tensor &indices)
        {
            Tensor x = as4d(self), g = as4d(grad), idx = as4d(indices);
            const int64_t N = x.size(0), C = x.size(1);
            Tensor gi = at::zeros({N, C, x.size(2) * x.size(3)}, grad.options());
            const int64_t L = g.size(2) * g.size(3);
            gi.scatter_add_(2, idx.reshape({N, C, L}), g.reshape({N, C, L}));
            gi = gi.view(x.sizes());
            return self.dim() == 3 ? gi.squeeze(0) : gi;
        }

        // Per-output divisor of avg_pool2d, shape [oh, ow].
        Tensor avg_divisor(const Tensor &x, const Pool2d &g, bool count_include_pad, std::optional<int64_t> divisor_override)
        {
            if (divisor_override.has_value())
                return at::full({g.oh, g.ow}, static_cast<double>(*divisor_override), x.options());
            const int64_t H = x.size(2), W = x.size(3);
            Tensor ones;
            if (count_include_pad)
                ones = at::constant_pad_nd(at::ones({1, 1, H + 2 * g.pad[0], W + 2 * g.pad[1]}, x.options()), {0, g.ew, 0, g.eh}, 0);
            else
                ones = at::constant_pad_nd(at::ones({1, 1, H, W}, x.options()),
                                           {g.pad[1], g.pad[1] + g.ew, g.pad[0], g.pad[0] + g.eh}, 0);
            return pool_windows(ones, g).sum(-1).view({g.oh, g.ow});
        }

        Tensor avg_pool2d_impl(const Tensor &self, at::IntArrayRef k, at::IntArrayRef st, at::IntArrayRef pad, bool ceil,
                               bool count_include_pad, std::optional<int64_t> divisor_override)
        {
            TORCH_CHECK(self.dim() == 3 || self.dim() == 4, "avg_pool2d: expected 3D or 4D input");
            Tensor x = as4d(self);
            auto g = pool2d_geometry(x, k, st, pad, {}, ceil);
            if (at::isFloatingType(x.scalar_type()))
            {
                TORCH_CHECK(!divisor_override.has_value() || *divisor_override != 0, "divisor must be not zero");
                Tensor out = at::empty({x.size(0), x.size(1), g.oh, g.ow}, x.options());
                kernels::run_avg_pool2d(x, out, {g.k[0], g.k[1], g.st[0], g.st[1], g.pad[0], g.pad[1], 1, 1},
                                        count_include_pad, divisor_override);
                return self.dim() == 3 ? out.squeeze(0) : out;
            }
            Tensor xp = at::constant_pad_nd(x, {g.pad[1], g.pad[1] + g.ew, g.pad[0], g.pad[0] + g.eh}, 0);
            Tensor out = pool_windows(xp, g).sum(-1) / avg_divisor(x, g, count_include_pad, divisor_override);
            out = out.to(self.scalar_type());
            return self.dim() == 3 ? out.squeeze(0) : out;
        }
        Tensor avg_pool2d_fn(const Tensor &self, at::IntArrayRef k, at::IntArrayRef st, at::IntArrayRef pad, bool ceil,
                             bool count_include_pad, std::optional<int64_t> divisor_override)
        {
            return avg_pool2d_impl(self, k, st, pad, ceil, count_include_pad, divisor_override);
        }
        Tensor &avg_pool2d_out_fn(const Tensor &self, at::IntArrayRef k, at::IntArrayRef st, at::IntArrayRef pad, bool ceil,
                                  bool count_include_pad, std::optional<int64_t> divisor_override, Tensor &out)
        {
            Tensor r = avg_pool2d_impl(self, k, st, pad, ceil, count_include_pad, divisor_override);
            at::native::resize_output(out, r.sizes());
            out.copy_(r);
            return out;
        }

        Tensor avg_pool2d_backward_fn(const Tensor &grad, const Tensor &self, at::IntArrayRef k, at::IntArrayRef st,
                                      at::IntArrayRef pad, bool ceil, bool count_include_pad,
                                      std::optional<int64_t> divisor_override)
        {
            Tensor x = as4d(self), go = as4d(grad);
            auto g = pool2d_geometry(x, k, st, pad, {}, ceil);
            const int64_t N = x.size(0), C = x.size(1), KK = g.k[0] * g.k[1];
            Tensor scaled = go / avg_divisor(x, g, count_include_pad, divisor_override);          // [N, C, oh, ow]
            Tensor cols = scaled.reshape({N, C, 1, g.oh * g.ow}).expand({N, C, KK, g.oh * g.ow}); // every tap
            const int64_t Hp = x.size(2) + 2 * g.pad[0] + g.eh, Wp = x.size(3) + 2 * g.pad[1] + g.ew;
            Tensor map = col_index_map(Hp, Wp, g.k, g.dil, g.st, x.options()); // [KK, L]
            Tensor out = at::zeros({N, C, Hp * Wp}, go.options());
            const int64_t L = KK * g.oh * g.ow;
            out.scatter_add_(2, map.reshape({1, 1, L}).expand({N, C, L}), cols.reshape({N, C, L}));
            Tensor gi = out.view({N, C, Hp, Wp})
                            .slice(2, g.pad[0], g.pad[0] + x.size(2))
                            .slice(3, g.pad[1], g.pad[1] + x.size(3))
                            .contiguous();
            return self.dim() == 3 ? gi.squeeze(0) : gi;
        }

        // ==================================================== resampling
        // Dense [out, in] matrices on the host, uploaded as float.
        Tensor host_matrix(const std::vector<double> &m, int64_t rows, int64_t cols, const at::TensorOptions &opts)
        {
            Tensor t = at::empty({rows, cols}, at::TensorOptions().dtype(at::kDouble));
            std::copy(m.begin(), m.end(), t.data_ptr<double>());
            return t.to(opts.device(), at::toOpMathType(opts.dtype().toScalarType()));
        }

        Tensor adaptive_avg_matrix(int64_t in, int64_t out, const at::TensorOptions &opts)
        {
            std::vector<double> m(out * in, 0.0);
            for (int64_t o = 0; o < out; ++o)
            {
                int64_t s = (o * in) / out, e = ((o + 1) * in + out - 1) / out;
                for (int64_t i = s; i < e; ++i)
                    m[o * in + i] = 1.0 / static_cast<double>(e - s);
            }
            return host_matrix(m, out, in, opts);
        }

        double scale_value(std::optional<double> scale, int64_t in, int64_t out)
        {
            return scale.has_value() && *scale > 0 ? 1.0 / *scale : static_cast<double>(in) / static_cast<double>(out);
        }

        // Linear interpolation matrix, matching area_pixel_compute_source_index.
        Tensor linear_matrix(int64_t in, int64_t out, bool align_corners, std::optional<double> scale,
                             const at::TensorOptions &opts)
        {
            std::vector<double> m(out * in, 0.0);
            double sc = align_corners ? (out > 1 ? static_cast<double>(in - 1) / static_cast<double>(out - 1) : 0.0)
                                      : scale_value(scale, in, out);
            // CPU computes these in float for float inputs.
            float scf = static_cast<float>(sc);
            for (int64_t o = 0; o < out; ++o)
            {
                float src = align_corners ? scf * o : std::max(scf * (o + 0.5f) - 0.5f, 0.0f);
                int64_t i0 = std::min<int64_t>(static_cast<int64_t>(src), in - 1);
                int64_t i1 = std::min<int64_t>(i0 + 1, in - 1);
                float l1 = std::min(std::max(src - static_cast<float>(i0), 0.0f), 1.0f);
                m[o * in + i0] += 1.0 - l1;
                m[o * in + i1] += l1;
            }
            return host_matrix(m, out, in, opts);
        }

        Tensor cubic_matrix(int64_t in, int64_t out, bool align_corners, std::optional<double> scale,
                            const at::TensorOptions &opts)
        {
            std::vector<double> m(out * in, 0.0);
            double sc = align_corners ? (out > 1 ? static_cast<double>(in - 1) / static_cast<double>(out - 1) : 0.0)
                                      : scale_value(scale, in, out);
            const double A = -0.75;
            auto c1 = [&](double x) { return ((A + 2) * x - (A + 3)) * x * x + 1; };
            auto c2 = [&](double x) { return ((A * x - 5 * A) * x + 8 * A) * x - 4 * A; };
            for (int64_t o = 0; o < out; ++o)
            {
                double src = align_corners ? sc * o : sc * (o + 0.5) - 0.5;
                int64_t i = static_cast<int64_t>(std::floor(src));
                double t = src - static_cast<double>(i);
                double w[4] = {c2(t + 1), c1(t), c1(1 - t), c2(2 - t)};
                for (int j = 0; j < 4; ++j)
                {
                    int64_t ii = std::min<int64_t>(std::max<int64_t>(i - 1 + j, 0), in - 1);
                    m[o * in + ii] += w[j];
                }
            }
            return host_matrix(m, out, in, opts);
        }

        // Anti-aliased resampling (CPU's _compute_weights_aa): a normalized
        // linear / cubic (a = -0.5) filter whose support widens by the
        // downsampling factor.
        Tensor aa_matrix(int64_t in, int64_t out, bool align_corners, std::optional<double> scale, bool cubic,
                         const at::TensorOptions &opts)
        {
            std::vector<double> m(out * in, 0.0);
            const double sc = align_corners ? (out > 1 ? static_cast<double>(in - 1) / static_cast<double>(out - 1) : 0.0)
                                            : scale_value(scale, in, out);
            const double interp = cubic ? 4.0 : 2.0;
            const double support = sc >= 1.0 ? interp * 0.5 * sc : interp * 0.5;
            const double invscale = sc >= 1.0 ? 1.0 / sc : 1.0;
            auto filter = [&](double x)
            {
                x = std::abs(x);
                if (!cubic)
                    return x < 1.0 ? 1.0 - x : 0.0;
                const double a = -0.5;
                if (x < 1.0)
                    return ((a + 2) * x - (a + 3)) * x * x + 1;
                if (x < 2.0)
                    return (((x - 5) * x + 8) * x - 4) * a;
                return 0.0;
            };
            for (int64_t o = 0; o < out; ++o)
            {
                const double center = sc * (o + 0.5);
                const int64_t xmin = std::max<int64_t>(static_cast<int64_t>(center - support + 0.5), 0);
                const int64_t xsize = std::min<int64_t>(static_cast<int64_t>(center + support + 0.5), in) - xmin;
                double total = 0.0;
                for (int64_t j = 0; j < xsize; ++j)
                {
                    const double w = filter((j + xmin - center + 0.5) * invscale);
                    m[o * in + xmin + j] = w;
                    total += w;
                }
                if (total != 0.0)
                    for (int64_t j = 0; j < xsize; ++j)
                        m[o * in + xmin + j] /= total;
            }
            return host_matrix(m, out, in, opts);
        }
        Tensor aa_linear_matrix(int64_t in, int64_t out, bool ac, std::optional<double> scale, const at::TensorOptions &o)
        {
            return aa_matrix(in, out, ac, scale, false, o);
        }
        Tensor aa_cubic_matrix(int64_t in, int64_t out, bool ac, std::optional<double> scale, const at::TensorOptions &o)
        {
            return aa_matrix(in, out, ac, scale, true, o);
        }

        Tensor nearest_index(int64_t in, int64_t out, std::optional<double> scale, bool exact, const at::TensorOptions &opts)
        {
            Tensor t = at::empty({out}, at::TensorOptions().dtype(at::kLong));
            auto *p = t.data_ptr<int64_t>();
            float sc = static_cast<float>(scale_value(scale, in, out));
            for (int64_t o = 0; o < out; ++o)
            {
                int64_t src = exact ? static_cast<int64_t>(std::floor((o + 0.5f) * sc)) : static_cast<int64_t>(std::floor(o * sc));
                p[o] = std::min<int64_t>(src, in - 1);
            }
            return t.to(opts.device());
        }

        // Applies A (per spatial dim) to the trailing dims of x: dims [2, 2+k).
        Tensor apply_separable(const Tensor &x, const std::vector<Tensor> &mats)
        {
            const auto acc = at::toOpMathType(x.scalar_type());
            Tensor y = x.to(acc);
            const int64_t nd = static_cast<int64_t>(mats.size());
            for (int64_t d = 0; d < nd; ++d)
            {
                int64_t dim = y.dim() - nd + d;
                // move dim last, multiply by A^T, move back
                y = at::matmul(y.transpose(dim, -1), mats[d].t()).transpose(dim, -1);
            }
            return y.to(x.scalar_type()).contiguous();
        }
        Tensor apply_separable_t(const Tensor &g, const std::vector<Tensor> &mats)
        {
            std::vector<Tensor> t;
            for (auto &m : mats)
                t.push_back(m.t());
            return apply_separable(g, t);
        }

        Tensor adaptive_avg_pool_nd(const Tensor &self, Ints out)
        {
            const int64_t nd = static_cast<int64_t>(out.size());
            std::vector<Tensor> mats;
            for (int64_t d = 0; d < nd; ++d)
                mats.push_back(adaptive_avg_matrix(self.size(self.dim() - nd + d), out[d], self.options()));
            return apply_separable(self, mats);
        }
        Tensor adaptive_avg_pool_nd_backward(const Tensor &grad, const Tensor &self, int64_t nd)
        {
            std::vector<Tensor> mats;
            for (int64_t d = 0; d < nd; ++d)
                mats.push_back(adaptive_avg_matrix(self.size(self.dim() - nd + d), grad.size(grad.dim() - nd + d), self.options()));
            return apply_separable_t(grad, mats);
        }
        Tensor adaptive_avg_pool2d_fn(const Tensor &self, c10::SymIntArrayRef out)
        {
            return adaptive_avg_pool_nd(self, vec(out));
        }
        Tensor &adaptive_avg_pool2d_out_fn(const Tensor &self, c10::SymIntArrayRef out_size, Tensor &out)
        {
            Tensor r = adaptive_avg_pool_nd(self, vec(out_size));
            at::native::resize_output(out, r.sizes());
            return out.copy_(r);
        }
        Tensor adaptive_avg_pool2d_backward_fn(const Tensor &grad, const Tensor &self)
        {
            return adaptive_avg_pool_nd_backward(grad, self, 2);
        }
        Tensor adaptive_avg_pool3d_fn(const Tensor &self, c10::SymIntArrayRef out)
        {
            return adaptive_avg_pool_nd(self, vec(out));
        }
        Tensor adaptive_avg_pool3d_backward_fn(const Tensor &grad, const Tensor &self)
        {
            return adaptive_avg_pool_nd_backward(grad, self, 3);
        }

        // ---------------------------------------------------- upsampling
        Ints out_shape_of(const Tensor &x, const Ints &spatial)
        {
            Ints s(x.sizes().begin(), x.sizes().end() - spatial.size());
            s.insert(s.end(), spatial.begin(), spatial.end());
            return s;
        }

#define WG_UPSAMPLE_LINEAR(NAME, ND, MATRIX, SCALE_PARAMS, SCALE_LIST)                                              \
    Tensor NAME##_fn(const Tensor &self, c10::SymIntArrayRef out_size, bool align_corners SCALE_PARAMS)             \
    {                                                                                                               \
        Ints os = vec(out_size);                                                                                    \
        std::vector<std::optional<double>> scale_list{SCALE_LIST};                                                      \
        std::vector<Tensor> mats;                                                                                   \
        for (int d = 0; d < ND; ++d)                                                                                \
            mats.push_back(MATRIX(self.size(self.dim() - ND + d), os[d], align_corners, scale_list[d], self.options())); \
        return apply_separable(self, mats);                                                                         \
    }                                                                                                               \
    Tensor &NAME##_out_fn(const Tensor &self, c10::SymIntArrayRef out_size, bool align_corners SCALE_PARAMS,         \
                          Tensor &out)                                                                              \
    {                                                                                                               \
        Ints os = vec(out_size);                                                                                    \
        std::vector<std::optional<double>> scale_list{SCALE_LIST};                                                      \
        std::vector<Tensor> mats;                                                                                   \
        for (int d = 0; d < ND; ++d)                                                                                \
            mats.push_back(MATRIX(self.size(self.dim() - ND + d), os[d], align_corners, scale_list[d], self.options())); \
        Tensor r = apply_separable(self, mats);                                                                     \
        at::native::resize_output(out, r.sizes());                                                                  \
        return out.copy_(r);                                                                                        \
    }                                                                                                               \
    Tensor NAME##_backward_fn(const Tensor &grad, c10::SymIntArrayRef out_size, c10::SymIntArrayRef in_size,        \
                              bool align_corners SCALE_PARAMS)                                                      \
    {                                                                                                               \
        Ints os = vec(out_size), is = vec(in_size);                                                                 \
        std::vector<std::optional<double>> scale_list{SCALE_LIST};                                                      \
        std::vector<Tensor> mats;                                                                                   \
        for (int d = 0; d < ND; ++d)                                                                                \
            mats.push_back(MATRIX(is[is.size() - ND + d], os[d], align_corners, scale_list[d], grad.options()));        \
        return apply_separable_t(grad, mats);                                                                       \
    }                                                                                                               \
    Tensor &NAME##_backward_out_fn(const Tensor &grad, c10::SymIntArrayRef out_size, c10::SymIntArrayRef in_size,   \
                                   bool align_corners SCALE_PARAMS, Tensor &gi)                                     \
    {                                                                                                               \
        Ints os = vec(out_size), is = vec(in_size);                                                                 \
        std::vector<std::optional<double>> scale_list{SCALE_LIST};                                                      \
        std::vector<Tensor> mats;                                                                                   \
        for (int d = 0; d < ND; ++d)                                                                                \
            mats.push_back(MATRIX(is[is.size() - ND + d], os[d], align_corners, scale_list[d], grad.options()));        \
        Tensor r = apply_separable_t(grad, mats);                                                                   \
        at::native::resize_output(gi, r.sizes());                                                                   \
        return gi.copy_(r);                                                                                         \
    }
#define COMMA ,
#define SC1 , std::optional<double> scales
#define SC2 , std::optional<double> scales_h, std::optional<double> scales_w
        WG_UPSAMPLE_LINEAR(upsample_linear1d, 1, linear_matrix, SC1, scales)
        WG_UPSAMPLE_LINEAR(upsample_bilinear2d, 2, linear_matrix, SC2, scales_h COMMA scales_w)
        WG_UPSAMPLE_LINEAR(upsample_bicubic2d, 2, cubic_matrix, SC2, scales_h COMMA scales_w)
#define SC3 , std::optional<double> scales_d, std::optional<double> scales_h, std::optional<double> scales_w
        WG_UPSAMPLE_LINEAR(upsample_trilinear3d, 3, linear_matrix, SC3, scales_d COMMA scales_h COMMA scales_w)
        WG_UPSAMPLE_LINEAR(_upsample_bilinear2d_aa, 2, aa_linear_matrix, SC2, scales_h COMMA scales_w)
        WG_UPSAMPLE_LINEAR(_upsample_bicubic2d_aa, 2, aa_cubic_matrix, SC2, scales_h COMMA scales_w)

        Tensor upsample_nearest(const Tensor &self, const Ints &os, const std::vector<std::optional<double>> &scales, bool exact)
        {
            const int64_t nd = static_cast<int64_t>(os.size());
            Tensor y = self;
            for (int64_t d = 0; d < nd; ++d)
            {
                int64_t dim = self.dim() - nd + d;
                y = y.index_select(dim, nearest_index(self.size(dim), os[d], scales[d], exact, self.options()));
            }
            return y;
        }
        Tensor upsample_nearest_backward(const Tensor &grad, const Ints &os, const Ints &is,
                                         const std::vector<std::optional<double>> &scales, bool exact)
        {
            const int64_t nd = static_cast<int64_t>(os.size());
            Tensor g = grad;
            for (int64_t d = nd - 1; d >= 0; --d)
            {
                int64_t dim = grad.dim() - nd + d;
                Ints shape = g.sizes().vec();
                shape[dim] = is[is.size() - nd + d];
                Tensor r = at::zeros(shape, grad.options());
                r.index_add_(dim, nearest_index(shape[dim], os[d], scales[d], exact, grad.options()), g);
                g = r;
            }
            return g;
        }

#define WG_UPSAMPLE_NEAREST(NAME, ND, EXACT, SCALE_PARAMS, SCALE_LIST)                                             \
    Tensor NAME##_fn(const Tensor &self, c10::SymIntArrayRef out_size SCALE_PARAMS)                                \
    {                                                                                                              \
        return upsample_nearest(self, vec(out_size), {SCALE_LIST}, EXACT);                                         \
    }                                                                                                              \
    Tensor &NAME##_out_fn(const Tensor &self, c10::SymIntArrayRef out_size SCALE_PARAMS, Tensor &out)              \
    {                                                                                                              \
        Tensor r = upsample_nearest(self, vec(out_size), {SCALE_LIST}, EXACT);                                     \
        at::native::resize_output(out, r.sizes());                                                                 \
        return out.copy_(r);                                                                                       \
    }                                                                                                              \
    Tensor NAME##_backward_fn(const Tensor &grad, c10::SymIntArrayRef out_size, c10::SymIntArrayRef in_size SCALE_PARAMS) \
    {                                                                                                              \
        return upsample_nearest_backward(grad, vec(out_size), vec(in_size), {SCALE_LIST}, EXACT);                  \
    }                                                                                                              \
    Tensor &NAME##_backward_out_fn(const Tensor &grad, c10::SymIntArrayRef out_size, c10::SymIntArrayRef in_size SCALE_PARAMS, \
                                   Tensor &gi)                                                                     \
    {                                                                                                              \
        Tensor r = upsample_nearest_backward(grad, vec(out_size), vec(in_size), {SCALE_LIST}, EXACT);              \
        at::native::resize_output(gi, r.sizes());                                                                  \
        return gi.copy_(r);                                                                                        \
    }
        WG_UPSAMPLE_NEAREST(upsample_nearest1d, 1, false, SC1, scales)
        WG_UPSAMPLE_NEAREST(upsample_nearest2d, 2, false, SC2, scales_h COMMA scales_w)
        WG_UPSAMPLE_NEAREST(upsample_nearest_exact1d, 1, true, SC1, scales)
        WG_UPSAMPLE_NEAREST(upsample_nearest_exact2d, 2, true, SC2, scales_h COMMA scales_w)
        WG_UPSAMPLE_NEAREST(upsample_nearest3d, 3, false, SC3, scales_d COMMA scales_h COMMA scales_w)
        WG_UPSAMPLE_NEAREST(_upsample_nearest_exact3d, 3, true, SC3, scales_d COMMA scales_h COMMA scales_w)

        // ============================================================ losses
        // Returns (picked log-prob per sample, weight per sample, safe target).
        std::tuple<Tensor, Tensor> nll_terms(const Tensor &self2, const Tensor &target1, const std::optional<Tensor> &weight,
                                             int64_t ignore_index, Tensor &safe)
        {
            Tensor valid = target1 != ignore_index;
            safe = at::where(valid, target1, at::zeros({}, target1.options()));
            Tensor picked = self2.gather(1, safe.unsqueeze(1)).squeeze(1);
            Tensor w = (weight.has_value() && weight->defined()) ? weight->index_select(0, safe)
                                                                 : at::ones({target1.size(0)}, self2.options());
            w = w * valid;
            return {picked, w};
        }

        std::tuple<Tensor, Tensor> nll_loss_forward_fn(const Tensor &self, const Tensor &target,
                                                       const std::optional<Tensor> &weight, int64_t reduction,
                                                       c10::SymInt ignore_index_s)
        {
            const int64_t ignore_index = ignore_index_s.expect_int();
            TORCH_CHECK(self.dim() >= 1 && self.dim() <= 2, "nll_loss: input tensor should be 1D or 2D");
            Tensor s2 = self.dim() == 1 ? self.unsqueeze(0) : self;
            Tensor t1 = target.dim() == 0 ? target.unsqueeze(0) : target;
            Tensor safe;
            auto [picked, w] = nll_terms(s2, t1, weight, ignore_index, safe);
            Tensor loss = -picked * w;
            Tensor total = w.sum();
            if (reduction == at::Reduction::None)
                return {self.dim() == 1 ? loss.squeeze(0) : loss, at::zeros({}, self.options())};
            if (reduction == at::Reduction::Sum)
                return {loss.sum(), total};
            return {loss.sum() / total, total};
        }

        Tensor nll_loss_backward_fn(const Tensor &grad, const Tensor &self, const Tensor &target,
                                    const std::optional<Tensor> &weight, int64_t reduction, c10::SymInt ignore_index_s,
                                    const Tensor &total_weight)
        {
            const int64_t ignore_index = ignore_index_s.expect_int();
            Tensor s2 = self.dim() == 1 ? self.unsqueeze(0) : self;
            Tensor t1 = target.dim() == 0 ? target.unsqueeze(0) : target;
            Tensor valid = t1 != ignore_index;
            Tensor safe = at::where(valid, t1, at::zeros({}, t1.options()));
            Tensor w = (weight.has_value() && weight->defined()) ? weight->index_select(0, safe)
                                                                 : at::ones({t1.size(0)}, s2.options());
            w = w * valid;
            Tensor g = reduction == at::Reduction::None ? grad.reshape({-1}) : grad;
            if (reduction == at::Reduction::Mean)
                g = g / total_weight;
            Tensor vals = -(w * g);
            Tensor gi = at::zeros_like(s2);
            gi.scatter_(1, safe.unsqueeze(1), vals.unsqueeze(1).to(gi.scalar_type()));
            return self.dim() == 1 ? gi.squeeze(0) : gi;
        }

        // nll_loss2d: [N, C, d...] -> [N*d..., C]
        std::tuple<Tensor, Tensor> nll_loss2d_forward_fn(const Tensor &self, const Tensor &target,
                                                         const std::optional<Tensor> &weight, int64_t reduction,
                                                         c10::SymInt ignore_index)
        {
            const int64_t C = self.size(1);
            Tensor s2 = self.movedim(1, -1).reshape({-1, C});
            Tensor t1 = target.reshape({-1});
            auto [loss, total] = nll_loss_forward_fn(s2, t1, weight, reduction, ignore_index);
            if (reduction == at::Reduction::None)
                loss = loss.view(target.sizes());
            return {loss, total};
        }
        Tensor nll_loss2d_backward_fn(const Tensor &grad, const Tensor &self, const Tensor &target,
                                      const std::optional<Tensor> &weight, int64_t reduction, c10::SymInt ignore_index,
                                      const Tensor &total_weight)
        {
            const int64_t C = self.size(1);
            Tensor s2 = self.movedim(1, -1).reshape({-1, C});
            Tensor gi = nll_loss_backward_fn(grad, s2, target.reshape({-1}), weight, reduction, ignore_index, total_weight);
            Ints moved = self.movedim(1, -1).sizes().vec();
            return gi.view(moved).movedim(-1, 1).contiguous();
        }

        Tensor mse_loss_fn(const Tensor &self, const Tensor &target, int64_t reduction)
        {
            Tensor d = self - target;
            return reduce_loss(d * d, reduction);
        }
        Tensor &mse_loss_out_fn(const Tensor &self, const Tensor &target, int64_t reduction, Tensor &out)
        {
            Tensor r = mse_loss_fn(self, target, reduction);
            at::native::resize_output(out, r.sizes());
            return out.copy_(r);
        }
        Tensor mse_loss_backward_fn(const Tensor &grad, const Tensor &self, const Tensor &target, int64_t reduction)
        {
            return (self - target) * grad * (2.0 * loss_norm(self, reduction));
        }
        Tensor &mse_loss_backward_out_fn(const Tensor &grad, const Tensor &self, const Tensor &target, int64_t reduction,
                                         Tensor &gi)
        {
            Tensor r = mse_loss_backward_fn(grad, self, target, reduction);
            at::native::resize_output(gi, r.sizes());
            return gi.copy_(r);
        }

        Tensor smooth_l1_loss_fn(const Tensor &self, const Tensor &target, int64_t reduction, double beta)
        {
            Tensor d = (self - target).abs();
            Tensor l = beta == 0 ? d : at::where(d < beta, d * d * (0.5 / beta), d - 0.5 * beta);
            return reduce_loss(l, reduction);
        }
        Tensor &smooth_l1_loss_out_fn(const Tensor &self, const Tensor &target, int64_t reduction, double beta, Tensor &out)
        {
            Tensor r = smooth_l1_loss_fn(self, target, reduction, beta);
            at::native::resize_output(out, r.sizes());
            return out.copy_(r);
        }
        Tensor &smooth_l1_loss_backward_out_fn(const Tensor &grad, const Tensor &self, const Tensor &target,
                                               int64_t reduction, double beta, Tensor &gi)
        {
            Tensor x = self - target;
            Tensor g = beta == 0 ? x.sign() : at::where(x.abs() < beta, x / beta, x.sign());
            Tensor r = g * grad * loss_norm(self, reduction);
            at::native::resize_output(gi, r.sizes());
            return gi.copy_(r);
        }

        Tensor huber_loss_fn(const Tensor &self, const Tensor &target, int64_t reduction, double delta)
        {
            Tensor d = (self - target).abs();
            Tensor l = at::where(d < delta, d * d * 0.5, (d - 0.5 * delta) * delta);
            return reduce_loss(l, reduction);
        }
        Tensor &huber_loss_out_fn(const Tensor &self, const Tensor &target, int64_t reduction, double delta, Tensor &out)
        {
            Tensor r = huber_loss_fn(self, target, reduction, delta);
            at::native::resize_output(out, r.sizes());
            return out.copy_(r);
        }
        Tensor &huber_loss_backward_out_fn(const Tensor &grad, const Tensor &self, const Tensor &target, int64_t reduction,
                                           double delta, Tensor &gi)
        {
            Tensor r = (self - target).clamp(-delta, delta) * grad * loss_norm(self, reduction);
            at::native::resize_output(gi, r.sizes());
            return gi.copy_(r);
        }

        Tensor binary_cross_entropy_fn(const Tensor &self, const Tensor &target, const std::optional<Tensor> &weight,
                                       int64_t reduction)
        {
            Tensor l = -(target * self.log().clamp_min(-100) + (1 - target) * (1 - self).log().clamp_min(-100));
            if (weight.has_value() && weight->defined())
                l = l * *weight;
            return reduce_loss(l, reduction);
        }
        Tensor &binary_cross_entropy_out_fn(const Tensor &self, const Tensor &target, const std::optional<Tensor> &weight,
                                            int64_t reduction, Tensor &out)
        {
            Tensor r = binary_cross_entropy_fn(self, target, weight, reduction);
            at::native::resize_output(out, r.sizes());
            return out.copy_(r);
        }
        Tensor binary_cross_entropy_backward_fn(const Tensor &grad, const Tensor &self, const Tensor &target,
                                                const std::optional<Tensor> &weight, int64_t reduction)
        {
            Tensor g = grad * (self - target) / ((1 - self) * self).clamp_min(1e-12) * loss_norm(self, reduction);
            if (weight.has_value() && weight->defined())
                g = g * *weight;
            return g;
        }

        std::tuple<Tensor, Tensor> log_sigmoid_forward_fn(const Tensor &self)
        {
            // log(sigmoid(x)) = min(x, 0) - log1p(exp(-|x|))
            Tensor out = self.clamp_max(0) - (-self.abs()).exp().log1p();
            return {out, at::empty({0}, self.options())};
        }
        Tensor log_sigmoid_backward_fn(const Tensor &grad, const Tensor &self, const Tensor &)
        {
            return grad * (-self).sigmoid();
        }

        Tensor glu_fn(const Tensor &self, int64_t dim)
        {
            TORCH_CHECK(self.dim() > 0, "glu does not support scalars because halving size must be even");
            dim = at::maybe_wrap_dim(dim, self.dim());
            TORCH_CHECK(self.size(dim) % 2 == 0, "Halving dimension must be even, but dimension ", dim, " is size ",
                        self.size(dim));
            auto parts = self.chunk(2, dim);
            return parts[0] * parts[1].sigmoid();
        }
        Tensor &glu_out_fn(const Tensor &self, int64_t dim, Tensor &out)
        {
            Tensor r = glu_fn(self, dim);
            at::native::resize_output(out, r.sizes());
            return out.copy_(r);
        }
        Tensor glu_backward_fn(const Tensor &grad, const Tensor &self, int64_t dim)
        {
            dim = at::maybe_wrap_dim(dim, self.dim());
            auto parts = self.chunk(2, dim);
            Tensor s = parts[1].sigmoid();
            return at::cat({grad * s, grad * parts[0] * s * (1 - s)}, dim);
        }

        Tensor prelu_kernel_fn(const Tensor &self, const Tensor &weight)
        {
            return at::where(self > 0, self, weight * self);
        }
        std::tuple<Tensor, Tensor> prelu_kernel_backward_fn(const Tensor &grad, const Tensor &self, const Tensor &weight)
        {
            Tensor pos = self > 0;
            Tensor gi = at::where(pos, grad, weight * grad);
            Tensor gw = at::where(pos, at::zeros({}, grad.options()), self * grad);
            return {gi, at::sum_to(gw, weight.sizes())};
        }

        Tensor logit_fn(const Tensor &self, std::optional<double> eps)
        {
            // Same order as the CPU kernel: x < lo ? lo : (x > hi ? hi : x)
            // (differs from clamp when eps > 0.5).
            Tensor x = self;
            if (eps.has_value())
            {
                double lo = *eps, hi = 1 - *eps;
                x = at::where(x < lo, at::scalar_tensor(lo, x.options()), at::where(x > hi, at::scalar_tensor(hi, x.options()), x));
            }
            return (x / (1 - x)).log();
        }
    }

#undef COMMA
    namespace
    {
        // Direct 2-d upsampling for floating inputs ([N, C, H, W] or [C, H, W]);
        // scales computed in float like ATen's compute_scales_value /
        // area_pixel_compute_scale.
        bool direct_upsample_ok(const Tensor &self) { return at::isFloatingType(self.scalar_type()) && (self.dim() == 4 || self.dim() == 3); }
        float nearest_scale(int64_t in, int64_t out, std::optional<double> scale)
        {
            return scale.has_value() && *scale > 0 ? static_cast<float>(1.0 / *scale)
                                                   : static_cast<float>(in) / static_cast<float>(out);
        }
        float linear_scale(int64_t in, int64_t out, bool align_corners, std::optional<double> scale)
        {
            if (align_corners)
                return out > 1 ? static_cast<float>(in - 1) / static_cast<float>(out - 1) : 0.f;
            return nearest_scale(in, out, scale);
        }
        Tensor upsample2d_direct(const Tensor &self, at::IntArrayRef os, kernels::Upsample mode, bool align_corners,
                                 std::optional<double> sh, std::optional<double> sw)
        {
            Tensor x = self.dim() == 3 ? self.unsqueeze(0) : self;
            TORCH_CHECK(os.size() == 2 && os[0] > 0 && os[1] > 0, "upsample: invalid output size ", os);
            TORCH_CHECK(x.size(2) > 0 && x.size(3) > 0, "upsample: input spatial size must be > 0, got ", x.sizes());
            Tensor out = at::empty({x.size(0), x.size(1), os[0], os[1]}, x.options());
            const int64_t H = x.size(2), W = x.size(3);
            float fh, fw;
            if (mode == kernels::Upsample::Bilinear)
            {
                fh = linear_scale(H, os[0], align_corners, sh);
                fw = linear_scale(W, os[1], align_corners, sw);
            }
            else
            {
                fh = nearest_scale(H, os[0], sh);
                fw = nearest_scale(W, os[1], sw);
            }
            kernels::run_upsample2d(x, out, mode, align_corners, fh, fw);
            return self.dim() == 3 ? out.squeeze(0) : out;
        }
        Tensor upsample_bilinear2d_direct_fn(const Tensor &self, c10::SymIntArrayRef out_size, bool align_corners,
                                             std::optional<double> sh, std::optional<double> sw)
        {
            if (!direct_upsample_ok(self))
                return upsample_bilinear2d_fn(self, out_size, align_corners, sh, sw);
            return upsample2d_direct(self, vec(out_size), kernels::Upsample::Bilinear, align_corners, sh, sw);
        }
        Tensor &upsample_bilinear2d_direct_out_fn(const Tensor &self, c10::SymIntArrayRef out_size, bool align_corners,
                                                  std::optional<double> sh, std::optional<double> sw, Tensor &out)
        {
            if (!direct_upsample_ok(self))
                return upsample_bilinear2d_out_fn(self, out_size, align_corners, sh, sw, out);
            Tensor r = upsample2d_direct(self, vec(out_size), kernels::Upsample::Bilinear, align_corners, sh, sw);
            at::native::resize_output(out, r.sizes());
            return out.copy_(r);
        }
#define WG_NEAREST_DIRECT(NAME, MODE)                                                                              \
    Tensor NAME##_direct_fn(const Tensor &self, c10::SymIntArrayRef out_size, std::optional<double> sh,             \
                            std::optional<double> sw)                                                              \
    {                                                                                                              \
        if (!direct_upsample_ok(self))                                                                             \
            return NAME##_fn(self, out_size, sh, sw);                                                              \
        return upsample2d_direct(self, vec(out_size), MODE, false, sh, sw);                                        \
    }                                                                                                              \
    Tensor &NAME##_direct_out_fn(const Tensor &self, c10::SymIntArrayRef out_size, std::optional<double> sh,        \
                                 std::optional<double> sw, Tensor &out)                                            \
    {                                                                                                              \
        if (!direct_upsample_ok(self))                                                                             \
            return NAME##_out_fn(self, out_size, sh, sw, out);                                                     \
        Tensor r = upsample2d_direct(self, vec(out_size), MODE, false, sh, sw);                                    \
        at::native::resize_output(out, r.sizes());                                                                 \
        return out.copy_(r);                                                                                       \
    }
        WG_NEAREST_DIRECT(upsample_nearest2d, kernels::Upsample::Nearest)
        WG_NEAREST_DIRECT(upsample_nearest_exact2d, kernels::Upsample::NearestExact)
#undef WG_NEAREST_DIRECT
    }

    TORCH_LIBRARY_IMPL(aten, PrivateUse1, m)
    {
        m.impl("native_batch_norm", TORCH_FN(native_batch_norm_fn));
        m.impl("_native_batch_norm_legit", TORCH_FN(bn_legit_fn));
        m.impl("_native_batch_norm_legit.no_stats", TORCH_FN(bn_legit_no_stats_fn));
        m.impl("_batch_norm_with_update", TORCH_FN(bn_with_update_fn));
        m.impl("native_batch_norm_backward", TORCH_FN(native_batch_norm_backward_fn));
        m.impl("batch_norm_backward", TORCH_FN(batch_norm_backward_fn));
        m.impl("native_group_norm", TORCH_FN(native_group_norm_fn));
        m.impl("native_group_norm_backward", TORCH_FN(native_group_norm_backward_fn));

        m.impl("im2col", TORCH_FN(im2col_fn));
        m.impl("col2im", TORCH_FN(col2im_fn));
        m.impl("max_pool2d_with_indices", TORCH_FN(max_pool2d_with_indices_fn));
        m.impl("max_pool2d_with_indices_backward", TORCH_FN(max_pool2d_with_indices_backward_fn));
        m.impl("avg_pool2d", TORCH_FN(avg_pool2d_fn));
        m.impl("avg_pool2d.out", TORCH_FN(avg_pool2d_out_fn));
        m.impl("avg_pool2d_backward", TORCH_FN(avg_pool2d_backward_fn));
        m.impl("_adaptive_avg_pool2d", TORCH_FN(adaptive_avg_pool2d_fn));
        m.impl("adaptive_avg_pool2d.out", TORCH_FN(adaptive_avg_pool2d_out_fn));
        m.impl("_adaptive_avg_pool2d_backward", TORCH_FN(adaptive_avg_pool2d_backward_fn));
        m.impl("_adaptive_avg_pool3d", TORCH_FN(adaptive_avg_pool3d_fn));
        m.impl("_adaptive_avg_pool3d_backward", TORCH_FN(adaptive_avg_pool3d_backward_fn));

        m.impl("upsample_linear1d", TORCH_FN(upsample_linear1d_fn));
        m.impl("upsample_linear1d.out", TORCH_FN(upsample_linear1d_out_fn));
        m.impl("upsample_linear1d_backward", TORCH_FN(upsample_linear1d_backward_fn));
        m.impl("upsample_linear1d_backward.grad_input", TORCH_FN(upsample_linear1d_backward_out_fn));
        m.impl("upsample_bilinear2d", TORCH_FN(upsample_bilinear2d_direct_fn));
        m.impl("upsample_bilinear2d.out", TORCH_FN(upsample_bilinear2d_direct_out_fn));
        m.impl("upsample_bilinear2d_backward", TORCH_FN(upsample_bilinear2d_backward_fn));
        m.impl("upsample_bilinear2d_backward.grad_input", TORCH_FN(upsample_bilinear2d_backward_out_fn));
        m.impl("upsample_bicubic2d", TORCH_FN(upsample_bicubic2d_fn));
        m.impl("upsample_bicubic2d.out", TORCH_FN(upsample_bicubic2d_out_fn));
        m.impl("upsample_bicubic2d_backward", TORCH_FN(upsample_bicubic2d_backward_fn));
        m.impl("upsample_bicubic2d_backward.grad_input", TORCH_FN(upsample_bicubic2d_backward_out_fn));
        m.impl("upsample_nearest1d", TORCH_FN(upsample_nearest1d_fn));
        m.impl("upsample_nearest1d.out", TORCH_FN(upsample_nearest1d_out_fn));
        m.impl("upsample_nearest1d_backward", TORCH_FN(upsample_nearest1d_backward_fn));
        m.impl("upsample_nearest1d_backward.grad_input", TORCH_FN(upsample_nearest1d_backward_out_fn));
#define WG_REG_UPSAMPLE(NAME)                                                     \
    m.impl(#NAME, TORCH_FN(NAME##_fn));                                           \
    m.impl(#NAME ".out", TORCH_FN(NAME##_out_fn));                                \
    m.impl(#NAME "_backward", TORCH_FN(NAME##_backward_fn));                      \
    m.impl(#NAME "_backward.grad_input", TORCH_FN(NAME##_backward_out_fn));
        WG_REG_UPSAMPLE(upsample_trilinear3d)
        WG_REG_UPSAMPLE(_upsample_bilinear2d_aa)
        WG_REG_UPSAMPLE(_upsample_bicubic2d_aa)
        WG_REG_UPSAMPLE(upsample_nearest3d)
        WG_REG_UPSAMPLE(_upsample_nearest_exact3d)
#undef WG_REG_UPSAMPLE
        m.impl("upsample_nearest2d", TORCH_FN(upsample_nearest2d_direct_fn));
        m.impl("upsample_nearest2d.out", TORCH_FN(upsample_nearest2d_direct_out_fn));
        m.impl("upsample_nearest2d_backward", TORCH_FN(upsample_nearest2d_backward_fn));
        m.impl("upsample_nearest2d_backward.grad_input", TORCH_FN(upsample_nearest2d_backward_out_fn));
        m.impl("_upsample_nearest_exact1d", TORCH_FN(upsample_nearest_exact1d_fn));
        m.impl("_upsample_nearest_exact1d.out", TORCH_FN(upsample_nearest_exact1d_out_fn));
        m.impl("_upsample_nearest_exact1d_backward", TORCH_FN(upsample_nearest_exact1d_backward_fn));
        m.impl("_upsample_nearest_exact1d_backward.grad_input", TORCH_FN(upsample_nearest_exact1d_backward_out_fn));
        m.impl("_upsample_nearest_exact2d", TORCH_FN(upsample_nearest_exact2d_direct_fn));
        m.impl("_upsample_nearest_exact2d.out", TORCH_FN(upsample_nearest_exact2d_direct_out_fn));
        m.impl("_upsample_nearest_exact2d_backward", TORCH_FN(upsample_nearest_exact2d_backward_fn));
        m.impl("_upsample_nearest_exact2d_backward.grad_input", TORCH_FN(upsample_nearest_exact2d_backward_out_fn));

        m.impl("nll_loss_forward", TORCH_FN(nll_loss_forward_fn));
        m.impl("nll_loss_backward", TORCH_FN(nll_loss_backward_fn));
        m.impl("nll_loss2d_forward", TORCH_FN(nll_loss2d_forward_fn));
        m.impl("nll_loss2d_backward", TORCH_FN(nll_loss2d_backward_fn));
        m.impl("mse_loss", TORCH_FN(mse_loss_fn));
        m.impl("mse_loss.out", TORCH_FN(mse_loss_out_fn));
        m.impl("mse_loss_backward", TORCH_FN(mse_loss_backward_fn));
        m.impl("mse_loss_backward.grad_input", TORCH_FN(mse_loss_backward_out_fn));
        m.impl("smooth_l1_loss", TORCH_FN(smooth_l1_loss_fn));
        m.impl("smooth_l1_loss.out", TORCH_FN(smooth_l1_loss_out_fn));
        m.impl("smooth_l1_loss_backward.grad_input", TORCH_FN(smooth_l1_loss_backward_out_fn));
        m.impl("huber_loss", TORCH_FN(huber_loss_fn));
        m.impl("huber_loss.out", TORCH_FN(huber_loss_out_fn));
        m.impl("huber_loss_backward.out", TORCH_FN(huber_loss_backward_out_fn));
        m.impl("binary_cross_entropy", TORCH_FN(binary_cross_entropy_fn));
        m.impl("binary_cross_entropy.out", TORCH_FN(binary_cross_entropy_out_fn));
        m.impl("binary_cross_entropy_backward", TORCH_FN(binary_cross_entropy_backward_fn));
        m.impl("log_sigmoid_forward", TORCH_FN(log_sigmoid_forward_fn));
        m.impl("log_sigmoid_backward", TORCH_FN(log_sigmoid_backward_fn));
        m.impl("glu", TORCH_FN(glu_fn));
        m.impl("glu.out", TORCH_FN(glu_out_fn));
        m.impl("glu_backward", TORCH_FN(glu_backward_fn));
        m.impl("_prelu_kernel", TORCH_FN(prelu_kernel_fn));
        m.impl("_prelu_kernel_backward", TORCH_FN(prelu_kernel_backward_fn));
        m.impl("logit", TORCH_FN(logit_fn));
    }
}
