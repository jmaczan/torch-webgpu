// N-d pooling beyond the 2-d kernels in nn.cpp: max_pool3d / avg_pool3d
// (+ backward), adaptive max pooling 2d/3d (+ backward) and max_unpool 2d/3d.
//
// Fixed windows are strided views of the (padded) input reduced along the
// last dim; variable adaptive bins are gathered to a fixed size by repeating
// each bin's first index (which cannot change a max). Backward passes
// scatter-add into flat input positions.

#include <ATen/ATen.h>
#include <ATen/OpMathType.h>
#include <ATen/native/Resize.h>
#include <ATen/ops/adaptive_max_pool2d_backward_meta.h>
#include <ATen/ops/adaptive_max_pool2d_meta.h>
#include <ATen/ops/adaptive_max_pool3d_backward_meta.h>
#include <ATen/ops/adaptive_max_pool3d_meta.h>
#include <ATen/ops/avg_pool3d_backward_meta.h>
#include <ATen/ops/avg_pool3d_meta.h>
#include <torch/library.h>

#include <limits>
#include <numeric>

#include "ops/common.h"

namespace torch_webgpu::ops
{
    using at::Tensor;
    using Ints = std::vector<int64_t>;

    namespace
    {
        Ints expand_n(at::IntArrayRef p, size_t n, const Ints &def)
        {
            if (p.empty())
                return def;
            if (p.size() == 1)
                return Ints(n, p[0]);
            TORCH_CHECK(p.size() == n, "expected ", n, " values, got ", p.size());
            return p.vec();
        }

        int64_t pool_out(int64_t in, int64_t k, int64_t pad, int64_t st, int64_t dil, bool ceil)
        {
            int64_t num = in + 2 * pad - dil * (k - 1) - 1 + (ceil ? st - 1 : 0);
            int64_t o = (num >= 0 ? num / st : -((-num + st - 1) / st)) + 1;
            if (ceil && (o - 1) * st >= in + pad)
                --o;
            return o;
        }

        struct Geometry
        {
            size_t nd;
            Ints in, k, st, pad, dil, out, extra; // extra: trailing padding needed for ceil mode
        };

        Geometry geometry(const Tensor &x, size_t nd, at::IntArrayRef k, at::IntArrayRef st, at::IntArrayRef pad,
                          at::IntArrayRef dil, bool ceil)
        {
            Geometry g;
            g.nd = nd;
            g.k = expand_n(k, nd, {});
            TORCH_CHECK(g.k.size() == nd, "kernel_size must have ", nd, " elements");
            g.st = expand_n(st, nd, g.k);
            g.pad = expand_n(pad, nd, Ints(nd, 0));
            g.dil = expand_n(dil, nd, Ints(nd, 1));
            for (size_t d = 0; d < nd; ++d)
            {
                TORCH_CHECK(g.k[d] > 0 && g.st[d] > 0 && g.dil[d] > 0, "kernel size, stride and dilation must be positive");
                TORCH_CHECK(g.pad[d] >= 0 && g.pad[d] <= g.k[d] / 2,
                            "pad should be at most half of effective kernel size, but got pad=", g.pad[d],
                            ", kernel_size=", g.k[d]);
                const int64_t in = x.size(x.dim() - nd + d);
                g.in.push_back(in);
                const int64_t o = pool_out(in, g.k[d], g.pad[d], g.st[d], g.dil[d], ceil);
                TORCH_CHECK(o > 0, "Given input size ", x.sizes(), ", calculated output size is too small");
                g.out.push_back(o);
                g.extra.push_back(std::max<int64_t>(0, (o - 1) * g.st[d] + g.dil[d] * (g.k[d] - 1) + 1 - (in + 2 * g.pad[d])));
            }
            return g;
        }

        // constant_pad_nd padding list for the trailing nd dims (last dim first).
        Ints pad_list(const Geometry &g, bool with_pad)
        {
            Ints p;
            for (size_t i = 0; i < g.nd; ++i)
            {
                const size_t d = g.nd - 1 - i;
                p.push_back(with_pad ? g.pad[d] : 0);
                p.push_back((with_pad ? g.pad[d] : 0) + g.extra[d]);
            }
            return p;
        }

        // [N, C, padded...] -> [N, C, out..., K] windows.
        Tensor windows(const Tensor &xp, const Geometry &g)
        {
            Ints sizes{xp.size(0), xp.size(1)}, strides{xp.stride(0), xp.stride(1)};
            for (size_t d = 0; d < g.nd; ++d)
            {
                sizes.push_back(g.out[d]);
                strides.push_back(xp.stride(2 + d) * g.st[d]);
            }
            int64_t K = 1;
            for (size_t d = 0; d < g.nd; ++d)
            {
                sizes.push_back(g.k[d]);
                strides.push_back(xp.stride(2 + d) * g.dil[d]);
                K *= g.k[d];
            }
            Ints flat(sizes.begin(), sizes.begin() + 2 + g.nd);
            flat.push_back(K);
            return xp.as_strided(sizes, strides, xp.storage_offset()).reshape(flat);
        }

        // Broadcastable [out_d] grid along output dim d of an [N, C, out...] tensor.
        Tensor out_grid(const Geometry &g, size_t d, const at::TensorOptions &lopt)
        {
            Ints shape(2 + g.nd, 1);
            shape[2 + d] = g.out[d];
            return at::arange(g.out[d], lopt).view(shape);
        }

        // Flat (unpadded) input position of local window tap `arg` per output.
        Tensor window_arg_to_input(const Tensor &arg, const Geometry &g)
        {
            auto lopt = arg.options().dtype(at::kLong);
            Tensor rem = arg, flat = at::zeros({}, lopt);
            Ints kstride(g.nd, 1);
            for (int64_t d = static_cast<int64_t>(g.nd) - 2; d >= 0; --d)
                kstride[d] = kstride[d + 1] * g.k[d + 1];
            for (size_t d = 0; d < g.nd; ++d)
            {
                Tensor kd = at::floor_divide(rem, kstride[d]);
                rem = at::remainder(rem, kstride[d]);
                Tensor coord = out_grid(g, d, lopt) * g.st[d] - g.pad[d] + kd * g.dil[d];
                flat = flat * g.in[d] + coord;
            }
            return flat;
        }

        Tensor batched(const Tensor &x, size_t nd) { return x.dim() == static_cast<int64_t>(nd) + 1 ? x.unsqueeze(0) : x; }
        Tensor unbatch(const Tensor &x, const Tensor &like, size_t nd)
        {
            return like.dim() == static_cast<int64_t>(nd) + 1 ? x.squeeze(0) : x;
        }

        at::Scalar lowest_of(at::ScalarType t)
        {
            switch (t)
            {
            case at::kLong: return std::numeric_limits<int64_t>::lowest();
            case at::kInt: return std::numeric_limits<int32_t>::lowest();
            case at::kShort: return std::numeric_limits<int16_t>::lowest();
            case at::kChar: return std::numeric_limits<int8_t>::lowest();
            case at::kByte: return 0;
            case at::kBool: return false;
            default: return -std::numeric_limits<double>::infinity();
            }
        }

        // Scatter-adds grad [N, C, L] at flat positions idx [N, C, L] into [N, C, spatial].
        Tensor scatter_grad(const Tensor &grad, const Tensor &idx, const Tensor &self_b, size_t nd)
        {
            const int64_t N = self_b.size(0), C = self_b.size(1);
            int64_t S = 1, L = 1;
            for (size_t d = 0; d < nd; ++d)
            {
                S *= self_b.size(2 + d);
                L *= grad.size(2 + d);
            }
            Tensor gi = at::zeros({N, C, S}, grad.options());
            if (gi.numel() > 0 && L > 0)
                gi.scatter_add_(2, idx.reshape({N, C, L}), grad.reshape({N, C, L}));
            return gi.view(self_b.sizes());
        }

        // ------------------------------------------------------- max_pool3d
        std::tuple<Tensor, Tensor> max_pool3d_with_indices_fn(const Tensor &self, at::IntArrayRef k, at::IntArrayRef st,
                                                              at::IntArrayRef pad, at::IntArrayRef dil, bool ceil)
        {
            TORCH_CHECK(self.dim() == 4 || self.dim() == 5, "max_pool3d: expected 4D or 5D input, got ", self.sizes());
            Tensor x = batched(self, 3);
            auto g = geometry(x, 3, k, st, pad, dil, ceil);
            Tensor xp = at::constant_pad_nd(x, pad_list(g, true), lowest_of(x.scalar_type()));
            auto [vals, arg] = windows(xp, g).max(-1);
            Tensor idx = window_arg_to_input(arg, g);
            return {unbatch(vals, self, 3), unbatch(idx, self, 3)};
        }
        Tensor max_pool3d_with_indices_backward_fn(const Tensor &grad, const Tensor &self, at::IntArrayRef, at::IntArrayRef,
                                                   at::IntArrayRef, at::IntArrayRef, bool, const Tensor &indices)
        {
            Tensor x = batched(self, 3);
            return unbatch(scatter_grad(batched(grad, 3), batched(indices, 3), x, 3), self, 3);
        }
        Tensor &max_pool3d_with_indices_backward_out_fn(const Tensor &grad, const Tensor &self, at::IntArrayRef k,
                                                        at::IntArrayRef st, at::IntArrayRef pad, at::IntArrayRef dil,
                                                        bool ceil, const Tensor &indices, Tensor &gi)
        {
            Tensor r = max_pool3d_with_indices_backward_fn(grad, self, k, st, pad, dil, ceil, indices);
            at::native::resize_output(gi, r.sizes());
            return gi.copy_(r);
        }
        std::tuple<Tensor &, Tensor &> max_pool3d_with_indices_out_fn(const Tensor &self, at::IntArrayRef k,
                                                                      at::IntArrayRef st, at::IntArrayRef pad,
                                                                      at::IntArrayRef dil, bool ceil, Tensor &out,
                                                                      Tensor &indices)
        {
            auto [v, i] = max_pool3d_with_indices_fn(self, k, st, pad, dil, ceil);
            at::native::resize_output(out, v.sizes());
            at::native::resize_output(indices, i.sizes());
            out.copy_(v);
            indices.copy_(i);
            return {out, indices};
        }

        // ------------------------------------------------------- avg_pool3d
        // Per-output divisor [out...].
        Tensor avg_divisor(const Tensor &x, const Geometry &g, bool count_include_pad, std::optional<int64_t> divisor_override)
        {
            Ints out(g.out.begin(), g.out.end());
            if (divisor_override.has_value())
                return at::full(out, static_cast<double>(*divisor_override), x.options());
            Ints ones_shape{1, 1};
            for (size_t d = 0; d < g.nd; ++d)
                ones_shape.push_back(g.in[d] + (count_include_pad ? 2 * g.pad[d] : 0));
            Tensor ones = at::ones(ones_shape, x.options());
            Ints pl;
            for (size_t i = 0; i < g.nd; ++i)
            {
                const size_t d = g.nd - 1 - i;
                pl.push_back(count_include_pad ? 0 : g.pad[d]);
                pl.push_back((count_include_pad ? 0 : g.pad[d]) + g.extra[d]);
            }
            return windows(at::constant_pad_nd(ones, pl, 0), g).sum(-1).view(out);
        }
        Tensor avg_pool_nd(const Tensor &self, size_t nd, at::IntArrayRef k, at::IntArrayRef st, at::IntArrayRef pad,
                           bool ceil, bool count_include_pad, std::optional<int64_t> divisor_override)
        {
            Tensor x = batched(self, nd);
            auto g = geometry(x, nd, k, st, pad, {}, ceil);
            const auto acc = at::toOpMathType(x.scalar_type());
            Tensor xp = at::constant_pad_nd(x.to(acc), pad_list(g, true), 0);
            Tensor out = windows(xp, g).sum(-1) / avg_divisor(xp, g, count_include_pad, divisor_override);
            if (!at::isFloatingType(x.scalar_type()))
                out = out.trunc();
            return unbatch(out.to(self.scalar_type()), self, nd);
        }
        // Flat padded positions of every window tap: [K, L].
        Tensor tap_map(const Geometry &g, const Ints &padded, const at::TensorOptions &lopt)
        {
            Ints pstride(g.nd, 1);
            for (int64_t d = static_cast<int64_t>(g.nd) - 2; d >= 0; --d)
                pstride[d] = pstride[d + 1] * padded[d + 1];
            Tensor pos = at::zeros({}, lopt);
            // dims: [k0, k1, ..., o0, o1, ...]
            const size_t nd = g.nd;
            for (size_t d = 0; d < nd; ++d)
            {
                Ints ks(2 * nd, 1), os(2 * nd, 1);
                ks[d] = g.k[d];
                os[nd + d] = g.out[d];
                Tensor kk = at::arange(g.k[d], lopt).view(ks), oo = at::arange(g.out[d], lopt).view(os);
                pos = pos + (oo * g.st[d] + kk * g.dil[d]) * pstride[d];
            }
            int64_t K = 1, L = 1;
            for (size_t d = 0; d < nd; ++d)
            {
                K *= g.k[d];
                L *= g.out[d];
            }
            Ints full;
            for (size_t d = 0; d < nd; ++d)
                full.push_back(g.k[d]);
            for (size_t d = 0; d < nd; ++d)
                full.push_back(g.out[d]);
            return pos.expand(full).reshape({K, L});
        }
        Tensor avg_pool_nd_backward(const Tensor &grad, const Tensor &self, size_t nd, at::IntArrayRef k,
                                    at::IntArrayRef st, at::IntArrayRef pad, bool ceil, bool count_include_pad,
                                    std::optional<int64_t> divisor_override)
        {
            Tensor x = batched(self, nd), go = batched(grad, nd);
            auto g = geometry(x, nd, k, st, pad, {}, ceil);
            const int64_t N = x.size(0), C = x.size(1);
            int64_t K = 1, L = 1, P = 1;
            Ints padded;
            for (size_t d = 0; d < nd; ++d)
            {
                K *= g.k[d];
                L *= g.out[d];
                padded.push_back(g.in[d] + 2 * g.pad[d] + g.extra[d]);
                P *= padded.back();
            }
            Tensor scaled = go / avg_divisor(x, g, count_include_pad, divisor_override);
            Tensor out = at::zeros({N, C, P}, go.options());
            if (out.numel() > 0 && L > 0)
            {
                Tensor cols = scaled.reshape({N, C, 1, L}).expand({N, C, K, L}).reshape({N, C, K * L});
                Tensor map = tap_map(g, padded, x.options().dtype(at::kLong)).reshape({1, 1, K * L}).expand({N, C, K * L});
                out.scatter_add_(2, map, cols);
            }
            Ints full{N, C};
            full.insert(full.end(), padded.begin(), padded.end());
            Tensor gi = out.view(full);
            for (size_t d = 0; d < nd; ++d)
                gi = gi.narrow(2 + d, g.pad[d], g.in[d]);
            return unbatch(gi.contiguous(), self, nd);
        }

        Tensor avg_pool3d_fn(const Tensor &self, at::IntArrayRef k, at::IntArrayRef st, at::IntArrayRef pad, bool ceil,
                             bool cip, std::optional<int64_t> div)
        {
            Functional<at::meta::structured_avg_pool3d> op;
            op.meta(self, k, st, pad, ceil, cip, div);
            op.outputs_[0].copy_(avg_pool_nd(self, 3, k, st, pad, ceil, cip, div));
            return std::move(op.outputs_[0]);
        }
        Tensor &avg_pool3d_out_fn(const Tensor &self, at::IntArrayRef k, at::IntArrayRef st, at::IntArrayRef pad, bool ceil,
                                  bool cip, std::optional<int64_t> div, Tensor &out)
        {
            Out<at::meta::structured_avg_pool3d> op(out);
            op.meta(self, k, st, pad, ceil, cip, div);
            op.maybe_get_output(0).copy_(avg_pool_nd(self, 3, k, st, pad, ceil, cip, div));
            op.finish();
            return out;
        }
        Tensor avg_pool3d_backward_fn(const Tensor &grad, const Tensor &self, at::IntArrayRef k, at::IntArrayRef st,
                                      at::IntArrayRef pad, bool ceil, bool cip, std::optional<int64_t> div)
        {
            Functional<at::meta::structured_avg_pool3d_backward> op;
            op.meta(grad, self, k, st, pad, ceil, cip, div);
            op.outputs_[0].copy_(avg_pool_nd_backward(grad, self, 3, k, st, pad, ceil, cip, div));
            return std::move(op.outputs_[0]);
        }
        Tensor &avg_pool3d_backward_out_fn(const Tensor &grad, const Tensor &self, at::IntArrayRef k, at::IntArrayRef st,
                                           at::IntArrayRef pad, bool ceil, bool cip, std::optional<int64_t> div,
                                           Tensor &gi)
        {
            Out<at::meta::structured_avg_pool3d_backward> op(gi);
            op.meta(grad, self, k, st, pad, ceil, cip, div);
            op.maybe_get_output(0).copy_(avg_pool_nd_backward(grad, self, 3, k, st, pad, ceil, cip, div));
            op.finish();
            return gi;
        }

        // ------------------------------------------------- adaptive max pool
        // Bin o covers [floor(o*in/out), ceil((o+1)*in/out)); returns [out, K]
        // input indices padded by repeating each bin's start.
        Tensor adaptive_bins(int64_t in, int64_t out, const at::TensorOptions &opt)
        {
            int64_t K = 0;
            for (int64_t o = 0; o < out; ++o)
                K = std::max(K, ((o + 1) * in + out - 1) / out - (o * in) / out);
            std::vector<int64_t> idx(out * K);
            for (int64_t o = 0; o < out; ++o)
            {
                const int64_t s = (o * in) / out, e = ((o + 1) * in + out - 1) / out;
                for (int64_t j = 0; j < K; ++j)
                    idx[o * K + j] = s + j < e ? s + j : s;
            }
            return at::tensor(idx, at::TensorOptions().dtype(at::kLong)).view({out, K}).to(opt.device());
        }

        void adaptive_max_impl(const Tensor &self, at::IntArrayRef out_size, size_t nd, const Tensor &out,
                               const Tensor &indices)
        {
            Tensor x = batched(self, nd);
            const int64_t N = x.size(0), C = x.size(1);
            if (out.numel() == 0)
                return;
            std::vector<Tensor> bins;
            Tensor y = x;
            Ints perm{0, 1}, view_shape{N, C}, Ks;
            for (size_t d = 0; d < nd; ++d)
            {
                bins.push_back(adaptive_bins(x.size(2 + d), out_size[d], x.options()));
                Ks.push_back(bins[d].size(1));
                y = y.index_select(2 + d, bins[d].reshape({-1}));
                view_shape.push_back(out_size[d]);
                view_shape.push_back(Ks[d]);
            }
            int64_t K = 1;
            for (size_t d = 0; d < nd; ++d)
            {
                perm.push_back(2 + 2 * d);
                K *= Ks[d];
            }
            for (size_t d = 0; d < nd; ++d)
                perm.push_back(3 + 2 * d);
            Ints red{N, C};
            red.insert(red.end(), out_size.begin(), out_size.end());
            red.push_back(K);
            auto [vals, arg] = y.reshape(view_shape).permute(perm).reshape(red).max(-1);
            // Window tap -> flat input index through the bin tables.
            auto lopt = x.options().dtype(at::kLong);
            Tensor rem = arg, flat = at::zeros({}, lopt);
            for (size_t d = 0; d < nd; ++d)
            {
                int64_t kstride = 1;
                for (size_t e = d + 1; e < nd; ++e)
                    kstride *= Ks[e];
                Tensor kd = at::floor_divide(rem, kstride);
                rem = at::remainder(rem, kstride);
                Ints shape(2 + nd, 1);
                shape[2 + d] = out_size[d];
                Tensor od = at::arange(out_size[d], lopt).view(shape);
                Tensor coord = at::take(bins[d], od * Ks[d] + kd);
                flat = flat * x.size(2 + d) + coord;
            }
            out.copy_(unbatch(vals, self, nd));
            indices.copy_(unbatch(flat.expand(vals.sizes()), self, nd));
        }

#define WG_ADAPTIVE_MAX(ND)                                                                                         \
    std::tuple<Tensor, Tensor> adaptive_max_pool##ND##d_fn(const Tensor &self, at::IntArrayRef out_size)            \
    {                                                                                                               \
        Functional<at::meta::structured_adaptive_max_pool##ND##d, 2> op;                                               \
        op.meta(self, out_size);                                                                                    \
        adaptive_max_impl(self, out_size, ND, op.outputs_[0], op.outputs_[1]);                                      \
        return {std::move(op.outputs_[0]), std::move(op.outputs_[1])};                                             \
    }                                                                                                               \
    std::tuple<Tensor &, Tensor &> adaptive_max_pool##ND##d_out_fn(const Tensor &self, at::IntArrayRef out_size,    \
                                                                   Tensor &out, Tensor &indices)                    \
    {                                                                                                               \
        Out<at::meta::structured_adaptive_max_pool##ND##d, 2> op(out, indices);                                        \
        op.meta(self, out_size);                                                                                    \
        adaptive_max_impl(self, out_size, ND, op.maybe_get_output(0), op.maybe_get_output(1));                      \
        op.finish();                                                                                                \
        return {out, indices};                                                                                      \
    }                                                                                                               \
    Tensor adaptive_max_pool##ND##d_backward_fn(const Tensor &grad, const Tensor &self, const Tensor &indices)      \
    {                                                                                                               \
        Functional<at::meta::structured_adaptive_max_pool##ND##d_backward> op;                                      \
        op.meta(grad, self, indices);                                                                               \
        op.outputs_[0].copy_(                                                                                       \
            unbatch(scatter_grad(batched(grad, ND), batched(indices, ND), batched(self, ND), ND), self, ND));       \
        return std::move(op.outputs_[0]);                                                                           \
    }                                                                                                               \
    Tensor &adaptive_max_pool##ND##d_backward_out_fn(const Tensor &grad, const Tensor &self, const Tensor &indices, \
                                                     Tensor &gi)                                                    \
    {                                                                                                               \
        Out<at::meta::structured_adaptive_max_pool##ND##d_backward> op(gi);                                         \
        op.meta(grad, self, indices);                                                                               \
        op.maybe_get_output(0).copy_(                                                                               \
            unbatch(scatter_grad(batched(grad, ND), batched(indices, ND), batched(self, ND), ND), self, ND));       \
        op.finish();                                                                                                \
        return gi;                                                                                                  \
    }
        WG_ADAPTIVE_MAX(2)
        WG_ADAPTIVE_MAX(3)
#undef WG_ADAPTIVE_MAX

        // ------------------------------------------------------ max_unpool
        Tensor max_unpool_nd(const Tensor &self, const Tensor &indices, at::IntArrayRef output_size, size_t nd)
        {
            TORCH_CHECK(indices.scalar_type() == at::kLong, "elements in indices should be type int64 but got: ",
                        indices.scalar_type());
            TORCH_CHECK(output_size.size() == nd, "There should be exactly ", nd,
                        " elements (depth, height, width) in output_size, but got ", output_size.size(), " elements.");
            TORCH_CHECK(self.dim() == static_cast<int64_t>(nd) + 1 || self.dim() == static_cast<int64_t>(nd) + 2,
                        "Input to max_unpooling", nd, "d should be a ", nd + 1, "d or ", nd + 2, "d Tensor, but got a tensor with ",
                        self.dim(), " dimensions.");
            TORCH_CHECK(self.sizes() == indices.sizes(), "Expected shape of indices to be same as that of the input tensor (",
                        self.sizes(), ") but got indices tensor with shape: ", indices.sizes());
            Tensor x = batched(self, nd), idx = batched(indices, nd);
            const int64_t N = x.size(0), C = x.size(1);
            int64_t S = 1, L = 1;
            for (size_t d = 0; d < nd; ++d)
            {
                S *= output_size[d];
                L *= x.size(2 + d);
            }
            Tensor out = at::zeros({N, C, S}, x.options());
            if (out.numel() > 0 && L > 0)
                out.scatter_(2, idx.reshape({N, C, L}), x.reshape({N, C, L}));
            Ints shape{N, C};
            shape.insert(shape.end(), output_size.begin(), output_size.end());
            return unbatch(out.view(shape), self, nd);
        }
        Tensor max_unpool2d_fn(const Tensor &self, const Tensor &indices, c10::SymIntArrayRef output_size)
        {
            return max_unpool_nd(self, indices, C10_AS_INTARRAYREF_SLOW(output_size), 2);
        }
        Tensor &max_unpool2d_out_fn(const Tensor &self, const Tensor &indices, c10::SymIntArrayRef output_size, Tensor &out)
        {
            Tensor r = max_unpool_nd(self, indices, C10_AS_INTARRAYREF_SLOW(output_size), 2);
            at::native::resize_output(out, r.sizes());
            return out.copy_(r);
        }
        Tensor max_unpool3d_fn(const Tensor &self, const Tensor &indices, c10::SymIntArrayRef output_size,
                               at::IntArrayRef, at::IntArrayRef)
        {
            return max_unpool_nd(self, indices, C10_AS_INTARRAYREF_SLOW(output_size), 3);
        }
        Tensor &max_unpool3d_out_fn(const Tensor &self, const Tensor &indices, c10::SymIntArrayRef output_size,
                                    at::IntArrayRef, at::IntArrayRef, Tensor &out)
        {
            Tensor r = max_unpool_nd(self, indices, C10_AS_INTARRAYREF_SLOW(output_size), 3);
            at::native::resize_output(out, r.sizes());
            return out.copy_(r);
        }
    }

    TORCH_LIBRARY_IMPL(aten, PrivateUse1, m)
    {
        m.impl("max_pool3d_with_indices", TORCH_FN(max_pool3d_with_indices_fn));
        m.impl("max_pool3d_with_indices.out", TORCH_FN(max_pool3d_with_indices_out_fn));
        m.impl("max_pool3d_with_indices_backward", TORCH_FN(max_pool3d_with_indices_backward_fn));
        m.impl("max_pool3d_with_indices_backward.grad_input", TORCH_FN(max_pool3d_with_indices_backward_out_fn));
        m.impl("avg_pool3d", TORCH_FN(avg_pool3d_fn));
        m.impl("avg_pool3d.out", TORCH_FN(avg_pool3d_out_fn));
        m.impl("avg_pool3d_backward", TORCH_FN(avg_pool3d_backward_fn));
        m.impl("avg_pool3d_backward.grad_input", TORCH_FN(avg_pool3d_backward_out_fn));
        m.impl("adaptive_max_pool2d", TORCH_FN(adaptive_max_pool2d_fn));
        m.impl("adaptive_max_pool2d.out", TORCH_FN(adaptive_max_pool2d_out_fn));
        m.impl("adaptive_max_pool2d_backward", TORCH_FN(adaptive_max_pool2d_backward_fn));
        m.impl("adaptive_max_pool2d_backward.grad_input", TORCH_FN(adaptive_max_pool2d_backward_out_fn));
        m.impl("adaptive_max_pool3d", TORCH_FN(adaptive_max_pool3d_fn));
        m.impl("adaptive_max_pool3d.out", TORCH_FN(adaptive_max_pool3d_out_fn));
        m.impl("adaptive_max_pool3d_backward", TORCH_FN(adaptive_max_pool3d_backward_fn));
        m.impl("adaptive_max_pool3d_backward.grad_input", TORCH_FN(adaptive_max_pool3d_backward_out_fn));
        m.impl("max_unpool2d", TORCH_FN(max_unpool2d_fn));
        m.impl("max_unpool2d.out", TORCH_FN(max_unpool2d_out_fn));
        m.impl("max_unpool3d", TORCH_FN(max_unpool3d_fn));
        m.impl("max_unpool3d.out", TORCH_FN(max_unpool3d_out_fn));
    }
}
