// grid_sampler_2d / grid_sampler_3d and their backward, composed from
// gathers and scatter-adds over flat spatial indices. Coordinate handling
// (unnormalize, border / reflection padding and their gradients) follows
// ATen/native/GridSampler.h exactly.

#include <ATen/ATen.h>
#include <ATen/OpMathType.h>
#include <c10/util/accumulate.h>
#include <torch/library.h>

#include <cmath>

#include "ops/common.h"

namespace torch_webgpu::ops
{
    using at::Tensor;

    namespace
    {
        enum Interp
        {
            Bilinear = 0,
            Nearest = 1,
            Bicubic = 2
        };
        enum Padding
        {
            Zeros = 0,
            Border = 1,
            Reflection = 2
        };

        // coordinate and d(coordinate)/d(grid) multiplier, [N, L] float.
        struct Coord
        {
            Tensor c, mult;
        };

        Coord unnormalize(const Tensor &g, int64_t size, bool align_corners)
        {
            if (align_corners)
                return {(g + 1) / 2 * static_cast<double>(size - 1), at::full_like(g, static_cast<double>(size - 1) / 2)};
            return {((g + 1) * static_cast<double>(size) - 1) / 2, at::full_like(g, static_cast<double>(size) / 2)};
        }

        // clip to [0, size-1]; gradient 0 where clipped (incl. the bounds).
        Coord clip(const Coord &x, int64_t size)
        {
            const double hi = static_cast<double>(size - 1);
            Tensor inside = (x.c > 0) & (x.c < hi);
            return {x.c.clamp(0, hi), x.mult * inside.to(x.mult.scalar_type())};
        }

        Coord reflect(const Coord &x, double twice_low, double twice_high)
        {
            if (twice_low == twice_high)
                return {at::zeros_like(x.c), at::zeros_like(x.mult)};
            const double mn = twice_low / 2, span = (twice_high - twice_low) / 2;
            Tensor in = x.c - mn;
            Tensor neg = in < 0;
            in = in.abs();
            Tensor extra = at::fmod(in, span);
            Tensor flips = at::floor(in / span);
            Tensor even = at::remainder(flips, 2) == 0;
            Tensor c = at::where(even, extra + mn, span - extra + mn);
            Tensor sign = at::where(neg, -at::ones_like(in), at::ones_like(in));
            Tensor m = at::where(even, sign, -sign);
            return {c, x.mult * m};
        }

        // CPU safe_downgrade_to_int_range: non-finite / huge -> -100 (out of bounds).
        Coord safe(const Coord &x)
        {
            Tensor ok = at::isfinite(x.c) & (x.c.abs() < 2147483000.0);
            return {at::where(ok, x.c, at::full_like(x.c, -100.0)), x.mult};
        }

        Coord source_index(const Tensor &g, int64_t size, int64_t padding, bool align_corners)
        {
            Coord x = unnormalize(g, size, align_corners);
            if (padding == Border)
                x = clip(x, size);
            else if (padding == Reflection)
            {
                x = align_corners ? reflect(x, 0, 2.0 * (size - 1)) : reflect(x, -1, 2.0 * size - 1);
                x = clip(x, size);
            }
            return safe(x);
        }

        // Padding transform of (integer-valued) bicubic tap coordinates.
        Tensor tap_coordinate(const Tensor &c, int64_t size, int64_t padding, bool align_corners)
        {
            if (padding == Zeros)
                return c;
            Coord x{c, at::ones_like(c)};
            if (padding == Reflection)
                x = align_corners ? reflect(x, 0, 2.0 * (size - 1)) : reflect(x, -1, 2.0 * size - 1);
            return clip(x, size).c;
        }

        // One sampling tap: per-axis integer coordinates, weight factors and
        // their derivatives.
        struct AxisTap
        {
            Tensor i;  // integer coordinate (float)
            Tensor w;  // weight factor
            Tensor dw; // d w / d coordinate
        };

        const double A = -0.75;
        Tensor cubic1(const Tensor &x) { return ((A + 2) * x - (A + 3)) * x * x + 1; }
        Tensor cubic2(const Tensor &x) { return ((A * x - 5 * A) * x + 8 * A) * x - 4 * A; }
        Tensor dcubic1(const Tensor &x) { return 3 * (A + 2) * x * x - 2 * (A + 3) * x; }
        Tensor dcubic2(const Tensor &x) { return 3 * A * x * x - 10 * A * x + 8 * A; }

        std::vector<AxisTap> axis_taps(const Tensor &c, int64_t interp)
        {
            if (interp == Nearest)
                return {{at::round(c), at::ones_like(c), at::zeros_like(c)}};
            Tensor f = at::floor(c);
            Tensor t = c - f;
            if (interp == Bilinear)
                return {{f, 1 - t, -at::ones_like(t)}, {f + 1, t, at::ones_like(t)}};
            return {{f - 1, cubic2(t + 1), dcubic2(t + 1)},
                    {f, cubic1(t), dcubic1(t)},
                    {f + 1, cubic1(1 - t), -dcubic1(1 - t)},
                    {f + 2, cubic2(2 - t), -dcubic2(2 - t)}};
        }

        struct Sampler
        {
            int64_t N, C, L, nd;
            std::vector<int64_t> sizes; // input spatial sizes (D, H, W) / (H, W)
            std::vector<Coord> coords;  // per axis, in input-dim order
            std::vector<std::vector<AxisTap>> taps;
            Tensor flat_input; // [N, C, S]
            int64_t interp, padding;
            bool align_corners;
        };

        // grid [N, out..., nd] with grid[..., 0] = x (last input dim).
        Sampler make_sampler(const Tensor &input, const Tensor &grid, int64_t interp, int64_t padding, bool align_corners)
        {
            Sampler s;
            s.nd = input.dim() - 2;
            s.N = input.size(0);
            s.C = input.size(1);
            s.L = 1;
            for (int64_t d = 0; d < s.nd; ++d)
                s.L *= grid.size(1 + d);
            for (int64_t d = 0; d < s.nd; ++d)
                s.sizes.push_back(input.size(2 + d));
            s.interp = interp;
            s.padding = padding;
            s.align_corners = align_corners;
            const auto ct = at::toOpMathType(input.scalar_type());
            Tensor g = grid.reshape({s.N, s.L, s.nd}).to(ct);
            for (int64_t d = 0; d < s.nd; ++d)
            {
                // input dim d (D/H/W order) is grid component nd-1-d (x, y, z order).
                Tensor gd = g.select(2, s.nd - 1 - d);
                Coord c = interp == Bicubic ? unnormalize(gd, s.sizes[d], align_corners)
                                            : source_index(gd, s.sizes[d], padding, align_corners);
                s.coords.push_back(c);
                s.taps.push_back(axis_taps(c.c, interp));
            }
            int64_t S = 1;
            for (auto v : s.sizes)
                S *= v;
            s.flat_input = input.reshape({s.N, s.C, S});
            return s;
        }

        // Iterates the tap combinations: callback(flat index [N, L] long,
        // weight*valid [N, L], per-axis dweight [N, L]).
        template <class F>
        void for_each_tap(const Sampler &s, bool need_dweight, F &&f)
        {
            const int64_t nd = s.nd;
            std::vector<size_t> sel(nd, 0);
            while (true)
            {
                Tensor idx, weight, valid;
                std::vector<Tensor> ws;
                for (int64_t d = 0; d < nd; ++d)
                {
                    const AxisTap &t = s.taps[d][sel[d]];
                    Tensor ic = t.i;
                    if (s.interp == Bicubic)
                        ic = tap_coordinate(ic, s.sizes[d], s.padding, s.align_corners);
                    Tensor v = (ic >= 0) & (ic < static_cast<double>(s.sizes[d]));
                    Tensor il = ic.clamp(0, static_cast<double>(s.sizes[d] - 1)).to(at::kLong);
                    idx = d == 0 ? il : idx * s.sizes[d] + il;
                    valid = d == 0 ? v : valid & v;
                    ws.push_back(t.w);
                }
                Tensor vf = valid.to(ws[0].scalar_type());
                weight = vf;
                for (auto &w : ws)
                    weight = weight * w;
                std::vector<Tensor> dws;
                if (need_dweight)
                    for (int64_t a = 0; a < nd; ++a)
                    {
                        Tensor dw = vf * s.taps[a][sel[a]].dw;
                        for (int64_t b = 0; b < nd; ++b)
                            if (b != a)
                                dw = dw * ws[b];
                        dws.push_back(dw);
                    }
                f(idx, weight, dws);
                int64_t d = nd - 1;
                while (d >= 0 && ++sel[d] == s.taps[d].size())
                    sel[d--] = 0;
                if (d < 0)
                    break;
            }
        }

        Tensor gather_tap(const Sampler &s, const Tensor &idx)
        {
            return s.flat_input.gather(2, idx.unsqueeze(1).expand({s.N, s.C, s.L}));
        }

        void check_grid_sampler(const Tensor &input, const Tensor &grid, int64_t nd, int64_t interp)
        {
            TORCH_CHECK(input.dim() == nd + 2 && grid.dim() == nd + 2, "grid_sampler(): expected ", nd + 2,
                        "D input and grid");
            TORCH_CHECK(grid.size(-1) == nd, "grid_sampler(): expected grid to have size ", nd,
                        " in last dimension, but got grid with sizes ", grid.sizes());
            TORCH_CHECK(input.size(0) == grid.size(0), "grid_sampler(): expected grid and input to have same batch size");
            TORCH_CHECK(interp >= Bilinear && interp <= Bicubic, "grid_sampler(): invalid interpolation mode ", interp);
        }

        Tensor grid_sampler_impl(const Tensor &input, const Tensor &grid, int64_t interp, int64_t padding,
                                 bool align_corners, int64_t nd)
        {
            check_grid_sampler(input, grid, nd, interp);
            std::vector<int64_t> out_shape{input.size(0), input.size(1)};
            for (int64_t d = 0; d < nd; ++d)
                out_shape.push_back(grid.size(1 + d));
            if (c10::multiply_integers(out_shape) == 0 || input.numel() == 0)
                return at::zeros(out_shape, input.options());
            Sampler s = make_sampler(input, grid, interp, padding, align_corners);
            const auto ct = at::toOpMathType(input.scalar_type());
            Tensor out = at::zeros({s.N, s.C, s.L}, input.options().dtype(ct));
            for_each_tap(s, false, [&](const Tensor &idx, const Tensor &w, const std::vector<Tensor> &)
                         { out.add_(gather_tap(s, idx).to(ct) * w.unsqueeze(1)); });
            return out.to(input.scalar_type()).view(out_shape);
        }

        std::tuple<Tensor, Tensor> grid_sampler_backward_impl(const Tensor &grad_output, const Tensor &input,
                                                              const Tensor &grid, int64_t interp, int64_t padding,
                                                              bool align_corners, std::array<bool, 2> mask, int64_t nd)
        {
            Tensor gi = mask[0] ? at::zeros_like(input) : Tensor();
            Tensor gg = mask[1] ? at::zeros_like(grid) : Tensor();
            if (input.numel() == 0 || grid.numel() == 0)
                return {gi, gg};
            Sampler s = make_sampler(input, grid, interp, padding, align_corners);
            const auto ct = at::toOpMathType(input.scalar_type());
            Tensor go = grad_output.reshape({s.N, s.C, s.L}).to(ct);
            Tensor gin = mask[0] ? at::zeros({s.N, s.C, s.flat_input.size(2)}, input.options().dtype(ct)) : Tensor();
            std::vector<Tensor> gc(nd);
            for (auto &t : gc)
                t = at::zeros({s.N, s.L}, input.options().dtype(ct));
            const bool need_grid = mask[1] && interp != Nearest;
            for_each_tap(s, need_grid,
                         [&](const Tensor &idx, const Tensor &w, const std::vector<Tensor> &dws)
                         {
                             if (mask[0])
                                 gin.scatter_add_(2, idx.unsqueeze(1).expand({s.N, s.C, s.L}), go * w.unsqueeze(1));
                             if (need_grid)
                             {
                                 Tensor gv = (go * gather_tap(s, idx).to(ct)).sum(1); // [N, L]
                                 for (int64_t d = 0; d < nd; ++d)
                                     gc[d].add_(gv * dws[d]);
                             }
                         });
            if (mask[0])
                gi = gin.view(input.sizes()).to(input.scalar_type());
            if (mask[1])
            {
                std::vector<Tensor> comps;
                for (int64_t k = 0; k < nd; ++k) // grid component k = input axis nd-1-k
                {
                    const int64_t d = nd - 1 - k;
                    comps.push_back(need_grid ? gc[d] * s.coords[d].mult : gc[d]);
                }
                gg = at::stack(comps, -1).view(grid.sizes()).to(grid.scalar_type());
            }
            return {gi, gg};
        }

        Tensor grid_sampler_2d_fn(const Tensor &input, const Tensor &grid, int64_t interp, int64_t padding,
                                  bool align_corners)
        {
            return grid_sampler_impl(input, grid, interp, padding, align_corners, 2);
        }
        Tensor grid_sampler_3d_fn(const Tensor &input, const Tensor &grid, int64_t interp, int64_t padding,
                                  bool align_corners)
        {
            return grid_sampler_impl(input, grid, interp, padding, align_corners, 3);
        }
        std::tuple<Tensor, Tensor> grid_sampler_2d_backward_fn(const Tensor &grad_output, const Tensor &input,
                                                               const Tensor &grid, int64_t interp, int64_t padding,
                                                               bool align_corners, std::array<bool, 2> mask)
        {
            return grid_sampler_backward_impl(grad_output, input, grid, interp, padding, align_corners, mask, 2);
        }
        std::tuple<Tensor, Tensor> grid_sampler_3d_backward_fn(const Tensor &grad_output, const Tensor &input,
                                                               const Tensor &grid, int64_t interp, int64_t padding,
                                                               bool align_corners, std::array<bool, 2> mask)
        {
            return grid_sampler_backward_impl(grad_output, input, grid, interp, padding, align_corners, mask, 3);
        }
    }

    TORCH_LIBRARY_IMPL(aten, PrivateUse1, m)
    {
        m.impl("grid_sampler_2d", TORCH_FN(grid_sampler_2d_fn));
        m.impl("grid_sampler_3d", TORCH_FN(grid_sampler_3d_fn));
        m.impl("grid_sampler_2d_backward", TORCH_FN(grid_sampler_2d_backward_fn));
        m.impl("grid_sampler_3d_backward", TORCH_FN(grid_sampler_3d_backward_fn));
    }
}
