// Convolutions (1-d/2-d/3-d, grouped, dilated, transposed) and their
// backward, reached through convolution_overrideable for PrivateUse1.
//
// Forward uses im2col: the padded input is viewed with as_strided as
// [N, G, C/G, K..., L...] patches (no copy), materialized once as a
// [N, G, C/G*K, L] matrix and multiplied with the [G, O/G, C/G*K] weight by
// the batched GEMM kernel. Transposed convolutions are expressed as regular
// convolutions over a zero-dilated input with flipped, transposed weights.

#include <ATen/ATen.h>
#include <torch/library.h>

#include "kernels/custom_gemm.h"
#include "kernels/wgsl.h"
#include "ops/common.h"

#include <numeric>

namespace torch_webgpu::ops
{
    using at::Tensor;

    namespace
    {
        using Ints = std::vector<int64_t>;

        Ints expand_param(at::IntArrayRef p, size_t n, const char *name)
        {
            if (p.size() == n)
                return p.vec();
            TORCH_CHECK(p.size() == 1, "torch_webgpu: expected ", name, " to have 1 or ", n, " elements, got ", p.size());
            return Ints(n, p[0]);
        }

        // im2col over a padded input x: [N, C, S...] -> [N, G, C/G * prod(K), prod(L)]
        Tensor im2col(const Tensor &x, const Ints &ksize, const Ints &stride, const Ints &dilation, int64_t groups,
                      const Ints &out_sz)
        {
            const int64_t nd = static_cast<int64_t>(ksize.size());
            const int64_t N = x.size(0), C = x.size(1);
            Ints shape{N, groups, C / groups};
            Ints strides{x.stride(0), x.stride(1) * (C / groups), x.stride(1)};
            for (int64_t d = 0; d < nd; ++d)
            {
                shape.push_back(ksize[d]);
                strides.push_back(x.stride(2 + d) * dilation[d]);
            }
            for (int64_t d = 0; d < nd; ++d)
            {
                shape.push_back(out_sz[d]);
                strides.push_back(x.stride(2 + d) * stride[d]);
            }
            Tensor patches = x.as_strided(shape, strides, x.storage_offset());
            const int64_t K = std::accumulate(ksize.begin(), ksize.end(), int64_t(1), std::multiplies<>());
            const int64_t L = std::accumulate(out_sz.begin(), out_sz.end(), int64_t(1), std::multiplies<>());
            return patches.reshape({N, groups, (C / groups) * K, L});
        }

        Tensor pad_spatial(const Tensor &x, const Ints &padding)
        {
            bool any = false;
            for (auto p : padding)
                any |= p != 0;
            if (!any)
                return x;
            Ints pads; // constant_pad_nd wants last dim first
            for (int64_t d = static_cast<int64_t>(padding.size()) - 1; d >= 0; --d)
            {
                pads.push_back(padding[d]);
                pads.push_back(padding[d]);
            }
            return at::constant_pad_nd(x, pads, 0);
        }

        // Regular (non-transposed) convolution, batched input [N, C, S...].
        Tensor conv_forward(const Tensor &input, const Tensor &weight, const std::optional<Tensor> &bias,
                            const Ints &stride, const Ints &padding, const Ints &dilation, int64_t groups)
        {
            const int64_t nd = weight.dim() - 2;
            const int64_t N = input.size(0), C = input.size(1), O = weight.size(0);
            TORCH_CHECK(C == weight.size(1) * groups, "Given groups=", groups, ", weight of size ", weight.sizes(),
                        ", expected input", input.sizes(), " to have ", weight.size(1) * groups,
                        " channels, but got ", C, " channels instead");
            Ints ksize(weight.sizes().begin() + 2, weight.sizes().end());
            Ints out_sz;
            for (int64_t d = 0; d < nd; ++d)
            {
                int64_t o = (input.size(2 + d) + 2 * padding[d] - dilation[d] * (ksize[d] - 1) - 1) / stride[d] + 1;
                TORCH_CHECK(o > 0, "Calculated output size is too small (input ", input.sizes(), ", kernel ", ksize, ")");
                out_sz.push_back(o);
            }
            Ints out_shape{N, O};
            out_shape.insert(out_shape.end(), out_sz.begin(), out_sz.end());
            if (N == 0)
                return at::empty(out_shape, input.options());

            Tensor xp = pad_spatial(input, padding);
            Tensor cols = im2col(xp, ksize, stride, dilation, groups, out_sz);   // [N, G, CK, L]
            Tensor w = weight.reshape({groups, O / groups, -1});                 // [G, O/G, CK]
            Tensor out = at::matmul(w.unsqueeze(0), cols);                       // [N, G, O/G, L]
            out = out.reshape(out_shape);
            if (bias.has_value() && bias->defined())
            {
                Ints bshape(out_shape.size(), 1);
                bshape[1] = O;
                out = out + bias->reshape(bshape);
            }
            return out;
        }

        // Zero-dilates x by `stride` and pads it so that a stride-1 convolution
        // with flipped weights computes the transposed convolution.
        Tensor transposed_input(const Tensor &x, const Ints &ksize, const Ints &stride, const Ints &padding,
                                const Ints &dilation, const Ints &output_padding)
        {
            const int64_t nd = static_cast<int64_t>(ksize.size());
            Ints shape{x.size(0), x.size(1)};
            for (int64_t d = 0; d < nd; ++d)
                shape.push_back((x.size(2 + d) - 1) * stride[d] + 1);
            Tensor up = x;
            bool strided = false;
            for (auto s : stride)
                strided |= s != 1;
            if (strided)
            {
                up = at::zeros(shape, x.options());
                Tensor view = up;
                for (int64_t d = 0; d < nd; ++d)
                    view = view.slice(2 + d, 0, shape[2 + d], stride[d]);
                view.copy_(x);
            }
            // Asymmetric padding: lo = dil*(k-1) - pad, hi = lo + output_padding.
            Ints pads;
            for (int64_t d = nd - 1; d >= 0; --d)
            {
                int64_t lo = dilation[d] * (ksize[d] - 1) - padding[d];
                pads.push_back(lo);
                pads.push_back(lo + output_padding[d]);
            }
            return at::constant_pad_nd(up, pads, 0); // negative pads crop
        }

        // weight [Cin, Cout/G, K...] -> flipped & transposed [Cout, Cin/G, K...]
        Tensor transposed_weight(const Tensor &weight, int64_t groups)
        {
            const int64_t nd = weight.dim() - 2;
            const int64_t Cin = weight.size(0), CoutG = weight.size(1);
            Ints flip_dims;
            for (int64_t d = 0; d < nd; ++d)
                flip_dims.push_back(2 + d);
            Tensor w = weight.flip(flip_dims);
            Ints ks(weight.sizes().begin() + 2, weight.sizes().end());
            Ints shape{groups, Cin / groups, CoutG};
            shape.insert(shape.end(), ks.begin(), ks.end());
            w = w.reshape(shape).transpose(1, 2); // [G, Cout/G, Cin/G, K...]
            Ints out_shape{groups * CoutG, Cin / groups};
            out_shape.insert(out_shape.end(), ks.begin(), ks.end());
            return w.reshape(out_shape);
        }

        Tensor conv_transposed_forward(const Tensor &input, const Tensor &weight, const std::optional<Tensor> &bias,
                                       const Ints &stride, const Ints &padding, const Ints &dilation,
                                       const Ints &output_padding, int64_t groups)
        {
            Ints ksize(weight.sizes().begin() + 2, weight.sizes().end());
            Tensor xin = transposed_input(input, ksize, stride, padding, dilation, output_padding);
            Ints ones(ksize.size(), 1), zeros(ksize.size(), 0);
            return conv_forward(xin, transposed_weight(weight, groups), bias, ones, zeros, dilation, groups);
        }

        // dW for a regular convolution: [O, C/G, K...]
        Tensor conv_weight_grad(const Tensor &input, const Tensor &grad_out, at::IntArrayRef wshape, const Ints &stride,
                                const Ints &padding, const Ints &dilation, int64_t groups)
        {
            const int64_t N = input.size(0), O = grad_out.size(1);
            Ints ksize(wshape.begin() + 2, wshape.end());
            Ints out_sz(grad_out.sizes().begin() + 2, grad_out.sizes().end());
            Tensor xp = pad_spatial(input, padding);
            Tensor cols = im2col(xp, ksize, stride, dilation, groups, out_sz);     // [N, G, CK, L]
            Tensor g = grad_out.reshape({N, groups, O / groups, -1});              // [N, G, O/G, L]
            Tensor dw = at::matmul(g, cols.transpose(-1, -2)).sum(0);              // [G, O/G, CK]
            return dw.reshape(wshape);
        }

        // ------------------------------------------------ implicit GEMM (2-d)
        // Convolutions as GEMMs whose patch operand is gathered on the fly
        // (bounds checks instead of padding): no im2col materialization.
        struct Conv2dGeom
        {
            int64_t N, G, Cg, Og, H, W, KH, KW, OH, OW, sh, sw, ph, pw, dh, dw;
        };
        Conv2dGeom geom2d(const Tensor &x, const Tensor &w, const Ints &stride, const Ints &padding, const Ints &dilation,
                          int64_t groups)
        {
            Conv2dGeom g{x.size(0), groups, x.size(1) / groups, w.size(0) / groups, x.size(2), x.size(3), w.size(2),
                         w.size(3), 0, 0, stride[0], stride[1], padding[0], padding[1], dilation[0], dilation[1]};
            g.OH = (g.H + 2 * g.ph - g.dh * (g.KH - 1) - 1) / g.sh + 1;
            g.OW = (g.W + 2 * g.pw - g.dw * (g.KW - 1) - 1) / g.sw + 1;
            return g;
        }
        bool implicit_ok(const Tensor &x, const Tensor &w, size_t nd, bool transposed)
        {
            return nd == 2 && !transposed && at::isFloatingType(x.scalar_type()) && wgsl::is_compute_dtype(x.scalar_type()) &&
                   w.scalar_type() == x.scalar_type();
        }
        // Parameter block shared by the three GEMMs (P.v[...]).
        std::vector<uint32_t> conv_params(const Conv2dGeom &g, const Tensor &x, const Tensor &w, const Tensor &go)
        {
            auto u = [](int64_t v) { return static_cast<uint32_t>(v); };
            std::vector<uint32_t> v{u(g.G), u(g.Cg), u(g.Og), u(g.H), u(g.W), u(g.KH), u(g.KW), u(g.OH), u(g.OW),
                                    u(g.sh), u(g.sw), u(g.ph), u(g.pw), u(g.dh), u(g.dw)};
            for (const Tensor *t : {&x, &w, &go})
                for (int d = 0; d < 4; ++d)
                    v.push_back(t->defined() ? u(t->stride(d)) : 0u);
            return v;
        }
        // WGSL names for the parameters above.
        const char *kConvLets =
            "    let G = pv(0u); let Cg = pv(1u); let Og = pv(2u); let H = pv(3u); let W = pv(4u);\n"
            "    let KH = pv(5u); let KW = pv(6u); let OH = pv(7u); let OW = pv(8u);\n"
            "    let sh = pv(9u); let sw = pv(10u); let ph = pv(11u); let pw = pv(12u); let dh = pv(13u); let dw = pv(14u);\n"
            "    let sx = vec4<u32>(pv(15u), pv(16u), pv(17u), pv(18u));\n"
            "    let swt = vec4<u32>(pv(19u), pv(20u), pv(21u), pv(22u));\n"
            "    let sg = vec4<u32>(pv(23u), pv(24u), pv(25u), pv(26u));\n";

        Tensor conv2d_implicit_forward(const Tensor &x, const Tensor &w, const std::optional<Tensor> &bias,
                                       const Ints &stride, const Ints &padding, const Ints &dilation, int64_t groups)
        {
            TORCH_CHECK(x.size(1) == w.size(1) * groups, "Given groups=", groups, ", weight of size ", w.sizes(),
                        ", expected input", x.sizes(), " to have ", w.size(1) * groups, " channels, but got ", x.size(1),
                        " channels instead");
            auto g = geom2d(x, w, stride, padding, dilation, groups);
            TORCH_CHECK(g.OH > 0 && g.OW > 0, "Calculated output size is too small (input ", x.sizes(), ", kernel ", w.sizes(), ")");
            Tensor out = at::empty({g.N, g.G * g.Og, g.OH, g.OW}, x.options());
            if (out.numel() == 0)
                return out;
            const bool has_bias = bias.has_value() && bias->defined();
            // GEMM: [Og] x [batch * OH * OW] per group (the batch is folded
            // into N so small feature maps still fill tiles), K = Cg * KH * KW.
            kernels::CustomGemm cg;
            cg.name = "conv2d_fwd";
            cg.key = has_bias ? "b" : "-";
            cg.M = g.Og;
            cg.N = g.N * g.OH * g.OW;
            cg.K = g.Cg * g.KH * g.KW;
            cg.Z = g.G;
            cg.operands = {{"x", x}, {"w", w}, {"out", out, true}};
            if (has_bias)
                cg.operands.push_back({"bias", *bias});
            cg.params = conv_params(g, x, w, Tensor());
            cg.load_a = std::string(kConvLets) +
                        "    let gi = z;\n    let c = k / (KH * KW);\n    let r = k % (KH * KW);\n"
                        "    return ld_w((gi * Og + m) * swt[0] + c * swt[1] + (r / KW) * swt[2] + (r % KW) * swt[3]);";
            cg.load_b = std::string(kConvLets) +
                        "    let gi = z;\n    let nn = n / (OH * OW);\n    let sp = n % (OH * OW);\n"
                        "    let c = k / (KH * KW);\n    let r = k % (KH * KW);\n"
                        "    let ih = i32((sp / OW) * sh + (r / KW) * dh) - i32(ph);\n"
                        "    let iw = i32((sp % OW) * sw + (r % KW) * dw) - i32(pw);\n"
                        "    if (ih < 0 || iw < 0 || ih >= i32(H) || iw >= i32(W)) { return 0.0; }\n"
                        "    return ld_x(nn * sx[0] + (gi * Cg + c) * sx[1] + u32(ih) * sx[2] + u32(iw) * sx[3]);";
            cg.store = std::string(kConvLets) +
                       "    let gi = z;\n    let nn = n / (OH * OW);\n    let sp = n % (OH * OW);\n    let co = gi * Og + m;\n" +
                       (has_bias ? "    st_out((nn * G * Og + co) * OH * OW + sp, v + ld_bias(co));"
                                 : "    st_out((nn * G * Og + co) * OH * OW + sp, v);");
            // Split-K when the tile grid alone can't fill the GPU (deep
            // layers: few output pixels, long reductions).
            const int64_t wgs = g.G * ((g.Og + 63) / 64) * ((cg.N + 63) / 64);
            const int64_t S = std::min<int64_t>(std::max<int64_t>(512 / std::max<int64_t>(wgs, 1), 1), cg.K / 256);
            if (S >= 2)
            {
                Tensor part = at::empty({S, g.G, g.Og, cg.N}, x.options().dtype(at::kFloat));
                // Partial products only: no output / bias bindings (unused
                // bindings would not be part of the shader's layout).
                kernels::CustomGemm sg = cg;
                sg.key += "|split";
                sg.operands = {{"x", x}, {"w", w}};
                sg.store = "";
                kernels::run_custom_gemm_split(sg, S, part);
                Tensor r = part.sum(0).view({g.G * g.Og, g.N, g.OH, g.OW}).permute({1, 0, 2, 3});
                if (has_bias)
                    r = r + bias->to(at::kFloat).view({1, -1, 1, 1});
                out.copy_(r);
            }
            else
                kernels::run_custom_gemm(cg);
            return out;
        }

        Tensor conv2d_implicit_grad_input(const Tensor &go, const Tensor &x, const Tensor &w, const Ints &stride,
                                          const Ints &padding, const Ints &dilation, int64_t groups)
        {
            auto g = geom2d(x, w, stride, padding, dilation, groups);
            Tensor gi = at::empty(x.sizes(), x.options().memory_format(at::MemoryFormat::Contiguous));
            if (gi.numel() == 0)
                return gi;
            // [Cg] x [batch * H * W] per group (batch folded into N).
            kernels::CustomGemm cg;
            cg.name = "conv2d_bwd_in";
            cg.M = g.Cg;
            cg.N = g.N * g.H * g.W;
            cg.K = g.Og * g.KH * g.KW;
            cg.Z = g.G;
            cg.operands = {{"go", go}, {"w", w}, {"gi", gi, true}};
            cg.params = conv_params(g, Tensor(), w, go);
            cg.load_a = std::string(kConvLets) +
                        "    let gz = z;\n    let co = k / (KH * KW);\n    let r = k % (KH * KW);\n"
                        "    return ld_w((gz * Og + co) * swt[0] + m * swt[1] + (r / KW) * swt[2] + (r % KW) * swt[3]);";
            cg.load_b = std::string(kConvLets) +
                        "    let gz = z;\n    let nn = n / (H * W);\n    let sp = n % (H * W);\n"
                        "    let co = k / (KH * KW);\n    let r = k % (KH * KW);\n"
                        "    let th = i32(sp / W + ph) - i32((r / KW) * dh);\n"
                        "    let tw = i32(sp % W + pw) - i32((r % KW) * dw);\n"
                        "    if (th < 0 || tw < 0) { return 0.0; }\n"
                        "    if (u32(th) % sh != 0u || u32(tw) % sw != 0u) { return 0.0; }\n"
                        "    let oh = u32(th) / sh;\n    let ow = u32(tw) / sw;\n"
                        "    if (oh >= OH || ow >= OW) { return 0.0; }\n"
                        "    return ld_go(nn * sg[0] + (gz * Og + co) * sg[1] + oh * sg[2] + ow * sg[3]);";
            cg.store = std::string(kConvLets) +
                       "    let gz = z;\n    let nn = n / (H * W);\n    let sp = n % (H * W);\n"
                       "    st_gi((nn * G * Cg + gz * Cg + m) * H * W + sp, v);";
            const int64_t wgs = g.G * ((g.Cg + 63) / 64) * ((cg.N + 63) / 64);
            const int64_t S = std::min<int64_t>(std::max<int64_t>(512 / std::max<int64_t>(wgs, 1), 1), cg.K / 256);
            if (S >= 2)
            {
                Tensor part = at::empty({S, g.G, g.Cg, cg.N}, x.options().dtype(at::kFloat));
                kernels::CustomGemm sgm = cg;
                sgm.key += "|split";
                sgm.operands = {{"go", go}, {"w", w}};
                sgm.store = "";
                kernels::run_custom_gemm_split(sgm, S, part);
                gi.copy_(part.sum(0).view({g.G * g.Cg, g.N, g.H, g.W}).permute({1, 0, 2, 3}));
            }
            else
                kernels::run_custom_gemm(cg);
            return gi;
        }

        Tensor conv2d_implicit_grad_weight(const Tensor &go, const Tensor &x, const Tensor &w, const Ints &stride,
                                           const Ints &padding, const Ints &dilation, int64_t groups)
        {
            auto g = geom2d(x, w, stride, padding, dilation, groups);
            Tensor gw = at::empty(w.sizes(), w.options().memory_format(at::MemoryFormat::Contiguous));
            if (gw.numel() == 0)
                return gw;
            kernels::CustomGemm cg;
            cg.name = "conv2d_bwd_w";
            cg.M = g.Og;
            cg.N = g.Cg * g.KH * g.KW;
            cg.K = g.N * g.OH * g.OW;
            cg.Z = g.G;
            cg.b_n_fast = false; // consecutive threads along (n, oh, ow): contiguous input rows
            cg.operands = {{"go", go}, {"x", x}, {"gw", gw, true}};
            cg.params = conv_params(g, x, Tensor(), go);
            cg.load_a = std::string(kConvLets) +
                        "    let nn = k / (OH * OW);\n    let r = k % (OH * OW);\n"
                        "    return ld_go(nn * sg[0] + (z * Og + m) * sg[1] + (r / OW) * sg[2] + (r % OW) * sg[3]);";
            cg.load_b = std::string(kConvLets) +
                        "    let nn = k / (OH * OW);\n    let r = k % (OH * OW);\n"
                        "    let ci = n / (KH * KW);\n    let q = n % (KH * KW);\n"
                        "    let ih = i32((r / OW) * sh + (q / KW) * dh) - i32(ph);\n"
                        "    let iw = i32((r % OW) * sw + (q % KW) * dw) - i32(pw);\n"
                        "    if (ih < 0 || iw < 0 || ih >= i32(H) || iw >= i32(W)) { return 0.0; }\n"
                        "    return ld_x(nn * sx[0] + (z * Cg + ci) * sx[1] + u32(ih) * sx[2] + u32(iw) * sx[3]);";
            cg.store = std::string(kConvLets) + "    st_gw((z * Og + m) * (Cg * KH * KW) + n, v);";
            // The reduction (N * OH * OW) is long and the output small: split K.
            const int64_t wgs = g.G * ((g.Og + 63) / 64) * ((cg.N + 63) / 64);
            const int64_t S = std::min<int64_t>(std::max<int64_t>(512 / std::max<int64_t>(wgs, 1), 1), cg.K / 256);
            if (S >= 2)
            {
                Tensor part = at::empty({S, g.G, g.Og, cg.N}, x.options().dtype(at::kFloat));
                kernels::run_custom_gemm_split(cg, S, part);
                gw.copy_(part.sum(0).view(gw.sizes()));
            }
            else
                kernels::run_custom_gemm(cg);
            return gw;
        }

        Tensor convolution_overrideable_fn(const Tensor &input, const Tensor &weight, const std::optional<Tensor> &bias,
                                           c10::SymIntArrayRef stride_s, c10::SymIntArrayRef padding_s,
                                           c10::SymIntArrayRef dilation_s, bool transposed,
                                           c10::SymIntArrayRef output_padding_s, c10::SymInt groups_s)
        {
            const int64_t groups = groups_s.expect_int();
            const size_t nd = static_cast<size_t>(weight.dim() - 2);
            TORCH_CHECK(nd >= 1 && nd <= 3, "torch_webgpu: only 1-d, 2-d and 3-d convolutions are supported");
            auto stride = expand_param(C10_AS_INTARRAYREF_SLOW(stride_s), nd, "stride");
            auto padding = expand_param(C10_AS_INTARRAYREF_SLOW(padding_s), nd, "padding");
            auto dilation = expand_param(C10_AS_INTARRAYREF_SLOW(dilation_s), nd, "dilation");
            auto output_padding = expand_param(C10_AS_INTARRAYREF_SLOW(output_padding_s), nd, "output_padding");
            const bool batched = input.dim() == weight.dim();
            Tensor x = batched ? input : input.unsqueeze(0);
            Tensor out = transposed ? conv_transposed_forward(x, weight, bias, stride, padding, dilation, output_padding, groups)
                         : implicit_ok(x, weight, nd, transposed)
                             ? conv2d_implicit_forward(x, weight, bias, stride, padding, dilation, groups)
                             : conv_forward(x, weight, bias, stride, padding, dilation, groups);
            return batched ? out : out.squeeze(0);
        }

        std::tuple<Tensor, Tensor, Tensor> convolution_backward_overrideable_fn(
            const Tensor &grad_output, const Tensor &input, const Tensor &weight, c10::SymIntArrayRef stride_s,
            c10::SymIntArrayRef padding_s, c10::SymIntArrayRef dilation_s, bool transposed,
            c10::SymIntArrayRef output_padding_s, c10::SymInt groups_s, std::array<bool, 3> output_mask)
        {
            const int64_t groups = groups_s.expect_int();
            const size_t nd = static_cast<size_t>(weight.dim() - 2);
            auto stride = expand_param(C10_AS_INTARRAYREF_SLOW(stride_s), nd, "stride");
            auto padding = expand_param(C10_AS_INTARRAYREF_SLOW(padding_s), nd, "padding");
            auto dilation = expand_param(C10_AS_INTARRAYREF_SLOW(dilation_s), nd, "dilation");
            auto output_padding = expand_param(C10_AS_INTARRAYREF_SLOW(output_padding_s), nd, "output_padding");
            const bool batched = input.dim() == weight.dim();
            Tensor x = batched ? input : input.unsqueeze(0);
            Tensor go = batched ? grad_output : grad_output.unsqueeze(0);
            Ints ksize(weight.sizes().begin() + 2, weight.sizes().end());

            Tensor gi, gw, gb;
            const bool implicit = implicit_ok(x, weight, nd, transposed) && go.scalar_type() == x.scalar_type();
            // (With few input channels the input-grad GEMM's M = C/G would
            // waste most of its 64-row tiles: the transposed-conv path is faster.)
            if (output_mask[0] && implicit && x.size(1) / groups >= 16)
            {
                gi = conv2d_implicit_grad_input(go, x, weight, stride, padding, dilation, groups);
                if (!batched)
                    gi = gi.squeeze(0);
            }
            else if (output_mask[0])
            {
                if (!transposed)
                {
                    // Input grad of a conv is a transposed conv; pick the
                    // output_padding that recovers the input size.
                    Ints op(nd);
                    for (size_t d = 0; d < nd; ++d)
                    {
                        int64_t base = (go.size(2 + d) - 1) * stride[d] - 2 * padding[d] + dilation[d] * (ksize[d] - 1) + 1;
                        op[d] = x.size(2 + d) - base;
                    }
                    gi = conv_transposed_forward(go, weight, std::nullopt, stride, padding, dilation, op, groups);
                }
                else
                {
                    gi = conv_forward(go, weight, std::nullopt, stride, padding, dilation, groups);
                    // With output_padding >= stride (possible when dilation >
                    // stride) the conv yields trailing positions past the input.
                    for (size_t d = 0; d < nd; ++d)
                        if (gi.size(2 + d) > x.size(2 + d))
                            gi = gi.narrow(2 + static_cast<int64_t>(d), 0, x.size(2 + d));
                }
                if (!batched)
                    gi = gi.squeeze(0);
            }
            if (output_mask[1] && implicit)
                gw = conv2d_implicit_grad_weight(go, x, weight, stride, padding, dilation, groups);
            else if (output_mask[1])
            {
                if (!transposed)
                    gw = conv_weight_grad(x, go, weight.sizes(), stride, padding, dilation, groups);
                else
                    gw = conv_weight_grad(go, x, weight.sizes(), stride, padding, dilation, groups);
            }
            if (output_mask[2])
            {
                Ints dims{0};
                for (size_t d = 0; d < nd; ++d)
                    dims.push_back(2 + static_cast<int64_t>(d));
                gb = go.sum(dims);
            }
            return {gi, gw, gb};
        }
    }

    TORCH_LIBRARY_IMPL(aten, PrivateUse1, m)
    {
        m.impl("convolution_overrideable", TORCH_FN(convolution_overrideable_fn));
        m.impl("convolution_backward_overrideable", TORCH_FN(convolution_backward_overrideable_fn));
    }
}
