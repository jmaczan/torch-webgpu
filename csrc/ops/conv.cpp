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
            if (output_mask[0])
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
            if (output_mask[1])
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
