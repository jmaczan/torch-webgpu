#pragma once
// Direct (thread per output element) kernels for 2-d pooling and
// upsampling over [N, C, H, W] tensors of any strides, computing in f32 and
// following ATen's CPU kernels (window bounds, NaN rules, source-index math).

#include <ATen/ATen.h>

#include <optional>

namespace torch_webgpu::kernels
{
    struct Pool2dParams
    {
        int64_t kh, kw, sh, sw, ph, pw, dh, dw;
    };

    // out [N, C, oh, ow]; indices (int64, flat h * W + w) may be undefined.
    void run_max_pool2d(const at::Tensor &input, const at::Tensor &out, const at::Tensor &indices, const Pool2dParams &p);
    void run_avg_pool2d(const at::Tensor &input, const at::Tensor &out, const Pool2dParams &p, bool count_include_pad,
                        std::optional<int64_t> divisor_override);

    enum class Upsample
    {
        Nearest,
        NearestExact,
        Bilinear,
    };
    // scale_h / scale_w: the source-index scale factors (input / output, or the
    // align_corners ratio), already computed as CPU does in float.
    void run_upsample2d(const at::Tensor &input, const at::Tensor &out, Upsample mode, bool align_corners, float scale_h,
                        float scale_w);
}
