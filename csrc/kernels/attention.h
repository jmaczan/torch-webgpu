#pragma once
// Fused (flash) scaled-dot-product attention forward.

#include <ATen/ATen.h>

#include <optional>

namespace torch_webgpu::kernels
{
    // Largest head dimension the fused kernel handles.
    constexpr int64_t kAttentionMaxHeadDim = 128;

    // Whether the device's shared memory fits the kernel for head dims d, dv.
    bool attention_fits(int64_t d, int64_t dv);

    // q [B, Hq, Lq, D], k [B, Hk, Lk, D], v [B, Hk, Lk, Dv] (any strides;
    // Hq % Hk == 0, query head h reads kv head h / (Hq / Hk)); bias
    // broadcastable to [B, Hq, Lq, Lk] (additive) or undefined.
    // out [B, Hq, Lq, Dv], lse [B, Hq, Lq] (float32): log-sum-exp of the
    // scaled, masked scores (-inf for fully masked rows, whose output is 0).
    // Causal masking is top-left aligned (key j visible to query i iff j <= i).
    void run_attention(const at::Tensor &q, const at::Tensor &k, const at::Tensor &v, const at::Tensor &bias,
                       bool causal, double scale, const at::Tensor &out, const at::Tensor &lse);
}
