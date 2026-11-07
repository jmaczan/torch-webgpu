#pragma once
// Batched GEMM:  out[b] = beta * bias[b] + alpha * (a[b] @ w[b])
//
// a: [B, M, K], w: [B, K, N], out: [B, M, N], bias broadcastable to out
// (stride-0 dims allowed). All operands may have arbitrary strides
// (transposed views need no copy). Accumulation is in f32 for floating
// dtypes and i32 for integral ones.

#include <ATen/ATen.h>
#include <c10/core/Scalar.h>

namespace torch_webgpu::kernels
{
    // 3-D operands. `bias` may be undefined (then beta is ignored).
    void run_matmul(const at::Tensor &a, const at::Tensor &w, const at::Tensor &out, const at::Tensor &bias,
                    const c10::Scalar &beta, const c10::Scalar &alpha);

    // Several linear layers on one input in a single dispatch (horizontal
    // fusion of decode GEMVs, e.g. q/k/v projections):
    //   out[:, N_0 + ... + N_{j-1} : ... + N_j] = x @ weights[j].T + biases[j]
    // x: [M, K] float32, weights[j]: [N_j, K] float32 contiguous, biases[j]
    // undefined or [N_j], out: [M, sum N_j] contiguous. Returns false (and
    // does nothing) when the shapes, layouts or device limits don't allow it.
    bool run_multi_gemv(const at::Tensor &x, const std::vector<at::Tensor> &weights,
                        const std::vector<at::Tensor> &biases, const at::Tensor &out);
}
