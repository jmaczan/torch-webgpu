#pragma once
// Indexing kernels (gather / scatter) on top of TensorIterator, following
// the scheme of PyTorch's CUDA index kernels:
//
//   gather : iter = [out, src_restrided, idx_0, ..., idx_{n-1}]
//            out[i] = src[off_src(i) + sum_j norm(idx_j(i)) * stride_j]
//   scatter: iter = [dst_restrided, value, idx_0, ..., idx_{n-1}]
//            dst[off_dst(i) + sum_j norm(idx_j(i)) * stride_j] (op)= value(i)
//
// where "restrided" tensors have stride 0 in the indexed dimensions (the
// index contribution is added explicitly), norm() wraps negative indices and
// out-of-bounds indices raise an asynchronous IndexError (see
// core::Stream::index_error_flag). Strides are in elements of the indexed
// tensor.

#include <ATen/TensorIterator.h>

namespace torch_webgpu::kernels
{
    enum class ScatterMode
    {
        Assign,
        Add,
        Mul,
        Max,
        Min,
    };

    // wrap_negative: Python-style negative indices (advanced indexing); otherwise
    // negative indices are out of bounds (index_select, gather, scatter, ...).
    void run_gather(at::TensorIteratorBase &iter, at::IntArrayRef index_sizes, at::IntArrayRef index_strides,
                    bool wrap_negative = false);
    void run_scatter(at::TensorIteratorBase &iter, at::IntArrayRef index_sizes, at::IntArrayRef index_strides,
                     ScatterMode mode, bool wrap_negative = false);

    // out[k, :] = unravel(i) for the k-th non-zero of a contiguous flat mask
    // (`positions` = inclusive cumsum of the mask as int32).
    void run_nonzero(const at::Tensor &mask_flat, const at::Tensor &positions, const at::Tensor &out,
                     at::IntArrayRef sizes);
}
