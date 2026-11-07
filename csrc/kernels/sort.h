#pragma once
// Row-wise stable sort and binary search.

#include <ATen/ATen.h>

namespace torch_webgpu::kernels
{
    // Sorts each row of `keys` ([R, n], contiguous, dtype float32 / int32 /
    // int64) in place along with `indices` ([R, n] int32, holding 0..n-1).
    // Stable: equal keys keep their original order (also when descending).
    // NaNs are treated as larger than every number.
    void run_sort_rows(const at::Tensor &keys, const at::Tensor &indices, bool descending);

    // out[r, i] = number of elements of sorted row seq[r] that are < v (or
    // <= v when `right`), with values[r, i]; seq/values contiguous [R, n] /
    // [R, m] in a common compute dtype. out is int32 or int64 [R, m].
    void run_searchsorted(const at::Tensor &seq, const at::Tensor &values, const at::Tensor &out, bool right);
}
