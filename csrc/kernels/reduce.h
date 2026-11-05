#pragma once
// Generic reductions over an arbitrary set of dimensions.
//
// A ReduceOp describes an accumulator in WGSL:
//   struct Acc { ... };                         (acc_struct)
//   fn acc_init() -> Acc                        (init: function body)
//   fn acc_step(a: Acc, x: IN, r: u32) -> Acc   (step: body; r = flat index
//                                                in the reduced dims, row-major)
//   fn acc_combine(a: Acc, b: Acc) -> Acc       (combine: body)
//   fn acc_out0(a: Acc, n: u32) -> OUT0         (out0: body; n = #reduced)
//   fn acc_out1(a: Acc, n: u32) -> OUT1         (optional second output)
// Scalars are available as s0, s1, ... in every function.
//
// Each output element is reduced by one workgroup (tree reduction in
// workgroup memory) or, for short reductions, by a single thread. Results are
// deterministic.

#include <ATen/ATen.h>

#include "kernels/wgsl.h"

#include <string>
#include <vector>

namespace torch_webgpu::kernels
{
    using wgsl::CT;

    struct ReduceOp
    {
        std::string name;
        std::string acc_struct;
        std::string init;
        std::string step;
        std::string combine;
        std::string out0;
        std::string out1;
        CT in_ct = CT::F32;
        CT out0_ct = CT::F32;
        CT out1_ct = CT::I32;
        std::vector<std::pair<c10::Scalar, CT>> scalars;
    };

    // Reduces `input` over `dims` (already wrapped; empty => all dims) into
    // `out0` (and `out1` if defined). Outputs must have the keepdim or
    // squeezed result shape.
    void run_reduce(const at::Tensor &input, at::IntArrayRef dims, const at::Tensor &out0,
                    const at::Tensor &out1, const ReduceOp &op);

    // Row-wise kernels (softmax family / normalization): `input` is viewed as
    // [outer..., dim] where the reduced dim may have any stride. The body runs
    // in a workgroup of 256 threads per row; see reduce.cpp for the contract.
    struct RowOp
    {
        std::string name;
        std::string body; // WGSL statements, see run_rows in reduce.cpp
        std::vector<std::pair<c10::Scalar, CT>> scalars;
        // Type of ld*/st*/wg_* values. Body helpers available:
        //   n (row length), row (row index), lid (0..255), T (alias of ct),
        //   ldK(j) -> T, stK(j, v: T), rstK(v: T) (per-row outputs),
        //   wg_sum(v) / wg_max(v) / wg_min(v) workgroup reductions,
        //   wg_scan(v) inclusive prefix sum over lid.
        CT ct = CT::F32;
    };
    // inputs/outputs all share the same shape (`dim` is the row dimension).
    // Extra per-row outputs (e.g. layer-norm mean/rstd) are passed in
    // row_outputs and indexed by row number (they must be contiguous).
    void run_rows(const std::vector<at::Tensor> &inputs, const std::vector<at::Tensor> &outputs,
                  const std::vector<at::Tensor> &row_outputs, int64_t dim, const RowOp &op);
}
