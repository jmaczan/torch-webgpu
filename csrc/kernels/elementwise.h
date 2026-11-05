#pragma once
// Generic elementwise kernels driven by at::TensorIterator.
//
// An ElementwiseOp is a WGSL expression over the iterator inputs, named
// x0, x1, ... (already converted to the op's compute type) and scalar
// parameters s0, s1, .... The engine generates a shader specialized for the
// operand dtypes, the number of (coalesced) dimensions and buffer aliasing,
// caches the pipeline and records the dispatch.

#include <ATen/TensorIterator.h>
#include <c10/core/Scalar.h>

#include "kernels/wgsl.h"

#include <optional>
#include <string>
#include <vector>

namespace torch_webgpu::kernels
{
    using wgsl::CT;

    struct ElementwiseOp
    {
        std::string name;
        // WGSL expression of type out_ct. If it contains "return", it is used
        // as a function body instead.
        std::string code;
        CT in_ct = CT::F32;
        CT out_ct = CT::F32;
        // Per-input compute type overrides (e.g. the bool condition of where).
        std::vector<std::optional<CT>> input_cts;
        // Scalar parameters s0, s1, ... with their types.
        std::vector<std::pair<c10::Scalar, CT>> scalars;
        // Extra module-scope WGSL (helper functions).
        std::string helpers;
        // Bit-exact copy (1 input, same element size); `code` is ignored.
        bool raw_copy = false;
        // Expose the linear iteration index to `code` as `idx: u32`. Only
        // meaningful for 1-D iterators (where it equals the element index).
        bool pass_index = false;
    };

    void run_elementwise(at::TensorIteratorBase &iter, const ElementwiseOp &op);

    // Convenience: compute type of the iterator's common dtype.
    CT common_ct(const at::TensorIteratorBase &iter);

    // Encodes a scalar as one 32-bit word of compute type `ct`.
    uint32_t scalar_word(const c10::Scalar &s, CT ct);
}
