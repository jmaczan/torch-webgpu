#pragma once
// Operand layout of a TensorIterator, shared by the elementwise and indexing
// kernel generators: per-operand element offsets/strides, de-duplicated
// buffer bindings (WebGPU forbids binding one buffer as both read-only and
// writable in a dispatch), CPU scalars passed as uniforms, and the WGSL
// pieces (Params struct, bindings, index prologue) that go with it.

#include <ATen/TensorIterator.h>
#include <c10/core/Scalar.h>
#include <webgpu/webgpu_cpp.h>

#include "core/kernel.h"
#include "kernels/wgsl.h"

#include <ostream>
#include <string>
#include <vector>

namespace torch_webgpu::kernels
{
    using wgsl::CT;

    constexpr int kIterMaxDims = 8;

    struct IterOperand
    {
        bool is_output = false;
        bool cpu_scalar = false;
        c10::ScalarType dtype = c10::kFloat;
        int binding = -1;
        uint32_t offset = 0;                 // element offset
        uint32_t strides[kIterMaxDims] = {}; // element strides
        int scalar_slot = -1;                // cpu scalars: slot in scalar_words
    };

    struct IterBinding
    {
        WGPUBuffer handle = nullptr;
        wgpu::Buffer buffer;
        bool written = false;
        bool atomic = false;
    };

    struct IterLayout
    {
        int ntensors = 0;
        int noutputs = 0;
        int ndim = 1; // >= 1
        uint32_t numel = 0;
        uint32_t shape[kIterMaxDims] = {};
        std::vector<IterOperand> ops;
        std::vector<IterBinding> bindings;
        std::vector<uint32_t> scalar_words;
        std::vector<CT> scalar_cts;
        int extra_words = 0; // user words appended to Params as `extra: array<vec4<u32>, N>`

        // `scalars` become s0, s1, ... (CPU-scalar operands are appended
        // after them). atomic_outputs forces output bindings to atomic<u32>.
        static IterLayout build(at::TensorIteratorBase &iter, const std::vector<std::pair<c10::Scalar, CT>> &scalars,
                                bool atomic_outputs = false);

        void write_key(std::ostream &os) const;
        // WGSL: Params struct + bindings + `var<uniform> P`.
        std::string declarations() const;
        // WGSL statements: computes `i` (linear id) and `oK` (element offset
        // of operand K); returns early when i >= numel.
        std::string prologue() const;
        std::string buffer_var(int k) const { return "buf" + std::to_string(ops[k].binding); }
        bool atomic(int k) const { return bindings[ops[k].binding].atomic; }
        std::string scalar_expr(int slot) const;

        void fill_params(core::Params &p, uint32_t row_stride, const std::vector<uint32_t> &extra = {}) const;
        std::vector<core::BufferBinding> buffers() const;
    };

    uint32_t checked_u32(int64_t v, const char *what);
}
