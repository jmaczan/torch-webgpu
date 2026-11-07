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
        int scalar_slot = -1;                // cpu scalars: scalar slot
    };

    struct IterBinding
    {
        WGPUBuffer handle = nullptr;
        wgpu::Buffer buffer;
        bool written = false;
        bool atomic = false;
        // Bound window [offset, offset + size) of the buffer: the bytes the
        // operands sharing it access (so tensors larger than the device's
        // maxStorageBufferBindingSize work as long as each dispatch's range fits).
        uint64_t offset = 0;
        uint64_t size = 0;
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
        std::vector<uint32_t> scalar_words; // two words per slot
        std::vector<CT> scalar_cts;          // one per slot
        int extra_words = 0; // user words appended to Params as `extra: array<vec4<u32>, N>`

        // `scalars` become s0, s1, ... (CPU-scalar operands are appended
        // after them). atomic_outputs forces output bindings to atomic<u32>.
        // cpu_scalar_cts[j] (if given) is the compute type CPU-scalar input j
        // is encoded in, so no conversion is needed in the shader.
        // extra_extent[k] (bytes) widens operand k's accessed range beyond the
        // iterator's own (e.g. the indexed dims of a gather source);
        // range_shift[k] (bytes) moves its start (and its element offset `oK`),
        // e.g. to the first row of an index chunk.
        static IterLayout build(at::TensorIteratorBase &iter, const std::vector<std::pair<c10::Scalar, CT>> &scalars,
                                bool atomic_outputs = false, const std::vector<CT> &cpu_scalar_cts = {},
                                const std::vector<int64_t> &extra_extent = {},
                                const std::vector<int64_t> &range_shift = {});

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

    // Largest per-buffer window (bytes) a dispatch over `iter` binds.
    uint64_t max_binding_window(at::TensorIteratorBase &iter, const std::vector<int64_t> &extra_extent = {},
                                const std::vector<int64_t> &range_shift = {});
    // Whether `iter` can run as one dispatch: 32-bit indexable and every
    // binding window within maxStorageBufferBindingSize.
    bool iter_fits_device(at::TensorIteratorBase &iter, const std::vector<int64_t> &extra_extent = {});

    // Calls f on sub-iterators that each fit the device (splitting the
    // largest dimension as needed), covering the whole iteration.
    template <class F>
    void for_each_fitting(at::TensorIteratorBase &iter, F &&f, const std::vector<int64_t> &extra_extent = {})
    {
        if (iter_fits_device(iter, extra_extent))
        {
            f(iter);
            return;
        }
        std::vector<std::unique_ptr<at::TensorIterator>> stack;
        stack.emplace_back(new at::TensorIterator(iter));
        while (!stack.empty())
        {
            auto it = std::move(stack.back());
            stack.pop_back();
            if (iter_fits_device(*it, extra_extent))
            {
                f(*it);
                continue;
            }
            TORCH_CHECK(it->numel() > 1 && it->ndim() > 0,
                        "torch_webgpu: an operand's accessed range exceeds the device's maxStorageBufferBindingSize");
            auto first = it->split(it->get_dim_to_split());
            stack.push_back(std::move(it));
            stack.push_back(std::move(first));
        }
    }
}
