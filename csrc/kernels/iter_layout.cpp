#include "kernels/iter_layout.h"

#include "core/allocator.h"
#include "core/context.h"
#include "kernels/elementwise.h"

#include <c10/util/Exception.h>

#include <sstream>

namespace torch_webgpu::kernels
{
    uint32_t checked_u32(int64_t v, const char *what)
    {
        TORCH_CHECK(v >= 0 && v <= static_cast<int64_t>(UINT32_MAX), "torch_webgpu: ", what, " (", v,
                    ") exceeds 32-bit indexing");
        return static_cast<uint32_t>(v);
    }

    namespace
    {
        // Byte range [lo, hi) operand k of `iter` accesses inside its allocation.
        std::pair<uint64_t, uint64_t> operand_range(at::TensorIteratorBase &iter, int k, const core::Allocation *alloc,
                                                    int64_t extra, int64_t shift = 0)
        {
            const uint64_t lo = reinterpret_cast<uintptr_t>(iter.data_ptr(k)) - alloc->va + static_cast<uint64_t>(shift);
            uint64_t ext = c10::elementSize(iter.dtype(k));
            if (iter.numel() == 0)
                return {lo, lo};
            auto strides = iter.strides(k);
            for (int d = 0; d < iter.ndim(); ++d)
                ext += static_cast<uint64_t>(iter.shape()[d] - 1) * static_cast<uint64_t>(strides[d]);
            return {lo, lo + ext + static_cast<uint64_t>(std::max<int64_t>(extra, 0))};
        }

        uint64_t offset_alignment()
        {
            static const uint64_t a = std::max<uint64_t>(core::Context::get().limits().minStorageBufferOffsetAlignment, 4);
            return a;
        }
    }

    uint64_t max_binding_window(at::TensorIteratorBase &iter, const std::vector<int64_t> &extra_extent,
                                const std::vector<int64_t> &range_shift)
    {
        // Per buffer: union of operand ranges, aligned like build() does.
        std::vector<std::tuple<WGPUBuffer, uint64_t, uint64_t>> wins;
        for (int k = 0; k < iter.ntensors(); ++k)
        {
            if (k >= iter.noutputs() && iter.is_cpu_scalar(k))
                continue;
            auto *alloc = core::allocation_of(iter.tensor_base(k));
            if (!alloc)
                continue;
            auto [lo, hi] = operand_range(iter, k, alloc, k < static_cast<int>(extra_extent.size()) ? extra_extent[k] : 0,
                                          k < static_cast<int>(range_shift.size()) ? range_shift[k] : 0);
            bool found = false;
            for (auto &[h, wl, wh] : wins)
                if (h == alloc->buffer.Get())
                {
                    wl = std::min(wl, lo);
                    wh = std::max(wh, hi);
                    found = true;
                }
            if (!found)
                wins.emplace_back(alloc->buffer.Get(), lo, hi);
        }
        uint64_t m = 0;
        const uint64_t al = offset_alignment();
        for (auto &[h, lo, hi] : wins)
            m = std::max(m, (hi + 3) / 4 * 4 - lo / al * al);
        return m;
    }

    bool iter_fits_device(at::TensorIteratorBase &iter, const std::vector<int64_t> &extra_extent)
    {
        return iter.can_use_32bit_indexing() &&
               max_binding_window(iter, extra_extent) <= core::Context::get().limits().maxStorageBufferBindingSize;
    }

    IterLayout IterLayout::build(at::TensorIteratorBase &iter, const std::vector<std::pair<c10::Scalar, CT>> &scalars,
                                 bool atomic_outputs, const std::vector<CT> &cpu_scalar_cts,
                                 const std::vector<int64_t> &extra_extent, const std::vector<int64_t> &range_shift)
    {
        std::vector<uint64_t> byte_offs(iter.ntensors(), 0);
        std::vector<std::pair<uint64_t, uint64_t>> ranges;
        IterLayout L;
        L.ntensors = iter.ntensors();
        L.noutputs = iter.noutputs();
        L.ndim = std::max<int>(iter.ndim(), 1);
        TORCH_CHECK(iter.ndim() <= kIterMaxDims, "torch_webgpu: kernel with ", iter.ndim(),
                    " non-coalescable dims is not supported (max ", kIterMaxDims, ")");
        L.numel = checked_u32(iter.numel(), "numel");
        for (int d = 0; d < kIterMaxDims; ++d)
            L.shape[d] = d < iter.ndim() ? checked_u32(iter.shape()[d], "size") : 1u;

        auto add_scalar = [&](const c10::Scalar &s, CT ct)
        {
            uint32_t w[2];
            wgsl::encode_scalar(s, ct, w);
            L.scalar_words.push_back(w[0]);
            L.scalar_words.push_back(w[1]);
            L.scalar_cts.push_back(ct);
            return static_cast<int>(L.scalar_cts.size() - 1);
        };
        for (auto &[s, ct] : scalars)
            add_scalar(s, ct);

        L.ops.resize(L.ntensors);
        for (int k = 0; k < L.ntensors; ++k)
        {
            auto &o = L.ops[k];
            o.is_output = k < L.noutputs;
            o.dtype = iter.dtype(k);
            o.cpu_scalar = !o.is_output && iter.is_cpu_scalar(k);
            if (o.cpu_scalar)
            {
                // CPU 0-dim tensors (e.g. Python numbers) are passed as
                // uniforms in their own compute type. Double/complex scalars
                // become f32, matching the f32 compute precision.
                TORCH_CHECK(wgsl::is_compute_dtype(o.dtype) || o.dtype == c10::kDouble ||
                                o.dtype == c10::kComplexDouble || o.dtype == c10::kComplexFloat,
                            "torch_webgpu: unsupported scalar dtype ", o.dtype);
                if (!wgsl::is_compute_dtype(o.dtype))
                    o.dtype = c10::kFloat;
                CT ct = wgsl::compute_type(o.dtype);
                const size_t j = static_cast<size_t>(k - L.noutputs);
                if (j < cpu_scalar_cts.size())
                {
                    // Encode directly in the requested type; record a dtype
                    // with that compute type so codegen sees no conversion.
                    ct = cpu_scalar_cts[j];
                    o.dtype = ct == CT::F32 ? c10::kFloat : ct == CT::I32 ? c10::kInt : ct == CT::I64 ? c10::kLong : c10::kBool;
                }
                o.scalar_slot = add_scalar(at::Tensor(iter.tensor_base(k)).item(), ct);
                continue;
            }
            const auto &t = iter.tensor_base(k);
            const int64_t esize = static_cast<int64_t>(c10::elementSize(o.dtype));
            auto *alloc = core::allocation_of(t);
            TORCH_CHECK(alloc != nullptr, "torch_webgpu: kernel operand ", k, " has no device storage");
            // data_ptr() is a virtual address inside the allocation (also for
            // sub-iterators). The element offset is made relative to the
            // binding window once all operands are known.
            auto range = operand_range(iter, k, alloc, k < static_cast<int>(extra_extent.size()) ? extra_extent[k] : 0,
                                       k < static_cast<int>(range_shift.size()) ? range_shift[k] : 0);
            byte_offs[k] = range.first;
            TORCH_INTERNAL_ASSERT(range.first % esize == 0);
            auto strides = iter.strides(k);
            for (int d = 0; d < iter.ndim(); ++d)
            {
                TORCH_INTERNAL_ASSERT(strides[d] % esize == 0);
                o.strides[d] = checked_u32(strides[d] / esize, "stride");
            }
            WGPUBuffer h = alloc->buffer.Get();
            int b = -1;
            for (size_t j = 0; j < L.bindings.size(); ++j)
                if (L.bindings[j].handle == h)
                    b = static_cast<int>(j);
            if (b < 0)
            {
                b = static_cast<int>(L.bindings.size());
                L.bindings.push_back({h, alloc->buffer});
                ranges.push_back(range);
            }
            else
            {
                ranges[b].first = std::min(ranges[b].first, range.first);
                ranges[b].second = std::max(ranges[b].second, range.second);
            }
            o.binding = b;
            if (o.is_output)
            {
                L.bindings[b].written = true;
                if (atomic_outputs || wgsl::is_subword(o.dtype))
                    L.bindings[b].atomic = true;
            }
        }
        // Binding windows, and element offsets relative to them.
        const uint64_t al = offset_alignment();
        const uint64_t max_bind = core::Context::get().limits().maxStorageBufferBindingSize;
        for (size_t b = 0; b < L.bindings.size(); ++b)
        {
            auto &bd = L.bindings[b];
            bd.offset = ranges[b].first / al * al;
            bd.size = std::max<uint64_t>((ranges[b].second + 3) / 4 * 4 - bd.offset, 4);
            TORCH_CHECK(bd.size <= max_bind, "torch_webgpu: an operand's accessed range (", bd.size,
                        " bytes) exceeds this device's maxStorageBufferBindingSize (", max_bind, " bytes)");
        }
        for (int k = 0; k < L.ntensors; ++k)
        {
            auto &o = L.ops[k];
            if (o.cpu_scalar)
                continue;
            const uint64_t esize = c10::elementSize(o.dtype);
            o.offset = checked_u32(static_cast<int64_t>((byte_offs[k] - L.bindings[o.binding].offset) / esize),
                                   "storage offset");
        }
        return L;
    }

    void IterLayout::write_key(std::string &key) const
    {
        key += char('0' + ndim);
        key += '|';
        key += std::to_string(extra_words);
        key += '|';
        for (auto ct : scalar_cts)
            key += char('0' + int(ct));
        key += '|';
        for (auto &o : ops)
        {
            key += std::to_string(int(o.dtype));
            key += o.cpu_scalar ? 's' : 't';
            key += char('0' + o.binding);
            key += ':';
            key += std::to_string(o.scalar_slot);
            key += ',';
        }
        key += '|';
        for (auto &b : bindings)
        {
            key += b.written ? 'w' : 'r';
            key += b.atomic ? 'a' : '-';
        }
    }

    std::string IterLayout::declarations() const
    {
        const int nscalar_vec = (static_cast<int>(scalar_words.size()) + 3) / 4;
        const int nops_vec = (ntensors + 3) / 4;
        std::ostringstream s;
        s << "struct Params {\n  numel: u32,\n  row_stride: u32,\n  _p0: u32,\n  _p1: u32,\n"
          << "  shape: array<vec4<u32>, 2>,\n"
          << "  offsets: array<vec4<u32>, " << nops_vec << ">,\n"
          << "  strides: array<vec4<u32>, " << 2 * ntensors << ">,\n";
        if (nscalar_vec > 0)
            s << "  scalars: array<vec4<u32>, " << nscalar_vec << ">,\n";
        if (extra_words > 0)
            s << "  extra: array<vec4<u32>, " << (extra_words + 3) / 4 << ">,\n";
        s << "};\n";
        for (size_t b = 0; b < bindings.size(); ++b)
            s << "@group(0) @binding(" << b << ") var<storage, " << (bindings[b].written ? "read_write" : "read")
              << "> buf" << b << ": array<" << (bindings[b].atomic ? "atomic<u32>" : "u32") << ">;\n";
        s << "@group(0) @binding(" << bindings.size() << ") var<uniform> P: Params;\n";
        return s.str();
    }

    std::string IterLayout::prologue() const
    {
        std::ostringstream s;
        s << "    let i = gid.x + gid.y * P.row_stride;\n"
          << "    if (i >= P.numel) { return; }\n";
        for (int k = 0; k < ntensors; ++k)
            if (!ops[k].cpu_scalar)
                s << "    var o" << k << " = P.offsets[" << k / 4 << "][" << k % 4 << "];\n";
        if (ndim == 1)
        {
            for (int k = 0; k < ntensors; ++k)
                if (!ops[k].cpu_scalar)
                    s << "    o" << k << " += i * P.strides[" << 2 * k << "][0];\n";
        }
        else
        {
            s << "    var rem = i;\n"
              << "    for (var d = 0u; d < " << ndim << "u; d++) {\n"
              << "        let sz = P.shape[d / 4u][d % 4u];\n"
              << "        let c = rem % sz;\n"
              << "        rem = rem / sz;\n";
            for (int k = 0; k < ntensors; ++k)
                if (!ops[k].cpu_scalar)
                    s << "        o" << k << " += c * P.strides[" << 2 * k << "u + d / 4u][d % 4u];\n";
            s << "    }\n";
        }
        return s.str();
    }

    std::string IterLayout::scalar_expr(int slot) const
    {
        return wgsl::scalar_slot_expr(slot, scalar_cts[slot]);
    }

    void IterLayout::fill_params(core::Params &p, uint32_t row_stride, const std::vector<uint32_t> &extra) const
    {
        const int nscalar_vec = (static_cast<int>(scalar_words.size()) + 3) / 4;
        const int nops_vec = (ntensors + 3) / 4;
        p.u32(numel).u32(row_stride).u32(0).u32(0);
        for (int d = 0; d < kIterMaxDims; ++d)
            p.u32(shape[d]);
        for (int k = 0; k < nops_vec * 4; ++k)
            p.u32(k < ntensors ? ops[k].offset : 0u);
        for (int k = 0; k < ntensors; ++k)
            for (int d = 0; d < kIterMaxDims; ++d)
                p.u32(ops[k].strides[d]);
        for (int j = 0; j < nscalar_vec * 4; ++j)
            p.u32(j < static_cast<int>(scalar_words.size()) ? scalar_words[j] : 0u);
        TORCH_INTERNAL_ASSERT(static_cast<int>(extra.size()) <= extra_words);
        for (int j = 0; j < (extra_words + 3) / 4 * 4; ++j)
            p.u32(j < static_cast<int>(extra.size()) ? extra[j] : 0u);
    }

    std::vector<core::BufferBinding> IterLayout::buffers() const
    {
        std::vector<core::BufferBinding> bufs;
        for (auto &b : bindings) // size 0: bindings added by hand, bound whole
            bufs.push_back({b.buffer, b.offset, b.size ? b.size : WGPU_WHOLE_SIZE});
        return bufs;
    }
}
