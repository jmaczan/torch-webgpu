#include "kernels/iter_layout.h"

#include "core/allocator.h"
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

    IterLayout IterLayout::build(at::TensorIteratorBase &iter, const std::vector<std::pair<c10::Scalar, CT>> &scalars,
                                 bool atomic_outputs, const std::vector<CT> &cpu_scalar_cts)
    {
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
            // 32-bit sub-iterators).
            int64_t byte_off = static_cast<int64_t>(reinterpret_cast<uintptr_t>(iter.data_ptr(k)) - alloc->va);
            TORCH_INTERNAL_ASSERT(byte_off % esize == 0);
            o.offset = checked_u32(byte_off / esize, "storage offset");
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
            }
            o.binding = b;
            if (o.is_output)
            {
                L.bindings[b].written = true;
                if (atomic_outputs || wgsl::is_subword(o.dtype))
                    L.bindings[b].atomic = true;
            }
        }
        return L;
    }

    void IterLayout::write_key(std::ostream &os) const
    {
        os << ndim << '|' << extra_words << '|';
        for (auto ct : scalar_cts)
            os << int(ct);
        os << '|';
        for (auto &o : ops)
            os << int(o.dtype) << (o.cpu_scalar ? 's' : 't') << o.binding << ':' << o.scalar_slot << ',';
        os << '|';
        for (auto &b : bindings)
            os << b.written << b.atomic;
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
        for (auto &b : bindings)
            bufs.push_back({b.buffer});
        return bufs;
    }
}
