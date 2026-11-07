#include "core/binding_set.h"

#include "core/context.h"

#include <c10/util/Exception.h>

namespace torch_webgpu::core
{
    uint64_t max_binding_size() { return Context::get().limits().maxStorageBufferBindingSize; }
    uint64_t binding_offset_alignment()
    {
        static const uint64_t a = std::max<uint64_t>(Context::get().limits().minStorageBufferOffsetAlignment, 4);
        return a;
    }

    std::pair<uint64_t, uint64_t> tensor_byte_range(const at::TensorBase &t, uint64_t extra_bytes)
    {
        const uint64_t lo = byte_offset_of(t);
        if (t.numel() == 0)
            return {lo, lo};
        uint64_t ext = t.element_size();
        for (int64_t d = 0; d < t.dim(); ++d)
        {
            TORCH_INTERNAL_ASSERT(t.stride(d) >= 0, "torch_webgpu: negative strides are not supported");
            ext += static_cast<uint64_t>(t.size(d) - 1) * static_cast<uint64_t>(t.stride(d)) * t.element_size();
        }
        return {lo, lo + ext + extra_bytes};
    }

    Window window_of(const at::TensorBase &t, uint64_t extra_bytes)
    {
        BindingSet bs;
        int r = bs.add(t, false, false, extra_bytes);
        bs.finalize();
        return {bs.buffers()[0], bs.offset(r)};
    }

    bool windows_fit(std::initializer_list<std::reference_wrapper<const at::TensorBase>> ts)
    {
        const uint64_t al = binding_offset_alignment(), limit = max_binding_size();
        for (const at::TensorBase &t : ts)
        {
            if (!t.defined() || t.numel() == 0)
                continue;
            auto [lo, hi] = tensor_byte_range(t);
            if ((hi + 3) / 4 * 4 - lo / al * al > limit)
                return false;
        }
        return true;
    }

    int BindingSet::add(const at::TensorBase &t, bool written, bool atomic, uint64_t extra_bytes)
    {
        TORCH_INTERNAL_ASSERT(!finalized_);
        const wgpu::Buffer &buf = buffer_of(t);
        auto [lo, hi] = tensor_byte_range(t, extra_bytes);
        int b = -1;
        for (size_t j = 0; j < bindings_.size(); ++j)
            if (bindings_[j].buffer.Get() == buf.Get())
                b = static_cast<int>(j);
        if (b < 0)
        {
            b = static_cast<int>(bindings_.size());
            Binding nb;
            nb.buffer = buf;
            nb.lo = lo;
            nb.hi = hi;
            bindings_.push_back(nb);
        }
        else
        {
            bindings_[b].lo = std::min(bindings_[b].lo, lo);
            bindings_[b].hi = std::max(bindings_[b].hi, hi);
        }
        bindings_[b].written |= written;
        bindings_[b].atomic |= atomic;
        regs_.push_back({b, lo, static_cast<uint64_t>(t.element_size())});
        return static_cast<int>(regs_.size() - 1);
    }

    uint64_t BindingSet::max_window() const
    {
        const uint64_t al = binding_offset_alignment();
        uint64_t m = 0;
        for (const auto &b : bindings_)
            m = std::max(m, (b.hi + 3) / 4 * 4 - b.lo / al * al);
        return m;
    }

    void BindingSet::finalize()
    {
        const uint64_t al = binding_offset_alignment(), limit = max_binding_size();
        for (auto &b : bindings_)
        {
            b.offset = b.lo / al * al;
            b.size = std::max<uint64_t>((b.hi + 3) / 4 * 4 - b.offset, 4);
            TORCH_CHECK(b.size <= limit, "torch_webgpu: a kernel operand's accessed range (", b.size,
                        " bytes) exceeds this device's maxStorageBufferBindingSize (", limit, " bytes)");
        }
        finalized_ = true;
    }

    uint32_t BindingSet::offset(int reg) const
    {
        TORCH_INTERNAL_ASSERT(finalized_);
        const auto &r = regs_[reg];
        const uint64_t rel = r.byte_lo - bindings_[r.binding].offset;
        TORCH_INTERNAL_ASSERT(rel % r.esize == 0);
        const uint64_t e = rel / r.esize;
        TORCH_CHECK(e <= UINT32_MAX, "torch_webgpu: element offset exceeds 32-bit indexing");
        return static_cast<uint32_t>(e);
    }

    std::vector<BufferBinding> BindingSet::buffers() const
    {
        TORCH_INTERNAL_ASSERT(finalized_);
        std::vector<BufferBinding> out;
        for (const auto &b : bindings_)
            out.push_back({b.buffer, b.offset, b.size});
        return out;
    }
}
