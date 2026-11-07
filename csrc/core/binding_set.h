#pragma once
// The storage buffers a dispatch binds. WebGPU forbids binding one buffer
// twice when either binding is writable, so tensors sharing a buffer share a
// binding; each binding is a *window* of its buffer covering the union of the
// bytes its tensors access. Binding windows instead of whole buffers lets
// tensors larger than maxStorageBufferBindingSize take part in dispatches
// whose accessed ranges fit.
//
//   BindingSet bs;
//   int a = bs.add(x), o = bs.add(out, /*written=*/true);
//   bs.finalize();                       // computes windows (checks limits)
//   params: bs.offset(a), bs.offset(o)   // element offsets inside windows
//   launch(..., bs.buffers(), ...);       // shader binding index: bs.binding(a)

#include <ATen/core/TensorBase.h>

#include "core/allocator.h"
#include "core/kernel.h"

#include <vector>

namespace torch_webgpu::core
{
    // Bytes [lo, hi) of its allocation tensor t accesses (all strides >= 0).
    std::pair<uint64_t, uint64_t> tensor_byte_range(const at::TensorBase &t, uint64_t extra_bytes = 0);
    // The device's binding limits.
    uint64_t max_binding_size();
    uint64_t binding_offset_alignment();

    // A single tensor's window and its element offset inside it (for kernels
    // binding each tensor separately).
    struct Window
    {
        BufferBinding binding;
        uint32_t offset = 0;
    };
    Window window_of(const at::TensorBase &t, uint64_t extra_bytes = 0);
    // Whether every tensor's window fits the device's binding limit.
    bool windows_fit(std::initializer_list<std::reference_wrapper<const at::TensorBase>> ts);

    class BindingSet
    {
    public:
        struct Binding
        {
            wgpu::Buffer buffer;
            bool written = false;
            bool atomic = false;
            uint64_t lo = 0, hi = 0;         // union of accessed ranges
            uint64_t offset = 0, size = 0;   // window (after finalize)
        };

        // Registers tensor t (+ extra_bytes past its last element); returns a
        // registration id for offset() / binding().
        int add(const at::TensorBase &t, bool written = false, bool atomic = false, uint64_t extra_bytes = 0);
        // Computes the windows; throws if one exceeds the device limit.
        void finalize();
        // Largest window (bytes) the registered tensors need.
        uint64_t max_window() const;
        bool fits() const { return max_window() <= max_binding_size(); }

        int binding(int reg) const { return regs_[reg].binding; }
        uint32_t offset(int reg) const;
        const std::vector<Binding> &bindings() const { return bindings_; }
        std::vector<BufferBinding> buffers() const;

    private:
        struct Reg
        {
            int binding;
            uint64_t byte_lo;
            uint64_t esize;
        };
        std::vector<Binding> bindings_;
        std::vector<Reg> regs_;
        bool finalized_ = false;
    };
}
