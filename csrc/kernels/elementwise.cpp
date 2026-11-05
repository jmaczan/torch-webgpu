#include "kernels/elementwise.h"

#include "core/allocator.h"
#include "core/kernel.h"

#include <c10/util/Exception.h>

#include <cstring>
#include <sstream>

namespace torch_webgpu::kernels
{
    namespace
    {
        constexpr int kMaxDims = 8;
        constexpr uint32_t kWorkgroup = 256;

        struct Operand
        {
            bool is_output;
            bool cpu_scalar;
            c10::ScalarType dtype;
            int binding = -1;
            uint32_t offset = 0;                 // element offset
            uint32_t strides[kMaxDims] = {0};    // element strides
            int scalar_slot = -1;                // index into scalar words for cpu scalars
        };

        struct BindingInfo
        {
            WGPUBuffer handle;
            wgpu::Buffer buffer;
            bool written = false;
            bool atomic = false;
        };

        uint32_t to_u32(int64_t v, const char *what)
        {
            TORCH_CHECK(v >= 0 && v <= static_cast<int64_t>(UINT32_MAX), "torch_webgpu: ", what, " (", v,
                        ") exceeds 32-bit indexing");
            return static_cast<uint32_t>(v);
        }

        void launch_one(at::TensorIteratorBase &iter, const ElementwiseOp &op)
        {
            const int ntensors = iter.ntensors();
            const int noutputs = iter.noutputs();
            const int ndim = std::max<int>(iter.ndim(), 1);
            TORCH_CHECK(noutputs == 1, "torch_webgpu: elementwise kernels support a single output");
            TORCH_CHECK(iter.ndim() <= kMaxDims, "torch_webgpu: elementwise kernel with ", iter.ndim(),
                        " non-coalescable dims is not supported (max ", kMaxDims, ")");
            const uint32_t numel = to_u32(iter.numel(), "numel");

            std::vector<Operand> ops(ntensors);
            std::vector<BindingInfo> bindings;
            std::vector<uint32_t> scalar_words;
            std::vector<CT> scalar_cts;

            for (auto &[s, ct] : op.scalars)
            {
                scalar_words.push_back(scalar_word(s, ct));
                scalar_cts.push_back(ct);
            }

            for (int k = 0; k < ntensors; ++k)
            {
                auto &o = ops[k];
                o.is_output = k < noutputs;
                o.dtype = iter.dtype(k);
                o.cpu_scalar = !o.is_output && iter.is_cpu_scalar(k);
                if (o.cpu_scalar)
                {
                    // CPU 0-dim tensors (e.g. Python numbers) are passed as
                    // uniforms in their own compute type and converted below
                    // like any other input. Double scalars become f32 here,
                    // matching the f32 compute precision.
                    TORCH_CHECK(wgsl::is_compute_dtype(o.dtype) || o.dtype == c10::kDouble ||
                                    o.dtype == c10::kComplexDouble || o.dtype == c10::kComplexFloat,
                                "torch_webgpu: unsupported scalar dtype ", o.dtype);
                    if (!wgsl::is_compute_dtype(o.dtype))
                        o.dtype = c10::kFloat;
                    CT own = wgsl::compute_type(o.dtype);
                    o.scalar_slot = static_cast<int>(scalar_words.size());
                    scalar_words.push_back(scalar_word(at::Tensor(iter.tensor_base(k)).item(), own));
                    scalar_cts.push_back(own);
                    continue;
                }
                const auto &t = iter.tensor_base(k);
                const size_t esize = c10::elementSize(o.dtype);
                auto *alloc = core::allocation_of(t);
                TORCH_CHECK(alloc != nullptr, "torch_webgpu: elementwise operand ", k, " has no device storage");
                // data_ptr() is allocation + byte offset (also for 32-bit sub-iterators).
                int64_t byte_off = static_cast<const char *>(iter.data_ptr(k)) -
                                   static_cast<const char *>(t.storage().data_ptr().get());
                TORCH_INTERNAL_ASSERT(byte_off % static_cast<int64_t>(esize) == 0);
                o.offset = to_u32(byte_off / static_cast<int64_t>(esize), "storage offset");
                auto strides = iter.strides(k);
                for (int d = 0; d < iter.ndim(); ++d)
                {
                    TORCH_INTERNAL_ASSERT(strides[d] % static_cast<int64_t>(esize) == 0);
                    o.strides[d] = to_u32(strides[d] / static_cast<int64_t>(esize), "stride");
                }

                WGPUBuffer h = alloc->buffer.Get();
                int b = -1;
                for (size_t j = 0; j < bindings.size(); ++j)
                    if (bindings[j].handle == h)
                        b = static_cast<int>(j);
                if (b < 0)
                {
                    b = static_cast<int>(bindings.size());
                    bindings.push_back({h, alloc->buffer});
                }
                o.binding = b;
                if (o.is_output)
                {
                    bindings[b].written = true;
                    if (wgsl::is_subword(o.dtype))
                        bindings[b].atomic = true;
                }
            }

            // ---- cache key ---------------------------------------------------
            std::ostringstream key;
            key << "ew|" << op.name << '|' << op.code << '|' << op.helpers << '|' << int(op.in_ct) << int(op.out_ct)
                << '|' << op.raw_copy << '|' << ndim << '|';
            for (auto &c : op.input_cts)
                key << (c ? int(*c) : 9);
            key << '|';
            for (auto ct : scalar_cts)
                key << int(ct);
            key << '|';
            for (auto &o : ops)
                key << int(o.dtype) << (o.cpu_scalar ? 's' : 't') << o.binding << o.scalar_slot << ',';
            key << '|';
            for (auto &b : bindings)
                key << b.written << b.atomic;

            const int nscalar_vec = (static_cast<int>(scalar_words.size()) + 3) / 4;
            const int nops_vec = (ntensors + 3) / 4;

            auto make_source = [&]() -> std::string
            {
                std::ostringstream s;
                s << wgsl::prelude();
                s << "struct Params {\n  numel: u32,\n  row_stride: u32,\n  _p0: u32,\n  _p1: u32,\n"
                  << "  shape: array<vec4<u32>, 2>,\n"
                  << "  offsets: array<vec4<u32>, " << nops_vec << ">,\n"
                  << "  strides: array<vec4<u32>, " << 2 * ntensors << ">,\n";
                if (nscalar_vec > 0)
                    s << "  scalars: array<vec4<u32>, " << nscalar_vec << ">,\n";
                s << "};\n";
                for (size_t b = 0; b < bindings.size(); ++b)
                {
                    s << "@group(0) @binding(" << b << ") var<storage, "
                      << (bindings[b].written ? "read_write" : "read") << "> buf" << b << ": array<"
                      << (bindings[b].atomic ? "atomic<u32>" : "u32") << ">;\n";
                }
                s << "@group(0) @binding(" << bindings.size() << ") var<uniform> P: Params;\n";

                for (int k = 0; k < ntensors; ++k)
                {
                    auto &o = ops[k];
                    if (o.cpu_scalar)
                        continue;
                    std::string var = "buf" + std::to_string(o.binding);
                    bool atomic = bindings[o.binding].atomic;
                    size_t es = c10::elementSize(o.dtype);
                    if (op.raw_copy)
                        s << (o.is_output ? wgsl::raw_store_fn("st" + std::to_string(k), var, es)
                                          : wgsl::raw_load_fn("ld" + std::to_string(k), var, es, atomic));
                    else
                        s << (o.is_output ? wgsl::store_fn("st" + std::to_string(k), var, o.dtype)
                                          : wgsl::load_fn("ld" + std::to_string(k), var, o.dtype, atomic));
                }
                s << op.helpers << "\n";

                auto scalar_expr = [&](int slot)
                {
                    std::string w = "P.scalars[" + std::to_string(slot / 4) + "][" + std::to_string(slot % 4) + "]";
                    return wgsl::scalar_from_word(w, scalar_cts[slot]);
                };

                // The op function.
                const int ninputs = ntensors - noutputs;
                if (!op.raw_copy)
                {
                    s << "fn op_fn(";
                    for (int j = 0; j < ninputs; ++j)
                    {
                        CT ct = op.input_cts.size() > size_t(j) && op.input_cts[j] ? *op.input_cts[j] : op.in_ct;
                        s << (j ? ", " : "") << "x" << j << ": " << wgsl::type_name(ct);
                    }
                    s << ") -> " << wgsl::type_name(op.out_ct) << " {\n";
                    for (size_t j = 0; j < op.scalars.size(); ++j)
                        s << "    let s" << j << " = " << scalar_expr(static_cast<int>(j)) << ";\n";
                    if (op.code.find("return") != std::string::npos)
                        s << op.code << "\n}\n";
                    else
                        s << "    return " << op.code << ";\n}\n";
                }

                s << "@compute @workgroup_size(" << kWorkgroup << ")\n"
                  << "fn main(@builtin(global_invocation_id) gid: vec3<u32>) {\n"
                  << "    let i = gid.x + gid.y * P.row_stride;\n"
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
                if (op.raw_copy && ntensors == 1)
                {
                    // Raw fill: the value's bits are in the first scalar words.
                    size_t es = c10::elementSize(ops[0].dtype);
                    if (es <= 4)
                        s << "    st0(o0, P.scalars[0][0]);\n";
                    else if (es == 8)
                        s << "    st0(o0, vec2<u32>(P.scalars[0][0], P.scalars[0][1]));\n";
                    else
                        s << "    st0(o0, P.scalars[0]);\n";
                }
                else if (op.raw_copy)
                {
                    s << "    st0(o0, ld1(o1));\n";
                }
                else
                {
                    std::vector<std::string> args;
                    for (int k = noutputs; k < ntensors; ++k)
                    {
                        int j = k - noutputs;
                        CT want = op.input_cts.size() > size_t(j) && op.input_cts[j] ? *op.input_cts[j] : op.in_ct;
                        std::string v;
                        CT have = wgsl::compute_type(ops[k].dtype);
                        if (ops[k].cpu_scalar)
                            v = scalar_expr(ops[k].scalar_slot);
                        else
                            v = "ld" + std::to_string(k) + "(o" + std::to_string(k) + ")";
                        args.push_back(wgsl::convert(v, have, want));
                    }
                    s << "    let r = op_fn(";
                    for (size_t j = 0; j < args.size(); ++j)
                        s << (j ? ", " : "") << args[j];
                    s << ");\n";
                    s << "    st0(o0, " << wgsl::convert("r", op.out_ct, wgsl::compute_type(ops[0].dtype)) << ");\n";
                }
                s << "}\n";
                return s.str();
            };

            const auto &pipeline = core::get_pipeline(key.str(), make_source, op.name.c_str());

            // ---- params --------------------------------------------------------
            auto grid = core::grid_1d(numel, kWorkgroup);
            core::Params p;
            p.u32(numel).u32(grid.row_stride).u32(0).u32(0);
            for (int d = 0; d < 8; ++d)
                p.u32(d < iter.ndim() ? to_u32(iter.shape()[d], "size") : 1u);
            for (int k = 0; k < nops_vec * 4; ++k)
                p.u32(k < ntensors ? ops[k].offset : 0u);
            for (int k = 0; k < ntensors; ++k)
                for (int d = 0; d < 8; ++d)
                    p.u32(ops[k].strides[d]);
            for (int j = 0; j < nscalar_vec * 4; ++j)
                p.u32(j < static_cast<int>(scalar_words.size()) ? scalar_words[j] : 0u);

            std::vector<core::BufferBinding> bufs;
            for (auto &b : bindings)
                bufs.push_back({b.buffer});
            core::launch(pipeline, bufs, p.data(), p.size(), grid);
        }
    }

    CT common_ct(const at::TensorIteratorBase &iter)
    {
        return wgsl::compute_type(iter.common_dtype());
    }

    uint32_t scalar_word(const c10::Scalar &s, CT ct)
    {
        uint32_t w = 0;
        switch (ct)
        {
        case CT::F32:
        {
            float f = s.isComplex() ? static_cast<float>(s.toComplexDouble().real())
                      : s.isBoolean() ? (s.toBool() ? 1.0f : 0.0f)
                                      : static_cast<float>(s.toDouble());
            std::memcpy(&w, &f, 4);
            break;
        }
        case CT::I32:
        {
            int64_t v = s.isFloatingPoint() ? static_cast<int64_t>(s.toDouble())
                        : s.isBoolean()     ? (s.toBool() ? 1 : 0)
                                            : s.toLong();
            int32_t i = static_cast<int32_t>(v);
            std::memcpy(&w, &i, 4);
            break;
        }
        case CT::Bool:
            w = s.isFloatingPoint() ? (s.toDouble() != 0.0) : s.isBoolean() ? s.toBool() : (s.toLong() != 0);
            break;
        }
        return w;
    }

    void run_elementwise(at::TensorIteratorBase &iter, const ElementwiseOp &op)
    {
        if (iter.numel() == 0)
            return;
        if (!iter.can_use_32bit_indexing())
        {
            for (auto &sub : iter.with_32bit_indexing())
                launch_one(sub, op);
            return;
        }
        launch_one(iter, op);
    }
}
