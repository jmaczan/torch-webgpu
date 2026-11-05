#include "kernels/elementwise.h"

#include "core/kernel.h"
#include "kernels/iter_layout.h"

#include <c10/util/Exception.h>

#include <cstring>
#include <sstream>

namespace torch_webgpu::kernels
{
    namespace
    {
        constexpr uint32_t kWorkgroup = 256;

        CT input_ct(const ElementwiseOp &op, int j)
        {
            return op.input_cts.size() > size_t(j) && op.input_cts[j] ? *op.input_cts[j] : op.in_ct;
        }

        void launch_one(at::TensorIteratorBase &iter, const ElementwiseOp &op)
        {
            TORCH_CHECK(iter.noutputs() == 1, "torch_webgpu: elementwise kernels support a single output");
            IterLayout L = IterLayout::build(iter, op.scalars);
            const int nt = L.ntensors, nout = L.noutputs;

            std::ostringstream key;
            key << "ew|" << op.name << '|' << op.code << '|' << std::hash<std::string>{}(op.helpers) << '|' << int(op.in_ct)
                << int(op.out_ct) << '|' << op.raw_copy << op.pass_index << '|';
            for (auto &c : op.input_cts)
                key << (c ? int(*c) : 9);
            key << '|';
            L.write_key(key);

            auto make_source = [&]() -> std::string
            {
                std::ostringstream s;
                s << wgsl::prelude() << L.declarations();
                for (int k = 0; k < nt; ++k)
                {
                    auto &o = L.ops[k];
                    if (o.cpu_scalar)
                        continue;
                    std::string var = L.buffer_var(k), kk = std::to_string(k);
                    size_t es = c10::elementSize(o.dtype);
                    if (op.raw_copy)
                        s << (o.is_output ? wgsl::raw_store_fn("st" + kk, var, es)
                                          : wgsl::raw_load_fn("ld" + kk, var, es, L.atomic(k)));
                    else
                        s << (o.is_output ? wgsl::store_fn("st" + kk, var, o.dtype)
                                          : wgsl::load_fn("ld" + kk, var, o.dtype, L.atomic(k)));
                }
                s << op.helpers << "\n";

                const int ninputs = nt - nout;
                if (!op.raw_copy)
                {
                    s << "fn op_fn(";
                    if (op.pass_index)
                        s << "idx: u32" << (ninputs ? ", " : "");
                    for (int j = 0; j < ninputs; ++j)
                        s << (j ? ", " : "") << "x" << j << ": " << wgsl::type_name(input_ct(op, j));
                    s << ") -> " << wgsl::type_name(op.out_ct) << " {\n";
                    for (size_t j = 0; j < op.scalars.size(); ++j)
                        s << "    let s" << j << " = " << L.scalar_expr(static_cast<int>(j)) << ";\n";
                    if (op.code.find("return") != std::string::npos)
                        s << op.code << "\n}\n";
                    else
                        s << "    return " << op.code << ";\n}\n";
                }

                s << "@compute @workgroup_size(" << kWorkgroup << ")\n"
                  << "fn main(@builtin(global_invocation_id) gid: vec3<u32>) {\n"
                  << L.prologue();
                if (op.raw_copy && nt == 1)
                {
                    // Raw fill: the value's bits are in the first scalar words.
                    size_t es = c10::elementSize(L.ops[0].dtype);
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
                    s << "    let r = op_fn(";
                    if (op.pass_index)
                        s << "i" << (nt > nout ? ", " : "");
                    for (int k = nout; k < nt; ++k)
                    {
                        CT have = wgsl::compute_type(L.ops[k].dtype);
                        std::string v = L.ops[k].cpu_scalar ? L.scalar_expr(L.ops[k].scalar_slot)
                                                            : "ld" + std::to_string(k) + "(o" + std::to_string(k) + ")";
                        s << (k > nout ? ", " : "") << wgsl::convert(v, have, input_ct(op, k - nout));
                    }
                    s << ");\n";
                    s << "    st0(o0, " << wgsl::convert("r", op.out_ct, wgsl::compute_type(L.ops[0].dtype)) << ");\n";
                }
                s << "}\n";
                return s.str();
            };

            const auto &pipeline = core::get_pipeline(key.str(), make_source, op.name.c_str());
            auto grid = core::grid_1d(L.numel, kWorkgroup);
            core::Params p;
            L.fill_params(p, grid.row_stride);
            core::launch(pipeline, L.buffers(), p.data(), p.size(), grid);
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
