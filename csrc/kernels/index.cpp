#include "kernels/index.h"

#include "core/allocator.h"
#include "core/kernel.h"
#include "core/stream.h"
#include "kernels/iter_layout.h"
#include "kernels/math.h"

#include <c10/util/Exception.h>

#include <sstream>

namespace torch_webgpu::kernels
{
    namespace
    {
        constexpr uint32_t kWG = 256;

        // Lane bits (already shifted down and masked) <-> compute value.
        std::string lane_decode(c10::ScalarType t, const std::string &b)
        {
            switch (t)
            {
            case c10::kFloat:
                return "bitcast<f32>(" + b + ")";
            case c10::kHalf:
                return "unpack2x16float(" + b + ")[0]";
            case c10::kBFloat16:
                return "bitcast<f32>(" + b + " << 16u)";
            case c10::kInt:
                return "bitcast<i32>(" + b + ")";
            case c10::kShort:
                return "(bitcast<i32>(" + b + " << 16u) >> 16u)";
            case c10::kChar:
                return "(bitcast<i32>(" + b + " << 24u) >> 24u)";
            case c10::kByte:
                return "i32(" + b + ")";
            case c10::kBool:
                return "(" + b + " != 0u)";
            default:
                TORCH_CHECK(false, "torch_webgpu: unsupported dtype for atomic update: ", t);
            }
        }
        std::string lane_encode(c10::ScalarType t, const std::string &v)
        {
            switch (t)
            {
            case c10::kFloat:
            case c10::kInt:
            case c10::kShort:
            case c10::kChar:
            case c10::kByte:
                return "bitcast<u32>(" + v + ")";
            case c10::kHalf:
                return "f32_to_f16_bits(" + v + ")";
            case c10::kBFloat16:
                return "f32_to_bf16_bits(" + v + ")";
            case c10::kBool:
                return "select(0u, 1u, " + v + ")";
            default:
                TORCH_CHECK(false, "torch_webgpu: unsupported dtype for atomic update: ", t);
            }
        }

        std::string combine_expr(ScatterMode mode, wgsl::CT ct)
        {
            using wgsl::CT;
            switch (mode)
            {
            case ScatterMode::Add:
                return ct == CT::Bool ? "cur || v" : "cur + v";
            case ScatterMode::Mul:
                return ct == CT::Bool ? "cur && v" : "cur * v";
            case ScatterMode::Max:
                return ct == CT::Bool ? "cur || v" : ct == CT::F32 ? "max_nan(cur, v)"
                                                                     : "max(cur, v)";
            case ScatterMode::Min:
                return ct == CT::Bool ? "cur && v" : ct == CT::F32 ? "min_nan(cur, v)"
                                                                     : "min(cur, v)";
            default:
                return "v";
            }
        }

        // Atomic read-modify-write of element i via compare-exchange on the
        // containing 32-bit word (works for every dtype up to 4 bytes).
        std::string rmw_fn(const std::string &var, c10::ScalarType t, ScatterMode mode)
        {
            const auto ct = wgsl::compute_type(t);
            const char *T = wgsl::type_name(ct);
            const size_t es = c10::elementSize(t);
            std::ostringstream s;
            s << "fn rmw0(i: u32, v: " << T << ") {\n";
            if (t == c10::kInt && mode == ScatterMode::Add)
            {
                s << "    atomicAdd(&" << var << "[i], bitcast<u32>(v));\n}\n";
                return s.str();
            }
            if (es == 4)
                s << "    let w = i;\n    let sh = 0u;\n    let mask = 0xffffffffu;\n";
            else if (es == 2)
                s << "    let w = i >> 1u;\n    let sh = (i & 1u) * 16u;\n    let mask = 0xffffu;\n";
            else
                s << "    let w = i >> 2u;\n    let sh = (i & 3u) * 8u;\n    let mask = 0xffu;\n";
            s << "    loop {\n"
              << "        let old = atomicLoad(&" << var << "[w]);\n"
              << "        let cur = " << lane_decode(t, "((old >> sh) & mask)") << ";\n"
              << "        let nv = " << combine_expr(mode, ct) << ";\n"
              << "        let nb = (old & ~(mask << sh)) | ((" << lane_encode(t, "nv") << " & mask) << sh);\n"
              << "        let r = atomicCompareExchangeWeak(&" << var << "[w], old, nb);\n"
              << "        if (r.exchanged) { break; }\n"
              << "    }\n}\n";
            return s.str();
        }

        void launch_index(at::TensorIteratorBase &iter, at::IntArrayRef sizes, at::IntArrayRef strides, bool scatter,
                          ScatterMode mode, bool wrap)
        {
            const int nidx = iter.ntensors() - 2;
            TORCH_CHECK(nidx >= 0 && nidx <= 16, "torch_webgpu: too many index tensors");
            TORCH_INTERNAL_ASSERT(static_cast<int>(sizes.size()) == nidx && static_cast<int>(strides.size()) == nidx);
            const bool rmw = scatter && mode != ScatterMode::Assign;
            IterLayout L = IterLayout::build(iter, {}, rmw);
            // The error flag is only bound when there are indices to check:
            // "auto" layouts drop bindings the shader never touches.
            int flag_b = -1;
            if (nidx > 0)
            {
                const auto &flag = core::Stream::get().index_error_flag();
                flag_b = static_cast<int>(L.bindings.size());
                L.bindings.push_back({flag.Get(), flag, true, true});
            }
            L.extra_words = 2 * nidx;

            const auto t0 = L.ops[0].dtype, t1 = L.ops[1].dtype;
            if (!rmw)
            {
                TORCH_CHECK(c10::elementSize(t0) == c10::elementSize(t1),
                            "torch_webgpu: index kernel expects matching element sizes");
            }
            else
            {
                TORCH_CHECK(c10::elementSize(t0) <= 4 && wgsl::is_compute_dtype(t0),
                            "torch_webgpu: accumulating index kernels support dtypes up to 32 bits, got ", t0);
            }

            std::ostringstream key;
            key << "idx|" << scatter << int(mode) << nidx << wrap << '|';
            L.write_key(key);

            auto make_source = [&]()
            {
                std::ostringstream s;
                s << wgsl::prelude() << math_wgsl() << L.declarations();
                const size_t es0 = c10::elementSize(t0), es1 = c10::elementSize(t1);
                if (!scatter)
                {
                    s << wgsl::raw_store_fn("st0", L.buffer_var(0), es0);
                    s << wgsl::raw_load_fn("ld1", L.buffer_var(1), es1, L.atomic(1));
                }
                else if (!rmw)
                {
                    s << wgsl::raw_store_fn("st0", L.buffer_var(0), es0);
                    s << wgsl::raw_load_fn("ld1", L.buffer_var(1), es1, L.atomic(1));
                }
                else
                {
                    s << rmw_fn(L.buffer_var(0), t0, mode);
                    s << wgsl::load_fn("ld1", L.buffer_var(1), t1, L.atomic(1));
                }
                for (int j = 0; j < nidx; ++j)
                {
                    int k = 2 + j;
                    TORCH_CHECK(at::isIntegralType(L.ops[k].dtype, false), "torch_webgpu: index tensors must be integral");
                    s << wgsl::load_fn("ld" + std::to_string(k), L.buffer_var(k), L.ops[k].dtype, L.atomic(k));
                }
                s << "@compute @workgroup_size(" << kWG << ")\n"
                  << "fn main(@builtin(global_invocation_id) gid: vec3<u32>) {\n"
                  << L.prologue()
                  << "    var ioff = 0u;\n";
                for (int j = 0; j < nidx; ++j)
                {
                    int k = 2 + j, js = nidx + j;
                    s << "    {\n"
                      << "        var v = ld" << k << "(o" << k << ");\n"
                      << "        let sz = i32(P.extra[" << j / 4 << "][" << j % 4 << "]);\n"
                      << (wrap ? "        if (v < 0) { v = v + sz; }\n" : "")
                      << "        if (v < 0 || v >= sz) { atomicOr(&buf" << flag_b << "[0], 1u); return; }\n"
                      << "        ioff += u32(v) * P.extra[" << js / 4 << "][" << js % 4 << "];\n"
                      << "    }\n";
                }
                if (!scatter)
                    s << "    st0(o0, ld1(o1 + ioff));\n";
                else if (!rmw)
                    s << "    st0(o0 + ioff, ld1(o1));\n";
                else
                    s << "    rmw0(o0 + ioff, "
                      << wgsl::convert("ld1(o1)", wgsl::compute_type(t1), wgsl::compute_type(t0)) << ");\n";
                s << "}\n";
                return s.str();
            };

            const auto &pipeline = core::get_pipeline(key.str(), make_source, scatter ? "index_scatter" : "index_gather");
            auto grid = core::grid_1d(L.numel, kWG);
            std::vector<uint32_t> extra;
            for (auto v : sizes)
                extra.push_back(checked_u32(v, "index size"));
            for (auto v : strides)
                extra.push_back(checked_u32(v, "index stride"));
            core::Params p;
            L.fill_params(p, grid.row_stride, extra);
            core::launch(pipeline, L.buffers(), p.data(), p.size(), grid);
        }

        void run(at::TensorIteratorBase &iter, at::IntArrayRef sizes, at::IntArrayRef strides, bool scatter, ScatterMode mode,
                 bool wrap)
        {
            if (iter.numel() == 0)
                return;
            if (!iter.can_use_32bit_indexing())
            {
                for (auto &sub : iter.with_32bit_indexing())
                    launch_index(sub, sizes, strides, scatter, mode, wrap);
                return;
            }
            launch_index(iter, sizes, strides, scatter, mode, wrap);
        }
    }

    void run_gather(at::TensorIteratorBase &iter, at::IntArrayRef sizes, at::IntArrayRef strides, bool wrap_negative)
    {
        run(iter, sizes, strides, false, ScatterMode::Assign, wrap_negative);
    }

    void run_scatter(at::TensorIteratorBase &iter, at::IntArrayRef sizes, at::IntArrayRef strides, ScatterMode mode,
                     bool wrap_negative)
    {
        run(iter, sizes, strides, true, mode, wrap_negative);
    }

    void run_nonzero(const at::Tensor &mask, const at::Tensor &positions, const at::Tensor &out, at::IntArrayRef sizes)
    {
        const int64_t n = mask.numel();
        const int nd = static_cast<int>(sizes.size());
        if (n == 0 || out.numel() == 0 || nd == 0)
            return;
        TORCH_INTERNAL_ASSERT(mask.is_contiguous() && positions.is_contiguous() && out.is_contiguous());
        TORCH_CHECK(nd <= 16, "torch_webgpu: nonzero supports at most 16 dims");
        std::ostringstream key;
        key << "nonzero|" << nd << int(mask.scalar_type()) << int(positions.scalar_type()) << int(out.scalar_type());
        auto make_source = [&]()
        {
            std::ostringstream s;
            s << wgsl::prelude();
            s << "struct Params { n: u32, row_stride: u32, nd: u32, _p: u32, off: vec4<u32>, sizes: array<vec4<u32>, 4>, };\n"
              << "@group(0) @binding(0) var<storage, read> bm: array<u32>;\n"
              << "@group(0) @binding(1) var<storage, read> bp: array<u32>;\n"
              << "@group(0) @binding(2) var<storage, read_write> bo: array<u32>;\n"
              << "@group(0) @binding(3) var<uniform> P: Params;\n";
            s << wgsl::load_fn("ld_m", "bm", mask.scalar_type(), false);
            s << wgsl::load_fn("ld_p", "bp", positions.scalar_type(), false);
            s << wgsl::store_fn("st_o", "bo", out.scalar_type());
            s << "@compute @workgroup_size(" << kWG << ")\n"
              << "fn main(@builtin(global_invocation_id) gid: vec3<u32>) {\n"
              << "    let i = gid.x + gid.y * P.row_stride;\n"
              << "    if (i >= P.n) { return; }\n"
              << "    if (!" << wgsl::convert("ld_m(P.off[0] + i)", wgsl::compute_type(mask.scalar_type()), wgsl::CT::Bool) << ") { return; }\n"
              << "    let pos = u32(ld_p(P.off[1] + i) - 1);\n"
              << "    var rem = i;\n";
            for (int d = nd - 1; d >= 0; --d)
                s << "    { let sz = P.sizes[" << d / 4 << "][" << d % 4 << "]; st_o(P.off[2] + pos * " << nd << "u + "
                  << d << "u, i32(rem % sz)); rem = rem / sz; }\n";
            s << "}\n";
            return s.str();
        };
        const auto &pipeline = core::get_pipeline(key.str(), make_source, "nonzero");
        auto grid = core::grid_1d(checked_u32(n, "numel"), kWG);
        core::Params p;
        p.u32(static_cast<uint32_t>(n)).u32(grid.row_stride).u32(nd).u32(0);
        p.u32(checked_u32(mask.storage_offset(), "offset")).u32(checked_u32(positions.storage_offset(), "offset"));
        p.u32(checked_u32(out.storage_offset(), "offset")).u32(0);
        for (int d = 0; d < 16; ++d)
            p.u32(d < nd ? checked_u32(sizes[d], "size") : 1u);
        core::launch(pipeline, {{core::buffer_of(mask)}, {core::buffer_of(positions)}, {core::buffer_of(out)}}, p.data(),
                     p.size(), grid);
    }
}
