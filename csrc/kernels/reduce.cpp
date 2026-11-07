#include "kernels/reduce.h"

#include "core/allocator.h"
#include "core/binding_set.h"
#include "core/context.h"
#include "core/kernel.h"
#include "kernels/elementwise.h"
#include "kernels/math.h"

#include <c10/util/Exception.h>

#include <algorithm>
#include <sstream>

namespace torch_webgpu::kernels
{
    namespace
    {
        constexpr int kMaxDims = 8;
        constexpr uint32_t kWG = 256;

        uint32_t u32_of(int64_t v, const char *what)
        {
            TORCH_CHECK(v >= 0 && v <= static_cast<int64_t>(UINT32_MAX), "torch_webgpu: ", what, " (", v,
                        ") exceeds 32-bit indexing");
            return static_cast<uint32_t>(v);
        }

        struct KeptDim
        {
            int64_t size, in_stride, out0_stride, out1_stride;
        };
        struct RedDim
        {
            int64_t size, stride;
        };

        std::string scalar_lets(const std::vector<std::pair<c10::Scalar, CT>> &scalars)
        {
            std::string s;
            for (size_t j = 0; j < scalars.size(); ++j)
                s += "    let s" + std::to_string(j) + " = " +
                     wgsl::scalar_slot_expr(static_cast<int>(j), scalars[j].second) + ";\n";
            return s;
        }

        void put_scalars(core::Params &p, const std::vector<std::pair<c10::Scalar, CT>> &scalars, int nsv)
        {
            std::vector<uint32_t> words;
            for (auto &[sc, ct] : scalars)
            {
                uint32_t w[2];
                wgsl::encode_scalar(sc, ct, w);
                words.push_back(w[0]);
                words.push_back(w[1]);
            }
            for (int j = 0; j < nsv * 4; ++j)
                p.u32(j < static_cast<int>(words.size()) ? words[j] : 0u);
        }

        // Byte size of `struct Acc` in a storage array (0 if it isn't
        // host-shareable, e.g. holds a bool).
        int64_t acc_storage_size(const std::string &acc_struct)
        {
            const auto b = acc_struct.find("struct Acc");
            if (b == std::string::npos)
                return 0;
            const auto open = acc_struct.find('{', b), close = acc_struct.find('}', open);
            const std::string body = acc_struct.substr(open + 1, close - open - 1);
            int64_t off = 0, align = 4;
            size_t pos = 0;
            while ((pos = body.find(':', pos)) != std::string::npos)
            {
                size_t e = body.find(',', pos);
                std::string ty = body.substr(pos + 1, (e == std::string::npos ? body.size() : e) - pos - 1);
                ty.erase(0, ty.find_first_not_of(' '));
                ty.erase(ty.find_last_not_of(' ') + 1);
                int64_t sz;
                if (ty == "f32" || ty == "i32" || ty == "u32")
                    sz = 4;
                else if (ty == "i64" || ty == "vec2<u32>")
                    sz = 8;
                else
                    return 0;
                off = (off + sz - 1) / sz * sz;
                off += sz;
                align = std::max(align, sz);
                pos = e == std::string::npos ? body.size() : e;
            }
            return (off + align - 1) / align * align;
        }

        uint32_t elem_offset(const at::Tensor &t)
        {
            return u32_of(core::element_offset_of(t), "storage offset");
        }
    }

    void run_reduce(const at::Tensor &input, at::IntArrayRef dims_in, const at::Tensor &out0,
                    const at::Tensor &out1, const ReduceOp &op)
    {
        const int64_t ndim = input.dim();
        if (out0.numel() == 0)
            return;
        std::vector<bool> reduced(std::max<int64_t>(ndim, 1), false);
        if (dims_in.empty())
            std::fill(reduced.begin(), reduced.end(), true);
        for (auto d : dims_in)
            reduced[at::maybe_wrap_dim(d, std::max<int64_t>(ndim, 1))] = true;

        const bool keepdim = out0.dim() == ndim && ndim > 0;
        const bool has_out1 = out1.defined();

        // Too large to bind? Split along a kept dim (outputs independent)...
        const at::Tensor no_tensor;
        if (!core::windows_fit({input, out0, has_out1 ? out1 : no_tensor}))
        {
            int64_t best = -1;
            for (int64_t d = 0; d < ndim; ++d)
                if (!reduced[d] && input.size(d) > 1 &&
                    (best < 0 || input.size(d) * input.stride(d) > input.size(best) * input.stride(best)))
                    best = d;
            if (best >= 0)
            {
                int64_t od = best;
                if (!keepdim)
                    for (int64_t d = 0; d < best; ++d)
                        od -= reduced[d] ? 1 : 0;
                const int64_t n = input.size(best), h = n / 2;
                for (auto [s0, len] : {std::pair<int64_t, int64_t>{0, h}, {h, n - h}})
                    run_reduce(input.narrow(best, s0, len), dims_in, out0.narrow(od, s0, len),
                               has_out1 ? out1.narrow(od, s0, len) : out1, op);
                return;
            }
            // ... or, for ops whose outputs combine, reduce halves of the
            // largest reduced dim into partials and reduce those.
            TORCH_CHECK(op.output_combinable && !has_out1,
                        "torch_webgpu: '", op.name, "' over a tensor larger than this device's maxStorageBufferBindingSize (",
                        core::max_binding_size(), " bytes) is not supported");
            int64_t rd = -1;
            for (int64_t d = 0; d < ndim; ++d)
                if (reduced[d] && input.size(d) > 1 &&
                    (rd < 0 || input.size(d) * input.stride(d) > input.size(rd) * input.stride(rd)))
                    rd = d;
            TORCH_CHECK(rd >= 0, "torch_webgpu: reduction element exceeds the binding limit");
            const int64_t n = input.size(rd), h = n / 2;
            std::vector<at::Tensor> parts;
            for (auto [s0, len] : {std::pair<int64_t, int64_t>{0, h}, {h, n - h}})
            {
                at::Tensor part = at::empty_like(out0);
                run_reduce(input.narrow(rd, s0, len), dims_in, part, at::Tensor(), op);
                parts.push_back(part);
            }
            at::Tensor stacked = at::stack(parts, 0);
            ReduceOp combine = op;
            combine.in_ct = op.out0_ct;
            run_reduce(stacked, {0}, out0, at::Tensor(), combine);
            return;
        }

        std::vector<KeptDim> kept;
        std::vector<RedDim> red;
        int64_t out_pos = 0;
        for (int64_t d = 0; d < ndim; ++d)
        {
            if (reduced[d])
            {
                if (keepdim)
                    out_pos++;
                if (input.size(d) != 1)
                    red.push_back({input.size(d), input.stride(d)});
                continue;
            }
            int64_t s0 = out0.dim() ? out0.stride(out_pos) : 0;
            int64_t s1 = has_out1 && out1.dim() ? out1.stride(out_pos) : 0;
            out_pos++;
            if (input.size(d) != 1)
                kept.push_back({input.size(d), input.stride(d), s0, s1});
        }
        // Coalesce adjacent dims (row-major order is preserved).
        auto coalesce_kept = [](std::vector<KeptDim> &v)
        {
            std::vector<KeptDim> r;
            for (auto &k : v)
            {
                if (!r.empty())
                {
                    auto &p = r.back();
                    if (p.in_stride == k.in_stride * k.size && p.out0_stride == k.out0_stride * k.size &&
                        p.out1_stride == k.out1_stride * k.size)
                    {
                        p = {p.size * k.size, k.in_stride, k.out0_stride, k.out1_stride};
                        continue;
                    }
                }
                r.push_back(k);
            }
            v.swap(r);
        };
        auto coalesce_red = [](std::vector<RedDim> &v)
        {
            std::vector<RedDim> r;
            for (auto &k : v)
            {
                if (!r.empty() && r.back().stride == k.stride * k.size)
                {
                    r.back() = {r.back().size * k.size, k.stride};
                    continue;
                }
                r.push_back(k);
            }
            v.swap(r);
        };
        coalesce_kept(kept);
        coalesce_red(red);
        TORCH_CHECK(kept.size() <= kMaxDims && red.size() <= kMaxDims,
                    "torch_webgpu: reduction with too many non-contiguous dims");

        int64_t n_out = 1, n_red = 1;
        for (auto &k : kept)
            n_out *= k.size;
        for (auto &r : red)
            n_red *= r.size;
        TORCH_INTERNAL_ASSERT(n_out == out0.numel());
        const uint32_t NOUT = u32_of(n_out, "numel"), NRED = u32_of(n_red, "reduction size");
        const bool wg_mode = n_red > 32;
        const int NK = static_cast<int>(kept.size()), NR = static_cast<int>(red.size());

        auto in_t = input.scalar_type();
        auto o0_t = out0.scalar_type();
        auto o1_t = has_out1 ? out1.scalar_type() : c10::kInt;
        TORCH_CHECK(!has_out1 || core::buffer_of(out0).Get() != core::buffer_of(out1).Get(),
                    "torch_webgpu: reduction outputs must not alias");

        const int nsv = (2 * static_cast<int>(op.scalars.size()) + 3) / 4;
        const uint32_t maxd = core::Context::get().limits().maxComputeWorkgroupsPerDimension;

        // Kernel modes:
        //  0 thread per output, sequential over the reduction (short reductions)
        //  1 workgroup per output, 256 threads strided over the reduction
        //  2 split-workgroup stage 1: (output, slice) workgroups -> partial Accs
        //  3 column stage 1: thread per (output, slice), coalesced across outputs
        //  4 stage 2 combining partials, workgroup per output (after 2)
        //  5 stage 2 combining partials, thread per output (after 3)
        // Few outputs over long reductions (2) or reductions along a strided dim
        // with contiguous outputs (3) would otherwise starve or uncoalesce the GPU.
        const int64_t acc_bytes = acc_storage_size(op.acc_struct);
        const bool column = NK > 0 && kept.back().in_stride == 1 && NRED >= 16 && NOUT >= 256 && acc_bytes > 0;
        const bool split_wg = !column && wg_mode && NOUT < 1024 && NRED >= 8192 && acc_bytes > 0;
        int mode = wg_mode ? 1 : 0;
        uint32_t S = 1, chunk = NRED;
        if (column)
        {
            mode = 3;
            S = static_cast<uint32_t>(std::clamp<int64_t>((65536 + NOUT - 1) / NOUT, 1, std::max<int64_t>(NRED / 32, 1)));
            chunk = (NRED + S - 1) / S;
            S = (NRED + chunk - 1) / chunk;
        }
        else if (split_wg)
        {
            mode = 2;
            S = static_cast<uint32_t>(std::clamp<int64_t>((NRED + 4095) / 4096, 1, std::min<uint32_t>(1024, maxd)));
            chunk = (NRED + S - 1) / S;
            S = (NRED + chunk - 1) / chunk;
        }
        if (S <= 1 && (mode == 2 || mode == 3))
            mode = wg_mode ? 1 : 0;

        auto source_for = [&](int m)
        {
            std::ostringstream s;
            s << wgsl::prelude() << math_wgsl();
            s << "struct Params {\n"
                 "  n_out: u32, n_red: u32, row_stride: u32, chunk: u32,\n"
                 "  off: vec4<u32>,\n"
                 "  k_size: array<vec4<u32>, 2>, k_in: array<vec4<u32>, 2>,\n"
                 "  k_o0: array<vec4<u32>, 2>, k_o1: array<vec4<u32>, 2>,\n"
                 "  r_size: array<vec4<u32>, 2>, r_in: array<vec4<u32>, 2>,\n"
                 "  n_split: u32, _q0: u32, _q1: u32, _q2: u32,\n";
            if (nsv)
                s << "  scalars: array<vec4<u32>, " << nsv << ">,\n";
            s << "};\n";
            const bool stage1 = m == 2 || m == 3, stage2 = m == 4 || m == 5;
            const char *IT = wgsl::type_name(op.in_ct);
            std::string lets = scalar_lets(op.scalars);
            s << wgsl::typed_ops(op.in_ct);
            s << op.acc_struct << "\n";
            int b = 0;
            if (!stage2)
                s << "@group(0) @binding(" << b++ << ") var<storage, read> bin: array<u32>;\n";
            if (stage1)
                s << "@group(0) @binding(" << b++ << ") var<storage, read_write> part: array<Acc>;\n";
            if (stage2)
                s << "@group(0) @binding(" << b++ << ") var<storage, read> part: array<Acc>;\n";
            if (!stage1)
            {
                s << "@group(0) @binding(" << b++ << ") var<storage, read_write> bo0: array<" << (wgsl::is_subword(o0_t) ? "atomic<u32>" : "u32") << ">;\n";
                if (has_out1)
                    s << "@group(0) @binding(" << b++ << ") var<storage, read_write> bo1: array<" << (wgsl::is_subword(o1_t) ? "atomic<u32>" : "u32") << ">;\n";
            }
            s << "@group(0) @binding(" << b << ") var<uniform> P: Params;\n";
            if (!stage2)
                s << wgsl::load_fn("ld_in", "bin", in_t, false);
            if (!stage1)
            {
                s << wgsl::store_fn("st_o0", "bo0", o0_t);
                if (has_out1)
                    s << wgsl::store_fn("st_o1", "bo1", o1_t);
            }
            s << "fn acc_init() -> Acc {\n" << lets << op.init << "\n}\n";
            s << "fn acc_step(a: Acc, x: " << IT << ", r: u32) -> Acc {\n" << lets << op.step << "\n}\n";
            s << "fn acc_combine(a: Acc, b: Acc) -> Acc {\n" << lets << op.combine << "\n}\n";
            s << "fn acc_out0(a: Acc, n: u32) -> " << wgsl::type_name(op.out0_ct) << " {\n" << lets << op.out0 << "\n}\n";
            if (has_out1)
                s << "fn acc_out1(a: Acc, n: u32) -> " << wgsl::type_name(op.out1_ct) << " {\n" << lets << op.out1 << "\n}\n";

            // in-offset of reduced element r
            s << "fn red_off(r0: u32) -> u32 {\n    var r = r0;\n    var off = 0u;\n";
            if (NR == 1)
                s << "    off = r * P.r_in[0][0];\n";
            else
                for (int d = NR - 1; d >= 0; --d)
                    s << "    { let sz = P.r_size[" << d / 4 << "][" << d % 4 << "]; off += (r % sz) * P.r_in[" << d / 4
                      << "][" << d % 4 << "]; r = r / sz; }\n";
            s << "    return off;\n}\n";

            auto in_ct_of = wgsl::compute_type(in_t);
            std::string load = wgsl::convert("ld_in(base + red_off(r))", in_ct_of, op.in_ct);

            auto emit_prologue = [&](const char *oexpr)
            {
                s << "    let o = " << oexpr << ";\n";
                s << "    var rem = o;\n    var base = P.off[0];\n    var o0 = P.off[1];\n    var o1 = P.off[2];\n";
                for (int d = NK - 1; d >= 0; --d)
                {
                    s << "    { let sz = P.k_size[" << d / 4 << "][" << d % 4 << "]; let c = rem % sz; rem = rem / sz;\n"
                      << "      base += c * P.k_in[" << d / 4 << "][" << d % 4 << "]; o0 += c * P.k_o0[" << d / 4 << "]["
                      << d % 4 << "]; o1 += c * P.k_o1[" << d / 4 << "][" << d % 4 << "]; }\n";
                }
            };
            auto emit_store = [&](const char *acc)
            {
                s << "        st_o0(o0, " << wgsl::convert(std::string("acc_out0(") + acc + ", P.n_red)", op.out0_ct, wgsl::compute_type(o0_t)) << ");\n";
                if (has_out1)
                    s << "        st_o1(o1, " << wgsl::convert(std::string("acc_out1(") + acc + ", P.n_red)", op.out1_ct, wgsl::compute_type(o1_t)) << ");\n";
            };
            auto emit_tree = [&]()
            {
                s << "    sh[lid.x] = acc;\n    workgroupBarrier();\n"
                  << "    for (var st = " << kWG / 2 << "u; st > 0u; st = st >> 1u) {\n"
                  << "        if (lid.x < st) { sh[lid.x] = acc_combine(sh[lid.x], sh[lid.x + st]); }\n"
                  << "        workgroupBarrier();\n    }\n";
            };
            const char *wg_main = "@compute @workgroup_size(256)\n"
                                  "fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>,\n"
                                  "        @builtin(num_workgroups) nwg: vec3<u32>) {\n";
            const char *th_main = "@compute @workgroup_size(256)\n"
                                  "fn main(@builtin(global_invocation_id) gid: vec3<u32>) {\n";
            switch (m)
            {
            case 0:
                s << th_main;
                emit_prologue("gid.x + gid.y * P.row_stride");
                s << "    if (o >= P.n_out) { return; }\n"
                  << "    var acc = acc_init();\n"
                  << "    for (var r = 0u; r < P.n_red; r++) { acc = acc_step(acc, " << load << ", r); }\n    {\n";
                emit_store("acc");
                s << "    }\n}\n";
                break;
            case 1:
                s << "var<workgroup> sh: array<Acc, " << kWG << ">;\n" << wg_main;
                emit_prologue("wid.x + wid.y * nwg.x");
                s << "    if (o >= P.n_out) { return; }\n"
                  << "    var acc = acc_init();\n"
                  << "    for (var r = lid.x; r < P.n_red; r += " << kWG << "u) { acc = acc_step(acc, " << load << ", r); }\n";
                emit_tree();
                s << "    if (lid.x == 0u) {\n        let res = sh[0];\n";
                emit_store("res");
                s << "    }\n}\n";
                break;
            case 2:
                s << "var<workgroup> sh: array<Acc, " << kWG << ">;\n" << wg_main;
                emit_prologue("wid.x");
                s << "    let sl = wid.y;\n"
                  << "    let r0 = sl * P.chunk;\n    let r1 = min(r0 + P.chunk, P.n_red);\n"
                  << "    var acc = acc_init();\n"
                  << "    for (var r = r0 + lid.x; r < r1; r += " << kWG << "u) { acc = acc_step(acc, " << load << ", r); }\n";
                emit_tree();
                s << "    if (lid.x == 0u) { part[o * P.n_split + sl] = sh[0]; }\n}\n";
                break;
            case 3:
                s << th_main
                  << "    let tid = gid.x + gid.y * P.row_stride;\n"
                  << "    if (tid >= P.n_out * P.n_split) { return; }\n"
                  << "    let sl = tid / P.n_out;\n";
                emit_prologue("tid % P.n_out");
                s << "    let r0 = sl * P.chunk;\n    let r1 = min(r0 + P.chunk, P.n_red);\n"
                  << "    var acc = acc_init();\n"
                  << "    for (var r = r0; r < r1; r++) { acc = acc_step(acc, " << load << ", r); }\n"
                  << "    part[o * P.n_split + sl] = acc;\n}\n";
                break;
            case 4:
                s << "var<workgroup> sh: array<Acc, " << kWG << ">;\n" << wg_main;
                emit_prologue("wid.x");
                s << "    var acc = acc_init();\n"
                  << "    for (var j = lid.x; j < P.n_split; j += " << kWG << "u) { acc = acc_combine(acc, part[o * P.n_split + j]); }\n";
                emit_tree();
                s << "    if (lid.x == 0u) {\n        let res = sh[0];\n";
                emit_store("res");
                s << "    }\n}\n";
                break;
            default: // 5
                s << th_main;
                emit_prologue("gid.x + gid.y * P.row_stride");
                s << "    if (o >= P.n_out) { return; }\n"
                  << "    var acc = acc_init();\n"
                  << "    for (var j = 0u; j < P.n_split; j++) { acc = acc_combine(acc, part[o * P.n_split + j]); }\n    {\n";
                emit_store("acc");
                s << "    }\n}\n";
                break;
            }
            return s.str();
        };
        auto pipeline_for = [&](int m) -> const core::Pipeline &
        {
            std::ostringstream key;
            key << "red|" << op.name << '|' << op.acc_struct << op.init << op.step << op.combine << op.out0 << op.out1
                << '|' << int(op.in_ct) << int(op.out0_ct) << int(op.out1_ct) << '|' << int(in_t) << ',' << int(o0_t)
                << ',' << int(o1_t) << '|' << NK << NR << m << has_out1 << nsv;
            for (auto &sc : op.scalars)
                key << int(sc.second);
            return core::get_pipeline(key.str(), [&] { return source_for(m); }, op.name.c_str());
        };
        auto grid_for = [&](int m) -> core::Grid
        {
            core::Grid g;
            if (m == 1 || m == 4)
            {
                g.x = std::min(NOUT, maxd);
                g.y = (NOUT + g.x - 1) / g.x;
                g.row_stride = 0;
            }
            else if (m == 2)
            {
                TORCH_CHECK(NOUT <= maxd, "torch_webgpu: reduction has too many outputs for split mode");
                g.x = NOUT;
                g.y = S;
            }
            else if (m == 3)
                g = core::grid_1d(static_cast<uint64_t>(NOUT) * S, kWG);
            else
                g = core::grid_1d(NOUT, kWG);
            return g;
        };

        const auto w_in = core::window_of(input), w_o0 = core::window_of(out0);
        const auto w_o1 = has_out1 ? core::window_of(out1) : core::Window{};
        auto params_for = [&](const core::Grid &grid)
        {
            core::Params p;
            p.u32(NOUT).u32(NRED).u32(grid.row_stride).u32(chunk);
            p.u32(w_in.offset).u32(w_o0.offset).u32(has_out1 ? w_o1.offset : 0).u32(0);
            auto put = [&](auto getter, size_t n)
            {
                for (size_t d = 0; d < 8; ++d)
                    p.u32(d < n ? getter(d) : 1u);
            };
            put([&](size_t d) { return u32_of(kept[d].size, "size"); }, kept.size());
            put([&](size_t d) { return u32_of(kept[d].in_stride, "stride"); }, kept.size());
            put([&](size_t d) { return u32_of(kept[d].out0_stride, "stride"); }, kept.size());
            put([&](size_t d) { return u32_of(kept[d].out1_stride, "stride"); }, kept.size());
            put([&](size_t d) { return u32_of(red[d].size, "size"); }, red.size());
            put([&](size_t d) { return u32_of(red[d].stride, "stride"); }, red.size());
            p.u32(S).u32(0).u32(0).u32(0);
            put_scalars(p, op.scalars, nsv);
            return p;
        };
        auto out_bufs = [&]()
        {
            std::vector<core::BufferBinding> b{w_o0.binding};
            if (has_out1)
                b.push_back(w_o1.binding);
            return b;
        };

        if (mode == 0 || mode == 1)
        {
            const auto &pipeline = pipeline_for(mode);
            auto grid = grid_for(mode);
            auto p = params_for(grid);
            std::vector<core::BufferBinding> bufs{w_in.binding};
            for (auto &b : out_bufs())
                bufs.push_back(b);
            core::launch(pipeline, bufs, p.data(), p.size(), grid);
            return;
        }
        // Two stages through partial accumulators.
        at::Tensor part = at::empty({static_cast<int64_t>(NOUT) * S * acc_bytes / 4}, input.options().dtype(at::kInt));
        const auto w_part = core::window_of(part);
        {
            const auto &pipeline = pipeline_for(mode);
            auto grid = grid_for(mode);
            auto p = params_for(grid);
            core::launch(pipeline, {w_in.binding, w_part.binding}, p.data(), p.size(), grid);
        }
        {
            const int m2 = mode == 2 ? 4 : 5;
            const auto &pipeline = pipeline_for(m2);
            auto grid = grid_for(m2);
            auto p = params_for(grid);
            std::vector<core::BufferBinding> bufs{w_part.binding};
            for (auto &b : out_bufs())
                bufs.push_back(b);
            core::launch(pipeline, bufs, p.data(), p.size(), grid);
        }
    }

    // ------------------------------------------------------------- rows
    void run_rows(const std::vector<at::Tensor> &inputs, const std::vector<at::Tensor> &outputs,
                  const std::vector<at::Tensor> &row_outputs, int64_t dim, const RowOp &op)
    {
        TORCH_INTERNAL_ASSERT(!inputs.empty() || !outputs.empty());
        const at::Tensor &ref = inputs.empty() ? outputs[0] : inputs[0];
        const int64_t ndim = std::max<int64_t>(ref.dim(), 1);
        dim = at::maybe_wrap_dim(dim, ndim);
        if (ref.numel() == 0)
            return;
        const int64_t n = ref.dim() ? ref.size(dim) : 1;
        const int64_t rows = ref.numel() / n;

        std::vector<at::Tensor> all;
        for (auto &t : inputs)
            all.push_back(t);
        for (auto &t : outputs)
            all.push_back(t);
        for (auto &t : all)
            TORCH_CHECK(t.sizes() == ref.sizes(), "torch_webgpu: row kernel operands must have the same shape");
        for (auto &t : row_outputs)
            TORCH_CHECK(t.is_contiguous() && t.numel() == rows, "torch_webgpu: row outputs must be contiguous");
        const int nt = static_cast<int>(all.size());

        // Too large to bind: split along the outermost non-row dim (its row
        // numbers form a contiguous block, so per-row outputs narrow too).
        {
            bool fit = true;
            for (auto &t : all)
                fit &= core::windows_fit({t});
            for (auto &t : row_outputs)
                fit &= core::windows_fit({t});
            bool contiguous = true;
            for (auto &t : all)
                contiguous &= t.is_contiguous();
            if (!fit && !contiguous)
            {
                // Strided operands (e.g. rows along a large-stride dim) span far
                // more memory than they hold: run on contiguous copies.
                std::vector<at::Tensor> in2, out2;
                for (auto &t : inputs)
                    in2.push_back(t.contiguous());
                for (auto &t : outputs)
                    out2.push_back(at::empty(t.sizes(), t.options()));
                run_rows(in2, out2, row_outputs, dim, op);
                for (size_t k = 0; k < outputs.size(); ++k)
                    const_cast<at::Tensor &>(outputs[k]).copy_(out2[k]);
                return;
            }
            if (!fit)
            {
                int64_t d0 = -1;
                for (int64_t d = 0; d < ref.dim() && d0 < 0; ++d)
                    if (d != dim && ref.size(d) > 1)
                        d0 = d;
                TORCH_CHECK(d0 >= 0, "torch_webgpu: '", op.name, "' row exceeds this device's maxStorageBufferBindingSize (",
                            core::max_binding_size(), " bytes)");
                const int64_t n0 = ref.size(d0), h = n0 / 2, inner = rows / n0;
                for (auto [s0, len] : {std::pair<int64_t, int64_t>{0, h}, {h, n0 - h}})
                {
                    std::vector<at::Tensor> in2, out2, ro2;
                    for (auto &t : inputs)
                        in2.push_back(t.narrow(d0, s0, len));
                    for (auto &t : outputs)
                        out2.push_back(t.narrow(d0, s0, len));
                    for (auto &t : row_outputs)
                        ro2.push_back(t.view({-1}).narrow(0, s0 * inner, len * inner));
                    run_rows(in2, out2, ro2, dim, op);
                }
                return;
            }
        }

        // Outer dims: all but `dim`, row-major. Per-tensor strides.
        std::vector<int64_t> osz;
        std::vector<std::vector<int64_t>> ost(nt);
        for (int64_t d = 0; d < ref.dim(); ++d)
        {
            if (d == dim || ref.size(d) == 1)
                continue;
            osz.push_back(ref.size(d));
            for (int k = 0; k < nt; ++k)
                ost[k].push_back(all[k].stride(d));
        }
        TORCH_CHECK(osz.size() <= kMaxDims, "torch_webgpu: row kernel supports at most 8 outer dims");
        const int NO = static_cast<int>(osz.size());

        // Unique buffers, bound as windows over the accessed bytes.
        core::BindingSet bs;
        std::vector<int> reg(nt + row_outputs.size()), bind(nt + row_outputs.size());
        for (int k = 0; k < nt; ++k)
        {
            const bool write = k >= static_cast<int>(inputs.size());
            reg[k] = bs.add(all[k], write, write && wgsl::is_subword(all[k].scalar_type()));
        }
        for (size_t k = 0; k < row_outputs.size(); ++k)
            reg[nt + k] = bs.add(row_outputs[k], true, wgsl::is_subword(row_outputs[k].scalar_type()));
        bs.finalize();
        for (size_t k = 0; k < reg.size(); ++k)
            bind[k] = bs.binding(reg[k]);
        const auto &bufs = bs.bindings();

        const int nsv = (2 * static_cast<int>(op.scalars.size()) + 3) / 4;
        const int nin = static_cast<int>(inputs.size());
        std::ostringstream key;
        key << "rows|" << op.name << '|' << op.body << '|' << op.helpers << '|' << NO << '|' << nsv << '|' << int(op.ct);
        for (auto &s : op.scalars)
            key << int(s.second);
        for (int k = 0; k < nt; ++k)
            key << int(all[k].scalar_type()) << ':' << bind[k] << ',';
        for (size_t k = 0; k < row_outputs.size(); ++k)
            key << int(row_outputs[k].scalar_type()) << ':' << bind[nt + k] << ',';
        for (auto &b : bufs)
            key << b.written << b.atomic;

        const int ntot = nt + static_cast<int>(row_outputs.size());
        const int nov = (ntot + 3) / 4;
        auto make_source = [&]()
        {
            std::ostringstream s;
            s << wgsl::prelude() << math_wgsl();
            s << "struct Params {\n  rows: u32, n: u32, _p0: u32, _p1: u32,\n"
              << "  osize: array<vec4<u32>, 2>,\n"
              << "  ostride: array<vec4<u32>, " << 2 * nt << ">,\n"
              << "  dstride: array<vec4<u32>, " << nov << ">,\n"
              << "  off: array<vec4<u32>, " << nov << ">,\n";
            if (nsv)
                s << "  scalars: array<vec4<u32>, " << nsv << ">,\n";
            s << "};\n";
            for (size_t b = 0; b < bufs.size(); ++b)
                s << "@group(0) @binding(" << b << ") var<storage, " << (bufs[b].written ? "read_write" : "read")
                  << "> buf" << b << ": array<" << (bufs[b].atomic ? "atomic<u32>" : "u32") << ">;\n";
            s << "@group(0) @binding(" << bufs.size() << ") var<uniform> P: Params;\n";
            const char *T = wgsl::type_name(op.ct);
            s << "alias T = " << T << ";\n";
            s << "var<workgroup> red_f: array<T, " << kWG << ">;\n";
            for (int k = 0; k < ntot; ++k)
                s << "var<private> base" << k << ": u32;\n";
            s << "var<private> row: u32;\n";
            s << "var<private> lid: u32;\n";
            for (int k = 0; k < nt; ++k)
            {
                std::string var = "buf" + std::to_string(bind[k]);
                std::string dk = "P.dstride[" + std::to_string(k / 4) + "][" + std::to_string(k % 4) + "]";
                if (k < nin)
                {
                    s << wgsl::load_fn("raw_ld" + std::to_string(k), var, all[k].scalar_type(), bufs[bind[k]].atomic);
                    s << "fn ld" << k << "(j: u32) -> T { return "
                      << wgsl::convert("raw_ld" + std::to_string(k) + "(base" + std::to_string(k) + " + j * " + dk + ")",
                                       wgsl::compute_type(all[k].scalar_type()), op.ct)
                      << "; }\n";
                }
                else
                {
                    int ok = k - nin;
                    s << wgsl::store_fn("raw_st" + std::to_string(ok), var, all[k].scalar_type());
                    s << "fn st" << ok << "(j: u32, v: T) { raw_st" << ok << "(base" << k << " + j * " << dk << ", "
                      << wgsl::convert("v", op.ct, wgsl::compute_type(all[k].scalar_type())) << "); }\n";
                }
            }
            for (size_t k = 0; k < row_outputs.size(); ++k)
            {
                int idx = nt + static_cast<int>(k);
                std::string var = "buf" + std::to_string(bind[idx]);
                s << wgsl::store_fn("raw_rst" + std::to_string(k), var, row_outputs[k].scalar_type());
                s << "fn rst" << k << "(v: T) { if (lid == 0u) { raw_rst" << k << "(base" << idx << " + row, "
                  << wgsl::convert("v", op.ct, wgsl::compute_type(row_outputs[k].scalar_type())) << "); } }\n";
            }
            s << wgsl::typed_ops(op.ct);
            auto wg_red = [&](const char *name, const char *combine)
            {
                s << "fn " << name << "(v: T) -> T {\n"
                  << "    red_f[lid] = v;\n    workgroupBarrier();\n"
                  << "    for (var st = " << kWG / 2 << "u; st > 0u; st = st >> 1u) {\n"
                  << "        if (lid < st) { let a = red_f[lid]; let b = red_f[lid + st]; red_f[lid] = " << combine << "; }\n"
                  << "        workgroupBarrier();\n    }\n"
                  << "    let r = red_f[0];\n    workgroupBarrier();\n    return r;\n}\n";
            };
            if (op.ct == CT::Bool)
            {
                wg_red("wg_any", "a || b");
                wg_red("wg_all", "a && b");
            }
            else
            {
                wg_red("wg_sum", "t_add(a, b)");
                wg_red("wg_max", "t_max(a, b)");
                wg_red("wg_min", "t_min(a, b)");
                // Hillis-Steele inclusive scan across the workgroup.
                s << "fn wg_scan(v: T) -> T {\n"
                  << "    red_f[lid] = v;\n    workgroupBarrier();\n"
                  << "    for (var st = 1u; st < " << kWG << "u; st = st << 1u) {\n"
                  << "        var t = red_f[lid];\n"
                  << "        if (lid >= st) { t = t_add(t, red_f[lid - st]); }\n"
                  << "        workgroupBarrier();\n"
                  << "        red_f[lid] = t;\n"
                  << "        workgroupBarrier();\n    }\n"
                  << "    let r = red_f[lid];\n    workgroupBarrier();\n    return r;\n}\n";
                s << "fn wg_scan_mul(v: T) -> T {\n"
                  << "    red_f[lid] = v;\n    workgroupBarrier();\n"
                  << "    for (var st = 1u; st < " << kWG << "u; st = st << 1u) {\n"
                  << "        var t = red_f[lid];\n"
                  << "        if (lid >= st) { t = t_mul(t, red_f[lid - st]); }\n"
                  << "        workgroupBarrier();\n"
                  << "        red_f[lid] = t;\n"
                  << "        workgroupBarrier();\n    }\n"
                  << "    let r = red_f[lid];\n    workgroupBarrier();\n    return r;\n}\n";
                s << "fn wg_bcast(v: T, src: u32) -> T {\n"
                  << "    if (lid == src) { red_f[0] = v; }\n    workgroupBarrier();\n"
                  << "    let r = red_f[0];\n    workgroupBarrier();\n    return r;\n}\n";
            }
            s << op.helpers << "\n";
            s << "@compute @workgroup_size(" << kWG << ")\n"
              << "fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) l: vec3<u32>,\n"
              << "        @builtin(num_workgroups) nwg: vec3<u32>) {\n"
              << "    let row_u = wid.x + wid.y * nwg.x;\n"
              << "    if (row_u >= P.rows) { return; }\n"
              << "    row = row_u;\n"
              << "    lid = l.x;\n"
              << "    let n = P.n;\n";
            s << scalar_lets(op.scalars);
            for (int k = 0; k < ntot; ++k)
                s << "    base" << k << " = P.off[" << k / 4 << "][" << k % 4 << "];\n";
            if (NO > 0)
            {
                s << "    var rem = row;\n";
                for (int d = NO - 1; d >= 0; --d)
                {
                    s << "    { let sz = P.osize[" << d / 4 << "][" << d % 4 << "]; let c = rem % sz; rem = rem / sz;\n";
                    for (int k = 0; k < nt; ++k)
                        s << "      base" << k << " += c * P.ostride[" << 2 * k + d / 4 << "][" << d % 4 << "];\n";
                    s << "    }\n";
                }
            }
            s << op.body << "\n}\n";
            return s.str();
        };

        const auto &pipeline = core::get_pipeline(key.str(), make_source, op.name.c_str());

        core::Params p;
        p.u32(u32_of(rows, "rows")).u32(u32_of(n, "row size")).u32(0).u32(0);
        for (int d = 0; d < 8; ++d)
            p.u32(d < NO ? u32_of(osz[d], "size") : 1u);
        for (int k = 0; k < nt; ++k)
            for (int d = 0; d < 8; ++d)
                p.u32(d < NO ? u32_of(ost[k][d], "stride") : 0u);
        for (int k = 0; k < nov * 4; ++k)
            p.u32(k < nt ? u32_of(all[k].dim() ? all[k].stride(dim) : 1, "stride") : 0u);
        for (int k = 0; k < nov * 4; ++k)
        {
            p.u32(k < ntot ? bs.offset(reg[k]) : 0u);
        }
        put_scalars(p, op.scalars, nsv);

        const uint32_t R = u32_of(rows, "rows");
        const uint32_t maxd = core::Context::get().limits().maxComputeWorkgroupsPerDimension;
        core::Grid grid;
        grid.x = std::min(R, maxd);
        grid.y = (R + grid.x - 1) / grid.x;
        core::launch(pipeline, bs.buffers(), p.data(), p.size(), grid);
    }
}
