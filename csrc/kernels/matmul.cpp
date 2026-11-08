#include "kernels/matmul.h"

#include "core/allocator.h"
#include "core/binding_set.h"
#include "core/context.h"
#include "core/kernel.h"
#include "kernels/elementwise.h"
#include "kernels/wgsl.h"

#include <c10/util/Exception.h>

#include <algorithm>
#include <cstdlib>
#include <sstream>

namespace torch_webgpu::kernels
{
    namespace
    {
        // Default tile (also used for split-K grids); large problems use
        // 128x128 tiles with 8x8 outputs per thread (more reuse per load).
        constexpr int BM = 64, BN = 64, BK = 16;
        struct Tile
        {
            int bm, bn, bk;
        };
        Tile pick_tile(int64_t M, int64_t N)
        {
            // Big tiles only while they still give enough workgroups to fill
            // the GPU (a few per compute unit).
            if (((M + 127) / 128) * ((N + 127) / 128) >= 256)
                return {128, 128, 8};
            return {BM, BN, BK};
        }

        uint32_t u32_of(int64_t v, const char *what)
        {
            TORCH_CHECK(v >= 0 && v <= static_cast<int64_t>(UINT32_MAX), "torch_webgpu: matmul ", what, " (", v,
                        ") exceeds 32-bit indexing");
            return static_cast<uint32_t>(v);
        }
    }

    namespace
    {
    // slices > 1: split-K inside the dispatch; `out` is then [slices * B, M, N]
    // receiving per-slice partial products (no bias).
    void launch_matmul(const at::Tensor &a, const at::Tensor &w, const at::Tensor &out, const at::Tensor &bias,
                       const c10::Scalar &beta, const c10::Scalar &alpha, bool has_bias, int64_t slices = 1)
    {
        const int64_t B = out.size(0) / slices;
        const int64_t M = out.size(1), N = out.size(2), K = a.size(2);
        const int64_t kc = (K + slices - 1) / slices;
        const Tile tile = slices > 1 ? Tile{BM, BN, BK} : pick_tile(M, N);
        const int tm = tile.bm / 16, tn = tile.bn / 16, la = tile.bm * tile.bk / 256, lw = tile.bk * tile.bn / 256;

        const CT ct = wgsl::compute_type(out.scalar_type());
        TORCH_CHECK(ct != CT::Bool, "torch_webgpu: matmul is not supported for bool tensors");
        const auto ta = a.scalar_type(), tw = w.scalar_type(), to = out.scalar_type();
        const auto tb = has_bias ? bias.scalar_type() : to;

        // Unique buffers (windows): a, w, bias are read-only; out is written.
        core::BindingSet bs;
        const int out_r = bs.add(out, true);
        const int a_r = bs.add(a), w_r = bs.add(w);
        const int bias_r = has_bias ? bs.add(bias) : -1;
        bs.finalize();
        const int out_b = bs.binding(out_r), a_b = bs.binding(a_r), w_b = bs.binding(w_r);
        const int bias_b = has_bias ? bs.binding(bias_r) : -1;
        const auto &binds = bs.bindings();
        TORCH_CHECK(a_b != out_b && w_b != out_b && bias_b != out_b,
                    "torch_webgpu: matmul output must not alias its inputs");

        // Choose load mappings so that consecutive threads read consecutive
        // addresses whenever one of the inner strides is 1.
        const bool a_k_contig = a.stride(2) == 1 || a.size(2) == 1;
        const bool w_n_contig = w.stride(2) == 1 || w.size(2) == 1;

        std::ostringstream key;
        key << "mm|" << int(ta) << int(tw) << int(to) << int(tb) << has_bias << a_k_contig << w_n_contig << '|'
            << a_b << w_b << bias_b << out_b << binds.size() << '|' << tile.bm << 'x' << tile.bn << 'x' << tile.bk;

        auto make_source = [&]()
        {
            const char *T = wgsl::type_name(ct);
            std::ostringstream s;
            s << wgsl::prelude();
            s << "struct Params {\n"
                 "  M: u32, N: u32, K: u32, nb: u32,\n"   // nb: batch count (grid z = slice * nb + batch)
                 "  off: vec4<u32>,\n"          // a, w, out, bias
                 "  sa: vec4<u32>, sw: vec4<u32>, so: vec4<u32>, sb: vec4<u32>,\n" // (batch, row, col, -)
                 "  alpha: u32, beta: u32, kc: u32, _q1: u32,\n" // kc: K-slice length (split-K)
                 "};\n";
            for (size_t i = 0; i < binds.size(); ++i)
            {
                bool is_out = static_cast<int>(i) == out_b;
                s << "@group(0) @binding(" << i << ") var<storage, " << (is_out ? "read_write" : "read") << "> buf" << i
                  << ": array<" << (is_out && wgsl::is_subword(to) ? "atomic<u32>" : "u32") << ">;\n";
            }
            s << "@group(0) @binding(" << binds.size() << ") var<uniform> P: Params;\n";
            s << wgsl::load_fn("ld_a", "buf" + std::to_string(a_b), ta, false);
            s << wgsl::load_fn("ld_w", "buf" + std::to_string(w_b), tw, false);
            if (has_bias)
                s << wgsl::load_fn("ld_b", "buf" + std::to_string(bias_b), tb, false);
            s << wgsl::store_fn("st_o", "buf" + std::to_string(out_b), to);
            s << "var<workgroup> As: array<array<" << T << ", " << tile.bm << ">, " << tile.bk << ">;\n";
            s << "var<workgroup> Ws: array<array<" << T << ", " << tile.bn << ">, " << tile.bk << ">;\n";

            auto cv = [&](const std::string &e, c10::ScalarType t)
            { return wgsl::convert(e, wgsl::compute_type(t), ct); };

            s << "@compute @workgroup_size(16, 16, 1)\n"
                 "fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {\n"
                 "    let tx = lid.x;\n    let ty = lid.y;\n    let tid = ty * 16u + tx;\n"
                 "    let bz = wid.z;\n    let b = bz % P.nb;\n    let sl = bz / P.nb;\n"
                 "    let kbeg = sl * P.kc;\n    let kend = min(P.K, kbeg + P.kc);\n"
                 "    let m0 = wid.y * " << tile.bm << "u;\n    let n0 = wid.x * " << tile.bn << "u;\n"
                 "    let a_base = P.off[0] + b * P.sa[0];\n"
                 "    let w_base = P.off[1] + b * P.sw[0];\n"
                 "    var acc: array<array<" << T << ", " << tn << ">, " << tm << ">;\n"
                 "    for (var i = 0u; i < " << tm << "u; i++) { for (var j = 0u; j < " << tn << "u; j++) { acc[i][j] = " << T << "(0); } }\n"
                 "    for (var k0 = kbeg; k0 < kend; k0 += " << tile.bk << "u) {\n"
                 "        for (var l = 0u; l < " << std::max(la, lw) << "u; l++) {\n"
                 "            let idx = tid + l * 256u;\n";
            // A tile: BM x BK
            if (a_k_contig)
                s << "            let ar = idx / " << tile.bk << "u;\n            let ac = idx % " << tile.bk << "u;\n";
            else
                s << "            let ar = idx % " << tile.bm << "u;\n            let ac = idx / " << tile.bm << "u;\n";
            s << "            let gm = m0 + ar;\n            let gk = k0 + ac;\n"
                 "            var av = " << T << "(0);\n"
                 "            if (gm < P.M && gk < kend) { av = " << cv("ld_a(a_base + gm * P.sa[1] + gk * P.sa[2])", ta) << "; }\n"
                 "            As[ac][ar] = av;\n";
            // W tile: BK x BN
            if (w_n_contig)
                s << "            let wr = idx / " << tile.bn << "u;\n            let wc = idx % " << tile.bn << "u;\n";
            else
                s << "            let wr = idx % " << tile.bk << "u;\n            let wc = idx / " << tile.bk << "u;\n";
            s << "            let gk2 = k0 + wr;\n            let gn = n0 + wc;\n"
                 "            var wv = " << T << "(0);\n"
                 "            if (gk2 < kend && gn < P.N) { wv = " << cv("ld_w(w_base + gk2 * P.sw[1] + gn * P.sw[2])", tw) << "; }\n"
                 "            Ws[wr][wc] = wv;\n"
                 "        }\n"
                 "        workgroupBarrier();\n"
                 "        for (var kk = 0u; kk < " << tile.bk << "u; kk++) {\n"
                 "            var av4: array<" << T << ", " << tm << ">;\n            var wv4: array<" << T << ", " << tn << ">;\n"
                 "            for (var i = 0u; i < " << tm << "u; i++) { av4[i] = As[kk][ty + 16u * i]; }\n"
                 "            for (var j = 0u; j < " << tn << "u; j++) { wv4[j] = Ws[kk][tx + 16u * j]; }\n"
                 "            for (var i = 0u; i < " << tm << "u; i++) { for (var j = 0u; j < " << tn << "u; j++) { acc[i][j] += av4[i] * wv4[j]; } }\n"
                 "        }\n"
                 "        workgroupBarrier();\n"
                 "    }\n";
            s << "    let alpha = " << wgsl::scalar_from_word("P.alpha", ct) << ";\n";
            if (has_bias)
                s << "    let beta = " << wgsl::scalar_from_word("P.beta", ct) << ";\n";
            s << "    for (var i = 0u; i < " << tm << "u; i++) {\n"
                 "        let gm = m0 + ty + 16u * i;\n"
                 "        if (gm >= P.M) { continue; }\n"
                 "        for (var j = 0u; j < " << tn << "u; j++) {\n"
                 "            let gn = n0 + tx + 16u * j;\n"
                 "            if (gn >= P.N) { continue; }\n"
                 "            var v = alpha * acc[i][j];\n";
            if (has_bias)
                s << "            v += beta * " << cv("ld_b(P.off[3] + b * P.sb[0] + gm * P.sb[1] + gn * P.sb[2])", tb) << ";\n";
            s << "            st_o(P.off[2] + bz * P.so[0] + gm * P.so[1] + gn * P.so[2], "
              << wgsl::convert("v", ct, wgsl::compute_type(to)) << ");\n"
                 "        }\n    }\n}\n";
            return s.str();
        };

        const auto &pipeline = core::get_pipeline(key.str(), make_source, "matmul");

        core::Params p;
        p.u32(u32_of(M, "M")).u32(u32_of(N, "N")).u32(u32_of(K, "K")).u32(u32_of(B, "batch"));
        p.u32(bs.offset(a_r)).u32(bs.offset(w_r));
        p.u32(bs.offset(out_r)).u32(has_bias ? bs.offset(bias_r) : 0u);
        auto strides = [&](const at::Tensor &t)
        {
            p.u32(u32_of(t.stride(0), "stride")).u32(u32_of(t.stride(1), "stride")).u32(u32_of(t.stride(2), "stride")).u32(0);
        };
        strides(a);
        strides(w);
        strides(out);
        if (has_bias)
            strides(bias);
        else
            p.u32(0).u32(0).u32(0).u32(0);
        p.u32(scalar_word(alpha, ct)).u32(has_bias ? scalar_word(beta, ct) : 0u).u32(u32_of(kc, "K slice")).u32(0);

        const uint32_t maxd = core::Context::get().limits().maxComputeWorkgroupsPerDimension;
        core::Grid grid;
        grid.x = static_cast<uint32_t>((N + tile.bn - 1) / tile.bn);
        grid.y = static_cast<uint32_t>((M + tile.bm - 1) / tile.bm);
        grid.z = static_cast<uint32_t>(B * slices);
        TORCH_CHECK(grid.x <= maxd && grid.y <= maxd && grid.z <= maxd, "torch_webgpu: matmul too large for one dispatch");

        core::launch(pipeline, bs.buffers(), p.data(), p.size(), grid);
    }

    // Skinny products (M <= 8, e.g. LLM decode GEMVs): the tiled GEMM would
    // launch N/64 workgroups and leave the GPU idle. Instead:
    //  - weight contiguous along K (nn.Linear): 8 output columns per
    //    workgroup, 32 lanes each reading K lane-strided (coalesced), then a
    //    shared-memory reduction;
    //  - weight contiguous along N (x @ W): 64 columns per workgroup (one per
    //    thread, coalesced), K split across 4 thread groups and reduced.
    // Returns false when neither layout applies.
    // (The shared reduction buffer is 256 * M floats: 16 KiB at M = 16, the
    // spec's minimum maxComputeWorkgroupStorageSize.)
    constexpr int64_t kSkinnyMaxM = 16;
    uint32_t gemv_lanes_per_row(int64_t N, int64_t K);

    // 16-byte vector loads for the vectorized GEMVs: float32 as vec4<f32>
    // (4 elements), float16 / bfloat16 as vec4<u32> (8 elements, unpacked
    // to f32). `vmad(acc, a, w)` adds the vectors' products, element pairs
    // summed into acc's lanes (the lanes are summed in the end).
    struct VecKind
    {
        int64_t ev = 0;   // elements per vector
        std::string type; // WGSL storage element type
        std::string helpers;
    };
    bool vec_kind(c10::ScalarType t, VecKind &v)
    {
        if (t == at::kFloat)
        {
            v.ev = 4;
            v.type = "vec4<f32>";
            v.helpers = "fn vmad(acc: vec4<f32>, a: vec4<f32>, w: vec4<f32>) -> vec4<f32> { return acc + a * w; }\n";
            return true;
        }
        if (t != at::kHalf && t != at::kBFloat16)
            return false;
        v.ev = 8;
        v.type = "vec4<u32>";
        v.helpers = t == at::kHalf
                        ? "fn up(w: u32) -> vec2<f32> { return unpack2x16float(w); }\n"
                        : "fn up(w: u32) -> vec2<f32> { return vec2<f32>(bitcast<f32>(w << 16u), bitcast<f32>(w & 0xffff0000u)); }\n";
        v.helpers += "fn vmad(acc: vec4<f32>, a: vec4<u32>, w: vec4<u32>) -> vec4<f32> {\n"
                     "    return acc + vec4<f32>(dot(up(a.x), up(w.x)), dot(up(a.y), up(w.y)), dot(up(a.z), up(w.z)), dot(up(a.w), up(w.w)));\n"
                     "}\n";
        return true;
    }

    // Vectorized GEMV for float32 nn.Linear weights (contiguous along K,
    // 16-byte aligned rows): decode is bound by streaming the weights, so
    // each lane reads vec4s, four loads in flight per iteration. TPR lanes
    // share a row (output column); TPR adapts to N so small layers still
    // spread over enough workgroups to keep the memory system busy.
    bool launch_gemv_vec4(const at::Tensor &a, const at::Tensor &w, const at::Tensor &out, const at::Tensor &bias,
                          const c10::Scalar &beta, const c10::Scalar &alpha, bool has_bias)
    {
        const int64_t B = out.size(0), M = out.size(1), N = out.size(2), K = a.size(2);
        VecKind vk;
        if (a.scalar_type() != w.scalar_type() || !vec_kind(w.scalar_type(), vk))
            return false;
        const int64_t EV = vk.ev;
        if (K % EV != 0 || w.stride(1) != 1 || a.stride(2) != 1 || w.stride(2) % EV != 0 || a.stride(1) % EV != 0 ||
            (B > 1 && (w.stride(0) % EV != 0 || a.stride(0) % EV != 0)))
            return false;
        const auto to = out.scalar_type();
        const auto tb = has_bias ? bias.scalar_type() : to;

        core::BindingSet bs;
        const int out_r = bs.add(out, true);
        const int a_r = bs.add(a), w_r = bs.add(w);
        const int bias_r = has_bias ? bs.add(bias) : -1;
        bs.finalize();
        const int out_b = bs.binding(out_r), a_b = bs.binding(a_r), w_b = bs.binding(w_r);
        const int bias_b = has_bias ? bs.binding(bias_r) : -1;
        const auto &binds = bs.bindings();
        if (a_b == w_b || a_b == out_b || w_b == out_b || bias_b == out_b || bias_b == w_b || bias_b == a_b)
            return false;
        if (bs.offset(a_r) % EV != 0 || bs.offset(w_r) % EV != 0)
            return false;

        // Lanes per row: enough rows per workgroup for large N, more lanes
        // (fewer rows) when N alone can't fill the GPU.
        const uint32_t tpr = gemv_lanes_per_row(N, K * 4 / EV);
        const uint32_t rows = 256 / tpr;

        std::ostringstream key;
        key << "gemv4|" << M << '|' << tpr << '|' << int(w.scalar_type()) << int(to) << int(tb) << has_bias << '|' << a_b << w_b << bias_b
            << out_b << binds.size();
        auto make_source = [&]()
        {
            std::ostringstream s;
            s << wgsl::prelude();
            s << "struct Params {\n"
                 "  M: u32, N: u32, K4: u32, _p: u32,\n"
                 "  off: vec4<u32>,\n"
                 "  sa: vec4<u32>, sw: vec4<u32>, so: vec4<u32>, sb: vec4<u32>,\n"
                 "  alpha: u32, beta: u32, _q0: u32, _q1: u32,\n"
                 "};\n";
            for (size_t i = 0; i < binds.size(); ++i)
            {
                const bool is_out = static_cast<int>(i) == out_b;
                const bool vec = static_cast<int>(i) == a_b || static_cast<int>(i) == w_b;
                s << "@group(0) @binding(" << i << ") var<storage, " << (is_out ? "read_write" : "read") << "> buf" << i
                  << ": array<" << (vec ? vk.type : is_out && wgsl::is_subword(to) ? "atomic<u32>" : "u32") << ">;\n";
            }
            s << "@group(0) @binding(" << binds.size() << ") var<uniform> P: Params;\n";
            s << vk.helpers;
            if (has_bias)
                s << wgsl::load_fn("ld_b", "buf" + std::to_string(bias_b), tb, false);
            s << wgsl::store_fn("st_o", "buf" + std::to_string(out_b), to);
            const std::string A = "buf" + std::to_string(a_b), W = "buf" + std::to_string(w_b);
            s << "var<workgroup> red: array<f32, " << 256 * M << ">;\n";
            s << "const TPR: u32 = " << tpr << "u;\n";
            s << "@compute @workgroup_size(256)\n"
                 "fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) l: vec3<u32>) {\n"
                 "    let t = l.x;\n    let b = wid.y;\n"
                 "    let lane = t % TPR;\n"
                 "    let n = wid.x * " << rows << "u + t / TPR;\n"
                 // vector indices (offsets and strides are multiples of EV)
                 "    let a0 = (P.off[0] + b * P.sa[0]) / " << EV << "u;\n"
                 "    let w0 = (P.off[1] + b * P.sw[0] + min(n, P.N - 1u) * P.sw[2]) / " << EV << "u;\n"
                 "    let sa1 = P.sa[1] / " << EV << "u;\n"
                 "    var acc: array<vec4<f32>, " << M << ">;\n"
                 "    var k = lane;\n"
                 "    if (n < P.N) {\n"
                 "        for (; k + 3u * TPR < P.K4; k += 4u * TPR) {\n"
                 "            let x0 = " << W << "[w0 + k];\n"
                 "            let x1 = " << W << "[w0 + k + TPR];\n"
                 "            let x2 = " << W << "[w0 + k + 2u * TPR];\n"
                 "            let x3 = " << W << "[w0 + k + 3u * TPR];\n";
            for (int64_t m = 0; m < M; ++m)
                s << "            { let ab = a0 + " << m << "u * sa1 + k;\n"
                     "              acc[" << m << "] = vmad(vmad(vmad(vmad(acc[" << m << "], " << A << "[ab], x0), " << A
                  << "[ab + TPR], x1), " << A << "[ab + 2u * TPR], x2), " << A << "[ab + 3u * TPR], x3); }\n";
            s << "        }\n"
                 "        for (; k < P.K4; k += TPR) {\n"
                 "            let x0 = " << W << "[w0 + k];\n";
            for (int64_t m = 0; m < M; ++m)
                s << "            acc[" << m << "] = vmad(acc[" << m << "], " << A << "[a0 + " << m << "u * sa1 + k], x0);\n";
            s << "        }\n    }\n";
            for (int64_t m = 0; m < M; ++m)
                s << "    red[t * " << M << "u + " << m << "u] = dot(acc[" << m << "], vec4<f32>(1.0));\n";
            s << "    workgroupBarrier();\n"
                 "    for (var st = TPR / 2u; st > 0u; st = st >> 1u) {\n"
                 "        if (lane < st) {\n"
                 "            for (var m = 0u; m < " << M << "u; m++) { red[t * " << M << "u + m] += red[(t + st) * " << M << "u + m]; }\n"
                 "        }\n"
                 "        workgroupBarrier();\n"
                 "    }\n"
                 "    if (lane != 0u || n >= P.N) { return; }\n"
                 "    let alpha = bitcast<f32>(P.alpha);\n";
            if (has_bias)
                s << "    let beta = bitcast<f32>(P.beta);\n";
            s << "    for (var m = 0u; m < " << M << "u; m++) {\n"
                 "        var v = alpha * red[t * " << M << "u + m];\n";
            if (has_bias)
                s << "        v += beta * " << wgsl::convert("ld_b(P.off[3] + b * P.sb[0] + m * P.sb[1] + n * P.sb[2])", wgsl::compute_type(tb), CT::F32) << ";\n";
            s << "        st_o(P.off[2] + b * P.so[0] + m * P.so[1] + n * P.so[2], " << wgsl::convert("v", CT::F32, wgsl::compute_type(to)) << ");\n"
                 "    }\n}\n";
            return s.str();
        };
        const auto &pipeline = core::get_pipeline(key.str(), make_source, "gemv4");

        core::Params p;
        p.u32(u32_of(M, "M")).u32(u32_of(N, "N")).u32(u32_of(K / EV, "K")).u32(0);
        p.u32(bs.offset(a_r)).u32(bs.offset(w_r)).u32(bs.offset(out_r)).u32(has_bias ? bs.offset(bias_r) : 0u);
        auto strides = [&](const at::Tensor &t)
        {
            p.u32(u32_of(t.stride(0), "stride")).u32(u32_of(t.stride(1), "stride")).u32(u32_of(t.stride(2), "stride")).u32(0);
        };
        strides(a);
        strides(w);
        strides(out);
        if (has_bias)
            strides(bias);
        else
            p.u32(0).u32(0).u32(0).u32(0);
        p.u32(scalar_word(alpha, CT::F32)).u32(has_bias ? scalar_word(beta, CT::F32) : 0u).u32(0).u32(0);

        const uint32_t maxd = core::Context::get().limits().maxComputeWorkgroupsPerDimension;
        core::Grid grid;
        grid.x = static_cast<uint32_t>((N + rows - 1) / rows);
        grid.y = static_cast<uint32_t>(B);
        if (grid.x > maxd || grid.y > maxd)
            return false;
        core::launch(pipeline, bs.buffers(), p.data(), p.size(), grid);
        return true;
    }

    // Lanes per output row for the vectorized GEMVs: enough rows per
    // workgroup for large N, more lanes (fewer rows) when N alone can't fill
    // the GPU.
    uint32_t gemv_lanes_per_row(int64_t N, int64_t K)
    {
        uint32_t tpr = 32;
        while (tpr < 256 && (N * tpr) / 256 < 1024 && uint64_t(K / 4) >= 2ull * tpr * 2)
            tpr *= 2;
        return tpr;
    }

    // Vectorized GEMV for x @ W with W contiguous along N (HF Conv1D,
    // x @ weight): each thread owns a quad of output columns (vec4 loads)
    // for a slice of K; QC quads x G K-groups per workgroup, QC shrinking
    // for narrow N so enough workgroups are in flight. M <= 4 (the K-group
    // reduction keeps 256 * M vec4 in shared memory).
    bool launch_gemv_vec4_n(const at::Tensor &a, const at::Tensor &w, const at::Tensor &out, const at::Tensor &bias,
                            const c10::Scalar &beta, const c10::Scalar &alpha, bool has_bias)
    {
        const int64_t B = out.size(0), M = out.size(1), N = out.size(2), K = a.size(2);
        if (M > 4 || a.scalar_type() != at::kFloat || w.scalar_type() != at::kFloat || N % 4 != 0)
            return false;
        if (w.stride(2) != 1 || w.stride(1) % 4 != 0 || (B > 1 && w.stride(0) % 4 != 0))
            return false;
        const auto ta = a.scalar_type(), to = out.scalar_type();
        const auto tb = has_bias ? bias.scalar_type() : to;

        core::BindingSet bs;
        const int out_r = bs.add(out, true);
        const int a_r = bs.add(a), w_r = bs.add(w);
        const int bias_r = has_bias ? bs.add(bias) : -1;
        bs.finalize();
        const int out_b = bs.binding(out_r), a_b = bs.binding(a_r), w_b = bs.binding(w_r);
        const int bias_b = has_bias ? bs.binding(bias_r) : -1;
        const auto &binds = bs.bindings();
        if (w_b == a_b || w_b == out_b || w_b == bias_b || a_b == out_b || bias_b == out_b)
            return false;
        if (bs.offset(w_r) % 4 != 0)
            return false;

        const int64_t quads = N / 4;
        uint32_t qc = 64;
        while (qc > 4 && (quads + qc - 1) / qc < 128)
            qc /= 2;
        const uint32_t groups = 256 / qc;

        std::ostringstream key;
        key << "gemv4n|" << M << '|' << qc << '|' << int(ta) << int(to) << int(tb) << has_bias << '|' << a_b << w_b
            << bias_b << out_b << binds.size();
        auto make_source = [&]()
        {
            std::ostringstream s;
            s << wgsl::prelude();
            s << "struct Params {\n"
                 "  M: u32, N: u32, K: u32, _p: u32,\n"
                 "  off: vec4<u32>,\n"
                 "  sa: vec4<u32>, sw: vec4<u32>, so: vec4<u32>, sb: vec4<u32>,\n"
                 "  alpha: u32, beta: u32, _q0: u32, _q1: u32,\n"
                 "};\n";
            for (size_t i = 0; i < binds.size(); ++i)
            {
                const bool is_out = static_cast<int>(i) == out_b;
                const bool vec = static_cast<int>(i) == w_b;
                s << "@group(0) @binding(" << i << ") var<storage, " << (is_out ? "read_write" : "read") << "> buf" << i
                  << ": array<" << (vec ? "vec4<f32>" : is_out && wgsl::is_subword(to) ? "atomic<u32>" : "u32") << ">;\n";
            }
            s << "@group(0) @binding(" << binds.size() << ") var<uniform> P: Params;\n";
            s << wgsl::load_fn("ld_a", "buf" + std::to_string(a_b), ta, false);
            if (has_bias)
                s << wgsl::load_fn("ld_b", "buf" + std::to_string(bias_b), tb, false);
            s << wgsl::store_fn("st_o", "buf" + std::to_string(out_b), to);
            const std::string W = "buf" + std::to_string(w_b);
            s << "var<workgroup> red: array<vec4<f32>, " << 256 * M << ">;\n";
            s << "const QC: u32 = " << qc << "u;\nconst G: u32 = " << groups << "u;\n";
            s << "@compute @workgroup_size(256)\n"
                 "fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) l: vec3<u32>) {\n"
                 "    let t = l.x;\n    let b = wid.y;\n"
                 "    let c = t % QC;\n    let g = t / QC;\n"
                 "    let q = wid.x * QC + c;\n"
                 "    let nq = P.N / 4u;\n"
                 "    let qq = min(q, nq - 1u);\n"
                 "    let a0 = P.off[0] + b * P.sa[0];\n"
                 "    let w0 = (P.off[1] + b * P.sw[0]) / 4u + qq;\n"
                 "    let swk = P.sw[1] / 4u;\n"
                 "    var acc: array<vec4<f32>, " << M << ">;\n"
                 "    if (q < nq) {\n"
                 "        var k = g;\n"
                 "        for (; k + 3u * G < P.K; k += 4u * G) {\n"
                 "            let v0 = " << W << "[w0 + k * swk];\n"
                 "            let v1 = " << W << "[w0 + (k + G) * swk];\n"
                 "            let v2 = " << W << "[w0 + (k + 2u * G) * swk];\n"
                 "            let v3 = " << W << "[w0 + (k + 3u * G) * swk];\n";
            for (int64_t m = 0; m < M; ++m)
                s << "            { let ab = a0 + " << m << "u * P.sa[1];\n"
                     "              acc[" << m << "] += " << wgsl::convert("ld_a(ab + k * P.sa[2])", wgsl::compute_type(ta), CT::F32)
                  << " * v0 + " << wgsl::convert("ld_a(ab + (k + G) * P.sa[2])", wgsl::compute_type(ta), CT::F32)
                  << " * v1 + " << wgsl::convert("ld_a(ab + (k + 2u * G) * P.sa[2])", wgsl::compute_type(ta), CT::F32)
                  << " * v2 + " << wgsl::convert("ld_a(ab + (k + 3u * G) * P.sa[2])", wgsl::compute_type(ta), CT::F32)
                  << " * v3; }\n";
            s << "        }\n"
                 "        for (; k < P.K; k += G) {\n"
                 "            let v0 = " << W << "[w0 + k * swk];\n";
            for (int64_t m = 0; m < M; ++m)
                s << "            acc[" << m << "] += " << wgsl::convert("ld_a(a0 + " + std::to_string(m) + "u * P.sa[1] + k * P.sa[2])", wgsl::compute_type(ta), CT::F32)
                  << " * v0;\n";
            s << "        }\n    }\n";
            for (int64_t m = 0; m < M; ++m)
                s << "    red[t * " << M << "u + " << m << "u] = acc[" << m << "];\n";
            s << "    workgroupBarrier();\n"
                 "    for (var st = G / 2u; st > 0u; st = st >> 1u) {\n"
                 "        if (g < st) {\n"
                 "            for (var m = 0u; m < " << M << "u; m++) { red[t * " << M << "u + m] += red[(t + st * QC) * " << M << "u + m]; }\n"
                 "        }\n"
                 "        workgroupBarrier();\n"
                 "    }\n"
                 "    if (g != 0u || q >= nq) { return; }\n"
                 "    let alpha = bitcast<f32>(P.alpha);\n";
            if (has_bias)
                s << "    let beta = bitcast<f32>(P.beta);\n";
            s << "    for (var m = 0u; m < " << M << "u; m++) {\n"
                 "        let r = red[t * " << M << "u + m];\n"
                 "        for (var e = 0u; e < 4u; e++) {\n"
                 "            let n = q * 4u + e;\n"
                 "            var v = alpha * r[e];\n";
            if (has_bias)
                s << "            v += beta * " << wgsl::convert("ld_b(P.off[3] + b * P.sb[0] + m * P.sb[1] + n * P.sb[2])", wgsl::compute_type(tb), CT::F32) << ";\n";
            s << "            st_o(P.off[2] + b * P.so[0] + m * P.so[1] + n * P.so[2], " << wgsl::convert("v", CT::F32, wgsl::compute_type(to)) << ");\n"
                 "        }\n"
                 "    }\n}\n";
            return s.str();
        };
        const auto &pipeline = core::get_pipeline(key.str(), make_source, "gemv4n");

        core::Params p;
        p.u32(u32_of(M, "M")).u32(u32_of(N, "N")).u32(u32_of(K, "K")).u32(0);
        p.u32(bs.offset(a_r)).u32(bs.offset(w_r)).u32(bs.offset(out_r)).u32(has_bias ? bs.offset(bias_r) : 0u);
        auto strides = [&](const at::Tensor &t)
        {
            p.u32(u32_of(t.stride(0), "stride")).u32(u32_of(t.stride(1), "stride")).u32(u32_of(t.stride(2), "stride")).u32(0);
        };
        strides(a);
        strides(w);
        strides(out);
        if (has_bias)
            strides(bias);
        else
            p.u32(0).u32(0).u32(0).u32(0);
        p.u32(scalar_word(alpha, CT::F32)).u32(has_bias ? scalar_word(beta, CT::F32) : 0u).u32(0).u32(0);

        const uint32_t maxd = core::Context::get().limits().maxComputeWorkgroupsPerDimension;
        core::Grid grid;
        grid.x = static_cast<uint32_t>((quads + qc - 1) / qc);
        grid.y = static_cast<uint32_t>(B);
        if (grid.x > maxd || grid.y > maxd)
            return false;
        core::launch(pipeline, bs.buffers(), p.data(), p.size(), grid);
        return true;
    }

    bool launch_skinny(const at::Tensor &a, const at::Tensor &w, const at::Tensor &out, const at::Tensor &bias,
                       const c10::Scalar &beta, const c10::Scalar &alpha, bool has_bias)
    {
        const int64_t B = out.size(0), M = out.size(1), N = out.size(2), K = a.size(2);
        const CT ct = wgsl::compute_type(out.scalar_type());
        if (M > kSkinnyMaxM || ct != CT::F32 || K == 0)
            return false;
        const bool k_contig = w.stride(1) == 1 && N > 1;
        const bool n_contig = w.stride(2) == 1;
        if (!k_contig && !n_contig)
            return false;
        // TORCH_WEBGPU_NO_GEMV4=1 forces the scalar kernel (for comparisons).
        static const bool no_vec4 = std::getenv("TORCH_WEBGPU_NO_GEMV4") != nullptr;
        if (k_contig && !no_vec4 && launch_gemv_vec4(a, w, out, bias, beta, alpha, has_bias))
            return true;
        if (!k_contig && !no_vec4 && launch_gemv_vec4_n(a, w, out, bias, beta, alpha, has_bias))
            return true;
        const auto ta = a.scalar_type(), tw = w.scalar_type(), to = out.scalar_type();
        const auto tb = has_bias ? bias.scalar_type() : to;

        core::BindingSet bs;
        const int out_r = bs.add(out, true);
        const int a_r = bs.add(a), w_r = bs.add(w);
        const int bias_r = has_bias ? bs.add(bias) : -1;
        bs.finalize();
        const int out_b = bs.binding(out_r), a_b = bs.binding(a_r), w_b = bs.binding(w_r);
        const int bias_b = has_bias ? bs.binding(bias_r) : -1;
        const auto &binds = bs.bindings();
        TORCH_CHECK(a_b != out_b && w_b != out_b && bias_b != out_b,
                    "torch_webgpu: matmul output must not alias its inputs");

        std::ostringstream key;
        key << "gemv|" << (k_contig ? 'k' : 'n') << M << '|' << int(ta) << int(tw) << int(to) << int(tb) << has_bias
            << '|' << a_b << w_b << bias_b << out_b << binds.size();
        auto make_source = [&]()
        {
            std::ostringstream s;
            s << wgsl::prelude();
            s << "struct Params {\n"
                 "  M: u32, N: u32, K: u32, _p: u32,\n"
                 "  off: vec4<u32>,\n"
                 "  sa: vec4<u32>, sw: vec4<u32>, so: vec4<u32>, sb: vec4<u32>,\n"
                 "  alpha: u32, beta: u32, _q0: u32, _q1: u32,\n"
                 "};\n";
            for (size_t i = 0; i < binds.size(); ++i)
            {
                bool is_out = static_cast<int>(i) == out_b;
                s << "@group(0) @binding(" << i << ") var<storage, " << (is_out ? "read_write" : "read") << "> buf" << i
                  << ": array<" << (is_out && wgsl::is_subword(to) ? "atomic<u32>" : "u32") << ">;\n";
            }
            s << "@group(0) @binding(" << binds.size() << ") var<uniform> P: Params;\n";
            s << wgsl::load_fn("ld_a", "buf" + std::to_string(a_b), ta, false);
            s << wgsl::load_fn("ld_w", "buf" + std::to_string(w_b), tw, false);
            if (has_bias)
                s << wgsl::load_fn("ld_b", "buf" + std::to_string(bias_b), tb, false);
            s << wgsl::store_fn("st_o", "buf" + std::to_string(out_b), to);
            s << "var<workgroup> red: array<f32, " << 256 * M << ">;\n";
            s << "@compute @workgroup_size(256)\n"
                 "fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) l: vec3<u32>) {\n"
                 "    let t = l.x;\n    let b = wid.y;\n"
                 "    let a_base = P.off[0] + b * P.sa[0];\n"
                 "    let w_base = P.off[1] + b * P.sw[0];\n"
                 "    var acc: array<f32, " << M << ">;\n";
            if (k_contig)
                s << "    let lane = t % 32u;\n    let col = t / 32u;\n"
                     "    let n = wid.x * 8u + col;\n"
                     "    if (n < P.N) {\n"
                     "        let wn = w_base + n * P.sw[2];\n"
                     "        for (var k = lane; k < P.K; k += 32u) {\n"
                     "            let wv = ld_w(wn + k);\n";
            else
                s << "    let lane = t / 64u;\n    let col = t % 64u;\n"
                     "    let n = wid.x * 64u + col;\n"
                     "    if (n < P.N) {\n"
                     "        for (var k = lane; k < P.K; k += 4u) {\n"
                     "            let wv = ld_w(w_base + k * P.sw[1] + n);\n";
            for (int64_t m = 0; m < M; ++m)
                s << "            acc[" << m << "] += ld_a(a_base + " << m << "u * P.sa[1] + k * P.sa[2]) * wv;\n";
            s << "        }\n    }\n";
            for (int64_t m = 0; m < M; ++m)
                s << "    red[t * " << M << "u + " << m << "u] = acc[" << m << "];\n";
            s << "    workgroupBarrier();\n";
            if (k_contig)
                // tree over the 32 lanes of each column (same 32-thread block)
                s << "    for (var st = 16u; st > 0u; st = st >> 1u) {\n"
                     "        if (lane < st) {\n"
                     "            for (var m = 0u; m < " << M << "u; m++) { red[t * " << M << "u + m] += red[(t + st) * " << M << "u + m]; }\n"
                     "        }\n"
                     "        workgroupBarrier();\n"
                     "    }\n"
                     "    if (lane != 0u || n >= P.N) { return; }\n";
            else
                // sum the 4 K-groups (threads col, col+64, col+128, col+192)
                s << "    if (lane != 0u || n >= P.N) { return; }\n"
                     "    for (var m = 0u; m < " << M << "u; m++) {\n"
                     "        red[t * " << M << "u + m] = red[t * " << M << "u + m] + red[(t + 64u) * " << M << "u + m] + "
                     "red[(t + 128u) * " << M << "u + m] + red[(t + 192u) * " << M << "u + m];\n"
                     "    }\n";
            s << "    let alpha = bitcast<f32>(P.alpha);\n";
            if (has_bias)
                s << "    let beta = bitcast<f32>(P.beta);\n";
            s << "    for (var m = 0u; m < " << M << "u; m++) {\n"
                 "        var v = alpha * red[t * " << M << "u + m];\n";
            if (has_bias)
                s << "        v += beta * " << wgsl::convert("ld_b(P.off[3] + b * P.sb[0] + m * P.sb[1] + n * P.sb[2])", wgsl::compute_type(tb), CT::F32) << ";\n";
            s << "        st_o(P.off[2] + b * P.so[0] + m * P.so[1] + n * P.so[2], " << wgsl::convert("v", CT::F32, wgsl::compute_type(to)) << ");\n"
                 "    }\n}\n";
            return s.str();
        };
        const auto &pipeline = core::get_pipeline(key.str(), make_source, "gemv");

        core::Params p;
        p.u32(u32_of(M, "M")).u32(u32_of(N, "N")).u32(u32_of(K, "K")).u32(0);
        p.u32(bs.offset(a_r)).u32(bs.offset(w_r)).u32(bs.offset(out_r)).u32(has_bias ? bs.offset(bias_r) : 0u);
        auto strides = [&](const at::Tensor &t)
        {
            p.u32(u32_of(t.stride(0), "stride")).u32(u32_of(t.stride(1), "stride")).u32(u32_of(t.stride(2), "stride")).u32(0);
        };
        strides(a);
        strides(w);
        strides(out);
        if (has_bias)
            strides(bias);
        else
            p.u32(0).u32(0).u32(0).u32(0);
        p.u32(scalar_word(alpha, CT::F32)).u32(has_bias ? scalar_word(beta, CT::F32) : 0u).u32(0).u32(0);

        const uint32_t maxd = core::Context::get().limits().maxComputeWorkgroupsPerDimension;
        core::Grid grid;
        grid.x = static_cast<uint32_t>(k_contig ? (N + 7) / 8 : (N + 63) / 64);
        grid.y = static_cast<uint32_t>(B);
        if (grid.x > maxd || grid.y > maxd)
            return false;
        core::launch(pipeline, bs.buffers(), p.data(), p.size(), grid);
        return true;
    }

    // Split-K: when the tiled GEMM's grid is too small to fill the GPU (e.g.
    // M = 16..128 against a large N x K weight), compute K-slices into
    // partial products and sum them (a column reduction), then apply
    // alpha / beta / bias.
    bool launch_split_k(const at::Tensor &a, const at::Tensor &w, const at::Tensor &out, const at::Tensor &bias,
                        const c10::Scalar &beta, const c10::Scalar &alpha, bool has_bias)
    {
        const int64_t B = out.size(0), M = out.size(1), N = out.size(2), K = a.size(2);
        const CT ct = wgsl::compute_type(out.scalar_type());
        if (ct != CT::F32 && ct != CT::I32)
            return false;
        const int64_t wgs = B * ((M + BM - 1) / BM) * ((N + BN - 1) / BN);
        constexpr int64_t kTargetWorkgroups = 512, kMinSlice = 256;
        if (wgs >= kTargetWorkgroups / 2 || K < 2 * kMinSlice)
            return false;
        const int64_t S = std::min((kTargetWorkgroups + wgs - 1) / wgs, K / kMinSlice);
        if (S < 2)
            return false;
        const auto pdt = ct == CT::F32 ? at::kFloat : at::kInt; // partials in the compute type
        at::Tensor part = at::empty({S, B, M, N}, out.options().dtype(pdt));
        launch_matmul(a, w, part.view({S * B, M, N}), at::Tensor(), 0, 1, false, S);
        at::Tensor res = part.sum(0, false, pdt);
        if (!(alpha.isIntegral(false) ? alpha.toLong() == 1 : alpha.toDouble() == 1.0))
            res.mul_(alpha);
        if (has_bias)
            res.add_(bias.to(pdt), beta);
        const_cast<at::Tensor &>(out).copy_(res);
        return true;
    }

    // Splits until every operand's window fits the device's binding limit:
    // batch first, then N (weight column blocks, e.g. an LM head), then M.
    void matmul_fitting(const at::Tensor &a, const at::Tensor &w, const at::Tensor &out, const at::Tensor &bias,
                        const c10::Scalar &beta, const c10::Scalar &alpha, bool has_bias)
    {
        const at::Tensor no_bias;
        if (core::windows_fit({a, w, out, has_bias ? bias : no_bias}))
        {
            if (!launch_skinny(a, w, out, bias, beta, alpha, has_bias) &&
                !launch_split_k(a, w, out, bias, beta, alpha, has_bias))
                launch_matmul(a, w, out, bias, beta, alpha, has_bias);
            return;
        }
        const int64_t B = out.size(0), M = out.size(1), N = out.size(2);
        auto half = [](int64_t n) { return n / 2; };
        auto rec = [&](int dim_out, int dim_a, int dim_w, int64_t n)
        {
            const int64_t h = half(n);
            for (auto [s0, len] : {std::pair<int64_t, int64_t>{0, h}, {h, n - h}})
            {
                at::Tensor aa = dim_a >= 0 ? a.narrow(dim_a, s0, len) : a;
                at::Tensor ww = dim_w >= 0 ? w.narrow(dim_w, s0, len) : w;
                at::Tensor bb = has_bias ? bias.narrow(dim_out, s0, len) : bias;
                matmul_fitting(aa, ww, out.narrow(dim_out, s0, len), bb, beta, alpha, has_bias);
            }
        };
        const bool w_or_out_big = !core::windows_fit({w}) || !core::windows_fit({out});
        if (B > 1)
            rec(0, a.size(0) > 1 ? 0 : -1, w.size(0) > 1 ? 0 : -1, B);
        else if (w_or_out_big && N > 1)
            rec(2, -1, 2, N);
        else if (M > 1)
            rec(1, 1, -1, M);
        else if (N > 1)
            rec(2, -1, 2, N);
        else
            TORCH_CHECK(false, "torch_webgpu: matmul operand exceeds this device's maxStorageBufferBindingSize (",
                        core::max_binding_size(), " bytes) along the reduction dimension");
    }
    }

    bool run_multi_gemv(const at::Tensor &x, const std::vector<at::Tensor> &weights,
                        const std::vector<at::Tensor> &biases, const at::Tensor &out)
    {
        const size_t J = weights.size();
        if (J == 0 || J > 4 || biases.size() != J || x.dim() != 2 || out.dim() != 2)
            return false;
        const int64_t M = x.size(0), K = x.size(1);
        const auto dt = x.scalar_type();
        VecKind vk;
        if (!vec_kind(dt, vk))
            return false;
        const int64_t EV = vk.ev;
        if (M > kSkinnyMaxM || K % EV != 0 || out.scalar_type() != dt || x.stride(1) != 1 || x.stride(0) % EV != 0 ||
            !out.is_contiguous())
            return false;
        int64_t total = 0;
        std::vector<int64_t> bounds; // cumulative row ends
        size_t nbias = 0;
        for (size_t j = 0; j < J; ++j)
        {
            const auto &w = weights[j];
            if (w.dim() != 2 || w.size(1) != K || w.scalar_type() != dt || !w.is_contiguous())
                return false;
            if (biases[j].defined())
            {
                if (biases[j].dim() != 1 || biases[j].size(0) != w.size(0) || biases[j].scalar_type() != dt ||
                    biases[j].stride(0) != 1)
                    return false;
                ++nbias;
            }
            total += w.size(0);
            bounds.push_back(total);
        }
        if (out.size(0) != M || out.size(1) != total || total == 0 || M == 0)
            return false;
        const auto &limits = core::Context::get().limits();
        if (2 + J + nbias > limits.maxStorageBuffersPerShaderStage)
            return false;

        core::BindingSet bs;
        const int out_r = bs.add(out, true);
        const int x_r = bs.add(x);
        std::vector<int> w_r, b_r;
        for (size_t j = 0; j < J; ++j)
        {
            w_r.push_back(bs.add(weights[j]));
            b_r.push_back(biases[j].defined() ? bs.add(biases[j]) : -1);
        }
        bs.finalize();
        // Weights and x are bound as vec4 arrays: every one needs a binding
        // of its own (no aliasing) and 16-byte aligned offsets.
        std::vector<int> seen;
        auto unique = [&](int r)
        {
            const int b = bs.binding(r);
            if (std::find(seen.begin(), seen.end(), b) != seen.end())
                return false;
            seen.push_back(b);
            return true;
        };
        if (!unique(out_r) || !unique(x_r) || bs.offset(x_r) % EV != 0)
            return false;
        for (size_t j = 0; j < J; ++j)
            if (!unique(w_r[j]) || bs.offset(w_r[j]) % EV != 0 || (b_r[j] >= 0 && !unique(b_r[j])))
                return false;
        const auto &binds = bs.bindings();

        const uint32_t tpr = gemv_lanes_per_row(total, K * 4 / EV);
        const uint32_t rows = 256 / tpr;
        std::ostringstream key;
        key << "multi_gemv|" << int(dt) << '|' << M << '|' << tpr << '|' << J << '|';
        for (size_t j = 0; j < J; ++j)
            key << bs.binding(w_r[j]) << ',' << (b_r[j] >= 0 ? bs.binding(b_r[j]) : -1) << ';';
        key << bs.binding(x_r) << ',' << bs.binding(out_r) << ',' << binds.size();

        auto make_source = [&]()
        {
            std::ostringstream s;
            s << wgsl::prelude();
            // offs: [x, out, w0..w3, b0..b3] element offsets
            s << "struct Params {\n"
                 "  M: u32, N: u32, K4: u32, sx: u32,\n"
                 "  ends: vec4<u32>,\n"
                 "  offs: array<vec4<u32>, 3>,\n"
                 "};\n";
            for (size_t i = 0; i < binds.size(); ++i)
            {
                const bool is_out = static_cast<int>(i) == bs.binding(out_r);
                bool vec = static_cast<int>(i) == bs.binding(x_r);
                for (size_t j = 0; j < J; ++j)
                    vec = vec || static_cast<int>(i) == bs.binding(w_r[j]);
                s << "@group(0) @binding(" << i << ") var<storage, " << (is_out ? "read_write" : "read") << "> buf" << i
                  << ": array<" << (vec ? vk.type : is_out && wgsl::is_subword(dt) ? "atomic<u32>" : "u32") << ">;\n";
            }
            s << "@group(0) @binding(" << binds.size() << ") var<uniform> P: Params;\n";
            s << "fn off(i: u32) -> u32 { return P.offs[i / 4u][i % 4u]; }\n";
            s << vk.helpers;
            s << wgsl::store_fn("st_o", "buf" + std::to_string(bs.binding(out_r)), dt);
            for (size_t j = 0; j < J; ++j)
                if (b_r[j] >= 0)
                    s << wgsl::load_fn("ld_b" + std::to_string(j), "buf" + std::to_string(bs.binding(b_r[j])), dt, false);
            s << "var<workgroup> red: array<f32, " << 256 * M << ">;\n";
            s << "const TPR: u32 = " << tpr << "u;\n";
            const std::string X = "buf" + std::to_string(bs.binding(x_r));
            s << "@compute @workgroup_size(256)\n"
                 "fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) l: vec3<u32>) {\n"
                 "    let t = l.x;\n"
                 "    let lane = t % TPR;\n"
                 "    let n = wid.x * " << rows << "u + t / TPR;\n"
                 "    let x0 = off(0u) / " << EV << "u;\n"
                 "    let sx = P.sx / " << EV << "u;\n"
                 "    var acc: array<vec4<f32>, " << M << ">;\n";
            for (size_t j = 0; j < J; ++j)
            {
                const std::string W = "buf" + std::to_string(bs.binding(w_r[j]));
                const std::string lo = j == 0 ? "0u" : "P.ends[" + std::to_string(j - 1) + "]";
                s << "    " << (j ? "else " : "") << "if (n < P.ends[" << j << "]) {\n"
                  << "        let w0 = off(" << 2 + j << "u) / " << EV << "u + (n - " << lo << ") * P.K4;\n"
                  << "        var k = lane;\n"
                  << "        for (; k + 3u * TPR < P.K4; k += 4u * TPR) {\n"
                  << "            let v0 = " << W << "[w0 + k];\n"
                  << "            let v1 = " << W << "[w0 + k + TPR];\n"
                  << "            let v2 = " << W << "[w0 + k + 2u * TPR];\n"
                  << "            let v3 = " << W << "[w0 + k + 3u * TPR];\n";
                for (int64_t m = 0; m < M; ++m)
                    s << "            { let ab = x0 + " << m << "u * sx + k;\n"
                         "              acc[" << m << "] = vmad(vmad(vmad(vmad(acc[" << m << "], " << X << "[ab], v0), " << X
                      << "[ab + TPR], v1), " << X << "[ab + 2u * TPR], v2), " << X << "[ab + 3u * TPR], v3); }\n";
                s << "        }\n"
                  << "        for (; k < P.K4; k += TPR) {\n"
                  << "            let v0 = " << W << "[w0 + k];\n";
                for (int64_t m = 0; m < M; ++m)
                    s << "            acc[" << m << "] = vmad(acc[" << m << "], " << X << "[x0 + " << m << "u * sx + k], v0);\n";
                s << "        }\n    }\n";
            }
            for (int64_t m = 0; m < M; ++m)
                s << "    red[t * " << M << "u + " << m << "u] = dot(acc[" << m << "], vec4<f32>(1.0));\n";
            s << "    workgroupBarrier();\n"
                 "    for (var st = TPR / 2u; st > 0u; st = st >> 1u) {\n"
                 "        if (lane < st) {\n"
                 "            for (var m = 0u; m < " << M << "u; m++) { red[t * " << M << "u + m] += red[(t + st) * " << M << "u + m]; }\n"
                 "        }\n"
                 "        workgroupBarrier();\n"
                 "    }\n"
                 "    if (lane != 0u || n >= P.N) { return; }\n"
                 "    var bias = 0.0f;\n";
            for (size_t j = 0; j < J; ++j)
            {
                if (b_r[j] < 0)
                    continue;
                const std::string lo = j == 0 ? "0u" : "P.ends[" + std::to_string(j - 1) + "]";
                s << "    if (n >= " << lo << " && n < P.ends[" << j << "]) { bias = "
                  << wgsl::convert("ld_b" + std::to_string(j) + "(off(" + std::to_string(6 + j) + "u) + n - " + lo + ")",
                                   wgsl::compute_type(dt), CT::F32)
                  << "; }\n";
            }
            s << "    for (var m = 0u; m < " << M << "u; m++) {\n"
                 "        st_o(off(1u) + m * P.N + n, " << wgsl::convert("red[t * " + std::to_string(M) + "u + m] + bias", CT::F32, wgsl::compute_type(dt)) << ");\n"
                 "    }\n}\n";
            return s.str();
        };
        const auto &pipeline = core::get_pipeline(key.str(), make_source, "multi_gemv");

        core::Params p;
        p.u32(u32_of(M, "M")).u32(u32_of(total, "N")).u32(u32_of(K / EV, "K")).u32(u32_of(x.stride(0), "stride"));
        for (size_t j = 0; j < 4; ++j)
            p.u32(j < J ? u32_of(bounds[j], "N") : u32_of(total, "N"));
        std::vector<uint32_t> offs(12, 0u);
        offs[0] = bs.offset(x_r);
        offs[1] = bs.offset(out_r);
        for (size_t j = 0; j < J; ++j)
        {
            offs[2 + j] = bs.offset(w_r[j]);
            offs[6 + j] = b_r[j] >= 0 ? bs.offset(b_r[j]) : 0u;
        }
        for (uint32_t o : offs)
            p.u32(o);

        core::Grid grid;
        grid.x = static_cast<uint32_t>((total + rows - 1) / rows);
        if (grid.x > limits.maxComputeWorkgroupsPerDimension)
            return false;
        core::launch(pipeline, bs.buffers(), p.data(), p.size(), grid);
        return true;
    }

    void run_matmul(const at::Tensor &a, const at::Tensor &w, const at::Tensor &out, const at::Tensor &bias,
                    const c10::Scalar &beta, const c10::Scalar &alpha)
    {
        TORCH_INTERNAL_ASSERT(a.dim() == 3 && w.dim() == 3 && out.dim() == 3);
        const int64_t B = out.size(0), M = out.size(1), N = out.size(2);
        if (B == 0 || M == 0 || N == 0)
            return;
        const bool has_bias = bias.defined() && !(beta.isIntegral(false) ? beta.toLong() == 0 : beta.toDouble() == 0.0);
        if (has_bias)
            TORCH_INTERNAL_ASSERT(bias.dim() == 3);
        matmul_fitting(a, w, out, bias, beta, alpha, has_bias);
    }
}
