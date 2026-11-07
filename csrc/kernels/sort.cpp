#include "kernels/sort.h"

#include "core/allocator.h"
#include "core/binding_set.h"
#include "core/context.h"
#include "core/kernel.h"
#include "kernels/elementwise.h"
#include "kernels/iter_layout.h"
#include "kernels/math.h"
#include "kernels/wgsl.h"

#include <c10/util/Exception.h>

#include <sstream>

namespace torch_webgpu::kernels
{
    namespace
    {
        constexpr uint32_t kWG = 256;
        // Elements per workgroup in shared-memory bitonic stages (keys up to
        // 8 bytes + 4-byte indices: 12 KiB, within the spec-minimum 16 KiB).
        constexpr int64_t kLocalBlock = 1024;

        // Key ordering helpers: NaN is the largest value.
        std::string key_ops(CT ct)
        {
            std::string T = wgsl::type_name(ct);
            if (ct == CT::F32)
                return "fn key_lt(x: f32, y: f32) -> bool { return (!isnan_f(x) && isnan_f(y)) || x < y; }\n"
                       "fn key_eq(x: f32, y: f32) -> bool { return x == y || (isnan_f(x) && isnan_f(y)); }\n";
            if (ct == CT::I64)
                return "fn key_lt(x: i64, y: i64) -> bool { return i64_lt(x, y); }\n"
                       "fn key_eq(x: i64, y: i64) -> bool { return i64_eq(x, y); }\n";
            return "fn key_lt(x: " + T + ", y: " + T + ") -> bool { return x < y; }\n"
                                                        "fn key_eq(x: " + T + ", y: " + T + ") -> bool { return x == y; }\n";
        }

        int64_t next_pow2(int64_t n)
        {
            int64_t p = 1;
            while (p < n)
                p <<= 1;
            return p;
        }
    }

    void run_sort_rows(const at::Tensor &keys, const at::Tensor &indices, bool descending)
    {
        TORCH_INTERNAL_ASSERT(keys.dim() == 2 && keys.is_contiguous() && indices.is_contiguous());
        TORCH_INTERNAL_ASSERT(indices.scalar_type() == at::kInt);
        const int64_t R = keys.size(0), n = keys.size(1);
        if (R == 0 || n == 0)
            return;
        const int64_t N = next_pow2(n);
        const CT ct = wgsl::compute_type(keys.scalar_type());
        TORCH_INTERNAL_ASSERT(c10::elementSize(keys.scalar_type()) >= 4, "sort keys must be 32 or 64 bit");
        checked_u32(R * N, "sort size");

        // Scratch rows padded to a power of two (fresh buffers: offset 0).
        // Indices start as 0..N-1; padding slots (index >= n) sort last.
        at::Tensor kp = at::empty({R, N}, keys.options());
        at::Tensor ip = at::empty({R, N}, indices.options());
        kp.narrow(1, 0, n).copy_(keys);
        ip.copy_(at::arange(N, indices.options()).expand({R, N}));

        std::ostringstream key;
        key << "bitonic|" << int(keys.scalar_type()) << descending;
        auto make_source = [&]()
        {
            std::ostringstream s;
            s << wgsl::prelude() << math_wgsl() << key_ops(ct);
            s << "struct Params { R: u32, N: u32, n: u32, k: u32, j: u32, row_stride: u32, half: u32, _p: u32, };\n"
              << "@group(0) @binding(0) var<storage, read_write> bk: array<u32>;\n"
              << "@group(0) @binding(1) var<storage, read_write> bi: array<u32>;\n"
              << "@group(0) @binding(2) var<uniform> P: Params;\n";
            s << wgsl::load_fn("ldk", "bk", keys.scalar_type(), false) << wgsl::store_fn("stk", "bk", keys.scalar_type());
            const char *T = wgsl::type_name(ct);
            // after(a, b): element a belongs after element b in the final order.
            s << "fn after(ka: " << T << ", ia: u32, kb: " << T << ", ib: u32) -> bool {\n"
              << "    if (ia >= P.n) { return ib < P.n || ia > ib; }\n"
              << "    if (ib >= P.n) { return false; }\n";
            if (descending)
                s << "    if (key_lt(ka, kb)) { return true; }\n    if (key_lt(kb, ka)) { return false; }\n";
            else
                s << "    if (key_lt(kb, ka)) { return true; }\n    if (key_lt(ka, kb)) { return false; }\n";
            s << "    return ia > ib;\n}\n";
            s << "@compute @workgroup_size(" << kWG << ")\n"
              << "fn main(@builtin(global_invocation_id) gid: vec3<u32>) {\n"
              << "    let t = gid.x + gid.y * P.row_stride;\n"
              << "    if (t >= P.R * P.half) { return; }\n"
              << "    let r = t / P.half;\n"
              << "    let tt = t % P.half;\n"
              << "    let l = (tt / P.j) * 2u * P.j + (tt % P.j);\n"
              << "    let h = l + P.j;\n"
              << "    let base = r * P.N;\n"
              << "    let up = (l & P.k) == 0u;\n"
              << "    let ka = ldk(base + l);\n    let kb = ldk(base + h);\n"
              << "    let ia = bi[base + l];\n    let ib = bi[base + h];\n"
              << "    let sw = select(after(kb, ib, ka, ia), after(ka, ia, kb, ib), up);\n"
              << "    if (sw) {\n"
              << "        stk(base + l, kb); stk(base + h, ka);\n"
              << "        bi[base + l] = ib; bi[base + h] = ia;\n"
              << "    }\n}\n";
            return s.str();
        };
        const auto &pipeline = core::get_pipeline(key.str(), make_source, "bitonic_sort");

        // Stages with stride j < LB run inside a workgroup on a block of LB
        // elements held in shared memory (one dispatch for all of them);
        // only larger strides need a global pass each.
        const int64_t LB = std::min<int64_t>(N, kLocalBlock);
        std::ostringstream lkey;
        lkey << "bitonic_local|" << int(keys.scalar_type()) << descending;
        auto make_local = [&]()
        {
            std::ostringstream s;
            s << wgsl::prelude() << math_wgsl() << key_ops(ct);
            s << "struct Params { R: u32, N: u32, n: u32, LB: u32, k_first: u32, k_last: u32, nblocks: u32, _p: u32, };\n"
              << "@group(0) @binding(0) var<storage, read_write> bk: array<u32>;\n"
              << "@group(0) @binding(1) var<storage, read_write> bi: array<u32>;\n"
              << "@group(0) @binding(2) var<uniform> P: Params;\n";
            s << wgsl::load_fn("ldk", "bk", keys.scalar_type(), false) << wgsl::store_fn("stk", "bk", keys.scalar_type());
            const char *T = wgsl::type_name(ct);
            s << "fn after(ka: " << T << ", ia: u32, kb: " << T << ", ib: u32) -> bool {\n"
              << "    if (ia >= P.n) { return ib < P.n || ia > ib; }\n"
              << "    if (ib >= P.n) { return false; }\n";
            if (descending)
                s << "    if (key_lt(ka, kb)) { return true; }\n    if (key_lt(kb, ka)) { return false; }\n";
            else
                s << "    if (key_lt(kb, ka)) { return true; }\n    if (key_lt(ka, kb)) { return false; }\n";
            s << "    return ia > ib;\n}\n";
            s << "var<workgroup> sk: array<" << T << ", " << kLocalBlock << ">;\n"
              << "var<workgroup> si: array<u32, " << kLocalBlock << ">;\n"
              << "@compute @workgroup_size(" << kWG << ")\n"
              << "fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>,\n"
              << "        @builtin(num_workgroups) nwg: vec3<u32>) {\n"
              << "    let blk = wid.x + wid.y * nwg.x;\n"
              << "    if (blk >= P.nblocks) { return; }\n"
              << "    let bpr = P.N / P.LB;\n"
              << "    let r = blk / bpr;\n    let b0 = (blk % bpr) * P.LB;\n"
              << "    let base = r * P.N + b0;\n"
              << "    for (var e = lid.x; e < P.LB; e += " << kWG << "u) { sk[e] = ldk(base + e); si[e] = bi[base + e]; }\n"
              << "    workgroupBarrier();\n"
              << "    for (var k = P.k_first; k <= P.k_last; k = k << 1u) {\n"
              << "        var j = min(k >> 1u, P.LB >> 1u);\n"
              << "        loop {\n"
              << "            if (j == 0u) { break; }\n"
              << "            for (var t = lid.x; t < P.LB / 2u; t += " << kWG << "u) {\n"
              << "                let l = (t / j) * 2u * j + (t % j);\n"
              << "                let h = l + j;\n"
              << "                let up = ((b0 + l) & k) == 0u;\n"
              << "                let ka = sk[l];\n                let kb = sk[h];\n"
              << "                let ia = si[l];\n                let ib = si[h];\n"
              << "                if (select(after(kb, ib, ka, ia), after(ka, ia, kb, ib), up)) {\n"
              << "                    sk[l] = kb; sk[h] = ka; si[l] = ib; si[h] = ia;\n"
              << "                }\n"
              << "            }\n"
              << "            workgroupBarrier();\n"
              << "            j = j >> 1u;\n"
              << "        }\n"
              << "    }\n"
              << "    for (var e = lid.x; e < P.LB; e += " << kWG << "u) { stk(base + e, sk[e]); bi[base + e] = si[e]; }\n"
              << "}\n";
            return s.str();
        };
        const auto &local = core::get_pipeline(lkey.str(), make_local, "bitonic_local");

        const uint32_t half = static_cast<uint32_t>(N / 2);
        auto grid = core::grid_1d(static_cast<uint64_t>(R) * half, kWG);
        const std::vector<core::BufferBinding> bufs{core::window_of(kp).binding, core::window_of(ip).binding};
        const uint32_t nblocks = static_cast<uint32_t>(R * (N / LB));
        const uint32_t maxd = core::Context::get().limits().maxComputeWorkgroupsPerDimension;
        core::Grid lgrid;
        lgrid.x = std::min(nblocks, maxd);
        lgrid.y = (nblocks + lgrid.x - 1) / lgrid.x;
        auto run_local = [&](int64_t k_first, int64_t k_last)
        {
            core::Params p;
            p.u32(static_cast<uint32_t>(R)).u32(static_cast<uint32_t>(N)).u32(static_cast<uint32_t>(n));
            p.u32(static_cast<uint32_t>(LB)).u32(static_cast<uint32_t>(k_first)).u32(static_cast<uint32_t>(k_last));
            p.u32(nblocks).u32(0);
            core::launch(local, bufs, p.data(), p.size(), lgrid);
        };
        run_local(2, LB);
        for (int64_t k = 2 * LB; k <= N; k <<= 1)
        {
            for (int64_t j = k / 2; j >= LB; j >>= 1)
            {
                core::Params p;
                p.u32(static_cast<uint32_t>(R)).u32(static_cast<uint32_t>(N)).u32(static_cast<uint32_t>(n));
                p.u32(static_cast<uint32_t>(k)).u32(static_cast<uint32_t>(j)).u32(grid.row_stride).u32(half).u32(0);
                core::launch(pipeline, bufs, p.data(), p.size(), grid);
            }
            run_local(k, k);
        }
        keys.copy_(kp.narrow(1, 0, n));
        indices.copy_(ip.narrow(1, 0, n));
    }

    void run_searchsorted(const at::Tensor &seq, const at::Tensor &values, const at::Tensor &out, bool right)
    {
        TORCH_INTERNAL_ASSERT(seq.dim() == 2 && values.dim() == 2 && out.dim() == 2);
        TORCH_INTERNAL_ASSERT(seq.is_contiguous() && values.is_contiguous() && out.is_contiguous());
        const int64_t R = values.size(0), m = values.size(1), n = seq.size(1);
        if (R == 0 || m == 0)
            return;
        const bool shared_seq = seq.size(0) == 1 && R != 1;
        const CT ct = wgsl::compute_type(seq.scalar_type());
        TORCH_INTERNAL_ASSERT(seq.scalar_type() == values.scalar_type());
        std::ostringstream key;
        key << "searchsorted|" << int(seq.scalar_type()) << int(out.scalar_type()) << right;
        auto make_source = [&]()
        {
            std::ostringstream s;
            s << wgsl::prelude() << math_wgsl() << key_ops(ct);
            s << "struct Params { R: u32, m: u32, n: u32, seq_row: u32, row_stride: u32, off_s: u32, off_v: u32, off_o: u32, };\n"
              << "@group(0) @binding(0) var<storage, read> bs: array<u32>;\n"
              << "@group(0) @binding(1) var<storage, read> bv: array<u32>;\n"
              << "@group(0) @binding(2) var<storage, read_write> bo: array<u32>;\n"
              << "@group(0) @binding(3) var<uniform> P: Params;\n";
            s << wgsl::load_fn("lds", "bs", seq.scalar_type(), false) << wgsl::load_fn("ldv", "bv", values.scalar_type(), false);
            s << wgsl::store_fn("sto", "bo", out.scalar_type());
            s << "@compute @workgroup_size(" << kWG << ")\n"
              << "fn main(@builtin(global_invocation_id) gid: vec3<u32>) {\n"
              << "    let t = gid.x + gid.y * P.row_stride;\n"
              << "    if (t >= P.R * P.m) { return; }\n"
              << "    let r = t / P.m;\n"
              << "    let v = ldv(P.off_v + t);\n"
              << "    let base = P.off_s + r * P.seq_row;\n"
              << "    var lo = 0u;\n    var hi = P.n;\n"
              << "    while (lo < hi) {\n"
              << "        let mid = (lo + hi) / 2u;\n"
              << "        let x = lds(base + mid);\n"
              // left: first index with x >= v (i.e. move right while x < v)
              << (right ? "        if (!key_lt(v, x)) { lo = mid + 1u; } else { hi = mid; }\n"
                        : "        if (key_lt(x, v)) { lo = mid + 1u; } else { hi = mid; }\n")
              << "    }\n"
              << "    sto(P.off_o + t, " << wgsl::convert("i32(lo)", CT::I32, wgsl::compute_type(out.scalar_type())) << ");\n"
              << "}\n";
            return s.str();
        };
        const auto &pipeline = core::get_pipeline(key.str(), make_source, "searchsorted");
        auto grid = core::grid_1d(static_cast<uint64_t>(R * m), kWG);
        core::Params p;
        p.u32(checked_u32(R, "rows")).u32(checked_u32(m, "size")).u32(checked_u32(n, "size"));
        p.u32(shared_seq ? 0u : checked_u32(n, "size")).u32(grid.row_stride);
        const auto ws = core::window_of(seq), wv = core::window_of(values), wo = core::window_of(out);
        p.u32(ws.offset).u32(wv.offset);
        p.u32(wo.offset);
        core::launch(pipeline, {ws.binding, wv.binding, wo.binding}, p.data(), p.size(), grid);
    }
}
