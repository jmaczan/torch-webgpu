#include "kernels/attention.h"

#include "core/binding_set.h"
#include "core/context.h"
#include "core/kernel.h"
#include "kernels/iter_layout.h"
#include "kernels/math.h"
#include "kernels/wgsl.h"

#include <cstring>
#include <sstream>

namespace torch_webgpu::kernels
{
    namespace
    {
        constexpr int kRows = 16; // query rows per workgroup (16 lanes each)

        // Shared memory (bytes) for key tile bc: Q rows, a K (padded) / V tile,
        // scores and validity flags.
        int64_t shared_bytes(int64_t d, int64_t dv, int64_t bc)
        {
            return 4 * (kRows * d + bc * std::max(d + 1, dv) + 2 * kRows * bc);
        }

        int64_t pick_bc(int64_t d, int64_t dv)
        {
            const int64_t avail = core::Context::get().limits().maxComputeWorkgroupStorageSize;
            for (int64_t bc : {32, 16})
                if (shared_bytes(d, dv, bc) <= avail)
                    return bc;
            return 0;
        }
    }

    bool attention_fits(int64_t d, int64_t dv)
    {
        return d >= 1 && dv >= 1 && d <= kAttentionMaxHeadDim && dv <= kAttentionMaxHeadDim && pick_bc(d, dv) > 0;
    }

    void run_attention(const at::Tensor &q, const at::Tensor &k, const at::Tensor &v, const at::Tensor &bias,
                       bool causal, double scale, const at::Tensor &out, const at::Tensor &lse)
    {
        const int64_t B = q.size(0), Hq = q.size(1), Lq = q.size(2), D = q.size(3);
        const int64_t Hk = k.size(1), Lk = k.size(2), Dv = v.size(3);
        if (out.numel() == 0)
            return;
        TORCH_INTERNAL_ASSERT(Hq % Hk == 0 && attention_fits(D, Dv));
        const int64_t BC = pick_bc(D, Dv);
        const int64_t KD = D + 1;                        // padded K row stride (bank conflicts)
        const int64_t KVS = BC * std::max(KD, Dv);
        const int64_t DVL = (Dv + 15) / 16;              // output dims per lane
        const bool has_bias = bias.defined();

        core::BindingSet bs;
        const int out_r = bs.add(out, true, wgsl::is_subword(out.scalar_type()));
        const int lse_r = bs.add(lse, true);
        const int q_r = bs.add(q), k_r = bs.add(k), v_r = bs.add(v);
        const int b_r = has_bias ? bs.add(bias) : -1;
        bs.finalize();
        auto bind = [&](int r) { return "buf" + std::to_string(bs.binding(r)); };

        std::ostringstream key;
        key << "attn|" << D << ',' << Dv << ',' << BC << '|' << int(q.scalar_type()) << int(k.scalar_type())
            << int(v.scalar_type()) << int(out.scalar_type()) << (has_bias ? int(bias.scalar_type()) : -1) << '|'
            << bs.binding(out_r) << bs.binding(lse_r) << bs.binding(q_r) << bs.binding(k_r) << bs.binding(v_r)
            << (has_bias ? bs.binding(b_r) : -1) << bs.bindings().size();
        auto make = [&]()
        {
            std::ostringstream s;
            s << wgsl::prelude() << math_wgsl();
            s << "struct Params {\n"
                 "  B: u32, Hq: u32, Hk: u32, Lq: u32,\n"
                 "  Lk: u32, causal: u32, _p0: u32, _p1: u32,\n"
                 "  sq: vec4<u32>, sk: vec4<u32>, sv: vec4<u32>, so: vec4<u32>, sb: vec4<u32>,\n"
                 "  offs: vec4<u32>, offs2: vec4<u32>,\n"
                 "  scale: f32, _f0: u32, _f1: u32, _f2: u32,\n"
                 "};\n";
            const auto &bb = bs.bindings();
            for (size_t i = 0; i < bb.size(); ++i)
                s << "@group(0) @binding(" << i << ") var<storage, " << (bb[i].written ? "read_write" : "read") << "> buf" << i
                  << ": array<" << (bb[i].atomic ? "atomic<u32>" : "u32") << ">;\n";
            s << "@group(0) @binding(" << bb.size() << ") var<uniform> P: Params;\n";
            auto ld = [&](const char *name, int r, const at::Tensor &t)
            {
                s << wgsl::load_fn(std::string("raw_") + name, bind(r), t.scalar_type(), false);
                s << "fn " << name << "(i: u32) -> f32 { return "
                  << wgsl::convert(std::string("raw_") + name + "(i)", wgsl::compute_type(t.scalar_type()), CT::F32) << "; }\n";
            };
            ld("ldq", q_r, q);
            ld("ldk", k_r, k);
            ld("ldv", v_r, v);
            if (has_bias)
                ld("ldb", b_r, bias);
            s << wgsl::store_fn("raw_sto", bind(out_r), out.scalar_type());
            s << wgsl::store_fn("raw_stl", bind(lse_r), at::kFloat);
            s << "var<workgroup> Qs: array<f32, " << kRows * D << ">;\n"
              << "var<workgroup> KV: array<f32, " << KVS << ">;\n"
              << "var<workgroup> Ps: array<f32, " << kRows * BC << ">;\n"
              << "var<workgroup> Pv: array<u32, " << kRows * BC << ">;\n";
            s << "@compute @workgroup_size(256)\n"
                 "fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) l3: vec3<u32>) {\n"
                 "    let lid = l3.x;\n    let r = lid / 16u;\n    let lane = lid % 16u;\n"
                 "    let qb = wid.x;\n    let h = wid.y;\n    let b = wid.z;\n"
                 "    let hk = h / (P.Hq / P.Hk);\n"
                 "    let qi = qb * 16u + r;\n    let qvalid = qi < P.Lq;\n"
                 "    let qbase = P.offs[0] + b * P.sq[0] + h * P.sq[1];\n"
                 "    let kbase = P.offs[1] + b * P.sk[0] + hk * P.sk[1];\n"
                 "    let vbase = P.offs[2] + b * P.sv[0] + hk * P.sv[1];\n"
              << "    for (var e = lid; e < " << kRows * D << "u; e += 256u) {\n"
              << "        let rr = e / " << D << "u;\n        let d = e % " << D << "u;\n        let qq = qb * 16u + rr;\n"
                 "        var x = 0.0;\n"
                 "        if (qq < P.Lq) { x = ldq(qbase + qq * P.sq[2] + d * P.sq[3]); }\n"
                 "        Qs[e] = x;\n"
                 "    }\n"
                 "    var m = 0.0;\n    var lsum = 0.0;\n    var started = false;\n"
              << "    var o: array<f32, " << DVL << ">;\n"
                 // keys a block can see (top-left causal): up to its last query row
                 "    var kend = P.Lk;\n"
                 "    if (P.causal != 0u) { kend = min(P.Lk, qb * 16u + 16u); }\n"
              << "    for (var kt = 0u; kt < kend; kt += " << BC << "u) {\n"
                 "        workgroupBarrier();\n"
              << "        for (var e = lid; e < " << BC * D << "u; e += 256u) {\n"
              << "            let jj = e / " << D << "u;\n            let d = e % " << D << "u;\n            let kj = kt + jj;\n"
                 "            var x = 0.0;\n"
                 "            if (kj < P.Lk) { x = ldk(kbase + kj * P.sk[2] + d * P.sk[3]); }\n"
              << "            KV[jj * " << KD << "u + d] = x;\n"
                 "        }\n"
                 "        workgroupBarrier();\n"
              << "        for (var t = 0u; t < " << BC / 16 << "u; t++) {\n"
                 "            let j = lane + 16u * t;\n            let kj = kt + j;\n"
                 "            var sc = 0.0;\n"
              << "            for (var d = 0u; d < " << D << "u; d++) { sc += Qs[r * " << D << "u + d] * KV[j * " << KD << "u + d]; }\n"
                 "            sc = sc * P.scale;\n"
                 "            let valid = kj < P.Lk && qvalid && (P.causal == 0u || kj <= qi);\n";
            if (has_bias)
                // A -inf bias masks the key (fully masked rows give 0, like
                // _safe_softmax); tested on the loaded bits (fast-math safe).
                s << "            var keep = valid;\n"
                     "            if (valid) {\n"
                     "                let bv = ldb(P.offs2[0] + b * P.sb[0] + h * P.sb[1] + qi * P.sb[2] + kj * P.sb[3]);\n"
                     "                if (isinf_f(bv) && signbit_f(bv)) { keep = false; } else { sc += bv; }\n"
                     "            }\n";
            else
                s << "            let keep = valid;\n";
            s << "            Ps[r * " << BC << "u + j] = sc;\n"
              << "            Pv[r * " << BC << "u + j] = select(0u, 1u, keep);\n"
                 "        }\n"
                 "        workgroupBarrier();\n"
                 // Row statistics (every lane of row r computes them). Masked
                 // entries are flagged, not -inf (driver fast-math).
                 "        var tmax = 0.0;\n        var any = false;\n"
              << "        for (var j = 0u; j < " << BC << "u; j++) {\n"
              << "            if (Pv[r * " << BC << "u + j] != 0u) {\n"
              << "                let x = Ps[r * " << BC << "u + j];\n"
                 "                if (!any || x > tmax) { tmax = x; }\n"
                 "                any = true;\n"
                 "            }\n"
                 "        }\n"
                 "        if (any) {\n"
                 "            var mn = tmax;\n"
                 "            if (started) { mn = max(m, tmax); }\n"
                 "            var alpha = 0.0;\n"
                 "            if (started) { alpha = exp(m - mn); }\n"
                 "            lsum = lsum * alpha;\n"
              << "            for (var t = 0u; t < " << DVL << "u; t++) { o[t] = o[t] * alpha; }\n"
                 "            m = mn;\n            started = true;\n"
                 "        }\n"
                 "        workgroupBarrier();\n"
              << "        for (var e = lid; e < " << BC * Dv << "u; e += 256u) {\n"
              << "            let jj = e / " << Dv << "u;\n            let d = e % " << Dv << "u;\n            let kj = kt + jj;\n"
                 "            var x = 0.0;\n"
                 "            if (kj < P.Lk) { x = ldv(vbase + kj * P.sv[2] + d * P.sv[3]); }\n"
              << "            KV[jj * " << Dv << "u + d] = x;\n"
                 "        }\n"
                 "        workgroupBarrier();\n"
                 "        if (any) {\n"
              << "            for (var j = 0u; j < " << BC << "u; j++) {\n"
              << "                if (Pv[r * " << BC << "u + j] == 0u) { continue; }\n"
              << "                let p = exp(Ps[r * " << BC << "u + j] - m);\n"
                 "                lsum += p;\n"
              << "                for (var t = 0u; t < " << DVL << "u; t++) {\n"
                 "                    let d = lane + 16u * t;\n"
              << "                    if (d < " << Dv << "u) { o[t] += p * KV[j * " << Dv << "u + d]; }\n"
                 "                }\n"
                 "            }\n"
                 "        }\n"
                 "    }\n"
                 "    if (!qvalid) { return; }\n"
                 "    let obase = P.offs[3] + b * P.so[0] + h * P.so[1] + qi * P.so[2];\n"
              << "    for (var t = 0u; t < " << DVL << "u; t++) {\n"
                 "        let d = lane + 16u * t;\n"
              << "        if (d < " << Dv << "u) {\n"
                 "            var y = 0.0;\n"
                 "            if (started) { y = o[t] / lsum; }\n"
              << "            raw_sto(obase + d * P.so[3], " << wgsl::convert("y", CT::F32, wgsl::compute_type(out.scalar_type())) << ");\n"
                 "        }\n"
                 "    }\n"
                 "    if (lane == 0u) {\n"
                 "        var lv = bitcast<f32>(INF_BITS | 0x80000000u);\n"
                 "        if (started) { lv = m + log(lsum); }\n"
                 "        raw_stl(P.offs2[1] + (b * P.Hq + h) * P.Lq + qi, lv);\n"
                 "    }\n"
                 "}\n";
            return s.str();
        };
        const auto &pipeline = core::get_pipeline(key.str(), make, "attention");

        auto u = [](int64_t x) { return checked_u32(x, "attention size"); };
        core::Params p;
        p.u32(u(B)).u32(u(Hq)).u32(u(Hk)).u32(u(Lq)).u32(u(Lk)).u32(causal ? 1u : 0u).u32(0).u32(0);
        for (const at::Tensor *t : {&q, &k, &v, &out})
            for (int d = 0; d < 4; ++d)
                p.u32(u(t->stride(d)));
        for (int d = 0; d < 4; ++d)
            p.u32(has_bias ? u(bias.stride(d)) : 0u);
        p.u32(bs.offset(q_r)).u32(bs.offset(k_r)).u32(bs.offset(v_r)).u32(bs.offset(out_r));
        p.u32(has_bias ? bs.offset(b_r) : 0u).u32(bs.offset(lse_r)).u32(0).u32(0);
        const float sc = static_cast<float>(scale);
        uint32_t scw;
        std::memcpy(&scw, &sc, 4);
        p.u32(scw).u32(0).u32(0).u32(0);
        const uint32_t maxd = core::Context::get().limits().maxComputeWorkgroupsPerDimension;
        core::Grid g;
        g.x = static_cast<uint32_t>((Lq + kRows - 1) / kRows);
        g.y = static_cast<uint32_t>(Hq);
        g.z = static_cast<uint32_t>(B);
        TORCH_CHECK(g.x <= maxd && g.y <= maxd && g.z <= maxd, "torch_webgpu: attention too large for one dispatch");
        core::launch(pipeline, bs.buffers(), p.data(), p.size(), g);
    }

    namespace
    {
        // Key capacity of the decode kernel's shared score buffer: a power of
        // two >= lk (at least 256), within the device's workgroup storage.
        int64_t decode_capacity(int64_t lk, int64_t d)
        {
            const int64_t avail = core::Context::get().limits().maxComputeWorkgroupStorageSize;
            int64_t cap = 256;
            while (cap < lk)
                cap *= 2;
            const int64_t bytes = 4 * (cap + d + 2 * 256);
            return bytes <= avail ? cap : 0;
        }
    }

    bool attention_decode_fits(int64_t lk, int64_t d, int64_t dv)
    {
        return d >= 1 && dv >= 1 && d <= 256 && dv <= 256 && lk >= 1 && decode_capacity(256, d) > 0;
    }

    void run_attention_decode(const at::Tensor &q, const at::Tensor &k, const at::Tensor &v, const at::Tensor &bias,
                              bool causal, double scale, const at::Tensor &out, const at::Tensor &lse)
    {
        const int64_t B = q.size(0), Hq = q.size(1), Lq = q.size(2), D = q.size(3);
        const int64_t Hk = k.size(1), Lk = k.size(2), Dv = v.size(3);
        if (out.numel() == 0)
            return;
        TORCH_INTERNAL_ASSERT(Hq % Hk == 0 && attention_decode_fits(Lk, D, Dv));
        // Long contexts: keys split into chunks of 128 across workgroups
        // (flash decoding); partial results are merged by a second kernel.
        // At most 64 splits (the merge reads every split's partials); chunks
        // grow in powers of two for very long contexts.
        int64_t chunk = 128;
        while ((Lk + chunk - 1) / chunk > 64)
            chunk *= 2;
        const int64_t NS = Lk > 128 ? (Lk + chunk - 1) / chunk : 1;
        const int64_t CAP = NS > 1 ? chunk : decode_capacity(Lk, D);
        TORCH_CHECK(CAP > 0 && 4 * (CAP + D + 2 * 256) <=
                                   int64_t(core::Context::get().limits().maxComputeWorkgroupStorageSize),
                    "torch_webgpu: attention key chunk exceeds the device's workgroup storage");
        int64_t DVP = 1;
        while (DVP < Dv)
            DVP *= 2;
        const int64_t G = 256 / DVP; // key groups in the PV product
        const bool has_bias = bias.defined();

        const bool split = NS > 1;
        // Partials per (b, h, query, split): [m, l, o_0 .. o_{Dv-1}].
        at::Tensor part = split ? at::empty({B, Hq, Lq, NS, Dv + 2}, q.options().dtype(at::kFloat)) : at::Tensor();

        core::BindingSet bs;
        const int out_r = split ? bs.add(part, true) : bs.add(out, true, wgsl::is_subword(out.scalar_type()));
        const int lse_r = split ? -1 : bs.add(lse, true);
        const int q_r = bs.add(q), k_r = bs.add(k), v_r = bs.add(v);
        const int b_r = has_bias ? bs.add(bias) : -1;
        bs.finalize();
        auto bind = [&](int r) { return "buf" + std::to_string(bs.binding(r)); };

        std::ostringstream key;
        key << "attn_decode|" << D << ',' << Dv << ',' << CAP << ',' << split << '|' << int(q.scalar_type()) << int(k.scalar_type())
            << int(v.scalar_type()) << int(out.scalar_type()) << (has_bias ? int(bias.scalar_type()) : -1) << '|'
            << bs.binding(out_r) << (split ? -1 : bs.binding(lse_r)) << bs.binding(q_r) << bs.binding(k_r)
            << bs.binding(v_r) << (has_bias ? bs.binding(b_r) : -1) << bs.bindings().size();
        auto make = [&]()
        {
            std::ostringstream s;
            s << wgsl::prelude() << math_wgsl();
            s << "struct Params {\n"
                 "  B: u32, Hq: u32, Hk: u32, Lq: u32,\n"
                 "  Lk: u32, causal: u32, NS: u32, _p1: u32,\n"
                 "  sq: vec4<u32>, sk: vec4<u32>, sv: vec4<u32>, so: vec4<u32>, sb: vec4<u32>,\n"
                 "  offs: vec4<u32>, offs2: vec4<u32>,\n"
                 "  scale: f32, _f0: u32, _f1: u32, _f2: u32,\n"
                 "};\n";
            const auto &bb = bs.bindings();
            for (size_t i = 0; i < bb.size(); ++i)
                s << "@group(0) @binding(" << i << ") var<storage, " << (bb[i].written ? "read_write" : "read") << "> buf" << i
                  << ": array<" << (bb[i].atomic ? "atomic<u32>" : "u32") << ">;\n";
            s << "@group(0) @binding(" << bb.size() << ") var<uniform> P: Params;\n";
            auto ld = [&](const char *name, int r, const at::Tensor &t)
            {
                s << wgsl::load_fn(std::string("raw_") + name, bind(r), t.scalar_type(), false);
                s << "fn " << name << "(i: u32) -> f32 { return "
                  << wgsl::convert(std::string("raw_") + name + "(i)", wgsl::compute_type(t.scalar_type()), CT::F32) << "; }\n";
            };
            ld("ldq", q_r, q);
            ld("ldk", k_r, k);
            ld("ldv", v_r, v);
            if (has_bias)
                ld("ldb", b_r, bias);
            if (split)
                s << "fn st_part(i: u32, v: f32) { " << bind(out_r) << "[i] = bitcast<u32>(v); }\n";
            else
            {
                s << wgsl::store_fn("raw_sto", bind(out_r), out.scalar_type());
                s << wgsl::store_fn("raw_stl", bind(lse_r), at::kFloat);
            }
            s << "const CHUNK: u32 = " << CAP << "u;\n";
            // S holds score bits; MASKED (the -inf pattern) marks invisible
            // keys and is only ever compared as an integer (fast-math safe).
            s << "const MASKED: u32 = 0xff800000u;\n"
              << "var<workgroup> Qs: array<f32, " << D << ">;\n"
              << "var<workgroup> S: array<u32, " << CAP << ">;\n"
              << "var<workgroup> R: array<f32, 256>;\n"
              << "var<workgroup> RC: array<u32, 256>;\n";
            s << "@compute @workgroup_size(256)\n"
                 "fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) l3: vec3<u32>) {\n"
                 "    let lid = l3.x;\n"
                 "    let qi = wid.x / P.NS;\n    let sid = wid.x % P.NS;\n    let h = wid.y;\n    let b = wid.z;\n"
                 "    let k0 = sid * CHUNK;\n    let k1 = min(P.Lk, k0 + CHUNK);\n"
                 "    let hk = h / (P.Hq / P.Hk);\n"
                 "    let qbase = P.offs[0] + b * P.sq[0] + h * P.sq[1] + qi * P.sq[2];\n"
                 "    let kbase = P.offs[1] + b * P.sk[0] + hk * P.sk[1];\n"
                 "    let vbase = P.offs[2] + b * P.sv[0] + hk * P.sv[1];\n"
              << "    for (var d = lid; d < " << D << "u; d += 256u) { Qs[d] = ldq(qbase + d * P.sq[3]); }\n"
                 "    workgroupBarrier();\n"
                 // scores and the running max of visible keys
                 "    var mx = 0.0;\n    var cnt = 0u;\n"
                 "    for (var j = k0 + lid; j < k1; j += 256u) {\n"
                 "        var visible = P.causal == 0u || j <= qi;\n"
                 "        var sc = 0.0;\n"
                 "        if (visible) {\n"
                 "            let kr = kbase + j * P.sk[2];\n"
              ;
            if (D % 8 == 0)
                // Eight independent partial sums: the key row's loads overlap
                // instead of forming one dependent chain.
                s << "            var s8 = array<f32, 8>(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);\n"
                  << "            for (var d = 0u; d < " << D << "u; d += 8u) {\n"
                     "                for (var e = 0u; e < 8u; e++) { s8[e] += Qs[d + e] * ldk(kr + (d + e) * P.sk[3]); }\n"
                     "            }\n"
                     "            sc = ((s8[0] + s8[1]) + (s8[2] + s8[3])) + ((s8[4] + s8[5]) + (s8[6] + s8[7]));\n";
            else
                s << "            for (var d = 0u; d < " << D << "u; d++) { sc += Qs[d] * ldk(kr + d * P.sk[3]); }\n";
            s << "            sc = sc * P.scale;\n";
            if (has_bias)
                s << "            let bv = ldb(P.offs2[0] + b * P.sb[0] + h * P.sb[1] + qi * P.sb[2] + j * P.sb[3]);\n"
                     "            if (isinf_f(bv) && signbit_f(bv)) { visible = false; } else { sc += bv; }\n";
            s << "        }\n"
                 "        if (visible) {\n"
                 "            if (cnt == 0u || sc > mx) { mx = sc; }\n"
                 "            cnt += 1u;\n"
                 "            S[j - k0] = bitcast<u32>(sc);\n"
                 "        } else {\n"
                 "            S[j - k0] = MASKED;\n"
                 "        }\n"
                 "    }\n"
                 "    R[lid] = mx;\n    RC[lid] = cnt;\n"
                 "    workgroupBarrier();\n"
                 "    for (var st = 128u; st > 0u; st >>= 1u) {\n"
                 "        if (lid < st) {\n"
                 "            let c2 = RC[lid + st];\n"
                 "            if (c2 > 0u && (RC[lid] == 0u || R[lid + st] > R[lid])) { R[lid] = R[lid + st]; }\n"
                 "            RC[lid] += c2;\n"
                 "        }\n"
                 "        workgroupBarrier();\n"
                 "    }\n"
                 "    let m = R[0];\n    let any = RC[0] > 0u;\n"
                 "    workgroupBarrier();\n"
                 // probabilities (unnormalized) and their sum
                 "    var ls = 0.0;\n"
                 "    for (var j = k0 + lid; j < k1; j += 256u) {\n"
                 "        var p = 0.0;\n"
                 "        if (S[j - k0] != MASKED) { p = exp(bitcast<f32>(S[j - k0]) - m); }\n"
                 "        S[j - k0] = bitcast<u32>(p);\n"
                 "        ls += p;\n"
                 "    }\n"
                 "    R[lid] = ls;\n"
                 "    workgroupBarrier();\n"
                 "    for (var st = 128u; st > 0u; st >>= 1u) {\n"
                 "        if (lid < st) { R[lid] += R[lid + st]; }\n"
                 "        workgroupBarrier();\n"
                 "    }\n"
                 "    let lsum = R[0];\n"
                 "    workgroupBarrier();\n"
                 // PV: thread (g, d) sums keys j = g (mod G) for output dim d
              << "    let g = lid / " << DVP << "u;\n    let d = lid % " << DVP << "u;\n"
                 "    var acc = 0.0;\n"
              << "    if (d < " << Dv << "u) {\n"
              << "        for (var j = k0 + g; j < k1; j += " << G << "u) {\n"
                 "            let p = bitcast<f32>(S[j - k0]);\n"
                 "            if (p != 0.0) { acc += p * ldv(vbase + j * P.sv[2] + d * P.sv[3]); }\n"
                 "        }\n"
                 "    }\n"
                 "    R[lid] = acc;\n"
                 "    workgroupBarrier();\n"
              << "    if (lid < " << DVP << "u && d < " << Dv << "u) {\n"
                 "        var y = 0.0;\n"
              << "        for (var gg = 0u; gg < " << G << "u; gg++) { y += R[gg * " << DVP << "u + d]; }\n";
            if (split)
                s << "        let pb = P.offs[3] + (((b * P.Hq + h) * P.Lq + qi) * P.NS + sid) * " << Dv + 2 << "u;\n"
                     "        st_part(pb + 2u + d, select(0.0, y, any));\n"
                     "        if (lid == 0u) { st_part(pb, select(0.0, m, any)); st_part(pb + 1u, select(0.0, lsum, any)); }\n"
                     "    }\n"
                     "}\n";
            else
                s << "        if (any) { y = y / lsum; } else { y = 0.0; }\n"
                     "        let obase = P.offs[3] + b * P.so[0] + h * P.so[1] + qi * P.so[2];\n"
                  << "        raw_sto(obase + d * P.so[3], " << wgsl::convert("y", CT::F32, wgsl::compute_type(out.scalar_type())) << ");\n"
                     "    }\n"
                     "    if (lid == 0u) {\n"
                     "        var lv = bitcast<f32>(INF_BITS | 0x80000000u);\n"
                     "        if (any) { lv = m + log(lsum); }\n"
                     "        raw_stl(P.offs2[1] + (b * P.Hq + h) * P.Lq + qi, lv);\n"
                     "    }\n"
                     "}\n";
            return s.str();
        };
        const auto &pipeline = core::get_pipeline(key.str(), make, "attention_decode");

        auto u = [](int64_t x) { return checked_u32(x, "attention size"); };
        core::Params p;
        p.u32(u(B)).u32(u(Hq)).u32(u(Hk)).u32(u(Lq)).u32(u(Lk)).u32(causal ? 1u : 0u).u32(u(NS)).u32(0);
        for (const at::Tensor *t : {&q, &k, &v, &out})
            for (int d = 0; d < 4; ++d)
                p.u32(u(t->stride(d)));
        for (int d = 0; d < 4; ++d)
            p.u32(has_bias ? u(bias.stride(d)) : 0u);
        p.u32(bs.offset(q_r)).u32(bs.offset(k_r)).u32(bs.offset(v_r)).u32(bs.offset(out_r));
        p.u32(has_bias ? bs.offset(b_r) : 0u).u32(split ? 0u : bs.offset(lse_r)).u32(0).u32(0);
        const float sc = static_cast<float>(scale);
        uint32_t scw;
        std::memcpy(&scw, &sc, 4);
        p.u32(scw).u32(0).u32(0).u32(0);
        const uint32_t maxd = core::Context::get().limits().maxComputeWorkgroupsPerDimension;
        core::Grid g;
        g.x = static_cast<uint32_t>(Lq * NS);
        g.y = static_cast<uint32_t>(Hq);
        g.z = static_cast<uint32_t>(B);
        TORCH_CHECK(g.x <= maxd && g.y <= maxd && g.z <= maxd, "torch_webgpu: attention too large for one dispatch");
        core::launch(pipeline, bs.buffers(), p.data(), p.size(), g);
        if (!split)
            return;

        // Merge the splits: m = max m_s, L = sum l_s e^(m_s - m),
        // out = sum o_s e^(m_s - m) / L (splits without visible keys have l = 0).
        core::BindingSet cbs;
        const int c_part = cbs.add(part);
        const int c_out = cbs.add(out, true, wgsl::is_subword(out.scalar_type()));
        const int c_lse = cbs.add(lse, true);
        cbs.finalize();
        auto cbind = [&](int r) { return "buf" + std::to_string(cbs.binding(r)); };
        std::ostringstream ckey;
        ckey << "attn_decode_combine|" << Dv << '|' << int(out.scalar_type()) << '|' << cbs.binding(c_part)
             << cbs.binding(c_out) << cbs.binding(c_lse) << cbs.bindings().size();
        auto cmake = [&]()
        {
            std::ostringstream s;
            s << wgsl::prelude() << math_wgsl();
            s << "struct Params {\n"
                 "  B: u32, Hq: u32, Lq: u32, NS: u32,\n"
                 "  so: vec4<u32>,\n"
                 "  offs: vec4<u32>,\n"
                 "};\n";
            const auto &bb = cbs.bindings();
            for (size_t i = 0; i < bb.size(); ++i)
                s << "@group(0) @binding(" << i << ") var<storage, " << (bb[i].written ? "read_write" : "read") << "> buf" << i
                  << ": array<" << (bb[i].atomic ? "atomic<u32>" : "u32") << ">;\n";
            s << "@group(0) @binding(" << bb.size() << ") var<uniform> P: Params;\n";
            s << "fn ldp(i: u32) -> f32 { return bitcast<f32>(" << cbind(c_part) << "[P.offs[0] + i]); }\n";
            s << wgsl::store_fn("raw_sto", cbind(c_out), out.scalar_type());
            s << wgsl::store_fn("raw_stl", cbind(c_lse), at::kFloat);
            s << "const W: u32 = " << Dv + 2 << "u;\n"
                 "@compute @workgroup_size(256)\n"
                 "fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) l3: vec3<u32>) {\n"
                 "    let lid = l3.x;\n    let qi = wid.x;\n    let h = wid.y;\n    let b = wid.z;\n"
                 "    let base = ((b * P.Hq + h) * P.Lq + qi) * P.NS * W;\n"
                 "    var m = 0.0;\n    var any = false;\n"
                 "    for (var s = 0u; s < P.NS; s++) {\n"
                 "        if (ldp(base + s * W + 1u) > 0.0) {\n"
                 "            let ms = ldp(base + s * W);\n"
                 "            if (!any || ms > m) { m = ms; }\n"
                 "            any = true;\n"
                 "        }\n"
                 "    }\n"
                 "    var L = 0.0;\n"
                 "    for (var s = 0u; s < P.NS; s++) {\n"
                 "        let ls = ldp(base + s * W + 1u);\n"
                 "        if (ls > 0.0) { L += ls * exp(ldp(base + s * W) - m); }\n"
                 "    }\n"
              << "    for (var d = lid; d < " << Dv << "u; d += 256u) {\n"
                 "        var y = 0.0;\n"
                 "        for (var s = 0u; s < P.NS; s++) {\n"
                 "            let ls = ldp(base + s * W + 1u);\n"
                 "            if (ls > 0.0) { y += ldp(base + s * W + 2u + d) * exp(ldp(base + s * W) - m); }\n"
                 "        }\n"
                 "        if (any) { y = y / L; } else { y = 0.0; }\n"
                 "        let obase = P.offs[1] + b * P.so[0] + h * P.so[1] + qi * P.so[2];\n"
              << "        raw_sto(obase + d * P.so[3], " << wgsl::convert("y", CT::F32, wgsl::compute_type(out.scalar_type())) << ");\n"
                 "    }\n"
                 "    if (lid == 0u) {\n"
                 "        var lv = bitcast<f32>(INF_BITS | 0x80000000u);\n"
                 "        if (any) { lv = m + log(L); }\n"
                 "        raw_stl(P.offs[2] + (b * P.Hq + h) * P.Lq + qi, lv);\n"
                 "    }\n"
                 "}\n";
            return s.str();
        };
        const auto &cpipe = core::get_pipeline(ckey.str(), cmake, "attention_decode_combine");
        core::Params cp;
        cp.u32(u(B)).u32(u(Hq)).u32(u(Lq)).u32(u(NS));
        for (int d = 0; d < 4; ++d)
            cp.u32(u(out.stride(d)));
        cp.u32(cbs.offset(c_part)).u32(cbs.offset(c_out)).u32(cbs.offset(c_lse)).u32(0);
        core::Grid cg;
        cg.x = static_cast<uint32_t>(Lq);
        cg.y = static_cast<uint32_t>(Hq);
        cg.z = static_cast<uint32_t>(B);
        core::launch(cpipe, cbs.buffers(), cp.data(), cp.size(), cg);
    }
}
