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
}
