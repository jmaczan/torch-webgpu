#include "kernels/custom_gemm.h"

#include "core/binding_set.h"
#include "core/context.h"
#include "core/kernel.h"
#include "kernels/iter_layout.h"
#include "kernels/wgsl.h"

#include <c10/util/Exception.h>

#include <sstream>

namespace torch_webgpu::kernels
{
    namespace
    {
        constexpr int BM = 64, BN = 64, BK = 16;

        void launch(const CustomGemm &g, int64_t slices, const at::Tensor &partial)
        {
            if (g.M == 0 || g.N == 0 || g.Z == 0)
                return;
            const bool split = slices > 1;
            // Split-K writes partials instead of storing: written operands are
            // only used by the store, so they aren't bound (auto layouts drop
            // bindings a shader never touches).
            std::vector<GemmOperand> ops;
            for (const auto &o : g.operands)
                if (!(split && o.written))
                    ops.push_back(o);
            if (split)
                ops.push_back({"partial", partial, true});
            core::BindingSet bs;
            std::vector<int> regs;
            for (auto &o : ops)
                regs.push_back(bs.add(o.tensor, o.written, o.written && wgsl::is_subword(o.tensor.scalar_type())));
            bs.finalize();

            std::ostringstream key;
            key << "cgemm|" << g.name << '|' << g.key << '|' << split << g.a_k_fast << g.b_n_fast << '|';
            for (size_t i = 0; i < ops.size(); ++i)
                key << ops[i].name << ':' << int(ops[i].tensor.scalar_type()) << ':' << bs.binding(regs[i]) << ',';
            key << bs.bindings().size();
            const int noffs = static_cast<int>(ops.size()), nparams = static_cast<int>(g.params.size());

            auto make_source = [&]()
            {
                std::ostringstream s;
                s << wgsl::prelude();
                s << "struct Params {\n  M: u32, N: u32, K: u32, Z: u32,\n  kc: u32, _p0: u32, _p1: u32, _p2: u32,\n"
                  << "  offs: array<vec4<u32>, " << (noffs + 3) / 4 << ">,\n";
                if (nparams)
                    s << "  v: array<vec4<u32>, " << (nparams + 3) / 4 << ">,\n";
                s << "};\n";
                const auto &binds = bs.bindings();
                for (size_t i = 0; i < binds.size(); ++i)
                    s << "@group(0) @binding(" << i << ") var<storage, " << (binds[i].written ? "read_write" : "read")
                      << "> buf" << i << ": array<" << (binds[i].atomic ? "atomic<u32>" : "u32") << ">;\n";
                s << "@group(0) @binding(" << binds.size() << ") var<uniform> P: Params;\n";
                // P.v[i] as a flat index
                if (nparams)
                    s << "fn pv(i: u32) -> u32 { return P.v[i / 4u][i % 4u]; }\n";
                for (size_t i = 0; i < ops.size(); ++i)
                {
                    const auto &o = ops[i];
                    const std::string var = "buf" + std::to_string(bs.binding(regs[i]));
                    const auto t = o.tensor.scalar_type();
                    const std::string off = "P.offs[" + std::to_string(i / 4) + "][" + std::to_string(i % 4) + "]";
                    s << wgsl::load_fn("raw_ld_" + o.name, var, t, bs.bindings()[bs.binding(regs[i])].atomic);
                    s << "fn ld_" << o.name << "(i: u32) -> f32 { return "
                      << wgsl::convert("raw_ld_" + o.name + "(" + off + " + i)", wgsl::compute_type(t), CT::F32) << "; }\n";
                    if (o.written)
                    {
                        s << wgsl::store_fn("raw_st_" + o.name, var, t);
                        s << "fn st_" << o.name << "(i: u32, v: f32) { raw_st_" << o.name << "(" << off << " + i, "
                          << wgsl::convert("v", CT::F32, wgsl::compute_type(t)) << "); }\n";
                    }
                }
                s << "fn load_a(z: u32, m: u32, k: u32) -> f32 {\n" << g.load_a << "\n}\n";
                s << "fn load_b(z: u32, k: u32, n: u32) -> f32 {\n" << g.load_b << "\n}\n";
                if (!split)
                    s << "fn store_c(z: u32, m: u32, n: u32, v: f32) {\n" << g.store << "\n}\n";
                s << "var<workgroup> As: array<array<f32, " << BM << ">, " << BK << ">;\n"
                  << "var<workgroup> Bs: array<array<f32, " << BN << ">, " << BK << ">;\n"
                  << "@compute @workgroup_size(16, 16, 1)\n"
                  << "fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {\n"
                  << "    let tx = lid.x;\n    let ty = lid.y;\n    let tid = ty * 16u + tx;\n"
                  << "    let z = wid.z % P.Z;\n    let sl = wid.z / P.Z;\n"
                  << "    let kbeg = sl * P.kc;\n    let kend = min(P.K, kbeg + P.kc);\n"
                  << "    let m0 = wid.y * " << BM << "u;\n    let n0 = wid.x * " << BN << "u;\n"
                  << "    var acc: array<array<f32, 4>, 4>;\n"
                  << "    for (var k0 = kbeg; k0 < kend; k0 += " << BK << "u) {\n"
                  << "        for (var l = 0u; l < 4u; l++) {\n"
                  << "            let idx = tid + l * 256u;\n";
                if (g.a_k_fast)
                    s << "            let ar = idx / " << BK << "u;\n            let ac = idx % " << BK << "u;\n";
                else
                    s << "            let ar = idx % " << BM << "u;\n            let ac = idx / " << BM << "u;\n";
                s << "            let gm = m0 + ar;\n            let gk = k0 + ac;\n"
                  << "            var av = 0.0;\n"
                  << "            if (gm < P.M && gk < kend) { av = load_a(z, gm, gk); }\n"
                  << "            As[ac][ar] = av;\n";
                if (g.b_n_fast)
                    s << "            let br = idx / " << BN << "u;\n            let bc = idx % " << BN << "u;\n";
                else
                    s << "            let br = idx % " << BK << "u;\n            let bc = idx / " << BK << "u;\n";
                s << "            let gk2 = k0 + br;\n            let gn = n0 + bc;\n"
                  << "            var bv = 0.0;\n"
                  << "            if (gk2 < kend && gn < P.N) { bv = load_b(z, gk2, gn); }\n"
                  << "            Bs[br][bc] = bv;\n"
                  << "        }\n"
                  << "        workgroupBarrier();\n"
                  << "        for (var kk = 0u; kk < " << BK << "u; kk++) {\n"
                  << "            var a4: array<f32, 4>;\n            var b4: array<f32, 4>;\n"
                  << "            for (var i = 0u; i < 4u; i++) { a4[i] = As[kk][ty + 16u * i]; }\n"
                  << "            for (var j = 0u; j < 4u; j++) { b4[j] = Bs[kk][tx + 16u * j]; }\n"
                  << "            for (var i = 0u; i < 4u; i++) { for (var j = 0u; j < 4u; j++) { acc[i][j] += a4[i] * b4[j]; } }\n"
                  << "        }\n"
                  << "        workgroupBarrier();\n"
                  << "    }\n"
                  << "    for (var i = 0u; i < 4u; i++) {\n"
                  << "        let gm = m0 + ty + 16u * i;\n"
                  << "        if (gm >= P.M) { continue; }\n"
                  << "        for (var j = 0u; j < 4u; j++) {\n"
                  << "            let gn = n0 + tx + 16u * j;\n"
                  << "            if (gn >= P.N) { continue; }\n";
                if (split)
                    s << "            st_partial(((sl * P.Z + z) * P.M + gm) * P.N + gn, acc[i][j]);\n";
                else
                    s << "            store_c(z, gm, gn, acc[i][j]);\n";
                s << "        }\n    }\n}\n";
                return s.str();
            };
            const auto &pipeline = core::get_pipeline(key.str(), make_source, g.name.c_str());

            const int64_t kc = (g.K + slices - 1) / slices;
            core::Params p;
            p.u32(checked_u32(g.M, "M")).u32(checked_u32(g.N, "N")).u32(checked_u32(g.K, "K")).u32(checked_u32(g.Z, "Z"));
            p.u32(checked_u32(kc, "K slice")).u32(0).u32(0).u32(0);
            for (int i = 0; i < (noffs + 3) / 4 * 4; ++i)
                p.u32(i < noffs ? bs.offset(regs[i]) : 0u);
            for (int i = 0; i < (nparams + 3) / 4 * 4; ++i)
                p.u32(i < nparams ? g.params[i] : 0u);
            const uint32_t maxd = core::Context::get().limits().maxComputeWorkgroupsPerDimension;
            core::Grid grid;
            grid.x = static_cast<uint32_t>((g.N + BN - 1) / BN);
            grid.y = static_cast<uint32_t>((g.M + BM - 1) / BM);
            grid.z = static_cast<uint32_t>(g.Z * slices);
            TORCH_CHECK(grid.x <= maxd && grid.y <= maxd && grid.z <= maxd, "torch_webgpu: '", g.name,
                        "' is too large for one dispatch");
            core::launch(pipeline, bs.buffers(), p.data(), p.size(), grid);
        }
    }

    void run_custom_gemm(const CustomGemm &g) { launch(g, 1, at::Tensor()); }

    void run_custom_gemm_split(const CustomGemm &g, int64_t slices, const at::Tensor &partial)
    {
        TORCH_INTERNAL_ASSERT(slices > 1 && partial.is_contiguous() && partial.scalar_type() == at::kFloat &&
                              partial.numel() == slices * g.Z * g.M * g.N);
        launch(g, slices, partial);
    }
}
