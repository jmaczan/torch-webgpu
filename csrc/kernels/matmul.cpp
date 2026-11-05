#include "kernels/matmul.h"

#include "core/allocator.h"
#include "core/context.h"
#include "core/kernel.h"
#include "kernels/elementwise.h"
#include "kernels/wgsl.h"

#include <c10/util/Exception.h>

#include <sstream>

namespace torch_webgpu::kernels
{
    namespace
    {
        constexpr int BM = 64, BN = 64, BK = 16;

        uint32_t u32_of(int64_t v, const char *what)
        {
            TORCH_CHECK(v >= 0 && v <= static_cast<int64_t>(UINT32_MAX), "torch_webgpu: matmul ", what, " (", v,
                        ") exceeds 32-bit indexing");
            return static_cast<uint32_t>(v);
        }
    }

    void run_matmul(const at::Tensor &a, const at::Tensor &w, const at::Tensor &out, const at::Tensor &bias,
                    const c10::Scalar &beta, const c10::Scalar &alpha)
    {
        TORCH_INTERNAL_ASSERT(a.dim() == 3 && w.dim() == 3 && out.dim() == 3);
        const int64_t B = out.size(0), M = out.size(1), N = out.size(2), K = a.size(2);
        if (B == 0 || M == 0 || N == 0)
            return;
        const bool has_bias = bias.defined() && !(beta.isIntegral(false) ? beta.toLong() == 0 : beta.toDouble() == 0.0);
        if (has_bias)
            TORCH_INTERNAL_ASSERT(bias.dim() == 3);

        const CT ct = wgsl::compute_type(out.scalar_type());
        TORCH_CHECK(ct != CT::Bool, "torch_webgpu: matmul is not supported for bool tensors");
        const auto ta = a.scalar_type(), tw = w.scalar_type(), to = out.scalar_type();
        const auto tb = has_bias ? bias.scalar_type() : to;

        // Unique buffers: a, w, bias are read-only; out is written.
        struct Bind
        {
            WGPUBuffer h;
            wgpu::Buffer b;
        };
        std::vector<Bind> binds;
        auto bind_of = [&](const at::Tensor &t)
        {
            auto &buf = core::buffer_of(t);
            for (size_t i = 0; i < binds.size(); ++i)
                if (binds[i].h == buf.Get())
                    return static_cast<int>(i);
            binds.push_back({buf.Get(), buf});
            return static_cast<int>(binds.size() - 1);
        };
        const int out_b = bind_of(out);
        const int a_b = bind_of(a), w_b = bind_of(w);
        const int bias_b = has_bias ? bind_of(bias) : -1;
        TORCH_CHECK(a_b != out_b && w_b != out_b && bias_b != out_b,
                    "torch_webgpu: matmul output must not alias its inputs");

        // Choose load mappings so that consecutive threads read consecutive
        // addresses whenever one of the inner strides is 1.
        const bool a_k_contig = a.stride(2) == 1 || a.size(2) == 1;
        const bool w_n_contig = w.stride(2) == 1 || w.size(2) == 1;

        std::ostringstream key;
        key << "mm|" << int(ta) << int(tw) << int(to) << int(tb) << has_bias << a_k_contig << w_n_contig << '|'
            << a_b << w_b << bias_b << out_b << binds.size();

        auto make_source = [&]()
        {
            const char *T = wgsl::type_name(ct);
            std::ostringstream s;
            s << wgsl::prelude();
            s << "struct Params {\n"
                 "  M: u32, N: u32, K: u32, _p: u32,\n"
                 "  off: vec4<u32>,\n"          // a, w, out, bias
                 "  sa: vec4<u32>, sw: vec4<u32>, so: vec4<u32>, sb: vec4<u32>,\n" // (batch, row, col, -)
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
            s << "var<workgroup> As: array<array<" << T << ", " << BM << ">, " << BK << ">;\n";
            s << "var<workgroup> Ws: array<array<" << T << ", " << BN << ">, " << BK << ">;\n";

            auto cv = [&](const std::string &e, c10::ScalarType t)
            { return wgsl::convert(e, wgsl::compute_type(t), ct); };

            s << "@compute @workgroup_size(16, 16, 1)\n"
                 "fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {\n"
                 "    let tx = lid.x;\n    let ty = lid.y;\n    let tid = ty * 16u + tx;\n"
                 "    let b = wid.z;\n"
                 "    let m0 = wid.y * " << BM << "u;\n    let n0 = wid.x * " << BN << "u;\n"
                 "    let a_base = P.off[0] + b * P.sa[0];\n"
                 "    let w_base = P.off[1] + b * P.sw[0];\n"
                 "    var acc: array<array<" << T << ", 4>, 4>;\n"
                 "    for (var i = 0u; i < 4u; i++) { for (var j = 0u; j < 4u; j++) { acc[i][j] = " << T << "(0); } }\n"
                 "    for (var k0 = 0u; k0 < P.K; k0 += " << BK << "u) {\n"
                 "        for (var l = 0u; l < 4u; l++) {\n"
                 "            let idx = tid + l * 256u;\n";
            // A tile: BM x BK
            if (a_k_contig)
                s << "            let ar = idx / " << BK << "u;\n            let ac = idx % " << BK << "u;\n";
            else
                s << "            let ar = idx % " << BM << "u;\n            let ac = idx / " << BM << "u;\n";
            s << "            let gm = m0 + ar;\n            let gk = k0 + ac;\n"
                 "            var av = " << T << "(0);\n"
                 "            if (gm < P.M && gk < P.K) { av = " << cv("ld_a(a_base + gm * P.sa[1] + gk * P.sa[2])", ta) << "; }\n"
                 "            As[ac][ar] = av;\n";
            // W tile: BK x BN
            if (w_n_contig)
                s << "            let wr = idx / " << BN << "u;\n            let wc = idx % " << BN << "u;\n";
            else
                s << "            let wr = idx % " << BK << "u;\n            let wc = idx / " << BK << "u;\n";
            s << "            let gk2 = k0 + wr;\n            let gn = n0 + wc;\n"
                 "            var wv = " << T << "(0);\n"
                 "            if (gk2 < P.K && gn < P.N) { wv = " << cv("ld_w(w_base + gk2 * P.sw[1] + gn * P.sw[2])", tw) << "; }\n"
                 "            Ws[wr][wc] = wv;\n"
                 "        }\n"
                 "        workgroupBarrier();\n"
                 "        for (var kk = 0u; kk < " << BK << "u; kk++) {\n"
                 "            var av4: array<" << T << ", 4>;\n            var wv4: array<" << T << ", 4>;\n"
                 "            for (var i = 0u; i < 4u; i++) { av4[i] = As[kk][ty + 16u * i]; }\n"
                 "            for (var j = 0u; j < 4u; j++) { wv4[j] = Ws[kk][tx + 16u * j]; }\n"
                 "            for (var i = 0u; i < 4u; i++) { for (var j = 0u; j < 4u; j++) { acc[i][j] += av4[i] * wv4[j]; } }\n"
                 "        }\n"
                 "        workgroupBarrier();\n"
                 "    }\n";
            s << "    let alpha = " << wgsl::scalar_from_word("P.alpha", ct) << ";\n";
            if (has_bias)
                s << "    let beta = " << wgsl::scalar_from_word("P.beta", ct) << ";\n";
            s << "    for (var i = 0u; i < 4u; i++) {\n"
                 "        let gm = m0 + ty + 16u * i;\n"
                 "        if (gm >= P.M) { continue; }\n"
                 "        for (var j = 0u; j < 4u; j++) {\n"
                 "            let gn = n0 + tx + 16u * j;\n"
                 "            if (gn >= P.N) { continue; }\n"
                 "            var v = alpha * acc[i][j];\n";
            if (has_bias)
                s << "            v += beta * " << cv("ld_b(P.off[3] + b * P.sb[0] + gm * P.sb[1] + gn * P.sb[2])", tb) << ";\n";
            s << "            st_o(P.off[2] + b * P.so[0] + gm * P.so[1] + gn * P.so[2], "
              << wgsl::convert("v", ct, wgsl::compute_type(to)) << ");\n"
                 "        }\n    }\n}\n";
            return s.str();
        };

        const auto &pipeline = core::get_pipeline(key.str(), make_source, "matmul");

        core::Params p;
        p.u32(u32_of(M, "M")).u32(u32_of(N, "N")).u32(u32_of(K, "K")).u32(0);
        p.u32(u32_of(a.storage_offset(), "offset")).u32(u32_of(w.storage_offset(), "offset"));
        p.u32(u32_of(out.storage_offset(), "offset")).u32(has_bias ? u32_of(bias.storage_offset(), "offset") : 0u);
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
        p.u32(scalar_word(alpha, ct)).u32(has_bias ? scalar_word(beta, ct) : 0u).u32(0).u32(0);

        const uint32_t maxd = core::Context::get().limits().maxComputeWorkgroupsPerDimension;
        core::Grid grid;
        grid.x = static_cast<uint32_t>((N + BN - 1) / BN);
        grid.y = static_cast<uint32_t>((M + BM - 1) / BM);
        grid.z = static_cast<uint32_t>(B);
        TORCH_CHECK(grid.x <= maxd && grid.y <= maxd && grid.z <= maxd, "torch_webgpu: matmul too large for one dispatch");

        std::vector<core::BufferBinding> bufs;
        for (auto &b : binds)
            bufs.push_back({b.b});
        core::launch(pipeline, bufs, p.data(), p.size(), grid);
    }
}
