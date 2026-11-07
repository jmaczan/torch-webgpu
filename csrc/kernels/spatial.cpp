#include "kernels/spatial.h"

#include "core/binding_set.h"
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
        constexpr uint32_t kWG = 256;

        // Shared Params: sizes, strides (elements), window parameters, offsets.
        // in: N C H W strides; out: oh ow + strides.
        std::string params_struct()
        {
            return "struct Params {\n"
                   "  n: u32, c: u32, h: u32, w: u32,\n"
                   "  oh: u32, ow: u32, total: u32, row_stride: u32,\n"
                   "  is: vec4<u32>, os: vec4<u32>,\n"
                   "  kh: u32, kw: u32, sh: u32, sw: u32,\n"
                   "  ph: u32, pw: u32, dh: u32, dw: u32,\n"
                   "  off_in: u32, off_out: u32, off_idx: u32, extra: u32,\n"
                   "  fa: f32, fb: f32, _f0: u32, _f1: u32,\n"
                   "};\n";
        }

        // Thread prologue: decodes (n, c, y, x) of output element t.
        std::string prologue()
        {
            return "    let t = gid.x + gid.y * P.row_stride;\n"
                   "    if (t >= P.total) { return; }\n"
                   "    let x = t % P.ow;\n    let y = (t / P.ow) % P.oh;\n"
                   "    let c = (t / (P.ow * P.oh)) % P.c;\n    let n = t / (P.ow * P.oh * P.c);\n"
                   "    let ib = P.off_in + n * P.is[0] + c * P.is[1];\n"
                   "    let ob = P.off_out + n * P.os[0] + c * P.os[1] + y * P.os[2] + x * P.os[3];\n";
        }

        struct Launch
        {
            core::BindingSet bs;
            int in_r = -1, out_r = -1, idx_r = -1;
        };

        void fill(core::Params &p, const at::Tensor &in, const at::Tensor &out, const Launch &L, uint32_t row_stride,
                  const Pool2dParams *pp, uint32_t extra, float fa, float fb)
        {
            auto u = [](int64_t v) { return checked_u32(v, "spatial size"); };
            p.u32(u(in.size(0))).u32(u(in.size(1))).u32(u(in.size(2))).u32(u(in.size(3)));
            p.u32(u(out.size(2))).u32(u(out.size(3))).u32(u(out.numel())).u32(row_stride);
            for (int d = 0; d < 4; ++d)
                p.u32(u(in.stride(d)));
            for (int d = 0; d < 4; ++d)
                p.u32(u(out.stride(d)));
            if (pp)
                p.u32(u(pp->kh)).u32(u(pp->kw)).u32(u(pp->sh)).u32(u(pp->sw)).u32(u(pp->ph)).u32(u(pp->pw)).u32(u(pp->dh)).u32(u(pp->dw));
            else
                for (int i = 0; i < 8; ++i)
                    p.u32(0);
            p.u32(L.bs.offset(L.in_r)).u32(L.bs.offset(L.out_r)).u32(L.idx_r >= 0 ? L.bs.offset(L.idx_r) : 0u).u32(extra);
            uint32_t wa, wb;
            std::memcpy(&wa, &fa, 4);
            std::memcpy(&wb, &fb, 4);
            p.u32(wa).u32(wb).u32(0).u32(0);
        }

        std::string bindings(const Launch &L, const at::Tensor &out, const at::Tensor &indices)
        {
            std::ostringstream s;
            const auto &b = L.bs.bindings();
            for (size_t i = 0; i < b.size(); ++i)
                s << "@group(0) @binding(" << i << ") var<storage, " << (b[i].written ? "read_write" : "read") << "> buf"
                  << i << ": array<" << (b[i].atomic ? "atomic<u32>" : "u32") << ">;\n";
            s << "@group(0) @binding(" << b.size() << ") var<uniform> P: Params;\n";
            return s.str();
        }
    }

    void run_max_pool2d(const at::Tensor &input, const at::Tensor &out, const at::Tensor &indices, const Pool2dParams &pp)
    {
        if (out.numel() == 0)
            return;
        const bool with_idx = indices.defined();
        Launch L;
        L.out_r = L.bs.add(out, true, wgsl::is_subword(out.scalar_type()));
        if (with_idx)
            L.idx_r = L.bs.add(indices, true);
        L.in_r = L.bs.add(input);
        L.bs.finalize();
        std::ostringstream key;
        key << "maxpool2d|" << int(input.scalar_type()) << int(out.scalar_type()) << with_idx << L.bs.binding(L.in_r)
            << L.bs.binding(L.out_r) << (with_idx ? L.bs.binding(L.idx_r) : -1);
        auto make = [&]()
        {
            std::ostringstream s;
            s << wgsl::prelude() << math_wgsl() << params_struct() << bindings(L, out, indices);
            s << wgsl::load_fn("ld_in", "buf" + std::to_string(L.bs.binding(L.in_r)), input.scalar_type(), false);
            s << wgsl::store_fn("st_out", "buf" + std::to_string(L.bs.binding(L.out_r)), out.scalar_type());
            if (with_idx)
                s << wgsl::store_fn("st_idx", "buf" + std::to_string(L.bs.binding(L.idx_r)), c10::kLong);
            s << "@compute @workgroup_size(" << kWG << ")\nfn main(@builtin(global_invocation_id) gid: vec3<u32>) {\n"
              << prologue()
              // As ATen's cpu_max_pool: window start moved into bounds by
              // whole dilation steps; the last NaN wins; index of the first
              // in-bounds position for an all -inf window.
              << "    var h0 = i32(y * P.sh) - i32(P.ph);\n    var w0 = i32(x * P.sw) - i32(P.pw);\n"
              << "    let h1 = min(h0 + i32((P.kh - 1u) * P.dh + 1u), i32(P.h));\n"
              << "    let w1 = min(w0 + i32((P.kw - 1u) * P.dw + 1u), i32(P.w));\n"
              << "    while (h0 < 0) { h0 += i32(P.dh); }\n    while (w0 < 0) { w0 += i32(P.dw); }\n"
              // CPU: maxval = -inf; update if (val > maxval || isnan(val)).
              // Without a -inf constant in comparisons (driver fast-math):
              // before the first update, any value but -inf qualifies.
              << "    var best_i = u32(h0) * P.w + u32(w0);\n"
              << "    var best = 0.0;\n    var have = false;\n"
              << "    for (var ih = h0; ih < h1; ih += i32(P.dh)) {\n"
              << "        for (var iw = w0; iw < w1; iw += i32(P.dw)) {\n"
              << "            let v = " << wgsl::convert("ld_in(ib + u32(ih) * P.is[2] + u32(iw) * P.is[3])", wgsl::compute_type(input.scalar_type()), CT::F32) << ";\n"
              << "            var upd = isnan_f(v);\n"
              << "            if (have) { upd = upd || v > best; } else { upd = upd || !(isinf_f(v) && signbit_f(v)); }\n"
              << "            if (upd) { best = v; best_i = u32(ih) * P.w + u32(iw); have = true; }\n"
              << "        }\n    }\n"
              << "    if (!have) { best = bitcast<f32>(INF_BITS | 0x80000000u); }\n"
              << "    st_out(ob, " << wgsl::convert("best", CT::F32, wgsl::compute_type(out.scalar_type())) << ");\n";
            if (with_idx)
                s << "    let ix = P.off_idx + n * P.extra + c * (P.oh * P.ow) + y * P.ow + x;\n"
                  << "    st_idx(ix, i64(best_i, 0u));\n";
            s << "}\n";
            return s.str();
        };
        const auto &pipeline = core::get_pipeline(key.str(), make, "max_pool2d");
        auto grid = core::grid_1d(out.numel(), kWG);
        core::Params p;
        // indices are contiguous [N, C, oh, ow]: per-batch stride in `extra`.
        TORCH_INTERNAL_ASSERT(!with_idx || indices.is_contiguous());
        fill(p, input, out, L, grid.row_stride, &pp, with_idx ? checked_u32(out.size(1) * out.size(2) * out.size(3), "size") : 0u, 0.f, 0.f);
        core::launch(pipeline, L.bs.buffers(), p.data(), p.size(), grid);
    }

    void run_avg_pool2d(const at::Tensor &input, const at::Tensor &out, const Pool2dParams &pp, bool count_include_pad,
                        std::optional<int64_t> divisor_override)
    {
        if (out.numel() == 0)
            return;
        Launch L;
        L.out_r = L.bs.add(out, true, wgsl::is_subword(out.scalar_type()));
        L.in_r = L.bs.add(input);
        L.bs.finalize();
        std::ostringstream key;
        key << "avgpool2d|" << int(input.scalar_type()) << int(out.scalar_type()) << count_include_pad
            << divisor_override.has_value() << L.bs.binding(L.in_r) << L.bs.binding(L.out_r);
        auto make = [&]()
        {
            std::ostringstream s;
            s << wgsl::prelude() << math_wgsl() << params_struct() << bindings(L, out, at::Tensor());
            s << wgsl::load_fn("ld_in", "buf" + std::to_string(L.bs.binding(L.in_r)), input.scalar_type(), false);
            s << wgsl::store_fn("st_out", "buf" + std::to_string(L.bs.binding(L.out_r)), out.scalar_type());
            s << "@compute @workgroup_size(" << kWG << ")\nfn main(@builtin(global_invocation_id) gid: vec3<u32>) {\n"
              << prologue()
              // As ATen's cpu_avg_pool: pool size counted with padding before
              // clipping the window to the input.
              << "    var h0 = i32(y * P.sh) - i32(P.ph);\n    var w0 = i32(x * P.sw) - i32(P.pw);\n"
              << "    var h1 = min(h0 + i32(P.kh), i32(P.h + P.ph));\n    var w1 = min(w0 + i32(P.kw), i32(P.w + P.pw));\n"
              << "    let pool = (h1 - h0) * (w1 - w0);\n"
              << "    h0 = max(h0, 0);\n    w0 = max(w0, 0);\n    h1 = min(h1, i32(P.h));\n    w1 = min(w1, i32(P.w));\n"
              << "    if (h0 >= h1 || w0 >= w1) { st_out(ob, " << wgsl::convert("0.0", CT::F32, wgsl::compute_type(out.scalar_type())) << "); return; }\n"
              << "    var sum = 0.0;\n"
              << "    for (var ih = h0; ih < h1; ih++) {\n"
              << "        for (var iw = w0; iw < w1; iw++) {\n"
              << "            sum += " << wgsl::convert("ld_in(ib + u32(ih) * P.is[2] + u32(iw) * P.is[3])", wgsl::compute_type(input.scalar_type()), CT::F32) << ";\n"
              << "        }\n    }\n";
            if (divisor_override.has_value())
                s << "    let div = P.fa;\n";
            else if (count_include_pad)
                s << "    let div = f32(pool);\n";
            else
                s << "    let div = f32((h1 - h0) * (w1 - w0));\n";
            s << "    st_out(ob, " << wgsl::convert("sum / div", CT::F32, wgsl::compute_type(out.scalar_type())) << ");\n}\n";
            return s.str();
        };
        const auto &pipeline = core::get_pipeline(key.str(), make, "avg_pool2d");
        auto grid = core::grid_1d(out.numel(), kWG);
        core::Params p;
        fill(p, input, out, L, grid.row_stride, &pp, 0u,
             divisor_override.has_value() ? static_cast<float>(*divisor_override) : 0.f, 0.f);
        core::launch(pipeline, L.bs.buffers(), p.data(), p.size(), grid);
    }

    void run_upsample2d(const at::Tensor &input, const at::Tensor &out, Upsample mode, bool align_corners, float scale_h,
                        float scale_w)
    {
        if (out.numel() == 0)
            return;
        Launch L;
        L.out_r = L.bs.add(out, true, wgsl::is_subword(out.scalar_type()));
        L.in_r = L.bs.add(input);
        L.bs.finalize();
        std::ostringstream key;
        key << "upsample2d|" << int(mode) << align_corners << int(input.scalar_type()) << int(out.scalar_type())
            << L.bs.binding(L.in_r) << L.bs.binding(L.out_r);
        auto make = [&]()
        {
            std::ostringstream s;
            s << wgsl::prelude() << math_wgsl() << params_struct() << bindings(L, out, at::Tensor());
            s << wgsl::load_fn("ld_in", "buf" + std::to_string(L.bs.binding(L.in_r)), input.scalar_type(), false);
            s << wgsl::store_fn("st_out", "buf" + std::to_string(L.bs.binding(L.out_r)), out.scalar_type());
            const std::string load = wgsl::convert("ld_in(ib + u32(yy) * P.is[2] + u32(xx) * P.is[3])",
                                                   wgsl::compute_type(input.scalar_type()), CT::F32);
            s << "fn px(ib: u32, yy: u32, xx: u32) -> f32 { return " << load << "; }\n";
            s << "@compute @workgroup_size(" << kWG << ")\nfn main(@builtin(global_invocation_id) gid: vec3<u32>) {\n"
              << prologue();
            if (mode == Upsample::Bilinear)
            {
                // area_pixel_compute_source_index + ATen's lambdas (in float).
                auto src = [&](const char *v, const char *sc)
                {
                    return align_corners ? std::string(sc) + " * f32(" + v + ")"
                                         : "max(" + std::string(sc) + " * (f32(" + v + ") + 0.5) - 0.5, 0.0)";
                };
                s << "    let hr = " << src("y", "P.fa") << ";\n    let wr = " << src("x", "P.fb") << ";\n"
                  << "    let h1 = u32(hr);\n    let w1 = u32(wr);\n"
                  << "    let h1p = select(0u, 1u, h1 < P.h - 1u);\n    let w1p = select(0u, 1u, w1 < P.w - 1u);\n"
                  << "    let h1l = hr - f32(h1);\n    let h0l = 1.0 - h1l;\n"
                  << "    let w1l = wr - f32(w1);\n    let w0l = 1.0 - w1l;\n"
                  << "    let v = h0l * (w0l * px(ib, h1, w1) + w1l * px(ib, h1, w1 + w1p)) +\n"
                  << "            h1l * (w0l * px(ib, h1 + h1p, w1) + w1l * px(ib, h1 + h1p, w1 + w1p));\n";
            }
            else
            {
                const char *off = mode == Upsample::NearestExact ? " + 0.5" : "";
                s << "    let sy = min(u32(floor((f32(y)" << off << ") * P.fa)), P.h - 1u);\n"
                  << "    let sx = min(u32(floor((f32(x)" << off << ") * P.fb)), P.w - 1u);\n"
                  << "    let v = px(ib, sy, sx);\n";
            }
            s << "    st_out(ob, " << wgsl::convert("v", CT::F32, wgsl::compute_type(out.scalar_type())) << ");\n}\n";
            return s.str();
        };
        const auto &pipeline = core::get_pipeline(key.str(), make, "upsample2d");
        auto grid = core::grid_1d(out.numel(), kWG);
        core::Params p;
        fill(p, input, out, L, grid.row_stride, nullptr, 0u, scale_h, scale_w);
        core::launch(pipeline, L.bs.buffers(), p.data(), p.size(), grid);
    }
}
