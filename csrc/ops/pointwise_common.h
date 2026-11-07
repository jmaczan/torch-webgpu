#pragma once
// Shared machinery for elementwise ATen ops: per-compute-type WGSL
// expressions, the run() launcher, and registration macros that wrap
// PyTorch's structured meta functions (functional / out / in-place).

#include <ATen/ATen.h>
#include <ATen/TensorIterator.h>

#include "kernels/elementwise.h"
#include "kernels/math.h"
#include "kernels/special.h"
#include "ops/common.h"

namespace torch_webgpu::ops
{
    using at::Scalar;
    using at::Tensor;
    using at::TensorIteratorBase;
    using kernels::CT;
    using kernels::ElementwiseOp;

    namespace pw
    {
        // Per-compute-type expressions; nullptr = unsupported for that type.
        struct Expr
        {
            const char *f = nullptr;
            const char *i = nullptr;
            const char *b = nullptr;
            const char *l = nullptr; // int64 (emulated i64)
        };

        struct Opts
        {
            std::vector<Scalar> scalars;
            std::optional<CT> out_ct;     // default: same as the input compute type
            std::optional<CT> scalar_ct;  // default: input compute type
            std::vector<std::optional<CT>> input_cts;
            std::optional<CT> in_ct;      // default: compute type of common dtype
            std::vector<std::string> special; // special functions used (kernels/special.h)
        };

        // Returns false (without launching) when the expression has no
        // variant for the compute type; callers then use the CPU fallback.
        inline bool run(TensorIteratorBase &it, const char *name, const Expr &e, Opts o = {})
        {
            CT ct = o.in_ct ? *o.in_ct : kernels::common_ct(it);
            const char *code = ct == CT::F32 ? e.f : ct == CT::I32 ? e.i
                                                 : ct == CT::I64   ? e.l
                                                                   : e.b;
            if (code == nullptr)
                return false;
            ElementwiseOp op;
            op.name = name;
            op.code = code;
            op.in_ct = ct;
            op.out_ct = o.out_ct.value_or(ct);
            op.input_cts = o.input_cts;
            CT sct = o.scalar_ct.value_or(ct);
            for (auto &s : o.scalars)
                op.scalars.push_back({s, sct});
            op.math = true;
            op.special = o.special;
            kernels::run_elementwise(it, op);
            return true;
        }

        // True when all iterator operands (except CPU scalars) and the common
        // dtype have GPU kernels.
        inline bool iter_ok(TensorIteratorBase &it)
        {
            for (int k = 0; k < it.ntensors(); ++k)
            {
                if (k >= it.noutputs() && it.is_cpu_scalar(k))
                    continue;
                if (!wgsl::is_compute_dtype(it.dtype(k)))
                    return false;
            }
            return true;
        }
    }

// ---------------------------------------------------------------------------
// Registration macros. KERNEL is a callable taking (TensorIteratorBase&).
//
// UNARY: op(self) / op.out(self, out) / op_(self), meta structured_<op>.
#define WG_UNARY(NAME, KERNEL)                                                            \
    Tensor NAME##_fn(const Tensor &self)                                                  \
    {                                                                                     \
        Functional<at::meta::structured_##NAME> op;                                       \
        op.meta(self);                                                                    \
        if (!iter_ok(op) || !KERNEL(op))                                                  \
            return WEBGPU_FALLBACK0(NAME)(self);                                          \
        return std::move(op.outputs_[0]);                                                 \
    }                                                                                     \
    Tensor &NAME##_out_fn(const Tensor &self, Tensor &out)                                \
    {                                                                                     \
        Out<at::meta::structured_##NAME> op(out);                                         \
        op.meta(self);                                                                    \
        if (!iter_ok(op) || !KERNEL(op))                                                  \
            return WEBGPU_FALLBACK(NAME, out)(self, out);                                 \
        op.finish();                                                                      \
        return out;                                                                       \
    }                                                                                     \
    Tensor &NAME##_inplace_fn(Tensor &self)                                               \
    {                                                                                     \
        Inplace<at::meta::structured_##NAME> op(self);                                    \
        op.meta(self);                                                                    \
        if (!iter_ok(op) || !KERNEL(op))                                                  \
            return WEBGPU_FALLBACK0(NAME##_)(self);                                       \
        op.finish();                                                                      \
        return self;                                                                      \
    }
#define WG_REG_UNARY(m, NAME)                                 \
    m.impl(#NAME, TORCH_FN(NAME##_fn));                       \
    m.impl(#NAME ".out", TORCH_FN(NAME##_out_fn));            \
    m.impl(#NAME "_", TORCH_FN(NAME##_inplace_fn));

// Generic: META class, argument list (types+names), call args, fallback
// tokens for the functional / out / inplace overloads.
#define WG_OP3(ID, META, KERNEL, PARAMS, ARGS, FB_F, FB_O, FB_I)                         \
    Tensor ID##_fn(const Tensor &self PARAMS)                                            \
    {                                                                                    \
        Functional<at::meta::META> op;                                                   \
        op.meta(self ARGS);                                                              \
        if (!iter_ok(op) || !KERNEL(op))                                                                \
            return FB_F(self ARGS);                                                                      \
        return std::move(op.outputs_[0]);                                                \
    }                                                                                    \
    Tensor &ID##_out_fn(const Tensor &self PARAMS, Tensor &out)                          \
    {                                                                                    \
        Out<at::meta::META> op(out);                                                     \
        op.meta(self ARGS);                                                              \
        if (!iter_ok(op) || !KERNEL(op))                                                                \
            return FB_O(self ARGS, out);                                                                      \
        op.finish();                                                                     \
        return out;                                                                      \
    }                                                                                    \
    Tensor &ID##_inplace_fn(Tensor &self PARAMS)                                         \
    {                                                                                    \
        Inplace<at::meta::META> op(self);                                                \
        op.meta(self ARGS);                                                              \
        if (!iter_ok(op) || !KERNEL(op))                                                                \
            return FB_I(self ARGS);                                                                      \
        op.finish();                                                                     \
        return self;                                                                     \
    }
// Same without in-place variant.
#define WG_OP2(ID, META, KERNEL, PARAMS, ARGS, FB_F, FB_O)                               \
    Tensor ID##_fn(const Tensor &self PARAMS)                                            \
    {                                                                                    \
        Functional<at::meta::META> op;                                                   \
        op.meta(self ARGS);                                                              \
        if (!iter_ok(op) || !KERNEL(op))                                                                \
            return FB_F(self ARGS);                                                                      \
        return std::move(op.outputs_[0]);                                                \
    }                                                                                    \
    Tensor &ID##_out_fn(const Tensor &self PARAMS, Tensor &out)                          \
    {                                                                                    \
        Out<at::meta::META> op(out);                                                     \
        op.meta(self ARGS);                                                              \
        if (!iter_ok(op) || !KERNEL(op))                                                                \
            return FB_O(self ARGS, out);                                                                      \
        op.finish();                                                                     \
        return out;                                                                      \
    }

#define P1(T1, A1) , T1 A1
#define P2(T1, A1, T2, A2) , T1 A1, T2 A2
#define P3(T1, A1, T2, A2, T3, A3) , T1 A1, T2 A2, T3 A3
#define P4(T1, A1, T2, A2, T3, A3, T4, A4) , T1 A1, T2 A2, T3 A3, T4 A4
#define P5(T1, A1, T2, A2, T3, A3, T4, A4, T5, A5) , T1 A1, T2 A2, T3 A3, T4 A4, T5 A5
#define A1(a) , a
#define A2(a, b) , a, b
#define A3(a, b, c) , a, b, c
#define A4(a, b, c, d) , a, b, c, d
#define A5(a, b, c, d, e) , a, b, c, d, e

    using pw::Expr;
    using pw::Opts;
    using pw::run;
    using pw::iter_ok;

// K(NAME, Expr{...}[, Opts]): a kernel callable for the macros above.
#define K(NAME, ...) [&](TensorIteratorBase &it) { return run(it, NAME, __VA_ARGS__); }
}
