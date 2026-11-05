// Pointwise (elementwise) ATen ops: unary, binary, ternary, comparisons,
// activations and their backward kernels.
//
// Most ops are structured: the meta function from PyTorch builds the
// TensorIterator (shape checks, broadcasting, type promotion, output
// allocation) and we only provide the WGSL expression.

#include <ATen/ATen.h>
#include <ATen/TensorIterator.h>
#include <ATen/ops/abs_native.h>
#include <ATen/ops/acos_meta.h>
#include <ATen/ops/acosh_meta.h>
#include <ATen/ops/add_meta.h>
#include <ATen/ops/addcdiv_meta.h>
#include <ATen/ops/addcmul_meta.h>
#include <ATen/ops/asin_meta.h>
#include <ATen/ops/asinh_meta.h>
#include <ATen/ops/atan2_meta.h>
#include <ATen/ops/atan_meta.h>
#include <ATen/ops/atanh_meta.h>
#include <ATen/ops/bitwise_and_meta.h>
#include <ATen/ops/bitwise_left_shift_meta.h>
#include <ATen/ops/bitwise_not_meta.h>
#include <ATen/ops/bitwise_or_meta.h>
#include <ATen/ops/bitwise_right_shift_meta.h>
#include <ATen/ops/bitwise_xor_meta.h>
#include <ATen/ops/ceil_meta.h>
#include <ATen/ops/clamp_max_meta.h>
#include <ATen/ops/clamp_meta.h>
#include <ATen/ops/clamp_min_meta.h>
#include <ATen/ops/copysign_meta.h>
#include <ATen/ops/cos_meta.h>
#include <ATen/ops/cosh_meta.h>
#include <ATen/ops/div_meta.h>
#include <ATen/ops/elu_backward_meta.h>
#include <ATen/ops/elu_meta.h>
#include <ATen/ops/eq_meta.h>
#include <ATen/ops/erf_meta.h>
#include <ATen/ops/erfc_meta.h>
#include <ATen/ops/erfinv_meta.h>
#include <ATen/ops/exp2_meta.h>
#include <ATen/ops/exp_meta.h>
#include <ATen/ops/expm1_meta.h>
#include <ATen/ops/floor_meta.h>
#include <ATen/ops/fmax_meta.h>
#include <ATen/ops/fmin_meta.h>
#include <ATen/ops/fmod_meta.h>
#include <ATen/ops/frac_meta.h>
#include <ATen/ops/gcd_meta.h>
#include <ATen/ops/ge_meta.h>
#include <ATen/ops/gelu_backward_meta.h>
#include <ATen/ops/gelu_meta.h>
#include <ATen/ops/gt_meta.h>
#include <ATen/ops/hardshrink_meta.h>
#include <ATen/ops/hardsigmoid_backward_meta.h>
#include <ATen/ops/hardsigmoid_meta.h>
#include <ATen/ops/heaviside_meta.h>
#include <ATen/ops/hypot_meta.h>
#include <ATen/ops/isneginf_meta.h>
#include <ATen/ops/isposinf_meta.h>
#include <ATen/ops/lcm_meta.h>
#include <ATen/ops/le_meta.h>
#include <ATen/ops/leaky_relu_backward_meta.h>
#include <ATen/ops/leaky_relu_meta.h>
#include <ATen/ops/lerp_meta.h>
#include <ATen/ops/log10_meta.h>
#include <ATen/ops/log1p_meta.h>
#include <ATen/ops/log2_meta.h>
#include <ATen/ops/log_meta.h>
#include <ATen/ops/logaddexp2_meta.h>
#include <ATen/ops/logaddexp_meta.h>
#include <ATen/ops/lt_meta.h>
#include <ATen/ops/maximum_meta.h>
#include <ATen/ops/minimum_meta.h>
#include <ATen/ops/mish_meta.h>
#include <ATen/ops/mul_meta.h>
#include <ATen/ops/ne_meta.h>
#include <ATen/ops/neg_meta.h>
#include <ATen/ops/pow_meta.h>
#include <ATen/ops/reciprocal_meta.h>
#include <ATen/ops/remainder_meta.h>
#include <ATen/ops/round_meta.h>
#include <ATen/ops/rsqrt_meta.h>
#include <ATen/ops/sgn_meta.h>
#include <ATen/ops/sigmoid_backward_meta.h>
#include <ATen/ops/sigmoid_meta.h>
#include <ATen/ops/sign_meta.h>
#include <ATen/ops/signbit_meta.h>
#include <ATen/ops/silu_backward_meta.h>
#include <ATen/ops/silu_meta.h>
#include <ATen/ops/sin_meta.h>
#include <ATen/ops/sinc_meta.h>
#include <ATen/ops/sinh_meta.h>
#include <ATen/ops/softplus_backward_meta.h>
#include <ATen/ops/softplus_meta.h>
#include <ATen/ops/softshrink_meta.h>
#include <ATen/ops/sqrt_meta.h>
#include <ATen/ops/sub_meta.h>
#include <ATen/ops/tan_meta.h>
#include <ATen/ops/tanh_backward_meta.h>
#include <ATen/ops/tanh_meta.h>
#include <ATen/ops/threshold_backward_meta.h>
#include <ATen/ops/threshold_meta.h>
#include <ATen/ops/trunc_meta.h>
#include <ATen/ops/xlogy_meta.h>
#include <torch/library.h>

#include "kernels/elementwise.h"
#include "kernels/math.h"
#include "ops/common.h"

namespace torch_webgpu::ops
{
    using at::Scalar;
    using at::Tensor;
    using at::TensorIteratorBase;
    using kernels::CT;
    using kernels::ElementwiseOp;

    namespace
    {
        // Per-compute-type expressions; nullptr = unsupported for that type.
        struct Expr
        {
            const char *f = nullptr;
            const char *i = nullptr;
            const char *b = nullptr;
        };

        struct Opts
        {
            std::vector<Scalar> scalars;
            std::optional<CT> out_ct;     // default: same as the input compute type
            std::optional<CT> scalar_ct;  // default: input compute type
            std::vector<std::optional<CT>> input_cts;
            std::optional<CT> in_ct;      // default: compute type of common dtype
        };

        void run(TensorIteratorBase &it, const char *name, const Expr &e, Opts o = {})
        {
            CT ct = o.in_ct ? *o.in_ct : kernels::common_ct(it);
            const char *code = ct == CT::F32 ? e.f : ct == CT::I32 ? e.i
                                                                   : e.b;
            TORCH_CHECK(code != nullptr, "torch_webgpu: '", name, "' is not implemented for dtype ",
                        ct == CT::Bool ? "bool" : ct == CT::I32 ? "integral"
                                                                : "floating");
            ElementwiseOp op;
            op.name = name;
            op.code = code;
            op.in_ct = ct;
            op.out_ct = o.out_ct.value_or(ct);
            op.input_cts = o.input_cts;
            CT sct = o.scalar_ct.value_or(ct);
            for (auto &s : o.scalars)
                op.scalars.push_back({s, sct});
            op.helpers = kernels::math_wgsl();
            kernels::run_elementwise(it, op);
        }

        // True when all iterator operands (except CPU scalars) and the common
        // dtype have GPU kernels.
        bool iter_ok(TensorIteratorBase &it)
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
        if (!iter_ok(op))                                                                 \
            return WEBGPU_FALLBACK0(NAME)(self);                                          \
        KERNEL(op);                                                                       \
        return std::move(op.outputs_[0]);                                                 \
    }                                                                                     \
    Tensor &NAME##_out_fn(const Tensor &self, Tensor &out)                                \
    {                                                                                     \
        Out<at::meta::structured_##NAME> op(out);                                         \
        op.meta(self);                                                                    \
        if (!iter_ok(op))                                                                 \
            return WEBGPU_FALLBACK(NAME, out)(self, out);                                 \
        KERNEL(op);                                                                       \
        op.finish();                                                                      \
        return out;                                                                       \
    }                                                                                     \
    Tensor &NAME##_inplace_fn(Tensor &self)                                               \
    {                                                                                     \
        Inplace<at::meta::structured_##NAME> op(self);                                    \
        op.meta(self);                                                                    \
        if (!iter_ok(op))                                                                 \
            return WEBGPU_FALLBACK0(NAME##_)(self);                                       \
        KERNEL(op);                                                                       \
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
        if (!iter_ok(op))                                                                \
            return FB_F(self ARGS);                                                      \
        KERNEL(op);                                                                      \
        return std::move(op.outputs_[0]);                                                \
    }                                                                                    \
    Tensor &ID##_out_fn(const Tensor &self PARAMS, Tensor &out)                          \
    {                                                                                    \
        Out<at::meta::META> op(out);                                                     \
        op.meta(self ARGS);                                                              \
        if (!iter_ok(op))                                                                \
            return FB_O(self ARGS, out);                                                 \
        KERNEL(op);                                                                      \
        op.finish();                                                                     \
        return out;                                                                      \
    }                                                                                    \
    Tensor &ID##_inplace_fn(Tensor &self PARAMS)                                         \
    {                                                                                    \
        Inplace<at::meta::META> op(self);                                                \
        op.meta(self ARGS);                                                              \
        if (!iter_ok(op))                                                                \
            return FB_I(self ARGS);                                                      \
        KERNEL(op);                                                                      \
        op.finish();                                                                     \
        return self;                                                                     \
    }
// Same without in-place variant.
#define WG_OP2(ID, META, KERNEL, PARAMS, ARGS, FB_F, FB_O)                               \
    Tensor ID##_fn(const Tensor &self PARAMS)                                            \
    {                                                                                    \
        Functional<at::meta::META> op;                                                   \
        op.meta(self ARGS);                                                              \
        if (!iter_ok(op))                                                                \
            return FB_F(self ARGS);                                                      \
        KERNEL(op);                                                                      \
        return std::move(op.outputs_[0]);                                                \
    }                                                                                    \
    Tensor &ID##_out_fn(const Tensor &self PARAMS, Tensor &out)                          \
    {                                                                                    \
        Out<at::meta::META> op(out);                                                     \
        op.meta(self ARGS);                                                              \
        if (!iter_ok(op))                                                                \
            return FB_O(self ARGS, out);                                                 \
        KERNEL(op);                                                                      \
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

    namespace
    {
        // ----------------------------------------------------------- unary
#define K(NAME, ...) [&](TensorIteratorBase &it) { run(it, NAME, __VA_ARGS__); }
        WG_UNARY(sin, K("sin", {"sin_f(x0)"}))
        WG_UNARY(cos, K("cos", {"cos_f(x0)"}))
        WG_UNARY(tan, K("tan", {"tan_f(x0)"}))
        WG_UNARY(asin, K("asin", {"asin(x0)"}))
        WG_UNARY(acos, K("acos", {"acos(x0)"}))
        WG_UNARY(atan, K("atan", {"atan(x0)"}))
        WG_UNARY(sinh, K("sinh", {"sinh(x0)"}))
        WG_UNARY(cosh, K("cosh", {"cosh(x0)"}))
        WG_UNARY(tanh, K("tanh", {"tanh_f(x0)"}))
        WG_UNARY(asinh, K("asinh", {"asinh(x0)"}))
        WG_UNARY(acosh, K("acosh", {"acosh(x0)"}))
        WG_UNARY(atanh, K("atanh", {"atanh(x0)"}))
        WG_UNARY(exp, K("exp", {"exp(x0)"}))
        WG_UNARY(exp2, K("exp2", {"exp2(x0)"}))
        WG_UNARY(expm1, K("expm1", {"expm1_f(x0)"}))
        WG_UNARY(log, K("log", {"log(x0)"}))
        WG_UNARY(log2, K("log2", {"log2(x0)"}))
        WG_UNARY(log10, K("log10", {"log2(x0) * 0.30102999566398120"}))
        WG_UNARY(log1p, K("log1p", {"log1p_f(x0)"}))
        WG_UNARY(sqrt, K("sqrt", {"sqrt(x0)"}))
        WG_UNARY(rsqrt, K("rsqrt", {"inverseSqrt(x0)"}))
        WG_UNARY(reciprocal, K("reciprocal", {"1.0 / x0"}))
        WG_UNARY(sigmoid, K("sigmoid", {"sigmoid_f(x0)"}))
        WG_UNARY(erf, K("erf", {"erf_f(x0)"}))
        WG_UNARY(erfc, K("erfc", {"erfc_f(x0)"}))
        WG_UNARY(erfinv, K("erfinv", {"erfinv_f(x0)"}))
        WG_UNARY(sinc, K("sinc", {"select(sin_f(3.14159265358979 * x0) / (3.14159265358979 * x0), 1.0, x0 == 0.0)"}))
        WG_UNARY(neg, K("neg", {"-x0", "-x0"}))
        WG_UNARY(floor, K("floor", {"floor(x0)", "x0", "x0"}))
        WG_UNARY(ceil, K("ceil", {"ceil(x0)", "x0", "x0"}))
        WG_UNARY(trunc, K("trunc", {"trunc(x0)", "x0", "x0"}))
        WG_UNARY(round, K("round", {"round(x0)", "x0", "x0"}))
        WG_UNARY(frac, K("frac", {"x0 - trunc(x0)"}))
        WG_UNARY(sign, K("sign", {"select(sign(x0), x0, isnan_f(x0))", "sign(x0)", "x0"}))
        WG_UNARY(sgn, K("sgn", {"select(sign(x0), x0, isnan_f(x0))", "sign(x0)", "x0"}))
        WG_UNARY(bitwise_not, K("bitwise_not", {nullptr, "~x0", "!x0"}))
        WG_UNARY(silu, K("silu", {"x0 / (1.0 + exp(-x0))"}))
        WG_UNARY(mish, K("mish", {"x0 * tanh_f(select(log1p_f(exp(x0)), x0, x0 > 20.0))"}))
        WG_UNARY(hardsigmoid, K("hardsigmoid", {"clamp(x0 / 6.0 + 0.5, 0.0, 1.0)"}))

        // signbit / isposinf / isneginf produce bool outputs.
        WG_OP2(signbit, structured_signbit,
               K("signbit", {"signbit_f(x0)", "x0 < 0", "false"}, Opts{.out_ct = CT::Bool}), , ,
               WEBGPU_FALLBACK0(signbit), WEBGPU_FALLBACK(signbit, out))
        WG_OP2(isposinf, structured_isposinf,
               K("isposinf", {"isinf_f(x0) && x0 > 0.0", "false", "false"}, Opts{.out_ct = CT::Bool}), , ,
               WEBGPU_FALLBACK0(isposinf), WEBGPU_FALLBACK(isposinf, out))
        WG_OP2(isneginf, structured_isneginf,
               K("isneginf", {"isinf_f(x0) && x0 < 0.0", "false", "false"}, Opts{.out_ct = CT::Bool}), , ,
               WEBGPU_FALLBACK0(isneginf), WEBGPU_FALLBACK(isneginf, out))

        // gelu(self, approximate)
        void gelu_kernel(TensorIteratorBase &it, c10::string_view approximate)
        {
            if (approximate == "tanh")
                run(it, "gelu_tanh", {"gelu_tanh(x0)"});
            else
                run(it, "gelu", {"gelu_erf(x0)"});
        }
        WG_OP3(gelu, structured_gelu, [&](TensorIteratorBase &it) { gelu_kernel(it, approximate); },
               P1(c10::string_view, approximate), A1(approximate), WEBGPU_FALLBACK0(gelu),
               WEBGPU_FALLBACK(gelu, out), WEBGPU_FALLBACK0(gelu_))

        // ------------------------------------------------- scalar-param unary
        WG_OP3(leaky_relu, structured_leaky_relu,
               K("leaky_relu", {"select(x0 * s0, x0, x0 > 0.0)"}, Opts{.scalars = {negative_slope}}),
               P1(const Scalar &, negative_slope), A1(negative_slope), WEBGPU_FALLBACK0(leaky_relu),
               WEBGPU_FALLBACK(leaky_relu, out), WEBGPU_FALLBACK0(leaky_relu_))
        WG_OP2(hardshrink, structured_hardshrink,
               K("hardshrink", {"select(x0, 0.0, x0 >= -s0 && x0 <= s0)"}, Opts{.scalars = {lambd}}),
               P1(const Scalar &, lambd), A1(lambd), WEBGPU_FALLBACK0(hardshrink), WEBGPU_FALLBACK(hardshrink, out))
        WG_OP2(softshrink, structured_softshrink,
               K("softshrink", {"select(select(x0 + s0, x0 - s0, x0 > s0), 0.0, x0 >= -s0 && x0 <= s0)"},
                 Opts{.scalars = {lambd}}),
               P1(const Scalar &, lambd), A1(lambd), WEBGPU_FALLBACK0(softshrink), WEBGPU_FALLBACK(softshrink, out))
        WG_OP2(softplus, structured_softplus,
               K("softplus", {"select(log1p_f(exp(x0 * s0)) / s0, x0, x0 * s0 > s1)"}, Opts{.scalars = {beta, threshold}}),
               P2(const Scalar &, beta, const Scalar &, threshold), A2(beta, threshold), WEBGPU_FALLBACK0(softplus),
               WEBGPU_FALLBACK(softplus, out))
        WG_OP3(threshold, structured_threshold,
               K("threshold", {"select(x0, s1, x0 <= s0)", "select(x0, s1, x0 <= s0)"}, Opts{.scalars = {threshold, value}}),
               P2(const Scalar &, threshold, const Scalar &, value), A2(threshold, value), WEBGPU_FALLBACK0(threshold),
               WEBGPU_FALLBACK(threshold, out), WEBGPU_FALLBACK0(threshold_))
        WG_OP3(elu, structured_elu,
               K("elu", {"select(s0 * s1 * expm1_f(x0 * s2), s1 * x0, x0 > 0.0)"},
                 Opts{.scalars = {alpha, scale, input_scale}}),
               P3(const Scalar &, alpha, const Scalar &, scale, const Scalar &, input_scale),
               A3(alpha, scale, input_scale), WEBGPU_FALLBACK0(elu), WEBGPU_FALLBACK(elu, out),
               WEBGPU_FALLBACK0(elu_))

        // ---------------------------------------------------------- binary
        void add_kernel(TensorIteratorBase &it, const Scalar &alpha, const char *name, bool subtract)
        {
            if (subtract)
                run(it, name, {"x0 - s0 * x1", "x0 - s0 * x1", nullptr}, Opts{.scalars = {alpha}});
            else
                run(it, name, {"x0 + s0 * x1", "x0 + s0 * x1", "x0 || (s0 && x1)"}, Opts{.scalars = {alpha}});
        }
        WG_OP3(add, structured_add_Tensor, [&](TensorIteratorBase &it) { add_kernel(it, alpha, "add", false); },
               P2(const Tensor &, other, const Scalar &, alpha), A2(other, alpha), WEBGPU_FALLBACK(add, Tensor),
               WEBGPU_FALLBACK(add, out), WEBGPU_FALLBACK(add_, Tensor))
        WG_OP3(sub, structured_sub_Tensor, [&](TensorIteratorBase &it) { add_kernel(it, alpha, "sub", true); },
               P2(const Tensor &, other, const Scalar &, alpha), A2(other, alpha), WEBGPU_FALLBACK(sub, Tensor),
               WEBGPU_FALLBACK(sub, out), WEBGPU_FALLBACK(sub_, Tensor))
        WG_OP3(mul, structured_mul_Tensor, K("mul", {"x0 * x1", "x0 * x1", "x0 && x1"}), P1(const Tensor &, other),
               A1(other), WEBGPU_FALLBACK(mul, Tensor), WEBGPU_FALLBACK(mul, out), WEBGPU_FALLBACK(mul_, Tensor))
        WG_OP3(div, structured_div_Tensor, K("div", {"x0 / x1"}), P1(const Tensor &, other), A1(other),
               WEBGPU_FALLBACK(div, Tensor), WEBGPU_FALLBACK(div, out), WEBGPU_FALLBACK(div_, Tensor))

        void div_mode_kernel(TensorIteratorBase &it, std::optional<c10::string_view> mode)
        {
            if (!mode.has_value())
                run(it, "div", {"x0 / x1"});
            else if (*mode == "trunc")
                run(it, "div_trunc", {"trunc(x0 / x1)", "select(x0 / x1, 0, x1 == 0)", nullptr});
            else if (*mode == "floor")
                run(it, "div_floor", {"div_floor_f(x0, x1)", "div_floor_i(x0, x1)", nullptr});
            else
                TORCH_CHECK(false, "div expected rounding_mode to be one of None, 'trunc', or 'floor' but found '",
                            *mode, "'");
        }
        WG_OP3(div_mode, structured_div_Tensor_mode, [&](TensorIteratorBase &it) { div_mode_kernel(it, rounding_mode); },
               P2(const Tensor &, other, std::optional<c10::string_view>, rounding_mode), A2(other, rounding_mode),
               WEBGPU_FALLBACK(div, Tensor_mode), WEBGPU_FALLBACK(div, out_mode), WEBGPU_FALLBACK(div_, Tensor_mode))

        WG_OP2(maximum, structured_maximum, K("maximum", {"max_nan(x0, x1)", "max(x0, x1)", "x0 || x1"}),
               P1(const Tensor &, other), A1(other), WEBGPU_FALLBACK0(maximum), WEBGPU_FALLBACK(maximum, out))
        WG_OP2(minimum, structured_minimum, K("minimum", {"min_nan(x0, x1)", "min(x0, x1)", "x0 && x1"}),
               P1(const Tensor &, other), A1(other), WEBGPU_FALLBACK0(minimum), WEBGPU_FALLBACK(minimum, out))
        WG_OP2(fmax, structured_fmax, K("fmax", {"fmax_f(x0, x1)", "max(x0, x1)", "x0 || x1"}), P1(const Tensor &, other),
               A1(other), WEBGPU_FALLBACK0(fmax), WEBGPU_FALLBACK(fmax, out))
        WG_OP2(fmin, structured_fmin, K("fmin", {"fmin_f(x0, x1)", "min(x0, x1)", "x0 && x1"}), P1(const Tensor &, other),
               A1(other), WEBGPU_FALLBACK0(fmin), WEBGPU_FALLBACK(fmin, out))
        WG_OP3(atan2, structured_atan2, K("atan2", {"atan2(x0, x1)"}), P1(const Tensor &, other), A1(other),
               WEBGPU_FALLBACK0(atan2), WEBGPU_FALLBACK(atan2, out), WEBGPU_FALLBACK0(atan2_))
        WG_OP3(hypot, structured_hypot, K("hypot", {"hypot_f(x0, x1)"}), P1(const Tensor &, other), A1(other),
               WEBGPU_FALLBACK0(hypot), WEBGPU_FALLBACK(hypot, out), WEBGPU_FALLBACK0(hypot_))
        WG_OP3(copysign, structured_copysign_Tensor, K("copysign", {"copysign_f(x0, x1)"}), P1(const Tensor &, other),
               A1(other), WEBGPU_FALLBACK(copysign, Tensor), WEBGPU_FALLBACK(copysign, out),
               WEBGPU_FALLBACK(copysign_, Tensor))
        WG_OP2(logaddexp, structured_logaddexp, K("logaddexp", {"logaddexp_f(x0, x1)"}), P1(const Tensor &, other),
               A1(other), WEBGPU_FALLBACK0(logaddexp), WEBGPU_FALLBACK(logaddexp, out))
        WG_OP2(logaddexp2, structured_logaddexp2, K("logaddexp2", {"logaddexp2_f(x0, x1)"}), P1(const Tensor &, other),
               A1(other), WEBGPU_FALLBACK0(logaddexp2), WEBGPU_FALLBACK(logaddexp2, out))
        WG_OP3(xlogy, structured_xlogy_Tensor,
               K("xlogy", {"select(select(x0 * log(x1), 0.0, x0 == 0.0), nan_f(), isnan_f(x1))"}),
               P1(const Tensor &, other), A1(other), WEBGPU_FALLBACK(xlogy, Tensor), WEBGPU_FALLBACK(xlogy, OutTensor),
               WEBGPU_FALLBACK(xlogy_, Tensor))
        WG_OP3(heaviside, structured_heaviside,
               K("heaviside", {"select(select(0.0, 1.0, x0 > 0.0), x1, x0 == 0.0)", "select(select(0, 1, x0 > 0), x1, x0 == 0)",
                               "select(x0, x1, !x0)"}),
               P1(const Tensor &, values), A1(values), WEBGPU_FALLBACK0(heaviside), WEBGPU_FALLBACK(heaviside, out),
               WEBGPU_FALLBACK0(heaviside_))
        WG_OP3(remainder, structured_remainder_Tensor, K("remainder", {"remainder_f(x0, x1)", "remainder_i(x0, x1)"}),
               P1(const Tensor &, other), A1(other), WEBGPU_FALLBACK(remainder, Tensor),
               WEBGPU_FALLBACK(remainder, Tensor_out), WEBGPU_FALLBACK(remainder_, Tensor))
        WG_OP3(fmod, structured_fmod_Tensor, K("fmod", {"fmod_f(x0, x1)", "select(x0 % x1, 0, x1 == 0)"}),
               P1(const Tensor &, other), A1(other), WEBGPU_FALLBACK(fmod, Tensor), WEBGPU_FALLBACK(fmod, Tensor_out),
               WEBGPU_FALLBACK(fmod_, Tensor))
        WG_OP3(pow_tt, structured_pow_Tensor_Tensor, K("pow", {"pow_f(x0, x1)", "ipow(x0, x1)"}),
               P1(const Tensor &, exponent), A1(exponent), WEBGPU_FALLBACK(pow, Tensor_Tensor),
               WEBGPU_FALLBACK(pow, Tensor_Tensor_out), WEBGPU_FALLBACK(pow_, Tensor))

        void pow_scalar_kernel(TensorIteratorBase &it, const Scalar &e)
        {
            if (kernels::common_ct(it) == CT::F32 && !e.isComplex())
            {
                double v = e.toDouble();
                if (v == 2.0)
                    return run(it, "pow2", {"x0 * x0"});
                if (v == 3.0)
                    return run(it, "pow3", {"x0 * x0 * x0"});
                if (v == 0.5)
                    return run(it, "pow_sqrt", {"sqrt(x0)"});
                if (v == -0.5)
                    return run(it, "pow_rsqrt", {"inverseSqrt(x0)"});
                if (v == -1.0)
                    return run(it, "pow_recip", {"1.0 / x0"});
                if (v == -2.0)
                    return run(it, "pow_m2", {"1.0 / (x0 * x0)"});
            }
            run(it, "pow_s", {"pow_f(x0, s0)", "ipow(x0, s0)"}, Opts{.scalars = {e}});
        }
        WG_OP3(pow_ts, structured_pow_Tensor_Scalar, [&](TensorIteratorBase &it) { pow_scalar_kernel(it, exponent); },
               P1(const Scalar &, exponent), A1(exponent), WEBGPU_FALLBACK(pow, Tensor_Scalar),
               WEBGPU_FALLBACK(pow, Tensor_Scalar_out), WEBGPU_FALLBACK(pow_, Scalar))

        WG_OP3(bitwise_and, structured_bitwise_and_Tensor, K("bitwise_and", {nullptr, "x0 & x1", "x0 & x1"}),
               P1(const Tensor &, other), A1(other), WEBGPU_FALLBACK(bitwise_and, Tensor),
               WEBGPU_FALLBACK(bitwise_and, Tensor_out), WEBGPU_FALLBACK(bitwise_and_, Tensor))
        WG_OP3(bitwise_or, structured_bitwise_or_Tensor, K("bitwise_or", {nullptr, "x0 | x1", "x0 | x1"}),
               P1(const Tensor &, other), A1(other), WEBGPU_FALLBACK(bitwise_or, Tensor),
               WEBGPU_FALLBACK(bitwise_or, Tensor_out), WEBGPU_FALLBACK(bitwise_or_, Tensor))
        WG_OP3(bitwise_xor, structured_bitwise_xor_Tensor, K("bitwise_xor", {nullptr, "x0 ^ x1", "x0 != x1"}),
               P1(const Tensor &, other), A1(other), WEBGPU_FALLBACK(bitwise_xor, Tensor),
               WEBGPU_FALLBACK(bitwise_xor, Tensor_out), WEBGPU_FALLBACK(bitwise_xor_, Tensor))
        WG_OP3(lshift, structured_bitwise_left_shift_Tensor,
               K("lshift", {nullptr, "select(x0 << u32(x1), 0, x1 < 0 || x1 > 31)"}), P1(const Tensor &, other), A1(other),
               WEBGPU_FALLBACK(bitwise_left_shift, Tensor), WEBGPU_FALLBACK(bitwise_left_shift, Tensor_out),
               WEBGPU_FALLBACK(bitwise_left_shift_, Tensor))
        WG_OP3(rshift, structured_bitwise_right_shift_Tensor,
               K("rshift", {nullptr, "select(x0 >> u32(x1), select(0, -1, x0 < 0), x1 < 0 || x1 > 31)"}),
               P1(const Tensor &, other), A1(other), WEBGPU_FALLBACK(bitwise_right_shift, Tensor),
               WEBGPU_FALLBACK(bitwise_right_shift, Tensor_out), WEBGPU_FALLBACK(bitwise_right_shift_, Tensor))
        WG_OP3(gcd, structured_gcd, K("gcd", {nullptr, "gcd_i(x0, x1)"}), P1(const Tensor &, other), A1(other),
               WEBGPU_FALLBACK0(gcd), WEBGPU_FALLBACK(gcd, out), WEBGPU_FALLBACK0(gcd_))
        WG_OP3(lcm, structured_lcm,
               K("lcm", {nullptr, "select(abs(x0 / gcd_i(x0, x1) * x1), 0, gcd_i(x0, x1) == 0)"}),
               P1(const Tensor &, other), A1(other), WEBGPU_FALLBACK0(lcm), WEBGPU_FALLBACK(lcm, out),
               WEBGPU_FALLBACK0(lcm_))

        // ------------------------------------------------------ comparisons
        const Opts kCmp{.out_ct = CT::Bool};
#define WG_CMP(NAME, EXPR_F, EXPR_I, EXPR_B)                                                               \
    WG_OP3(NAME##_t, structured_##NAME##_Tensor, K(#NAME, {EXPR_F, EXPR_I, EXPR_B}, kCmp),                    \
           P1(const Tensor &, other), A1(other), WEBGPU_FALLBACK(NAME, Tensor), WEBGPU_FALLBACK(NAME, Tensor_out), \
           WEBGPU_FALLBACK(NAME##_, Tensor))                                                                  \
    WG_OP3(NAME##_s, structured_##NAME##_Scalar, K(#NAME, {EXPR_F, EXPR_I, EXPR_B}, kCmp),                    \
           P1(const Scalar &, other), A1(other), WEBGPU_FALLBACK(NAME, Scalar), WEBGPU_FALLBACK(NAME, Scalar_out), \
           WEBGPU_FALLBACK(NAME##_, Scalar))
        WG_CMP(eq, "x0 == x1", "x0 == x1", "x0 == x1")
        WG_CMP(ne, "x0 != x1", "x0 != x1", "x0 != x1")
        WG_CMP(lt, "x0 < x1", "x0 < x1", "!x0 && x1")
        WG_CMP(le, "x0 <= x1", "x0 <= x1", "!x0 || x1")
        WG_CMP(gt, "x0 > x1", "x0 > x1", "x0 && !x1")
        WG_CMP(ge, "x0 >= x1", "x0 >= x1", "x0 || !x1")

        // ---------------------------------------------------------- clamp
        // Inputs: x0 = self, then the optional bounds in order.
        void clamp_scalar_kernel(TensorIteratorBase &it, const std::optional<Scalar> &lo, const std::optional<Scalar> &hi)
        {
            if (lo && hi)
                run(it, "clamp", {"select(min(max(x0, s0), s1), x0, isnan_f(x0))", "min(max(x0, s0), s1)", "(x0 || s0) && s1"},
                    Opts{.scalars = {*lo, *hi}});
            else if (lo)
                run(it, "clamp_min", {"select(max(x0, s0), x0, isnan_f(x0))", "max(x0, s0)", "x0 || s0"}, Opts{.scalars = {*lo}});
            else
                run(it, "clamp_max", {"select(min(x0, s0), x0, isnan_f(x0))", "min(x0, s0)", "x0 && s0"}, Opts{.scalars = {*hi}});
        }

        Tensor clamp_fn(const Tensor &self, const std::optional<Scalar> &min, const std::optional<Scalar> &max)
        {
            Functional<at::meta::structured_clamp> op;
            op.meta(self, at::OptionalScalarRef(min ? &*min : nullptr), at::OptionalScalarRef(max ? &*max : nullptr));
            if (!iter_ok(op))
                return WEBGPU_FALLBACK0(clamp)(self, min, max);
            clamp_scalar_kernel(op, min, max);
            return std::move(op.outputs_[0]);
        }
        Tensor &clamp_out_fn(const Tensor &self, const std::optional<Scalar> &min, const std::optional<Scalar> &max, Tensor &out)
        {
            Out<at::meta::structured_clamp> op(out);
            op.meta(self, at::OptionalScalarRef(min ? &*min : nullptr), at::OptionalScalarRef(max ? &*max : nullptr));
            if (!iter_ok(op))
                return WEBGPU_FALLBACK(clamp, out)(self, min, max, out);
            clamp_scalar_kernel(op, min, max);
            op.finish();
            return out;
        }
        Tensor &clamp_inplace_fn(Tensor &self, const std::optional<Scalar> &min, const std::optional<Scalar> &max)
        {
            Inplace<at::meta::structured_clamp> op(self);
            op.meta(self, at::OptionalScalarRef(min ? &*min : nullptr), at::OptionalScalarRef(max ? &*max : nullptr));
            if (!iter_ok(op))
                return WEBGPU_FALLBACK0(clamp_)(self, min, max);
            clamp_scalar_kernel(op, min, max);
            op.finish();
            return self;
        }

        WG_OP3(clamp_min_s, structured_clamp_min,
               [&](TensorIteratorBase &it) { clamp_scalar_kernel(it, min, std::nullopt); }, P1(const Scalar &, min),
               A1(min), WEBGPU_FALLBACK0(clamp_min), WEBGPU_FALLBACK(clamp_min, out), WEBGPU_FALLBACK0(clamp_min_))
        WG_OP3(clamp_max_s, structured_clamp_max,
               [&](TensorIteratorBase &it) { clamp_scalar_kernel(it, std::nullopt, max); }, P1(const Scalar &, max),
               A1(max), WEBGPU_FALLBACK0(clamp_max), WEBGPU_FALLBACK(clamp_max, out), WEBGPU_FALLBACK0(clamp_max_))
        WG_OP3(clamp_min_t, structured_clamp_min_Tensor,
               K("clamp_min_t", {"select(max(x0, x1), x0, isnan_f(x0) || isnan_f(x1))", "max(x0, x1)", "x0 || x1"}),
               P1(const Tensor &, min), A1(min), WEBGPU_FALLBACK(clamp_min, Tensor), WEBGPU_FALLBACK(clamp_min, Tensor_out),
               WEBGPU_FALLBACK(clamp_min_, Tensor))
        WG_OP3(clamp_max_t, structured_clamp_max_Tensor,
               K("clamp_max_t", {"select(min(x0, x1), x0, isnan_f(x0) || isnan_f(x1))", "min(x0, x1)", "x0 && x1"}),
               P1(const Tensor &, max), A1(max), WEBGPU_FALLBACK(clamp_max, Tensor), WEBGPU_FALLBACK(clamp_max, Tensor_out),
               WEBGPU_FALLBACK(clamp_max_, Tensor))

        // ------------------------------------------------------- ternary ops
        WG_OP3(addcmul, structured_addcmul, K("addcmul", {"x0 + s0 * x1 * x2", "x0 + s0 * x1 * x2"}, Opts{.scalars = {value}}),
               P3(const Tensor &, tensor1, const Tensor &, tensor2, const Scalar &, value), A3(tensor1, tensor2, value),
               WEBGPU_FALLBACK0(addcmul), WEBGPU_FALLBACK(addcmul, out), WEBGPU_FALLBACK0(addcmul_))
        WG_OP3(addcdiv, structured_addcdiv, K("addcdiv", {"x0 + s0 * x1 / x2"}, Opts{.scalars = {value}}),
               P3(const Tensor &, tensor1, const Tensor &, tensor2, const Scalar &, value), A3(tensor1, tensor2, value),
               WEBGPU_FALLBACK0(addcdiv), WEBGPU_FALLBACK(addcdiv, out), WEBGPU_FALLBACK0(addcdiv_))
        WG_OP3(lerp_s, structured_lerp_Scalar,
               K("lerp_s", {"select(x1 - (x1 - x0) * (1.0 - s0), x0 + s0 * (x1 - x0), abs(s0) < 0.5)"}, Opts{.scalars = {weight}}),
               P2(const Tensor &, end, const Scalar &, weight), A2(end, weight), WEBGPU_FALLBACK(lerp, Scalar),
               WEBGPU_FALLBACK(lerp, Scalar_out), WEBGPU_FALLBACK(lerp_, Scalar))
        WG_OP3(lerp_t, structured_lerp_Tensor,
               K("lerp_t", {"select(x1 - (x1 - x0) * (1.0 - x2), x0 + x2 * (x1 - x0), abs(x2) < 0.5)"}),
               P2(const Tensor &, end, const Tensor &, weight), A2(end, weight), WEBGPU_FALLBACK(lerp, Tensor),
               WEBGPU_FALLBACK(lerp, Tensor_out), WEBGPU_FALLBACK(lerp_, Tensor))

        // --------------------------------------------------------- backward
        // These ops have only functional and grad_input (out) variants.
#define WG_BWD(ID, META, KERNEL, PARAMS, ARGS, FB_F, FB_O)                     \
    Tensor ID##_fn(const Tensor &grad PARAMS)                                  \
    {                                                                          \
        Functional<at::meta::META> op;                                         \
        op.meta(grad ARGS);                                                    \
        if (!iter_ok(op))                                                      \
            return FB_F(grad ARGS);                                            \
        KERNEL(op);                                                            \
        return std::move(op.outputs_[0]);                                      \
    }                                                                          \
    Tensor &ID##_out_fn(const Tensor &grad PARAMS, Tensor &grad_input)         \
    {                                                                          \
        Out<at::meta::META> op(grad_input);                                    \
        op.meta(grad ARGS);                                                    \
        if (!iter_ok(op))                                                      \
            return FB_O(grad ARGS, grad_input);                                \
        KERNEL(op);                                                            \
        op.finish();                                                           \
        return grad_input;                                                     \
    }
        WG_BWD(silu_backward, structured_silu_backward,
               K("silu_backward", {"x0 * sigmoid_f(x1) * (1.0 + x1 * (1.0 - sigmoid_f(x1)))"}), P1(const Tensor &, self),
               A1(self), WEBGPU_FALLBACK0(silu_backward), WEBGPU_FALLBACK(silu_backward, grad_input))
        WG_BWD(tanh_backward, structured_tanh_backward, K("tanh_backward", {"x0 * (1.0 - x1 * x1)"}),
               P1(const Tensor &, output), A1(output), WEBGPU_FALLBACK0(tanh_backward),
               WEBGPU_FALLBACK(tanh_backward, grad_input))
        WG_BWD(sigmoid_backward, structured_sigmoid_backward, K("sigmoid_backward", {"x0 * x1 * (1.0 - x1)"}),
               P1(const Tensor &, output), A1(output), WEBGPU_FALLBACK0(sigmoid_backward),
               WEBGPU_FALLBACK(sigmoid_backward, grad_input))
        WG_BWD(hardsigmoid_backward, structured_hardsigmoid_backward,
               K("hardsigmoid_backward", {"select(0.0, x0 / 6.0, x1 > -3.0 && x1 < 3.0)"}), P1(const Tensor &, self),
               A1(self), WEBGPU_FALLBACK0(hardsigmoid_backward), WEBGPU_FALLBACK(hardsigmoid_backward, grad_input))
        WG_BWD(threshold_backward, structured_threshold_backward,
               // The meta puts `self` before `grad` (it reuses the threshold kernel).
               K("threshold_backward", {"select(x1, 0.0, x0 <= s0)", "select(x1, 0, x0 <= s0)"}, Opts{.scalars = {threshold}}),
               P2(const Tensor &, self, const Scalar &, threshold), A2(self, threshold),
               WEBGPU_FALLBACK0(threshold_backward), WEBGPU_FALLBACK(threshold_backward, grad_input))
        WG_BWD(leaky_relu_backward, structured_leaky_relu_backward,
               // The meta orders the inputs (self, grad_output).
               K("leaky_relu_backward", {"select(x1 * s0, x1, x0 > 0.0)"}, Opts{.scalars = {negative_slope}}),
               P3(const Tensor &, self, const Scalar &, negative_slope, bool, self_is_result),
               A3(self, negative_slope, self_is_result), WEBGPU_FALLBACK0(leaky_relu_backward),
               WEBGPU_FALLBACK(leaky_relu_backward, grad_input))
        WG_BWD(softplus_backward, structured_softplus_backward,
               K("softplus_backward", {"select(x0 * exp(x1 * s0) / (exp(x1 * s0) + 1.0), x0, x1 * s0 > s1)"},
                 Opts{.scalars = {beta, threshold}}),
               P3(const Tensor &, self, const Scalar &, beta, const Scalar &, threshold), A3(self, beta, threshold),
               WEBGPU_FALLBACK0(softplus_backward), WEBGPU_FALLBACK(softplus_backward, grad_input))

        void gelu_backward_kernel(TensorIteratorBase &it, c10::string_view approximate)
        {
            if (approximate == "tanh")
                run(it, "gelu_backward_tanh", {R"(
    let k_beta = 0.7978845608028654;
    let k_kappa = 0.044715;
    let x = x1;
    let inner = k_beta * (x + k_kappa * x * x * x);
    let t = tanh_f(inner);
    let left = 0.5 * x;
    let right = 1.0 + t;
    let left_d = 0.5 * right;
    let right_d = left * (1.0 - t * t) * k_beta * (1.0 + 3.0 * k_kappa * x * x);
    return x0 * (left_d + right_d);)"});
            else
                run(it, "gelu_backward", {R"(
    let x = x1;
    let cdf = 0.5 * (1.0 + erf_f(x * 0.7071067811865476));
    let pdf = exp(-0.5 * x * x) * 0.3989422804014327;
    return x0 * (cdf + x * pdf);)"});
        }
        WG_BWD(gelu_backward, structured_gelu_backward, [&](TensorIteratorBase &it) { gelu_backward_kernel(it, approximate); },
               P2(const Tensor &, self, c10::string_view, approximate), A2(self, approximate),
               WEBGPU_FALLBACK0(gelu_backward), WEBGPU_FALLBACK(gelu_backward, grad_input))

        Tensor elu_backward_fn(const Tensor &grad, const Scalar &alpha, const Scalar &scale, const Scalar &input_scale,
                               bool is_result, const Tensor &self_or_result)
        {
            Functional<at::meta::structured_elu_backward> op;
            op.meta(grad, alpha, scale, input_scale, is_result, self_or_result);
            if (!iter_ok(op))
                return WEBGPU_FALLBACK0(elu_backward)(grad, alpha, scale, input_scale, is_result, self_or_result);
            if (is_result)
                run(op, "elu_backward_r", {"select(x0 * s1, x0 * s2 * (x1 + s0 * s1), x1 <= 0.0)"},
                    Opts{.scalars = {alpha, scale, input_scale}});
            else
                run(op, "elu_backward", {"select(x0 * s1, x0 * s2 * s0 * s1 * exp(x1 * s2), x1 <= 0.0)"},
                    Opts{.scalars = {alpha, scale, input_scale}});
            return std::move(op.outputs_[0]);
        }

        // --------------------------------------------- non-structured ops
        // abs: unary_op TensorIterator, keeps dtype.
        Tensor &abs_out_fn(const Tensor &self, Tensor &out)
        {
            if (!gpu_dtypes_ok({self, out}))
                return WEBGPU_FALLBACK(abs, out)(self, out);
            auto it = at::TensorIterator::unary_op(out, self);
            run(it, "abs", {"abs(x0)", "abs(x0)", "x0"});
            return out;
        }
        Tensor abs_fn(const Tensor &self)
        {
            Tensor out = at::empty({0}, self.options());
            return abs_out_fn(self, out);
        }
        Tensor &abs_inplace_fn(Tensor &self) { return abs_out_fn(self, self); }

        // relu: in-place-able, keeps dtype (bool is rejected by PyTorch).
        Tensor &relu_inplace_fn(Tensor &self)
        {
            if (!gpu_dtypes_ok({self}))
                return WEBGPU_FALLBACK0(relu_)(self);
            TORCH_CHECK(self.scalar_type() != at::kBool, "Boolean inputs not supported for relu");
            auto it = at::TensorIterator::unary_op(self, self);
            run(it, "relu", {"select(max(x0, 0.0), x0, isnan_f(x0))", "max(x0, 0)"});
            return self;
        }
        Tensor relu_fn(const Tensor &self)
        {
            if (!gpu_dtypes_ok({self}))
                return WEBGPU_FALLBACK0(relu)(self);
            TORCH_CHECK(self.scalar_type() != at::kBool, "Boolean inputs not supported for relu");
            Tensor out;
            auto it = at::TensorIterator::unary_op(out, self);
            run(it, "relu", {"select(max(x0, 0.0), x0, isnan_f(x0))", "max(x0, 0)"});
            return it.output();
        }

        // hardtanh / hardswish (+ backward)
        Tensor &hardtanh_out_fn(const Tensor &self, const Scalar &lo, const Scalar &hi, Tensor &out)
        {
            if (!gpu_dtypes_ok({self, out}))
                return WEBGPU_FALLBACK(hardtanh, out)(self, lo, hi, out);
            TORCH_CHECK(self.scalar_type() != at::kBool, "Bool inputs not supported for hardtanh");
            auto it = at::TensorIterator::unary_op(out, self);
            run(it, "hardtanh", {"select(min(max(x0, s0), s1), x0, isnan_f(x0))", "min(max(x0, s0), s1)"},
                Opts{.scalars = {lo, hi}});
            return out;
        }
        Tensor hardtanh_fn(const Tensor &self, const Scalar &lo, const Scalar &hi)
        {
            Tensor out = at::empty({0}, self.options());
            return hardtanh_out_fn(self, lo, hi, out);
        }
        Tensor &hardtanh_inplace_fn(Tensor &self, const Scalar &lo, const Scalar &hi)
        {
            return hardtanh_out_fn(self, lo, hi, self);
        }
        Tensor &hardtanh_backward_out_fn(const Tensor &grad, const Tensor &self, const Scalar &lo, const Scalar &hi,
                                         Tensor &grad_input)
        {
            if (!gpu_dtypes_ok({grad, self, grad_input}))
                return WEBGPU_FALLBACK(hardtanh_backward, grad_input)(grad, self, lo, hi, grad_input);
            auto it = at::TensorIterator::borrowing_binary_op(grad_input, grad, self);
            run(it, "hardtanh_backward", {"select(x0, 0.0, x1 <= s0 || x1 >= s1)"}, Opts{.scalars = {lo, hi}});
            return grad_input;
        }
        Tensor hardtanh_backward_fn(const Tensor &grad, const Tensor &self, const Scalar &lo, const Scalar &hi)
        {
            Tensor out;
            if (!gpu_dtypes_ok({grad, self}))
                return WEBGPU_FALLBACK0(hardtanh_backward)(grad, self, lo, hi);
            auto it = at::TensorIterator::borrowing_binary_op(out, grad, self);
            run(it, "hardtanh_backward", {"select(x0, 0.0, x1 <= s0 || x1 >= s1)"}, Opts{.scalars = {lo, hi}});
            return it.output();
        }

        const char *kHardswish = "x0 * clamp(x0 + 3.0, 0.0, 6.0) / 6.0";
        Tensor &hardswish_out_fn(const Tensor &self, Tensor &out)
        {
            if (!gpu_dtypes_ok({self, out}))
                return WEBGPU_FALLBACK(hardswish, out)(self, out);
            auto it = at::TensorIterator::unary_op(out, self);
            run(it, "hardswish", {kHardswish});
            return out;
        }
        Tensor hardswish_fn(const Tensor &self)
        {
            Tensor out = at::empty({0}, self.options());
            return hardswish_out_fn(self, out);
        }
        Tensor &hardswish_inplace_fn(Tensor &self) { return hardswish_out_fn(self, self); }
        Tensor hardswish_backward_fn(const Tensor &grad, const Tensor &self)
        {
            if (!gpu_dtypes_ok({grad, self}))
                return WEBGPU_FALLBACK0(hardswish_backward)(grad, self);
            Tensor out;
            auto it = at::TensorIterator::borrowing_binary_op(out, grad, self);
            run(it, "hardswish_backward", {"select(select(x0 * (x1 / 3.0 + 0.5), x0, x1 >= 3.0), 0.0, x1 <= -3.0)"});
            return it.output();
        }

        // where(condition, self, other)
        Tensor &where_out_fn(const Tensor &cond, const Tensor &self, const Tensor &other, Tensor &out)
        {
            if (!gpu_dtypes_ok({cond, self, other, out}))
                return WEBGPU_FALLBACK(where, self_out)(cond, self, other, out);
            auto result_type = at::result_type(self, other);
            TORCH_CHECK(out.scalar_type() == result_type, "Expected out type to be ", result_type, " but got ",
                        out.scalar_type());
            auto it = at::TensorIteratorConfig()
                          .check_all_same_dtype(false)
                          .check_all_same_device(false)
                          .add_output(out)
                          .add_const_input(cond)
                          .add_const_input(self)
                          .add_const_input(other)
                          .build();
            CT ct = wgsl::compute_type(result_type);
            run(it, "where", {"select(x2, x1, x0)", "select(x2, x1, x0)", "select(x2, x1, x0)"},
                Opts{.input_cts = {CT::Bool, ct, ct}, .in_ct = ct});
            return out;
        }
        Tensor where_fn(const Tensor &cond, const Tensor &self, const Tensor &other)
        {
            auto result_type = at::result_type(self, other);
            auto device = is_webgpu(self) ? self.device() : is_webgpu(other) ? other.device()
                                                                             : cond.device();
            Tensor out = at::empty({0}, self.options().dtype(result_type).device(device));
            return where_out_fn(cond, self, other, out);
        }

        // masked_fill_(self, mask, value)
        Tensor &masked_fill_scalar_fn(Tensor &self, const Tensor &mask, const Scalar &value)
        {
            if (!gpu_dtypes_ok({self, mask}))
                return WEBGPU_FALLBACK(masked_fill_, Scalar)(self, mask, value);
            TORCH_CHECK(mask.scalar_type() == at::kBool, "masked_fill_ only supports boolean masks, but got mask with dtype ",
                        mask.scalar_type());
            if (self.numel() == 0)
                return self;
            auto it = at::TensorIteratorConfig()
                          .set_check_mem_overlap(false)
                          .check_all_same_dtype(false)
                          .resize_outputs(false)
                          .add_output(self)
                          .add_const_input(self)
                          .add_const_input(mask)
                          .build();
            CT ct = wgsl::compute_type(self.scalar_type());
            run(it, "masked_fill", {"select(x0, s0, x1)", "select(x0, s0, x1)", "select(x0, s0, x1)"},
                Opts{.scalars = {value}, .input_cts = {ct, CT::Bool}, .in_ct = ct});
            return self;
        }
        Tensor &masked_fill_tensor_fn(Tensor &self, const Tensor &mask, const Tensor &value)
        {
            TORCH_CHECK(value.dim() == 0, "masked_fill_ only supports a 0-dimensional value tensor, but got tensor with ",
                        value.dim(), " dimension(s).");
            if (!is_webgpu(value))
                return masked_fill_scalar_fn(self, mask, value.item());
            if (!gpu_dtypes_ok({self, mask, value}))
                return WEBGPU_FALLBACK(masked_fill_, Tensor)(self, mask, value);
            if (self.numel() == 0)
                return self;
            Tensor v = value.to(self.scalar_type());
            auto it = at::TensorIteratorConfig()
                          .set_check_mem_overlap(false)
                          .check_all_same_dtype(false)
                          .resize_outputs(false)
                          .add_output(self)
                          .add_const_input(self)
                          .add_const_input(mask)
                          .add_const_input(v)
                          .build();
            CT ct = wgsl::compute_type(self.scalar_type());
            run(it, "masked_fill_t", {"select(x0, x2, x1)", "select(x0, x2, x1)", "select(x0, x2, x1)"},
                Opts{.input_cts = {ct, CT::Bool, ct}, .in_ct = ct});
            return self;
        }

        // logical_* : output dtype is the out tensor's (bool for functional).
        Tensor &logical_binary(const Tensor &self, const Tensor &other, Tensor &out, const char *name, const char *expr)
        {
            auto it = at::TensorIteratorConfig()
                          .check_all_same_dtype(false)
                          .add_output(out)
                          .add_const_input(self)
                          .add_const_input(other)
                          .build();
            // Convert every input to bool.
            ElementwiseOp op;
            op.name = name;
            op.code = expr;
            op.in_ct = CT::Bool;
            op.out_ct = CT::Bool;
            kernels::run_elementwise(it, op);
            return out;
        }
#define WG_LOGICAL(NAME, EXPR)                                                                    \
    Tensor &logical_##NAME##_out_fn(const Tensor &self, const Tensor &other, Tensor &out)        \
    {                                                                                             \
        if (!gpu_dtypes_ok({self, other, out}))                                                   \
            return WEBGPU_FALLBACK(logical_##NAME, out)(self, other, out);                        \
        return logical_binary(self, other, out, "logical_" #NAME, EXPR);                          \
    }                                                                                             \
    Tensor logical_##NAME##_fn(const Tensor &self, const Tensor &other)                           \
    {                                                                                             \
        Tensor out = at::empty({0}, self.options().dtype(at::kBool).device(is_webgpu(self) ? self.device() : other.device())); \
        return logical_##NAME##_out_fn(self, other, out);                                         \
    }                                                                                             \
    Tensor &logical_##NAME##_inplace_fn(Tensor &self, const Tensor &other)                        \
    {                                                                                             \
        return logical_##NAME##_out_fn(self, other, self);                                        \
    }
        WG_LOGICAL(and, "x0 && x1")
        WG_LOGICAL(or, "x0 || x1")
        WG_LOGICAL(xor, "x0 != x1")

        Tensor &logical_not_out_fn(const Tensor &self, Tensor &out)
        {
            if (!gpu_dtypes_ok({self, out}))
                return WEBGPU_FALLBACK(logical_not, out)(self, out);
            auto it = at::TensorIteratorConfig()
                          .check_all_same_dtype(false)
                          .add_output(out)
                          .add_const_input(self)
                          .build();
            ElementwiseOp op;
            op.name = "logical_not";
            op.code = "!x0";
            op.in_ct = CT::Bool;
            op.out_ct = CT::Bool;
            kernels::run_elementwise(it, op);
            return out;
        }
        Tensor logical_not_fn(const Tensor &self)
        {
            Tensor out = at::empty({0}, self.options().dtype(at::kBool));
            return logical_not_out_fn(self, out);
        }
        Tensor &logical_not_inplace_fn(Tensor &self) { return logical_not_out_fn(self, self); }

        // isnan: bool output
        Tensor isnan_fn(const Tensor &self)
        {
            if (!gpu_dtypes_ok({self}))
                return WEBGPU_FALLBACK0(isnan)(self);
            Tensor out = at::empty(self.sizes(), self.options().dtype(at::kBool));
            if (!self.is_floating_point())
                return out.fill_(false);
            auto it = at::TensorIteratorConfig().check_all_same_dtype(false).add_output(out).add_const_input(self).build();
            run(it, "isnan", {"isnan_f(x0)"}, Opts{.out_ct = CT::Bool, .in_ct = CT::F32});
            return out;
        }

        // nan_to_num(self, nan, posinf, neginf)
        Tensor &nan_to_num_out_fn(const Tensor &self, std::optional<double> nan, std::optional<double> posinf,
                                  std::optional<double> neginf, Tensor &out)
        {
            if (!gpu_dtypes_ok({self, out}))
                return WEBGPU_FALLBACK(nan_to_num, out)(self, nan, posinf, neginf, out);
            TORCH_CHECK(self.scalar_type() == out.scalar_type(), "nan_to_num: dtype of out: ", out.scalar_type(),
                        " should be same as input: ", self.scalar_type());
            if (!self.is_floating_point())
            {
                at::native::resize_output(out, self.sizes());
                out.copy_(self);
                return out;
            }
            double lim = self.scalar_type() == at::kHalf       ? 65504.0
                         : self.scalar_type() == at::kBFloat16 ? 3.3895313892515355e38
                                                               : 3.4028234663852886e38;
            auto it = at::TensorIterator::unary_op(out, self);
            run(it, "nan_to_num", {"select(select(select(x0, s2, x0 == -inf_f()), s1, x0 == inf_f()), s0, isnan_f(x0))"},
                Opts{.scalars = {nan.value_or(0.0), posinf.value_or(lim), neginf.value_or(-lim)}});
            return out;
        }
        Tensor nan_to_num_fn(const Tensor &self, std::optional<double> nan, std::optional<double> posinf,
                             std::optional<double> neginf)
        {
            Tensor out = at::empty({0}, self.options());
            return nan_to_num_out_fn(self, nan, posinf, neginf, out);
        }
        Tensor &nan_to_num_inplace_fn(Tensor &self, std::optional<double> nan, std::optional<double> posinf,
                                      std::optional<double> neginf)
        {
            return nan_to_num_out_fn(self, nan, posinf, neginf, self);
        }
#undef K
    }

    TORCH_LIBRARY_IMPL(aten, PrivateUse1, m)
    {
        WG_REG_UNARY(m, sin)
        WG_REG_UNARY(m, cos)
        WG_REG_UNARY(m, tan)
        WG_REG_UNARY(m, asin)
        WG_REG_UNARY(m, acos)
        WG_REG_UNARY(m, atan)
        WG_REG_UNARY(m, sinh)
        WG_REG_UNARY(m, cosh)
        WG_REG_UNARY(m, tanh)
        WG_REG_UNARY(m, asinh)
        WG_REG_UNARY(m, acosh)
        WG_REG_UNARY(m, atanh)
        WG_REG_UNARY(m, exp)
        WG_REG_UNARY(m, exp2)
        WG_REG_UNARY(m, expm1)
        WG_REG_UNARY(m, log)
        WG_REG_UNARY(m, log2)
        WG_REG_UNARY(m, log10)
        WG_REG_UNARY(m, log1p)
        WG_REG_UNARY(m, sqrt)
        WG_REG_UNARY(m, rsqrt)
        WG_REG_UNARY(m, reciprocal)
        WG_REG_UNARY(m, sigmoid)
        WG_REG_UNARY(m, erf)
        WG_REG_UNARY(m, erfc)
        WG_REG_UNARY(m, erfinv)
        WG_REG_UNARY(m, sinc)
        WG_REG_UNARY(m, neg)
        WG_REG_UNARY(m, floor)
        WG_REG_UNARY(m, ceil)
        WG_REG_UNARY(m, trunc)
        WG_REG_UNARY(m, round)
        WG_REG_UNARY(m, frac)
        WG_REG_UNARY(m, sign)
        WG_REG_UNARY(m, sgn)
        WG_REG_UNARY(m, bitwise_not)
        WG_REG_UNARY(m, silu)
        WG_REG_UNARY(m, mish)
        WG_REG_UNARY(m, hardsigmoid)

        m.impl("signbit", TORCH_FN(signbit_fn));
        m.impl("signbit.out", TORCH_FN(signbit_out_fn));
        m.impl("isposinf", TORCH_FN(isposinf_fn));
        m.impl("isposinf.out", TORCH_FN(isposinf_out_fn));
        m.impl("isneginf", TORCH_FN(isneginf_fn));
        m.impl("isneginf.out", TORCH_FN(isneginf_out_fn));
        m.impl("isnan", TORCH_FN(isnan_fn));

        m.impl("gelu", TORCH_FN(gelu_fn));
        m.impl("gelu.out", TORCH_FN(gelu_out_fn));
        m.impl("gelu_", TORCH_FN(gelu_inplace_fn));
        m.impl("leaky_relu", TORCH_FN(leaky_relu_fn));
        m.impl("leaky_relu.out", TORCH_FN(leaky_relu_out_fn));
        m.impl("leaky_relu_", TORCH_FN(leaky_relu_inplace_fn));
        m.impl("hardshrink", TORCH_FN(hardshrink_fn));
        m.impl("hardshrink.out", TORCH_FN(hardshrink_out_fn));
        m.impl("softshrink", TORCH_FN(softshrink_fn));
        m.impl("softshrink.out", TORCH_FN(softshrink_out_fn));
        m.impl("softplus", TORCH_FN(softplus_fn));
        m.impl("softplus.out", TORCH_FN(softplus_out_fn));
        m.impl("threshold", TORCH_FN(threshold_fn));
        m.impl("threshold.out", TORCH_FN(threshold_out_fn));
        m.impl("threshold_", TORCH_FN(threshold_inplace_fn));
        m.impl("elu", TORCH_FN(elu_fn));
        m.impl("elu.out", TORCH_FN(elu_out_fn));
        m.impl("elu_", TORCH_FN(elu_inplace_fn));

#define REG3(NAME, ID, F, O, I)            \
    m.impl(F, TORCH_FN(ID##_fn));          \
    m.impl(O, TORCH_FN(ID##_out_fn));      \
    m.impl(I, TORCH_FN(ID##_inplace_fn));
#define REG2(ID, F, O)                     \
    m.impl(F, TORCH_FN(ID##_fn));          \
    m.impl(O, TORCH_FN(ID##_out_fn));
        REG3(add, add, "add.Tensor", "add.out", "add_.Tensor")
        REG3(sub, sub, "sub.Tensor", "sub.out", "sub_.Tensor")
        REG3(mul, mul, "mul.Tensor", "mul.out", "mul_.Tensor")
        REG3(div, div, "div.Tensor", "div.out", "div_.Tensor")
        REG3(div, div_mode, "div.Tensor_mode", "div.out_mode", "div_.Tensor_mode")
        REG2(maximum, "maximum", "maximum.out")
        REG2(minimum, "minimum", "minimum.out")
        REG2(fmax, "fmax", "fmax.out")
        REG2(fmin, "fmin", "fmin.out")
        REG3(atan2, atan2, "atan2", "atan2.out", "atan2_")
        REG3(hypot, hypot, "hypot", "hypot.out", "hypot_")
        REG3(copysign, copysign, "copysign.Tensor", "copysign.out", "copysign_.Tensor")
        REG2(logaddexp, "logaddexp", "logaddexp.out")
        REG2(logaddexp2, "logaddexp2", "logaddexp2.out")
        REG3(xlogy, xlogy, "xlogy.Tensor", "xlogy.OutTensor", "xlogy_.Tensor")
        REG3(heaviside, heaviside, "heaviside", "heaviside.out", "heaviside_")
        REG3(remainder, remainder, "remainder.Tensor", "remainder.Tensor_out", "remainder_.Tensor")
        REG3(fmod, fmod, "fmod.Tensor", "fmod.Tensor_out", "fmod_.Tensor")
        REG3(pow, pow_tt, "pow.Tensor_Tensor", "pow.Tensor_Tensor_out", "pow_.Tensor")
        REG3(pow, pow_ts, "pow.Tensor_Scalar", "pow.Tensor_Scalar_out", "pow_.Scalar")
        REG3(bitwise_and, bitwise_and, "bitwise_and.Tensor", "bitwise_and.Tensor_out", "bitwise_and_.Tensor")
        REG3(bitwise_or, bitwise_or, "bitwise_or.Tensor", "bitwise_or.Tensor_out", "bitwise_or_.Tensor")
        REG3(bitwise_xor, bitwise_xor, "bitwise_xor.Tensor", "bitwise_xor.Tensor_out", "bitwise_xor_.Tensor")
        REG3(lshift, lshift, "bitwise_left_shift.Tensor", "bitwise_left_shift.Tensor_out", "bitwise_left_shift_.Tensor")
        REG3(rshift, rshift, "bitwise_right_shift.Tensor", "bitwise_right_shift.Tensor_out", "bitwise_right_shift_.Tensor")
        REG3(gcd, gcd, "gcd", "gcd.out", "gcd_")
        REG3(lcm, lcm, "lcm", "lcm.out", "lcm_")
#define REG_CMP(NAME)                                                                    \
    REG3(NAME, NAME##_t, #NAME ".Tensor", #NAME ".Tensor_out", #NAME "_.Tensor")         \
    REG3(NAME, NAME##_s, #NAME ".Scalar", #NAME ".Scalar_out", #NAME "_.Scalar")
        REG_CMP(eq)
        REG_CMP(ne)
        REG_CMP(lt)
        REG_CMP(le)
        REG_CMP(gt)
        REG_CMP(ge)

        m.impl("clamp", TORCH_FN(clamp_fn));
        m.impl("clamp.out", TORCH_FN(clamp_out_fn));
        m.impl("clamp_", TORCH_FN(clamp_inplace_fn));
        REG3(clamp_min, clamp_min_s, "clamp_min", "clamp_min.out", "clamp_min_")
        REG3(clamp_max, clamp_max_s, "clamp_max", "clamp_max.out", "clamp_max_")
        REG3(clamp_min, clamp_min_t, "clamp_min.Tensor", "clamp_min.Tensor_out", "clamp_min_.Tensor")
        REG3(clamp_max, clamp_max_t, "clamp_max.Tensor", "clamp_max.Tensor_out", "clamp_max_.Tensor")

        REG3(addcmul, addcmul, "addcmul", "addcmul.out", "addcmul_")
        REG3(addcdiv, addcdiv, "addcdiv", "addcdiv.out", "addcdiv_")
        REG3(lerp, lerp_s, "lerp.Scalar", "lerp.Scalar_out", "lerp_.Scalar")
        REG3(lerp, lerp_t, "lerp.Tensor", "lerp.Tensor_out", "lerp_.Tensor")

        REG2(silu_backward, "silu_backward", "silu_backward.grad_input")
        REG2(tanh_backward, "tanh_backward", "tanh_backward.grad_input")
        REG2(sigmoid_backward, "sigmoid_backward", "sigmoid_backward.grad_input")
        REG2(hardsigmoid_backward, "hardsigmoid_backward", "hardsigmoid_backward.grad_input")
        REG2(threshold_backward, "threshold_backward", "threshold_backward.grad_input")
        REG2(leaky_relu_backward, "leaky_relu_backward", "leaky_relu_backward.grad_input")
        REG2(softplus_backward, "softplus_backward", "softplus_backward.grad_input")
        REG2(gelu_backward, "gelu_backward", "gelu_backward.grad_input")
        m.impl("elu_backward", TORCH_FN(elu_backward_fn));

        m.impl("abs", TORCH_FN(abs_fn));
        m.impl("abs.out", TORCH_FN(abs_out_fn));
        m.impl("abs_", TORCH_FN(abs_inplace_fn));
        m.impl("relu", TORCH_FN(relu_fn));
        m.impl("relu_", TORCH_FN(relu_inplace_fn));
        m.impl("hardtanh", TORCH_FN(hardtanh_fn));
        m.impl("hardtanh.out", TORCH_FN(hardtanh_out_fn));
        m.impl("hardtanh_", TORCH_FN(hardtanh_inplace_fn));
        m.impl("hardtanh_backward", TORCH_FN(hardtanh_backward_fn));
        m.impl("hardtanh_backward.grad_input", TORCH_FN(hardtanh_backward_out_fn));
        m.impl("hardswish", TORCH_FN(hardswish_fn));
        m.impl("hardswish.out", TORCH_FN(hardswish_out_fn));
        m.impl("hardswish_", TORCH_FN(hardswish_inplace_fn));
        m.impl("hardswish_backward", TORCH_FN(hardswish_backward_fn));
        m.impl("where.self", TORCH_FN(where_fn));
        m.impl("where.self_out", TORCH_FN(where_out_fn));
        m.impl("masked_fill_.Scalar", TORCH_FN(masked_fill_scalar_fn));
        m.impl("masked_fill_.Tensor", TORCH_FN(masked_fill_tensor_fn));
        m.impl("logical_and", TORCH_FN(logical_and_fn));
        m.impl("logical_and.out", TORCH_FN(logical_and_out_fn));
        m.impl("logical_and_", TORCH_FN(logical_and_inplace_fn));
        m.impl("logical_or", TORCH_FN(logical_or_fn));
        m.impl("logical_or.out", TORCH_FN(logical_or_out_fn));
        m.impl("logical_or_", TORCH_FN(logical_or_inplace_fn));
        m.impl("logical_xor", TORCH_FN(logical_xor_fn));
        m.impl("logical_xor.out", TORCH_FN(logical_xor_out_fn));
        m.impl("logical_xor_", TORCH_FN(logical_xor_inplace_fn));
        m.impl("logical_not", TORCH_FN(logical_not_fn));
        m.impl("logical_not.out", TORCH_FN(logical_not_out_fn));
        m.impl("logical_not_", TORCH_FN(logical_not_inplace_fn));
        m.impl("nan_to_num", TORCH_FN(nan_to_num_fn));
        m.impl("nan_to_num.out", TORCH_FN(nan_to_num_out_fn));
        m.impl("nan_to_num_", TORCH_FN(nan_to_num_inplace_fn));
    }
}
