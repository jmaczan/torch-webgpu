// Special functions and less common pointwise ops: gamma family, Bessel,
// normal-distribution helpers, clamp(Tensor), round(decimals), nextafter,
// frexp, angle, rsub.

#include <ATen/ATen.h>
#include <ATen/native/Resize.h>
#include <ATen/ScalarOps.h>
#include <ATen/WrapDimUtils.h>
#include <ATen/ops/clamp_meta.h>
#include <ATen/ops/linalg_cross_meta.h>
#include <ATen/ops/pow_meta.h>
#include <ATen/ops/digamma_meta.h>
#include <ATen/ops/i0_meta.h>
#include <ATen/ops/lgamma_meta.h>
#include <ATen/ops/nextafter_meta.h>
#include <ATen/ops/polygamma_meta.h>
#include <ATen/ops/round_meta.h>
#include <ATen/ops/special_entr_meta.h>
#include <ATen/ops/special_erfcx_meta.h>
#include <ATen/ops/special_i0e_meta.h>
#include <ATen/ops/special_i1_meta.h>
#include <ATen/ops/special_i1e_meta.h>
#include <ATen/ops/special_log_ndtr_meta.h>
#include <ATen/ops/special_ndtri_meta.h>
#include <ATen/ops/special_xlog1py_meta.h>
#include <ATen/ops/special_zeta_meta.h>
#include <torch/library.h>

#include <cmath>

#include "ops/pointwise_common.h"

namespace torch_webgpu::ops
{
    namespace
    {
        Opts uses(std::vector<std::string> fns) { return Opts{.special = std::move(fns)}; }

        // ------------------------------------------------- gamma family
        WG_UNARY(lgamma, K("lgamma", {"lgamma_f(x0)"}, uses({"lgamma_f"})))
        WG_UNARY(digamma, K("digamma", {"digamma_f(x0)"}, uses({"digamma_f"})))
        WG_UNARY(i0, K("i0", {"i0_f(x0)"}, uses({"i0_f"})))

#define WG_SPECIAL1(NAME, EXPR, FN)                                                                         \
    WG_OP2(NAME, structured_##NAME, K(#NAME, {EXPR}, uses({FN})), , , WEBGPU_FALLBACK0(NAME), \
           WEBGPU_FALLBACK(NAME, out))
        WG_SPECIAL1(special_i0e, "i0e_f(x0)", "i0e_f")
        WG_SPECIAL1(special_i1, "i1_f(x0)", "i1_f")
        WG_SPECIAL1(special_i1e, "i1e_f(x0)", "i1e_f")
        WG_SPECIAL1(special_ndtri, "ndtri_f(x0)", "ndtri_f")
        WG_SPECIAL1(special_log_ndtr, "log_ndtr_f(x0)", "log_ndtr_f")
        WG_SPECIAL1(special_erfcx, "erfcx_f(x0)", "erfcx_f")
        WG_SPECIAL1(special_entr, "entr_f(x0)", "entr_f")
#undef WG_SPECIAL1

        WG_OP2(special_xlog1py, structured_special_xlog1py, K("xlog1py", {"xlog1py_f(x0, x1)"}, uses({"xlog1py_f"})),
               P1(const Tensor &, other), A1(other), WEBGPU_FALLBACK0(special_xlog1py),
               WEBGPU_FALLBACK(special_xlog1py, out))
        WG_OP2(special_zeta, structured_special_zeta, K("zeta", {"zeta_f(x0, x1)"}, uses({"zeta_f"})),
               P1(const Tensor &, other), A1(other), WEBGPU_FALLBACK0(special_zeta), WEBGPU_FALLBACK(special_zeta, out))

        // polygamma(n, x) = (-1)^(n+1) n! zeta(n+1, x) for n >= 2.
        bool polygamma_kernel(TensorIteratorBase &it, int64_t n)
        {
            if (n == 0)
                return run(it, "digamma", {"digamma_f(x0)"}, uses({"digamma_f"}));
            if (n == 1)
                return run(it, "trigamma", {"trigamma_f(x0)"}, uses({"trigamma_f"}));
            const double factor = ((n % 2) ? 1.0 : -1.0) * std::exp(std::lgamma(static_cast<double>(n) + 1.0));
            Opts o = uses({"polygamma_f"});
            o.scalars = {Scalar(static_cast<double>(n + 1)), Scalar(factor)};
            o.scalar_ct = CT::F32;
            return run(it, "polygamma", {"polygamma_f(s0, s1, x0)"}, o);
        }
        Tensor polygamma_fn(int64_t n, const Tensor &self)
        {
            Functional<at::meta::structured_polygamma> op;
            op.meta(n, self);
            if (!iter_ok(op) || !polygamma_kernel(op, n))
                return WEBGPU_FALLBACK0(polygamma)(n, self);
            return std::move(op.outputs_[0]);
        }
        Tensor &polygamma_out_fn(int64_t n, const Tensor &self, Tensor &out)
        {
            Out<at::meta::structured_polygamma> op(out);
            op.meta(n, self);
            if (!iter_ok(op) || !polygamma_kernel(op, n))
                return WEBGPU_FALLBACK(polygamma, out)(n, self, out);
            op.finish();
            return out;
        }

        // ------------------------------------------------- clamp(Tensor)
        // Both bounds: NaN in any operand gives NaN (CPU/CUDA clamp_stub);
        // one bound: maximum/minimum semantics (NaN-propagating).
        bool clamp_tensor_kernel(TensorIteratorBase &it, bool has_min, bool has_max)
        {
            if (has_min && has_max)
                return run(it, "clamp_t",
                           {"select(min(max(x0, x1), x2), nan_f(), isnan_f(x0) || isnan_f(x1) || isnan_f(x2))",
                            "min(max(x0, x1), x2)", "(x0 || x1) && x2", "i64_min(i64_max(x0, x1), x2)"});
            if (has_min)
                return run(it, "maximum", {"select(max(x0, x1), nan_f(), isnan_f(x0) || isnan_f(x1))", "max(x0, x1)",
                                           "x0 || x1", "i64_max(x0, x1)"});
            return run(it, "minimum", {"select(min(x0, x1), nan_f(), isnan_f(x0) || isnan_f(x1))", "min(x0, x1)",
                                       "x0 && x1", "i64_min(x0, x1)"});
        }
        bool defined(const std::optional<Tensor> &t) { return t.has_value() && t->defined(); }
        at::OptionalTensorRef ref(const std::optional<Tensor> &t)
        {
            return defined(t) ? at::OptionalTensorRef(*t) : at::OptionalTensorRef();
        }
        Tensor clamp_t_fn(const Tensor &self, const std::optional<Tensor> &min, const std::optional<Tensor> &max)
        {
            Functional<at::meta::structured_clamp_Tensor> op;
            op.meta(self, ref(min), ref(max));
            if (!iter_ok(op) || !clamp_tensor_kernel(op, defined(min), defined(max)))
                return WEBGPU_FALLBACK(clamp, Tensor)(self, min, max);
            return std::move(op.outputs_[0]);
        }
        Tensor &clamp_t_out_fn(const Tensor &self, const std::optional<Tensor> &min, const std::optional<Tensor> &max,
                               Tensor &out)
        {
            Out<at::meta::structured_clamp_Tensor> op(out);
            op.meta(self, ref(min), ref(max));
            if (!iter_ok(op) || !clamp_tensor_kernel(op, defined(min), defined(max)))
                return WEBGPU_FALLBACK(clamp, Tensor_out)(self, min, max, out);
            op.finish();
            return out;
        }
        Tensor &clamp_t_inplace_fn(Tensor &self, const std::optional<Tensor> &min, const std::optional<Tensor> &max)
        {
            Inplace<at::meta::structured_clamp_Tensor> op(self);
            op.meta(self, ref(min), ref(max));
            if (!iter_ok(op) || !clamp_tensor_kernel(op, defined(min), defined(max)))
                return WEBGPU_FALLBACK(clamp_, Tensor)(self, min, max);
            op.finish();
            return self;
        }

        // ------------------------------------------------ round(decimals)
        bool round_decimals_kernel(TensorIteratorBase &it, int64_t decimals)
        {
            // As CPU/CUDA: scale by 10^|decimals| in the compute type, round
            // half to even, scale back.
            const bool neg = decimals < 0;
            const float ten_pow = static_cast<float>(std::pow(10.0, static_cast<double>(neg ? -decimals : decimals)));
            Opts o;
            o.scalars = {Scalar(static_cast<double>(ten_pow))};
            o.scalar_ct = CT::F32;
            return run(it, neg ? "round_dec_neg" : "round_dec",
                       {neg ? "round(x0 / s0) * s0" : "round(x0 * s0) / s0"}, o);
        }
        WG_OP3(round_decimals, structured_round_decimals,
               [&](TensorIteratorBase &it) { return round_decimals_kernel(it, decimals); }, P1(int64_t, decimals),
               A1(decimals), WEBGPU_FALLBACK(round, decimals), WEBGPU_FALLBACK(round, decimals_out),
               WEBGPU_FALLBACK(round_, decimals))

        // ------------------------------------------------------- nextafter
        // Bit stepping is exact only for float32 storage; half types use
        // the CPU (stepping must happen in the 16-bit representation).
        bool nextafter_kernel(TensorIteratorBase &it)
        {
            if (it.common_dtype() != at::kFloat)
                return false;
            return run(it, "nextafter",
                       {"select(select(select(bitcast<f32>(select(bitcast<u32>(x0) - 1u, bitcast<u32>(x0) + 1u, "
                        "(x0 < x1) == (x0 > 0.0))), bitcast<f32>(select(1u, 0x80000001u, x1 < 0.0)), x0 == 0.0), "
                        "x1, x0 == x1), nan_f(), isnan_f(x0) || isnan_f(x1))"});
        }
        WG_OP3(nextafter, structured_nextafter, nextafter_kernel, P1(const Tensor &, other), A1(other),
               WEBGPU_FALLBACK0(nextafter), WEBGPU_FALLBACK(nextafter, out), WEBGPU_FALLBACK0(nextafter_))

        // ----------------------------------------------------------- frexp
        std::tuple<Tensor &, Tensor &> frexp_out_fn(const Tensor &self, Tensor &mantissa, Tensor &exponent)
        {
            TORCH_CHECK(at::isFloatingType(self.scalar_type()),
                        "torch.frexp() only supports floating-point dtypes");
            TORCH_CHECK(mantissa.dtype() == self.dtype(), "torch.frexp() expects mantissa to have dtype ",
                        self.dtype(), " but got ", mantissa.dtype());
            TORCH_CHECK(exponent.dtype() == at::kInt, "torch.frexp() expects exponent to have int dtype but got ",
                        exponent.dtype());
            if (!gpu_dtypes_ok({self}))
                return WEBGPU_FALLBACK(frexp, Tensor_out)(self, mantissa, exponent);
            at::native::resize_output(mantissa, self.sizes());
            at::native::resize_output(exponent, self.sizes());
            auto mit = at::TensorIterator::unary_op(mantissa, self);
            run(mit, "frexp_m", {"bitcast<f32>(frexp_f(x0).x)"}, uses({"frexp_f"}));
            auto eit = at::TensorIteratorConfig()
                           .add_output(exponent)
                           .add_const_input(self)
                           .check_all_same_dtype(false)
                           .build();
            run(eit, "frexp_e", {"bitcast<i32>(frexp_f(x0).y)"},
                Opts{.out_ct = CT::I32, .in_ct = CT::F32, .special = {"frexp_f"}});
            return {mantissa, exponent};
        }

        // ----------------------------------------------------------- angle
        // Real inputs: pi for negatives, 0 otherwise, NaN kept; integers
        // promote to the default float dtype.
        Tensor &angle_out_fn(const Tensor &self, Tensor &out)
        {
            if (self.is_complex() || !gpu_dtypes_ok({self, out}))
                return WEBGPU_FALLBACK(angle, out)(self, out);
            auto it = at::TensorIterator::unary_float_op(out, self);
            if (!run(it, "angle", {"select(select(0.0, 3.14159265358979323846, x0 < 0.0), x0, isnan_f(x0))"},
                     Opts{.in_ct = CT::F32}))
                return WEBGPU_FALLBACK(angle, out)(self, out);
            return out;
        }
        Tensor angle_fn(const Tensor &self)
        {
            if (self.is_complex() || !gpu_dtypes_ok({self}))
                return WEBGPU_FALLBACK0(angle)(self);
            Tensor out;
            auto it = at::TensorIterator::unary_float_op(out, self);
            run(it, "angle", {"select(select(0.0, 3.14159265358979323846, x0 < 0.0), x0, isnan_f(x0))"},
                Opts{.in_ct = CT::F32});
            return it.output();
        }

        // ------------------------------------------------ pow(Scalar, Tensor)
        // As CPU: base 1 fills ones; otherwise pow.Tensor_Tensor with the base
        // as a wrapped scalar tensor (which keeps result_type semantics).
        void pow_scalar_impl(const Scalar &base, const Tensor &exponent, const Tensor &out)
        {
            if (base.equal(1.0))
                const_cast<Tensor &>(out).fill_(1);
            else
                at::pow_out(const_cast<Tensor &>(out), at::native::wrapped_scalar_tensor(base),
                            exponent);
        }
        Tensor pow_scalar_fn(const Scalar &base, const Tensor &exponent)
        {
            Functional<at::meta::structured_pow_Scalar> op;
            op.meta(base, exponent);
            pow_scalar_impl(base, exponent, op.outputs_[0]);
            return std::move(op.outputs_[0]);
        }
        Tensor &pow_scalar_out_fn(const Scalar &base, const Tensor &exponent, Tensor &out)
        {
            Out<at::meta::structured_pow_Scalar> op(out);
            op.meta(base, exponent);
            pow_scalar_impl(base, exponent, op.maybe_get_output(0));
            op.finish();
            return out;
        }

        // ---------------------------------------------------- linalg_cross
        // out_i = a_{i+1} b_{i+2} - a_{i+2} b_{i+1} along `dim` (size 3).
        Tensor &linalg_cross_out_fn(const Tensor &self, const Tensor &other, int64_t dim, Tensor &out)
        {
            Out<at::meta::structured_linalg_cross> op(out);
            op.meta(self, other, dim);
            const Tensor &o = op.maybe_get_output(0);
            if (o.numel() > 0)
            {
                const int64_t d = at::maybe_wrap_dim(dim, o.dim());
                Tensor a = self.expand(o.sizes()), b = other.expand(o.sizes());
                Tensor a1 = a.roll(-1, d), a2 = a.roll(-2, d), b1 = b.roll(-1, d), b2 = b.roll(-2, d);
                at::sub_out(const_cast<Tensor &>(o), a1 * b2, a2 * b1);
            }
            op.finish();
            return out;
        }
        Tensor linalg_cross_fn(const Tensor &self, const Tensor &other, int64_t dim)
        {
            Tensor out = at::empty({0}, self.options().dtype(at::result_type(self, other)));
            return linalg_cross_out_fn(self, other, dim, out);
        }

        // ------------------------------------------------------------ rsub
        Tensor rsub_fn(const Tensor &self, const Tensor &other, const Scalar &alpha)
        {
            return at::sub(other, self, alpha);
        }
#undef K
    }

    TORCH_LIBRARY_IMPL(aten, PrivateUse1, m)
    {
        WG_REG_UNARY(m, lgamma)
        WG_REG_UNARY(m, digamma)
        WG_REG_UNARY(m, i0)
        m.impl("special_i0e", TORCH_FN(special_i0e_fn));
        m.impl("special_i0e.out", TORCH_FN(special_i0e_out_fn));
        m.impl("special_i1", TORCH_FN(special_i1_fn));
        m.impl("special_i1.out", TORCH_FN(special_i1_out_fn));
        m.impl("special_i1e", TORCH_FN(special_i1e_fn));
        m.impl("special_i1e.out", TORCH_FN(special_i1e_out_fn));
        m.impl("special_ndtri", TORCH_FN(special_ndtri_fn));
        m.impl("special_ndtri.out", TORCH_FN(special_ndtri_out_fn));
        m.impl("special_log_ndtr", TORCH_FN(special_log_ndtr_fn));
        m.impl("special_log_ndtr.out", TORCH_FN(special_log_ndtr_out_fn));
        m.impl("special_erfcx", TORCH_FN(special_erfcx_fn));
        m.impl("special_erfcx.out", TORCH_FN(special_erfcx_out_fn));
        m.impl("special_entr", TORCH_FN(special_entr_fn));
        m.impl("special_entr.out", TORCH_FN(special_entr_out_fn));
        m.impl("special_xlog1py", TORCH_FN(special_xlog1py_fn));
        m.impl("special_xlog1py.out", TORCH_FN(special_xlog1py_out_fn));
        m.impl("special_zeta", TORCH_FN(special_zeta_fn));
        m.impl("special_zeta.out", TORCH_FN(special_zeta_out_fn));
        m.impl("polygamma", TORCH_FN(polygamma_fn));
        m.impl("polygamma.out", TORCH_FN(polygamma_out_fn));
        m.impl("clamp.Tensor", TORCH_FN(clamp_t_fn));
        m.impl("clamp.Tensor_out", TORCH_FN(clamp_t_out_fn));
        m.impl("clamp_.Tensor", TORCH_FN(clamp_t_inplace_fn));
        m.impl("round.decimals", TORCH_FN(round_decimals_fn));
        m.impl("round.decimals_out", TORCH_FN(round_decimals_out_fn));
        m.impl("round_.decimals", TORCH_FN(round_decimals_inplace_fn));
        m.impl("nextafter", TORCH_FN(nextafter_fn));
        m.impl("nextafter.out", TORCH_FN(nextafter_out_fn));
        m.impl("nextafter_", TORCH_FN(nextafter_inplace_fn));
        m.impl("frexp.Tensor_out", TORCH_FN(frexp_out_fn));
        m.impl("angle", TORCH_FN(angle_fn));
        m.impl("angle.out", TORCH_FN(angle_out_fn));
        m.impl("rsub.Tensor", TORCH_FN(rsub_fn));
        m.impl("pow.Scalar", TORCH_FN(pow_scalar_fn));
        m.impl("pow.Scalar_out", TORCH_FN(pow_scalar_out_fn));
        m.impl("linalg_cross", TORCH_FN(linalg_cross_fn));
        m.impl("linalg_cross.out", TORCH_FN(linalg_cross_out_fn));
    }
}
