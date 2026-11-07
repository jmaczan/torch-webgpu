// Reductions, softmax family, normalization and scans.

#include <ATen/ATen.h>
#include <ATen/OpMathType.h>
#include <ATen/native/ReduceOpsUtils.h>
#include <ATen/ops/_log_softmax_backward_data_meta.h>
#include <ATen/ops/_log_softmax_meta.h>
#include <ATen/ops/_softmax_backward_data_meta.h>
#include <ATen/ops/_softmax_meta.h>
#include <ATen/ops/all_meta.h>
#include <ATen/ops/amax_meta.h>
#include <ATen/ops/amin_meta.h>
#include <ATen/ops/aminmax_meta.h>
#include <ATen/ops/any_meta.h>
#include <ATen/ops/argmax_meta.h>
#include <ATen/ops/argmin_meta.h>
#include <ATen/ops/cumprod_meta.h>
#include <ATen/ops/cumsum_meta.h>
#include <ATen/ops/linalg_vector_norm_meta.h>
#include <ATen/ops/max_meta.h>
#include <ATen/ops/mean_meta.h>
#include <ATen/ops/min_meta.h>
#include <ATen/ops/prod_meta.h>
#include <ATen/ops/sum_meta.h>
#include <torch/library.h>

#include "kernels/elementwise.h"
#include "kernels/reduce.h"
#include "ops/common.h"

#include <cmath>
#include <limits>

namespace torch_webgpu::ops
{
    using at::Scalar;
    using at::Tensor;
    using kernels::CT;
    using kernels::ReduceOp;
    using kernels::RowOp;

    namespace
    {
        const char *tname(CT ct) { return wgsl::type_name(ct); }

        std::vector<int64_t> dims_of(at::OptionalIntArrayRef dim)
        {
            if (!dim.has_value())
                return {};
            return {dim->begin(), dim->end()};
        }

        // ----------------------------------------------------- op builders
        // Generic accumulators use the typed helpers (t_add, t_mul, ...) so
        // they work for every compute type, including emulated int64.
        ReduceOp sum_like(const char *name, CT ct, const char *step, const char *out)
        {
            ReduceOp op;
            op.name = name;
            op.in_ct = ct;
            op.out0_ct = ct;
            op.acc_struct = std::string("struct Acc { v: ") + tname(ct) + ", };";
            op.init = "    return Acc(t_zero());";
            op.step = step;
            op.combine = "    return Acc(t_add(a.v, b.v));";
            op.out0 = out;
            return op;
        }

        ReduceOp sum_op(CT ct) { return sum_like("sum", ct, "    return Acc(t_add(a.v, x));", "    return a.v;"); }
        ReduceOp nansum_op(CT ct)
        {
            return sum_like("nansum", ct, ct == CT::F32 ? "    return Acc(select(a.v + x, a.v, isnan_f(x)));" : "    return Acc(t_add(a.v, x));",
                            "    return a.v;");
        }
        ReduceOp mean_op() { return sum_like("mean", CT::F32, "    return Acc(a.v + x);", "    return a.v / f32(n);"); }

        ReduceOp prod_op(CT ct)
        {
            ReduceOp op;
            op.name = "prod";
            op.in_ct = ct;
            op.out0_ct = ct;
            op.acc_struct = std::string("struct Acc { v: ") + tname(ct) + ", };";
            op.init = "    return Acc(t_one());";
            op.step = "    return Acc(t_mul(a.v, x));";
            op.combine = "    return Acc(t_mul(a.v, b.v));";
            op.out0 = "    return a.v;";
            return op;
        }

        // amax / amin (NaN propagating)
        ReduceOp minmax_op(CT ct, bool is_max)
        {
            ReduceOp op;
            op.name = is_max ? "amax" : "amin";
            op.in_ct = ct;
            op.out0_ct = ct;
            op.acc_struct = std::string("struct Acc { v: ") + tname(ct) + ", };";
            if (ct == CT::F32)
            {
                op.init = is_max ? "    return Acc(-inf_f());" : "    return Acc(inf_f());";
                op.step = is_max ? "    return Acc(max_nan(a.v, x));" : "    return Acc(min_nan(a.v, x));";
                op.combine = is_max ? "    return Acc(max_nan(a.v, b.v));" : "    return Acc(min_nan(a.v, b.v));";
            }
            else if (ct == CT::I32)
            {
                op.init = is_max ? "    return Acc(-2147483647 - 1);" : "    return Acc(2147483647);";
                op.step = is_max ? "    return Acc(max(a.v, x));" : "    return Acc(min(a.v, x));";
                op.combine = is_max ? "    return Acc(max(a.v, b.v));" : "    return Acc(min(a.v, b.v));";
            }
            else if (ct == CT::I64)
            {
                op.init = is_max ? "    return Acc(i64(0u, 0x80000000u));" : "    return Acc(i64(0xffffffffu, 0x7fffffffu));";
                op.step = is_max ? "    return Acc(i64_max(a.v, x));" : "    return Acc(i64_min(a.v, x));";
                op.combine = is_max ? "    return Acc(i64_max(a.v, b.v));" : "    return Acc(i64_min(a.v, b.v));";
            }
            else
            {
                op.init = is_max ? "    return Acc(false);" : "    return Acc(true);";
                op.step = is_max ? "    return Acc(a.v || x);" : "    return Acc(a.v && x);";
                op.combine = is_max ? "    return Acc(a.v || b.v);" : "    return Acc(a.v && b.v);";
            }
            op.out0 = "    return a.v;";
            return op;
        }

        // Value + index (max.dim / min.dim / argmax / argmin). Ties and NaNs
        // resolve to the first index, NaN counts as the extreme value.
        ReduceOp arg_op(CT ct, bool is_max, bool values)
        {
            ReduceOp op;
            op.name = std::string(is_max ? "argmax" : "argmin") + (values ? "_v" : "");
            op.in_ct = ct;
            const char *T = tname(ct);
            op.acc_struct = std::string("struct Acc { v: ") + T + ", i: u32, };\n";
            std::string better, eq;
            if (ct == CT::F32)
            {
                better = is_max ? "(x > y || (isnan_f(x) && !isnan_f(y)))" : "(x < y || (isnan_f(x) && !isnan_f(y)))";
                eq = "(x == y || (isnan_f(x) && isnan_f(y)))";
            }
            else if (ct == CT::I32)
            {
                better = is_max ? "(x > y)" : "(x < y)";
                eq = "(x == y)";
            }
            else if (ct == CT::I64)
            {
                better = is_max ? "i64_gt(x, y)" : "i64_lt(x, y)";
                eq = "i64_eq(x, y)";
            }
            else
            {
                better = is_max ? "(x && !y)" : "(!x && y)";
                eq = "(x == y)";
            }
            op.acc_struct += std::string("fn arg_better(x: ") + T + ", y: " + T + ") -> bool { return " + better + "; }\n" +
                             "fn arg_eq(x: " + T + ", y: " + T + ") -> bool { return " + eq + "; }\n";
            op.init = "    return Acc(t_zero(), 0xffffffffu);";
            op.step = "    if (a.i == 0xffffffffu || arg_better(x, a.v)) { return Acc(x, r); }\n    return a;";
            op.combine = "    if (b.i == 0xffffffffu) { return a; }\n"
                         "    if (a.i == 0xffffffffu) { return b; }\n"
                         "    if (arg_better(b.v, a.v) || (arg_eq(b.v, a.v) && b.i < a.i)) { return b; }\n"
                         "    return a;";
            if (values)
            {
                op.out0_ct = ct;
                op.out0 = "    return a.v;";
                op.out1_ct = CT::I32;
                op.out1 = "    return i32(a.i);";
            }
            else
            {
                op.out0_ct = CT::I32;
                op.out0 = "    return i32(a.i);";
            }
            return op;
        }

        ReduceOp anyall_op(bool is_any)
        {
            ReduceOp op;
            op.name = is_any ? "any" : "all";
            op.in_ct = CT::Bool;
            op.out0_ct = CT::Bool;
            op.acc_struct = "struct Acc { v: bool, };";
            op.init = is_any ? "    return Acc(false);" : "    return Acc(true);";
            op.step = is_any ? "    return Acc(a.v || x);" : "    return Acc(a.v && x);";
            op.combine = is_any ? "    return Acc(a.v || b.v);" : "    return Acc(a.v && b.v);";
            op.out0 = "    return a.v;";
            return op;
        }

        // Welford mean/variance (Chan et al. parallel combine).
        ReduceOp welford_op(double correction, bool take_sqrt, bool with_mean)
        {
            ReduceOp op;
            op.name = take_sqrt ? "std" : "var";
            op.in_ct = CT::F32;
            op.out0_ct = CT::F32;
            op.acc_struct = "struct Acc { n: f32, mean: f32, m2: f32, };";
            op.init = "    return Acc(0.0, 0.0, 0.0);";
            op.step = "    let n1 = a.n + 1.0;\n    let d = x - a.mean;\n    let m = a.mean + d / n1;\n"
                      "    return Acc(n1, m, a.m2 + d * (x - m));";
            op.combine = "    let n = a.n + b.n;\n    if (n == 0.0) { return a; }\n    let d = b.mean - a.mean;\n"
                         "    return Acc(n, a.mean + d * b.n / n, a.m2 + b.m2 + d * d * a.n * b.n / n);";
            op.scalars = {{Scalar(correction), CT::F32}};
            op.out0 = take_sqrt ? "    return sqrt(a.m2 / max(a.n - s0, 0.0));" : "    return a.m2 / max(a.n - s0, 0.0);";
            if (with_mean)
            {
                op.out1_ct = CT::F32;
                op.out1 = "    return select(a.mean, nan_f(), a.n == 0.0);";
            }
            return op;
        }

        ReduceOp norm_op(double ord)
        {
            ReduceOp op;
            op.in_ct = CT::F32;
            op.out0_ct = CT::F32;
            op.acc_struct = "struct Acc { v: f32, };";
            op.combine = "    return Acc(a.v + b.v);";
            op.init = "    return Acc(0.0);";
            if (ord == 2.0)
            {
                op.name = "norm2";
                op.step = "    return Acc(a.v + x * x);";
                op.out0 = "    return sqrt(a.v);";
            }
            else if (ord == 1.0)
            {
                op.name = "norm1";
                op.step = "    return Acc(a.v + abs(x));";
                op.out0 = "    return a.v;";
            }
            else if (ord == 0.0)
            {
                op.name = "norm0";
                op.step = "    return Acc(a.v + select(0.0, 1.0, (bitcast<u32>(x) & 0x7fffffffu) != 0u));";
                op.out0 = "    return a.v;";
            }
            else if (std::isinf(ord))
            {
                op.name = ord > 0 ? "norminf" : "normninf";
                op.init = ord > 0 ? "    return Acc(0.0);" : "    return Acc(inf_f());";
                op.step = ord > 0 ? "    return Acc(max_nan(a.v, abs(x)));" : "    return Acc(min_nan(a.v, abs(x)));";
                op.combine = ord > 0 ? "    return Acc(max_nan(a.v, b.v));" : "    return Acc(min_nan(a.v, b.v));";
                op.out0 = "    return a.v;";
            }
            else
            {
                op.name = "normp";
                op.scalars = {{Scalar(ord), CT::F32}};
                op.step = "    return Acc(a.v + pow_f(abs(x), s0));";
                op.out0 = "    return pow_f(a.v, 1.0 / s0);";
            }
            return op;
        }

        bool reduce_ok(std::initializer_list<std::reference_wrapper<const Tensor>> ts) { return gpu_dtypes_ok(ts); }

        // ------------------------------------------------------------- sum
        void sum_impl(const Tensor &self, at::OptionalIntArrayRef dim, const Tensor &out)
        {
            CT ct = wgsl::compute_type(out.scalar_type());
            kernels::run_reduce(self, dims_of(dim), out, Tensor(), sum_op(ct));
        }
        Tensor sum_dim_fn(const Tensor &self, at::OptionalIntArrayRef dim, bool keepdim, std::optional<at::ScalarType> dtype)
        {
            Functional<at::meta::structured_sum_dim_IntList> op;
            op.meta(self, dim, keepdim, dtype);
            if (!reduce_ok({self, op.outputs_[0]}))
                return WEBGPU_FALLBACK(sum, dim_IntList)(self, dim, keepdim, dtype);
            sum_impl(self, dim, op.outputs_[0]);
            return std::move(op.outputs_[0]);
        }
        Tensor &sum_out_fn(const Tensor &self, at::OptionalIntArrayRef dim, bool keepdim, std::optional<at::ScalarType> dtype,
                           Tensor &out)
        {
            Out<at::meta::structured_sum_dim_IntList> op(out);
            op.meta(self, dim, keepdim, dtype);
            if (!reduce_ok({self, out}))
                return WEBGPU_FALLBACK(sum, IntList_out)(self, dim, keepdim, dtype, out);
            sum_impl(self, dim, op.maybe_get_output(0));
            op.finish();
            return out;
        }

        Tensor nansum_fn(const Tensor &self, at::OptionalIntArrayRef dim, bool keepdim, std::optional<at::ScalarType> dtype)
        {
            if (!self.is_floating_point())
                return at::sum(self, dim, keepdim, dtype);
            Functional<at::meta::structured_sum_dim_IntList> op;
            op.meta(self, dim, keepdim, dtype);
            if (!reduce_ok({self, op.outputs_[0]}))
                return WEBGPU_FALLBACK0(nansum)(self, dim, keepdim, dtype);
            kernels::run_reduce(self, dims_of(dim), op.outputs_[0], Tensor(), nansum_op(CT::F32));
            return std::move(op.outputs_[0]);
        }

        // ------------------------------------------------------------ mean
        Tensor mean_dim_fn(const Tensor &self, at::OptionalIntArrayRef dim, bool keepdim, std::optional<at::ScalarType> dtype)
        {
            Functional<at::meta::structured_mean_dim> op;
            op.meta(self, dim, keepdim, dtype);
            if (!reduce_ok({self, op.outputs_[0]}))
                return WEBGPU_FALLBACK(mean, dim)(self, dim, keepdim, dtype);
            kernels::run_reduce(self, dims_of(dim), op.outputs_[0], Tensor(), mean_op());
            return std::move(op.outputs_[0]);
        }
        Tensor &mean_out_fn(const Tensor &self, at::OptionalIntArrayRef dim, bool keepdim, std::optional<at::ScalarType> dtype,
                            Tensor &out)
        {
            Out<at::meta::structured_mean_dim> op(out);
            op.meta(self, dim, keepdim, dtype);
            if (!reduce_ok({self, out}))
                return WEBGPU_FALLBACK(mean, out)(self, dim, keepdim, dtype, out);
            kernels::run_reduce(self, dims_of(dim), op.maybe_get_output(0), Tensor(), mean_op());
            op.finish();
            return out;
        }

        // ------------------------------------------------------------ prod
        Tensor prod_dim_fn(const Tensor &self, int64_t dim, bool keepdim, std::optional<at::ScalarType> dtype)
        {
            Functional<at::meta::structured_prod_dim_int> op;
            op.meta(self, dim, keepdim, dtype);
            if (!reduce_ok({self, op.outputs_[0]}))
                return WEBGPU_FALLBACK(prod, dim_int)(self, dim, keepdim, dtype);
            int64_t d[1] = {dim};
            kernels::run_reduce(self, d, op.outputs_[0], Tensor(), prod_op(wgsl::compute_type(op.outputs_[0].scalar_type())));
            return std::move(op.outputs_[0]);
        }
        Tensor prod_fn(const Tensor &self, std::optional<at::ScalarType> dtype)
        {
            auto out_dtype = dtype.value_or(at::isIntegralType(self.scalar_type(), true) ? at::kLong : self.scalar_type());
            if (!gpu_dtypes_ok({self}) || !wgsl::is_compute_dtype(out_dtype))
                return WEBGPU_FALLBACK0(prod)(self, dtype);
            Tensor out = at::empty({}, self.options().dtype(out_dtype));
            kernels::run_reduce(self, {}, out, Tensor(), prod_op(wgsl::compute_type(out_dtype)));
            return out;
        }

        // ------------------------------------------------------- amax/amin
#define WG_MINMAX(NAME, IS_MAX)                                                                         \
    Tensor NAME##_fn(const Tensor &self, at::IntArrayRef dim, bool keepdim)                             \
    {                                                                                                   \
        Functional<at::meta::structured_##NAME> op;                                                     \
        op.meta(self, dim, keepdim);                                                                    \
        if (!reduce_ok({self}))                                                                         \
            return WEBGPU_FALLBACK0(NAME)(self, dim, keepdim);                                          \
        kernels::run_reduce(self, dim, op.outputs_[0], Tensor(),                                        \
                            minmax_op(wgsl::compute_type(self.scalar_type()), IS_MAX));                 \
        return std::move(op.outputs_[0]);                                                               \
    }                                                                                                   \
    Tensor &NAME##_out_fn(const Tensor &self, at::IntArrayRef dim, bool keepdim, Tensor &out)           \
    {                                                                                                   \
        Out<at::meta::structured_##NAME> op(out);                                                       \
        op.meta(self, dim, keepdim);                                                                    \
        if (!reduce_ok({self, out}))                                                                    \
            return WEBGPU_FALLBACK(NAME, out)(self, dim, keepdim, out);                                 \
        kernels::run_reduce(self, dim, op.maybe_get_output(0), Tensor(),                                \
                            minmax_op(wgsl::compute_type(self.scalar_type()), IS_MAX));                 \
        op.finish();                                                                                    \
        return out;                                                                                     \
    }
        WG_MINMAX(amax, true)
        WG_MINMAX(amin, false)

        // max() / min() over all elements
        Tensor max_all_fn(const Tensor &self)
        {
            TORCH_CHECK(self.numel() > 0, "max(): Expected reduction dim to be specified for input.numel() == 0. "
                                          "Specify the reduction dim with the 'dim' argument.");
            if (!reduce_ok({self}))
                return WEBGPU_FALLBACK0(max)(self);
            Tensor out = at::empty({}, self.options());
            kernels::run_reduce(self, {}, out, Tensor(), minmax_op(wgsl::compute_type(self.scalar_type()), true));
            return out;
        }
        Tensor min_all_fn(const Tensor &self)
        {
            TORCH_CHECK(self.numel() > 0, "min(): Expected reduction dim to be specified for input.numel() == 0. "
                                          "Specify the reduction dim with the 'dim' argument.");
            if (!reduce_ok({self}))
                return WEBGPU_FALLBACK0(min)(self);
            Tensor out = at::empty({}, self.options());
            kernels::run_reduce(self, {}, out, Tensor(), minmax_op(wgsl::compute_type(self.scalar_type()), false));
            return out;
        }

        // ------------------------------------------------- argmax / argmin
#define WG_ARG(NAME, IS_MAX)                                                                            \
    void NAME##_impl(const Tensor &self, std::optional<int64_t> dim, const Tensor &out)                 \
    {                                                                                                   \
        auto op_ = arg_op(wgsl::compute_type(self.scalar_type()), IS_MAX, false);                       \
        if (dim.has_value())                                                                            \
        {                                                                                               \
            int64_t d[1] = {*dim};                                                                      \
            if (self.dim() == 0)                                                                        \
                kernels::run_reduce(self, {}, out, Tensor(), op_);                                      \
            else                                                                                        \
                kernels::run_reduce(self, d, out, Tensor(), op_);                                       \
        }                                                                                               \
        else                                                                                            \
            kernels::run_reduce(self, {}, out, Tensor(), op_);                                          \
    }                                                                                                   \
    Tensor NAME##_fn(const Tensor &self, std::optional<int64_t> dim, bool keepdim)                      \
    {                                                                                                   \
        Functional<at::meta::structured_##NAME> op;                                                     \
        op.meta(self, dim, keepdim);                                                                    \
        if (!reduce_ok({self}))                                                                         \
            return WEBGPU_FALLBACK0(NAME)(self, dim, keepdim);                                          \
        NAME##_impl(self, dim, op.outputs_[0]);                                                         \
        return std::move(op.outputs_[0]);                                                               \
    }                                                                                                   \
    Tensor &NAME##_out_fn(const Tensor &self, std::optional<int64_t> dim, bool keepdim, Tensor &out)    \
    {                                                                                                   \
        Out<at::meta::structured_##NAME> op(out);                                                       \
        op.meta(self, dim, keepdim);                                                                    \
        if (!reduce_ok({self}))                                                                         \
            return WEBGPU_FALLBACK(NAME, out)(self, dim, keepdim, out);                                 \
        NAME##_impl(self, dim, op.maybe_get_output(0));                                                 \
        op.finish();                                                                                    \
        return out;                                                                                     \
    }
        WG_ARG(argmax, true)
        WG_ARG(argmin, false)

        // ---------------------------------------------- max.dim / min.dim
#define WG_MAXDIM(NAME, IS_MAX)                                                                                   \
    std::tuple<Tensor, Tensor> NAME##_dim_fn(const Tensor &self, int64_t dim, bool keepdim)                       \
    {                                                                                                             \
        Functional<at::meta::structured_##NAME##_dim, 2> op;                                                      \
        op.meta(self, dim, keepdim);                                                                              \
        if (!reduce_ok({self}))                                                                                   \
            return WEBGPU_FALLBACK(NAME, dim)(self, dim, keepdim);                                                \
        int64_t d[1] = {dim};                                                                                     \
        kernels::run_reduce(self, self.dim() == 0 ? at::IntArrayRef() : at::IntArrayRef(d), op.outputs_[0],      \
                            op.outputs_[1], arg_op(wgsl::compute_type(self.scalar_type()), IS_MAX, true));        \
        return {std::move(op.outputs_[0]), std::move(op.outputs_[1])};                                            \
    }                                                                                                             \
    std::tuple<Tensor &, Tensor &> NAME##_dim_out_fn(const Tensor &self, int64_t dim, bool keepdim, Tensor &values, \
                                                     Tensor &indices)                                             \
    {                                                                                                             \
        Out<at::meta::structured_##NAME##_dim, 2> op(values, indices);                                            \
        op.meta(self, dim, keepdim);                                                                              \
        if (!reduce_ok({self, values}))                                                                           \
            return WEBGPU_FALLBACK(NAME, dim_##NAME)(self, dim, keepdim, values, indices);                        \
        int64_t d[1] = {dim};                                                                                     \
        kernels::run_reduce(self, self.dim() == 0 ? at::IntArrayRef() : at::IntArrayRef(d), op.maybe_get_output(0), \
                            op.maybe_get_output(1), arg_op(wgsl::compute_type(self.scalar_type()), IS_MAX, true)); \
        op.finish();                                                                                              \
        return {values, indices};                                                                                 \
    }
        WG_MAXDIM(max, true)
        WG_MAXDIM(min, false)

        // ------------------------------------------------------- any / all
#define WG_ANYALL(NAME, IS_ANY)                                                                              \
    Tensor NAME##_dims_fn(const Tensor &self, at::OptionalIntArrayRef dim, bool keepdim)                     \
    {                                                                                                        \
        Functional<at::meta::structured_##NAME##_dims> op;                                                   \
        op.meta(self, dim, keepdim);                                                                         \
        if (!reduce_ok({self, op.outputs_[0]}))                                                              \
            return WEBGPU_FALLBACK(NAME, dims)(self, dim, keepdim);                                          \
        kernels::run_reduce(self, dims_of(dim), op.outputs_[0], Tensor(), anyall_op(IS_ANY));                \
        return std::move(op.outputs_[0]);                                                                    \
    }                                                                                                        \
    Tensor NAME##_dim_fn(const Tensor &self, int64_t dim, bool keepdim)                                      \
    {                                                                                                        \
        Functional<at::meta::structured_##NAME##_dim> op;                                                    \
        op.meta(self, dim, keepdim);                                                                         \
        if (!reduce_ok({self, op.outputs_[0]}))                                                              \
            return WEBGPU_FALLBACK(NAME, dim)(self, dim, keepdim);                                           \
        int64_t d[1] = {dim};                                                                                \
        kernels::run_reduce(self, self.dim() == 0 ? at::IntArrayRef() : at::IntArrayRef(d), op.outputs_[0], \
                            Tensor(), anyall_op(IS_ANY));                                                    \
        return std::move(op.outputs_[0]);                                                                    \
    }                                                                                                        \
    Tensor NAME##_all_fn(const Tensor &self)                                                                 \
    {                                                                                                        \
        Functional<at::meta::structured_##NAME> op;                                                          \
        op.meta(self);                                                                                       \
        if (!reduce_ok({self, op.outputs_[0]}))                                                              \
            return WEBGPU_FALLBACK0(NAME)(self);                                                             \
        kernels::run_reduce(self, {}, op.outputs_[0], Tensor(), anyall_op(IS_ANY));                          \
        return std::move(op.outputs_[0]);                                                                    \
    }
        WG_ANYALL(any, true)
        WG_ANYALL(all, false)

        // ------------------------------------------------------- var / std
        std::vector<int64_t> reduced_shape(const Tensor &self, const std::vector<int64_t> &dims, bool keepdim)
        {
            std::vector<bool> red(std::max<int64_t>(self.dim(), 1), dims.empty());
            for (auto d : dims)
                red[at::maybe_wrap_dim(d, std::max<int64_t>(self.dim(), 1))] = true;
            std::vector<int64_t> shape;
            for (int64_t d = 0; d < self.dim(); ++d)
            {
                if (!red[d])
                    shape.push_back(self.size(d));
                else if (keepdim)
                    shape.push_back(1);
            }
            return shape;
        }

        double correction_value(const std::optional<Scalar> &correction)
        {
            return correction.has_value() ? correction->toDouble() : 1.0;
        }

        Tensor var_fn(const Tensor &self, at::OptionalIntArrayRef dim, const std::optional<Scalar> &correction, bool keepdim)
        {
            if (!reduce_ok({self}) || !self.is_floating_point())
                return WEBGPU_FALLBACK(var, correction)(self, dim, correction, keepdim);
            auto dims = dims_of(dim);
            Tensor out = at::empty(reduced_shape(self, dims, keepdim), self.options());
            kernels::run_reduce(self, dims, out, Tensor(), welford_op(correction_value(correction), false, false));
            return out;
        }
        Tensor std_fn(const Tensor &self, at::OptionalIntArrayRef dim, const std::optional<Scalar> &correction, bool keepdim)
        {
            if (!reduce_ok({self}) || !self.is_floating_point())
                return WEBGPU_FALLBACK(std, correction)(self, dim, correction, keepdim);
            auto dims = dims_of(dim);
            Tensor out = at::empty(reduced_shape(self, dims, keepdim), self.options());
            kernels::run_reduce(self, dims, out, Tensor(), welford_op(correction_value(correction), true, false));
            return out;
        }
        std::tuple<Tensor, Tensor> var_mean_fn(const Tensor &self, at::OptionalIntArrayRef dim,
                                               const std::optional<Scalar> &correction, bool keepdim)
        {
            if (!reduce_ok({self}) || !self.is_floating_point())
                return WEBGPU_FALLBACK(var_mean, correction)(self, dim, correction, keepdim);
            auto dims = dims_of(dim);
            auto shape = reduced_shape(self, dims, keepdim);
            Tensor v = at::empty(shape, self.options()), m = at::empty(shape, self.options());
            kernels::run_reduce(self, dims, v, m, welford_op(correction_value(correction), false, true));
            return {v, m};
        }
        std::tuple<Tensor, Tensor> std_mean_fn(const Tensor &self, at::OptionalIntArrayRef dim,
                                               const std::optional<Scalar> &correction, bool keepdim)
        {
            if (!reduce_ok({self}) || !self.is_floating_point())
                return WEBGPU_FALLBACK(std_mean, correction)(self, dim, correction, keepdim);
            auto dims = dims_of(dim);
            auto shape = reduced_shape(self, dims, keepdim);
            Tensor v = at::empty(shape, self.options()), m = at::empty(shape, self.options());
            kernels::run_reduce(self, dims, v, m, welford_op(correction_value(correction), true, true));
            return {v, m};
        }

        // ---------------------------------------------- linalg_vector_norm
        Tensor &vector_norm_out_fn(const Tensor &self, const Scalar &ord, at::OptionalIntArrayRef dim, bool keepdim,
                                   std::optional<at::ScalarType> dtype, Tensor &out)
        {
            Out<at::meta::structured_linalg_vector_norm> op(out);
            op.meta(self, ord, dim, keepdim, dtype);
            if (!reduce_ok({self, out}) || self.is_complex())
                return WEBGPU_FALLBACK(linalg_vector_norm, out)(self, ord, dim, keepdim, dtype, out);
            kernels::run_reduce(self, dims_of(dim), op.maybe_get_output(0), Tensor(), norm_op(ord.toDouble()));
            op.finish();
            return out;
        }
        Tensor vector_norm_fn(const Tensor &self, const Scalar &ord, at::OptionalIntArrayRef dim, bool keepdim,
                              std::optional<at::ScalarType> dtype)
        {
            Functional<at::meta::structured_linalg_vector_norm> op;
            op.meta(self, ord, dim, keepdim, dtype);
            if (!reduce_ok({self, op.outputs_[0]}) || self.is_complex())
                return WEBGPU_FALLBACK0(linalg_vector_norm)(self, ord, dim, keepdim, dtype);
            kernels::run_reduce(self, dims_of(dim), op.outputs_[0], Tensor(), norm_op(ord.toDouble()));
            return std::move(op.outputs_[0]);
        }

        // --------------------------------------------------------- softmax
        const char *kSoftmax = R"(
    var m = -inf_f();
    for (var j = lid; j < n; j += 256u) { m = max_nan(m, ld0(j)); }
    m = wg_max(m);
    var s = 0.0;
    for (var j = lid; j < n; j += 256u) { s += exp(ld0(j) - m); }
    s = wg_sum(s);
    let inv = 1.0 / s;
    for (var j = lid; j < n; j += 256u) { st0(j, exp(ld0(j) - m) * inv); })";
        const char *kLogSoftmax = R"(
    var m = -inf_f();
    for (var j = lid; j < n; j += 256u) { m = max_nan(m, ld0(j)); }
    m = wg_max(m);
    var s = 0.0;
    for (var j = lid; j < n; j += 256u) { s += exp(ld0(j) - m); }
    let lse = m + log(wg_sum(s));
    for (var j = lid; j < n; j += 256u) { st0(j, ld0(j) - lse); })";
        // inputs: grad, output
        const char *kSoftmaxBwd = R"(
    var acc = 0.0;
    for (var j = lid; j < n; j += 256u) { acc += ld0(j) * ld1(j); }
    let s = wg_sum(acc);
    for (var j = lid; j < n; j += 256u) { st0(j, ld1(j) * (ld0(j) - s)); })";
        const char *kLogSoftmaxBwd = R"(
    var acc = 0.0;
    for (var j = lid; j < n; j += 256u) { acc += ld0(j); }
    let s = wg_sum(acc);
    for (var j = lid; j < n; j += 256u) { st0(j, ld0(j) - exp(ld1(j)) * s); })";

#define WG_SOFTMAX(ID, META, BODY, FB_F, FB_O)                                                          \
    Tensor ID##_fn(const Tensor &self, int64_t dim, bool half_to_float)                                 \
    {                                                                                                   \
        Functional<at::meta::META> op;                                                                  \
        op.meta(self, dim, half_to_float);                                                              \
        if (!reduce_ok({self}))                                                                         \
            return FB_F(self, dim, half_to_float);                                                      \
        if (self.dim() == 0)                                                                            \
        {                                                                                               \
            kernels::run_rows({self.reshape({1})}, {op.outputs_[0].reshape({1})}, {}, 0, RowOp{#ID, BODY});    \
            return std::move(op.outputs_[0]);                                                           \
        }                                                                                               \
        kernels::run_rows({self}, {op.outputs_[0]}, {}, dim, RowOp{#ID, BODY});                         \
        return std::move(op.outputs_[0]);                                                               \
    }                                                                                                   \
    Tensor &ID##_out_fn(const Tensor &self, int64_t dim, bool half_to_float, Tensor &out)               \
    {                                                                                                   \
        Out<at::meta::META> op(out);                                                                    \
        op.meta(self, dim, half_to_float);                                                              \
        if (!reduce_ok({self, out}))                                                                    \
            return FB_O(self, dim, half_to_float, out);                                                 \
        const Tensor &o = op.maybe_get_output(0);                                                       \
        if (self.dim() == 0)                                                                            \
            kernels::run_rows({self.reshape({1})}, {o.reshape({1})}, {}, 0, RowOp{#ID, BODY});                 \
        else                                                                                            \
            kernels::run_rows({self}, {o}, {}, dim, RowOp{#ID, BODY});                                  \
        op.finish();                                                                                    \
        return out;                                                                                     \
    }
        WG_SOFTMAX(softmax, structured__softmax, kSoftmax, WEBGPU_FALLBACK0(_softmax), WEBGPU_FALLBACK(_softmax, out))
        WG_SOFTMAX(log_softmax, structured__log_softmax, kLogSoftmax, WEBGPU_FALLBACK0(_log_softmax),
                   WEBGPU_FALLBACK(_log_softmax, out))

#define WG_SOFTMAX_BWD(ID, META, BODY, FB_F, FB_O)                                                            \
    Tensor ID##_fn(const Tensor &grad, const Tensor &output, int64_t dim, at::ScalarType input_dtype)         \
    {                                                                                                         \
        Functional<at::meta::META> op;                                                                        \
        op.meta(grad, output, dim, input_dtype);                                                              \
        if (!reduce_ok({grad, output}))                                                                       \
            return FB_F(grad, output, dim, input_dtype);                                                      \
        if (grad.dim() == 0)                                                                                  \
            kernels::run_rows({grad.reshape({1}), output.reshape({1})}, {op.outputs_[0].reshape({1})}, {}, 0, RowOp{#ID, BODY}); \
        else                                                                                                  \
            kernels::run_rows({grad, output}, {op.outputs_[0]}, {}, dim, RowOp{#ID, BODY});                   \
        return std::move(op.outputs_[0]);                                                                     \
    }                                                                                                         \
    Tensor &ID##_out_fn(const Tensor &grad, const Tensor &output, int64_t dim, at::ScalarType input_dtype,    \
                        Tensor &out)                                                                          \
    {                                                                                                         \
        Out<at::meta::META> op(out);                                                                          \
        op.meta(grad, output, dim, input_dtype);                                                              \
        if (!reduce_ok({grad, output, out}))                                                                  \
            return FB_O(grad, output, dim, input_dtype, out);                                                 \
        const Tensor &o = op.maybe_get_output(0);                                                             \
        if (grad.dim() == 0)                                                                                  \
            kernels::run_rows({grad.reshape({1}), output.reshape({1})}, {o.reshape({1})}, {}, 0, RowOp{#ID, BODY});    \
        else                                                                                                  \
            kernels::run_rows({grad, output}, {o}, {}, dim, RowOp{#ID, BODY});                                \
        op.finish();                                                                                          \
        return out;                                                                                           \
    }
        WG_SOFTMAX_BWD(softmax_bwd, structured__softmax_backward_data, kSoftmaxBwd,
                       WEBGPU_FALLBACK0(_softmax_backward_data), WEBGPU_FALLBACK(_softmax_backward_data, out))
        WG_SOFTMAX_BWD(log_softmax_bwd, structured__log_softmax_backward_data, kLogSoftmaxBwd,
                       WEBGPU_FALLBACK0(_log_softmax_backward_data), WEBGPU_FALLBACK(_log_softmax_backward_data, out))

        // ------------------------------------------------------ layer norm
        // Rows of length N over the flattened normalized dims.
        std::tuple<Tensor, Tensor, Tensor> native_layer_norm_fn(const Tensor &input, c10::SymIntArrayRef normalized_shape_sym,
                                                                const std::optional<Tensor> &weight_opt,
                                                                const std::optional<Tensor> &bias_opt, double eps)
        {
            auto normalized_shape = C10_AS_INTARRAYREF_SLOW(normalized_shape_sym);
            const Tensor &weight = weight_opt.has_value() ? *weight_opt : Tensor();
            const Tensor &bias = bias_opt.has_value() ? *bias_opt : Tensor();
            if (!gpu_dtypes_ok({input, weight, bias}))
                return WEBGPU_FALLBACK0(native_layer_norm)(input, normalized_shape, weight_opt, bias_opt, eps);
            const int64_t nd = static_cast<int64_t>(normalized_shape.size());
            const int64_t axis = input.dim() - nd;
            TORCH_CHECK(axis >= 0 && input.sizes().slice(axis).equals(normalized_shape),
                        "Given normalized_shape=", normalized_shape, ", expected input with shape [*, ",
                        normalized_shape, "], but got input of size", input.sizes());
            int64_t N = 1;
            for (auto s : normalized_shape)
                N *= s;
            const int64_t M = N == 0 ? 0 : input.numel() / N;

            Tensor x = input.contiguous();
            Tensor y = at::empty(input.sizes(), input.options().memory_format(c10::MemoryFormat::Contiguous));
            // mean/rstd: shape [*outer, 1, ..., 1], dtype float for reduced precision inputs (like CUDA).
            auto stat_dtype = at::toOpMathType(input.scalar_type());
            std::vector<int64_t> stat_shape(input.sizes().begin(), input.sizes().begin() + axis);
            for (int64_t i = 0; i < nd; ++i)
                stat_shape.push_back(1);
            Tensor mean = at::empty(stat_shape, input.options().dtype(stat_dtype));
            Tensor rstd = at::empty(stat_shape, input.options().dtype(stat_dtype));
            if (M == 0 || N == 0)
                return {y, mean, rstd};

            Tensor x2 = x.view({M, N});
            Tensor y2 = y.view({M, N});
            std::vector<Tensor> ins{x2};
            std::string wexpr = "1.0", bexpr = "0.0";
            if (weight.defined())
            {
                ins.push_back(weight.contiguous().view({1, N}).expand({M, N}));
                wexpr = "ld" + std::to_string(ins.size() - 1) + "(j)";
            }
            if (bias.defined())
            {
                ins.push_back(bias.contiguous().view({1, N}).expand({M, N}));
                bexpr = "ld" + std::to_string(ins.size() - 1) + "(j)";
            }
            std::string body = R"(
    var s = 0.0;
    for (var j = lid; j < n; j += 256u) { s += ld0(j); }
    let mean = wg_sum(s) / f32(n);
    var v = 0.0;
    for (var j = lid; j < n; j += 256u) { let d = ld0(j) - mean; v += d * d; }
    let rstd = inverseSqrt(wg_sum(v) / f32(n) + s0);
    for (var j = lid; j < n; j += 256u) { st0(j, (ld0(j) - mean) * rstd * )" +
                               wexpr + " + " + bexpr + R"(); }
    rst0(mean);
    rst1(rstd);)";
            RowOp op{"layer_norm", body, {{Scalar(eps), CT::F32}}};
            kernels::run_rows(ins, {y2}, {mean.view({M}), rstd.view({M})}, 1, op);
            return {y, mean, rstd};
        }

        std::tuple<Tensor, Tensor, Tensor> native_layer_norm_backward_fn(
            const Tensor &grad_out, const Tensor &input, c10::SymIntArrayRef normalized_shape_sym, const Tensor &mean,
            const Tensor &rstd, const std::optional<Tensor> &weight_opt, const std::optional<Tensor> &bias_opt,
            std::array<bool, 3> output_mask)
        {
            auto normalized_shape = C10_AS_INTARRAYREF_SLOW(normalized_shape_sym);
            const Tensor &weight = weight_opt.has_value() ? *weight_opt : Tensor();
            const Tensor &bias = bias_opt.has_value() ? *bias_opt : Tensor();
            if (!gpu_dtypes_ok({grad_out, input, mean, rstd, weight, bias}))
                return WEBGPU_FALLBACK0(native_layer_norm_backward)(grad_out, input, normalized_shape, mean, rstd,
                                                                    weight_opt, bias_opt, output_mask);
            const int64_t nd = static_cast<int64_t>(normalized_shape.size());
            const int64_t axis = input.dim() - nd;
            int64_t N = 1;
            for (auto s : normalized_shape)
                N *= s;
            const int64_t M = N == 0 ? 0 : input.numel() / N;
            Tensor dx, dw, db;
            Tensor x2 = input.contiguous().view({M, N});
            Tensor g2 = grad_out.contiguous().view({M, N});
            Tensor mean2 = mean.contiguous().view({M, 1}).expand({M, N});
            Tensor rstd2 = rstd.contiguous().view({M, 1}).expand({M, N});
            if (output_mask[0])
            {
                dx = at::empty(input.sizes(), input.options().memory_format(c10::MemoryFormat::Contiguous));
                if (M > 0 && N > 0)
                {
                    std::vector<Tensor> ins{g2, x2, mean2, rstd2};
                    std::string wexpr = "1.0";
                    if (weight.defined())
                    {
                        ins.push_back(weight.contiguous().view({1, N}).expand({M, N}));
                        wexpr = "ld4(j)";
                    }
                    std::string body = R"(
    let mean = ld2(0u);
    let rstd = ld3(0u);
    var a = 0.0;
    var b = 0.0;
    for (var j = lid; j < n; j += 256u) {
        let g = ld0(j) * )" + wexpr + R"(;
        let xh = (ld1(j) - mean) * rstd;
        a += g;
        b += g * xh;
    }
    let mg = wg_sum(a) / f32(n);
    let mgx = wg_sum(b) / f32(n);
    for (var j = lid; j < n; j += 256u) {
        let g = ld0(j) * )" + wexpr + R"(;
        let xh = (ld1(j) - mean) * rstd;
        st0(j, rstd * (g - mg - xh * mgx));
    })";
                    kernels::run_rows(ins, {dx.view({M, N})}, {}, 1, RowOp{"layer_norm_bwd", body});
                }
            }
            if (output_mask[1] && weight.defined())
            {
                if (M == 0)
                    dw = at::zeros_like(weight);
                else
                {
                    Tensor xhat = (x2 - mean2) * rstd2;
                    dw = (g2 * xhat).sum(0).to(weight.scalar_type()).view(weight.sizes());
                }
            }
            if (output_mask[2] && bias.defined())
            {
                db = M == 0 ? at::zeros_like(bias) : g2.sum(0).to(bias.scalar_type()).view(bias.sizes());
            }
            return {dx, dw, db};
        }

        // ------------------------------------------------- cumsum/cumprod
        const char *kScan = R"(
    var carry = t_zero();
    for (var t0 = 0u; t0 < n; t0 = t0 + 256u) {
        let j = t0 + lid;
        var v = t_zero();
        if (j < n) { v = ld0(j); }
        let sc = wg_scan(v);
        if (j < n) { st0(j, t_add(sc, carry)); }
        carry = t_add(carry, wg_bcast(sc, 255u));
    })";
        const char *kScanMul = R"(
    var carry = t_one();
    for (var t0 = 0u; t0 < n; t0 = t0 + 256u) {
        let j = t0 + lid;
        var v = t_one();
        if (j < n) { v = ld0(j); }
        let sc = wg_scan_mul(v);
        if (j < n) { st0(j, t_mul(sc, carry)); }
        carry = t_mul(carry, wg_bcast(sc, 255u));
    })";

        void scan_impl(const Tensor &self, int64_t dim, const Tensor &out, bool mul)
        {
            if (self.numel() == 0)
                return;
            CT ct = wgsl::compute_type(out.scalar_type());
            if (ct == CT::Bool)
                ct = CT::I32;
            RowOp op{mul ? "cumprod" : "cumsum", mul ? kScanMul : kScan, {}, ct};
            if (self.dim() == 0)
                kernels::run_rows({self.reshape({1})}, {out.reshape({1})}, {}, 0, op);
            else
                kernels::run_rows({self}, {out}, {}, dim, op);
        }

#define WG_SCAN(NAME, MUL)                                                                                \
    Tensor NAME##_fn(const Tensor &self, int64_t dim, std::optional<at::ScalarType> dtype)                \
    {                                                                                                     \
        Functional<at::meta::structured_##NAME> op;                                                       \
        op.meta(self, dim, dtype);                                                                        \
        if (!reduce_ok({self, op.outputs_[0]}))                                                           \
            return WEBGPU_FALLBACK0(NAME)(self, dim, dtype);                                              \
        scan_impl(self, dim, op.outputs_[0], MUL);                                                        \
        return std::move(op.outputs_[0]);                                                                 \
    }                                                                                                     \
    Tensor &NAME##_out_fn(const Tensor &self, int64_t dim, std::optional<at::ScalarType> dtype, Tensor &out) \
    {                                                                                                     \
        Out<at::meta::structured_##NAME> op(out);                                                         \
        op.meta(self, dim, dtype);                                                                        \
        if (!reduce_ok({self, out}))                                                                      \
            return WEBGPU_FALLBACK(NAME, out)(self, dim, dtype, out);                                     \
        scan_impl(self, dim, op.maybe_get_output(0), MUL);                                                \
        op.finish();                                                                                      \
        return out;                                                                                       \
    }                                                                                                     \
    Tensor &NAME##_inplace_fn(Tensor &self, int64_t dim, std::optional<at::ScalarType> dtype)             \
    {                                                                                                     \
        Inplace<at::meta::structured_##NAME> op(self);                                                    \
        op.meta(self, dim, dtype);                                                                        \
        if (!reduce_ok({self}))                                                                           \
            return WEBGPU_FALLBACK0(NAME##_)(self, dim, dtype);                                           \
        /* in-place scans read and write the same row: go through a temporary */                         \
        Tensor tmp = at::empty_like(self);                                                                \
        scan_impl(self, dim, tmp, MUL);                                                                   \
        self.copy_(tmp);                                                                                  \
        return self;                                                                                      \
    }
        WG_SCAN(cumsum, false)
        WG_SCAN(cumprod, true)
    }

    TORCH_LIBRARY_IMPL(aten, PrivateUse1, m)
    {
        m.impl("sum.dim_IntList", TORCH_FN(sum_dim_fn));
        m.impl("sum.IntList_out", TORCH_FN(sum_out_fn));
        m.impl("nansum", TORCH_FN(nansum_fn));
        m.impl("mean.dim", TORCH_FN(mean_dim_fn));
        m.impl("mean.out", TORCH_FN(mean_out_fn));
        m.impl("prod.dim_int", TORCH_FN(prod_dim_fn));
        m.impl("prod", TORCH_FN(prod_fn));
        m.impl("amax", TORCH_FN(amax_fn));
        m.impl("amax.out", TORCH_FN(amax_out_fn));
        m.impl("amin", TORCH_FN(amin_fn));
        m.impl("amin.out", TORCH_FN(amin_out_fn));
        m.impl("max", TORCH_FN(max_all_fn));
        m.impl("min", TORCH_FN(min_all_fn));
        m.impl("argmax", TORCH_FN(argmax_fn));
        m.impl("argmax.out", TORCH_FN(argmax_out_fn));
        m.impl("argmin", TORCH_FN(argmin_fn));
        m.impl("argmin.out", TORCH_FN(argmin_out_fn));
        m.impl("max.dim", TORCH_FN(max_dim_fn));
        m.impl("max.dim_max", TORCH_FN(max_dim_out_fn));
        m.impl("min.dim", TORCH_FN(min_dim_fn));
        m.impl("min.dim_min", TORCH_FN(min_dim_out_fn));
        m.impl("any.dims", TORCH_FN(any_dims_fn));
        m.impl("any.dim", TORCH_FN(any_dim_fn));
        m.impl("any", TORCH_FN(any_all_fn));
        m.impl("all.dims", TORCH_FN(all_dims_fn));
        m.impl("all.dim", TORCH_FN(all_dim_fn));
        m.impl("all", TORCH_FN(all_all_fn));
        m.impl("var.correction", TORCH_FN(var_fn));
        m.impl("std.correction", TORCH_FN(std_fn));
        m.impl("var_mean.correction", TORCH_FN(var_mean_fn));
        m.impl("std_mean.correction", TORCH_FN(std_mean_fn));
        m.impl("linalg_vector_norm", TORCH_FN(vector_norm_fn));
        m.impl("linalg_vector_norm.out", TORCH_FN(vector_norm_out_fn));
        m.impl("_softmax", TORCH_FN(softmax_fn));
        m.impl("_softmax.out", TORCH_FN(softmax_out_fn));
        m.impl("_log_softmax", TORCH_FN(log_softmax_fn));
        m.impl("_log_softmax.out", TORCH_FN(log_softmax_out_fn));
        m.impl("_softmax_backward_data", TORCH_FN(softmax_bwd_fn));
        m.impl("_softmax_backward_data.out", TORCH_FN(softmax_bwd_out_fn));
        m.impl("_log_softmax_backward_data", TORCH_FN(log_softmax_bwd_fn));
        m.impl("_log_softmax_backward_data.out", TORCH_FN(log_softmax_bwd_out_fn));
        m.impl("native_layer_norm", TORCH_FN(native_layer_norm_fn));
        m.impl("native_layer_norm_backward", TORCH_FN(native_layer_norm_backward_fn));
        m.impl("cumsum", TORCH_FN(cumsum_fn));
        m.impl("cumsum.out", TORCH_FN(cumsum_out_fn));
        m.impl("cumsum_", TORCH_FN(cumsum_inplace_fn));
        m.impl("cumprod", TORCH_FN(cumprod_fn));
        m.impl("cumprod.out", TORCH_FN(cumprod_out_fn));
        m.impl("cumprod_", TORCH_FN(cumprod_inplace_fn));
    }
}
