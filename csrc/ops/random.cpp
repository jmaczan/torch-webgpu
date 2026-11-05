// Random sampling: uniform_, normal_, bernoulli_, random_, exponential_,
// geometric_, log_normal_, cauchy_, native_dropout.
//
// Values come from a counter-based Philox4x32-10 generator evaluated in the
// kernel (like CUDA): key = generator seed, counter = (element/4, kernel
// offset). Each launch advances the generator offset, so results are
// reproducible with torch.manual_seed and independent between calls.

#include <ATen/ATen.h>
#include <ATen/TensorIterator.h>
#include <torch/library.h>

#include "core/generator.h"
#include "kernels/elementwise.h"
#include "kernels/math.h"
#include "ops/common.h"

namespace torch_webgpu::ops
{
    using at::Scalar;
    using at::Tensor;
    using kernels::CT;
    using kernels::ElementwiseOp;

    namespace
    {
        const std::string &philox_wgsl()
        {
            static const std::string s = R"(
fn mulhilo(a: u32, b: u32) -> vec2<u32> {
    let a_lo = a & 0xffffu; let a_hi = a >> 16u;
    let b_lo = b & 0xffffu; let b_hi = b >> 16u;
    let lolo = a_lo * b_lo;
    let lohi = a_lo * b_hi;
    let hilo = a_hi * b_lo;
    let hihi = a_hi * b_hi;
    let mid = (lolo >> 16u) + (lohi & 0xffffu) + (hilo & 0xffffu);
    let hi = hihi + (lohi >> 16u) + (hilo >> 16u) + (mid >> 16u);
    let lo = (mid << 16u) | (lolo & 0xffffu);
    return vec2<u32>(hi, lo);
}
fn philox(ctr0: vec4<u32>, key0: vec2<u32>) -> vec4<u32> {
    var c = ctr0;
    var k = key0;
    for (var r = 0u; r < 10u; r++) {
        let p0 = mulhilo(0xD2511F53u, c.x);
        let p1 = mulhilo(0xCD9E8D57u, c.z);
        c = vec4<u32>(p1.x ^ c.y ^ k.x, p1.y, p0.x ^ c.w ^ k.y, p0.y);
        k = k + vec2<u32>(0x9E3779B9u, 0xBB67AE85u);
    }
    return c;
}
// Random words for linear index idx: lanes of one Philox call per 4 elements.
fn rng4(idx: u32, seed_lo: u32, seed_hi: u32, off_lo: u32, off_hi: u32) -> vec4<u32> {
    return philox(vec4<u32>(idx >> 2u, 0u, off_lo, off_hi), vec2<u32>(seed_lo, seed_hi));
}
fn rng_u32(idx: u32, seed_lo: u32, seed_hi: u32, off_lo: u32, off_hi: u32) -> u32 {
    return rng4(idx, seed_lo, seed_hi, off_lo, off_hi)[idx & 3u];
}
// Uniform in [0, 1) with 24 bits of randomness.
fn u01(w: u32) -> f32 { return f32(w >> 8u) * 5.9604644775390625e-8; }
// Uniform in (0, 1].
fn u01_open0(w: u32) -> f32 { return (f32(w >> 8u) + 1.0) * 5.9604644775390625e-8; }
fn randn(idx: u32, seed_lo: u32, seed_hi: u32, off_lo: u32, off_hi: u32) -> f32 {
    let r = rng4(idx, seed_lo, seed_hi, off_lo, off_hi);
    let lane = idx & 3u;
    var a = r.x; var b = r.y;
    if (lane >= 2u) { a = r.z; b = r.w; }
    let rad = sqrt(-2.0 * log(u01_open0(a)));
    let th = 6.283185307179586 * u01(b);
    return select(rad * cos_f(th), rad * sin_f(th), (lane & 1u) == 1u);
}
)";
            return s;
        }

        // Runs `code` (which may call RNG(...) helpers via the s0..s3 seed/offset
        // scalars) over self in place, then advances the generator.
        void rng_fill(Tensor &self, const std::optional<at::Generator> &gen, const char *name, const std::string &code_f,
                      const std::string &code_i, std::vector<std::pair<Scalar, CT>> extra_scalars, CT out_ct,
                      const Tensor &input = Tensor())
        {
            if (self.numel() == 0)
                return;
            auto *g = core::get_generator(gen);
            uint64_t seed, offset;
            {
                std::lock_guard<std::mutex> lock(g->mutex_);
                std::tie(seed, offset) = g->philox_engine_inputs(4);
            }
            // Contiguous scratch so the linear index is the element index.
            Tensor out = self.is_contiguous() ? self : at::empty(self.sizes(), self.options());
            at::TensorIteratorConfig cfg;
            cfg.add_output(out).resize_outputs(false).check_all_same_dtype(false).set_check_mem_overlap(false);
            if (input.defined())
                cfg.add_const_input(input);
            auto iter = cfg.build();
            ElementwiseOp op;
            op.name = name;
            op.pass_index = true;
            op.in_ct = CT::F32;
            op.out_ct = out_ct;
            op.code = out_ct == CT::F32 ? code_f : code_i;
            op.helpers = kernels::math_wgsl() + philox_wgsl();
            auto word = [](uint64_t v, int hi)
            { return Scalar(static_cast<int64_t>(static_cast<int32_t>(static_cast<uint32_t>(hi ? v >> 32 : v)))); };
            op.scalars = {{word(seed, 0), CT::I32}, {word(seed, 1), CT::I32}, {word(offset, 0), CT::I32}, {word(offset, 1), CT::I32}};
            for (auto &s : extra_scalars)
                op.scalars.push_back(s);
            TORCH_CHECK(iter.can_use_32bit_indexing(), "torch_webgpu: random fill of > 2^32 elements is not supported");
            kernels::run_elementwise(iter, op);
            if (!out.is_same(self))
                self.copy_(out);
        }

#define RNG_ARGS "idx, bitcast<u32>(s0), bitcast<u32>(s1), bitcast<u32>(s2), bitcast<u32>(s3)"

        bool rng_dtype_ok(const Tensor &t) { return wgsl::is_compute_dtype(t.scalar_type()); }

        Tensor &uniform_fn(Tensor &self, double from, double to, std::optional<at::Generator> gen)
        {
            if (!rng_dtype_ok(self) || !self.is_floating_point())
                return WEBGPU_FALLBACK0(uniform_)(self, from, to, gen);
            TORCH_CHECK(from <= to, "uniform_ expects to return a [from, to) range, but found from=", from, " > to=", to);
            rng_fill(self, gen, "uniform", "s4 + (s5 - s4) * u01(rng_u32(" RNG_ARGS "))", "", {{from, CT::F32}, {to, CT::F32}},
                     CT::F32);
            return self;
        }

        Tensor &normal_fn(Tensor &self, double mean, double std, std::optional<at::Generator> gen)
        {
            if (!rng_dtype_ok(self) || !self.is_floating_point())
                return WEBGPU_FALLBACK0(normal_)(self, mean, std, gen);
            TORCH_CHECK(std >= 0.0, "normal expects std >= 0.0, but found std ", std);
            rng_fill(self, gen, "normal", "s4 + s5 * randn(" RNG_ARGS ")", "", {{mean, CT::F32}, {std, CT::F32}}, CT::F32);
            return self;
        }

        Tensor &log_normal_fn(Tensor &self, double mean, double std, std::optional<at::Generator> gen)
        {
            if (!rng_dtype_ok(self) || !self.is_floating_point())
                return WEBGPU_FALLBACK0(log_normal_)(self, mean, std, gen);
            TORCH_CHECK(std > 0.0, "log_normal_ expects std > 0.0, but found std=", std);
            rng_fill(self, gen, "log_normal", "exp(s4 + s5 * randn(" RNG_ARGS "))", "", {{mean, CT::F32}, {std, CT::F32}},
                     CT::F32);
            return self;
        }

        Tensor &exponential_fn(Tensor &self, double lambd, std::optional<at::Generator> gen)
        {
            if (!rng_dtype_ok(self) || !self.is_floating_point())
                return WEBGPU_FALLBACK0(exponential_)(self, lambd, gen);
            TORCH_CHECK(lambd > 0.0, "exponential_ expects lambda > 0.0, but found lambda=", lambd);
            rng_fill(self, gen, "exponential", "-log(u01_open0(rng_u32(" RNG_ARGS "))) / s4", "", {{lambd, CT::F32}}, CT::F32);
            return self;
        }

        Tensor &geometric_fn(Tensor &self, double p, std::optional<at::Generator> gen)
        {
            if (!rng_dtype_ok(self) || self.scalar_type() == at::kBool)
                return WEBGPU_FALLBACK0(geometric_)(self, p, gen);
            TORCH_CHECK(0 < p && p < 1, "geometric_ expects p to be in (0, 1), but got p=", p);
            const char *e = "ceil(log(u01_open0(rng_u32(" RNG_ARGS "))) / log(1.0 - s4))";
            rng_fill(self, gen, "geometric", e, std::string("i32(") + e + ")", {{p, CT::F32}},
                     self.is_floating_point() ? CT::F32 : CT::I32);
            return self;
        }

        Tensor &cauchy_fn(Tensor &self, double median, double sigma, std::optional<at::Generator> gen)
        {
            if (!rng_dtype_ok(self) || !self.is_floating_point())
                return WEBGPU_FALLBACK0(cauchy_)(self, median, sigma, gen);
            rng_fill(self, gen, "cauchy", "s4 + s5 * tan_f(3.141592653589793 * (u01(rng_u32(" RNG_ARGS ")) - 0.5))", "",
                     {{median, CT::F32}, {sigma, CT::F32}}, CT::F32);
            return self;
        }

        Tensor &bernoulli_float_fn(Tensor &self, double p, std::optional<at::Generator> gen)
        {
            if (!rng_dtype_ok(self))
                return WEBGPU_FALLBACK(bernoulli_, float)(self, p, gen);
            TORCH_CHECK(0 <= p && p <= 1, "bernoulli_ expects p to be in [0, 1], but got p=", p);
            CT ct = self.is_floating_point() ? CT::F32 : CT::I32;
            rng_fill(self, gen, "bernoulli", "select(0.0, 1.0, u01(rng_u32(" RNG_ARGS ")) < s4)",
                     "select(0, 1, u01(rng_u32(" RNG_ARGS ")) < s4)", {{p, CT::F32}}, ct);
            return self;
        }

        Tensor &bernoulli_tensor_fn(Tensor &self, const Tensor &p, std::optional<at::Generator> gen)
        {
            if (!rng_dtype_ok(self) || !gpu_dtypes_ok({p}))
                return WEBGPU_FALLBACK(bernoulli_, Tensor)(self, p, gen);
            Tensor prob = p.to(self.device(), at::kFloat).expand(self.sizes()).contiguous();
            CT ct = self.is_floating_point() ? CT::F32 : CT::I32;
            rng_fill(self, gen, "bernoulli_t", "select(0.0, 1.0, u01(rng_u32(" RNG_ARGS ")) < x0)",
                     "select(0, 1, u01(rng_u32(" RNG_ARGS ")) < x0)", {}, ct, prob);
            return self;
        }

        Tensor bernoulli_fn(const Tensor &self, std::optional<at::Generator> gen)
        {
            Tensor out = at::empty_like(self);
            return bernoulli_tensor_fn(out, self, gen);
        }

        // random_(from, to): integers in [from, to). Range must fit in 32 bits.
        Tensor &random_from_fn(Tensor &self, int64_t from, std::optional<int64_t> to, std::optional<at::Generator> gen)
        {
            if (!rng_dtype_ok(self))
                return WEBGPU_FALLBACK(random_, from)(self, from, to, gen);
            // Default upper bound: the dtype's range (floats: 2^mantissa+1), capped at
            // 2^31 - 1 because integer compute is 32-bit.
            int64_t hi = 0;
            if (to.has_value())
                hi = *to;
            else
            {
                switch (self.scalar_type())
                {
                case at::kFloat: hi = int64_t(1) << 24; break;
                case at::kHalf: hi = int64_t(1) << 11; break;
                case at::kBFloat16: hi = int64_t(1) << 8; break;
                case at::kBool: hi = 2; break;
                case at::kByte: hi = 256; break;
                case at::kChar: hi = 128; break;
                case at::kShort: hi = 32768; break;
                default: hi = std::numeric_limits<int32_t>::max(); break;
                }
            }
            TORCH_CHECK(from < hi, "random_ expects 'from' to be less than 'to', but got from=", from, " >= to=", hi);
            const int64_t range = hi - from;
            TORCH_CHECK(range <= (int64_t(1) << 32) && from >= std::numeric_limits<int32_t>::min() &&
                            from <= std::numeric_limits<int32_t>::max(),
                        "torch_webgpu: random_ ranges beyond 32 bits are not supported");
            CT ct = self.is_floating_point() ? CT::F32 : CT::I32;
            // s4 = from, s5 = range (as u32 bits; range == 2^32 encodes as 0)
            rng_fill(self, gen, "random",
                     "f32(s4) + f32(select(rng_u32(" RNG_ARGS ") % bitcast<u32>(s5), rng_u32(" RNG_ARGS "), bitcast<u32>(s5) == 0u))",
                     "s4 + bitcast<i32>(select(rng_u32(" RNG_ARGS ") % bitcast<u32>(s5), rng_u32(" RNG_ARGS "), bitcast<u32>(s5) == 0u))",
                     {{Scalar(from), CT::I32}, {Scalar(static_cast<int64_t>(static_cast<int32_t>(static_cast<uint32_t>(range)))), CT::I32}}, ct);
            return self;
        }
        Tensor &random_to_fn(Tensor &self, int64_t to, std::optional<at::Generator> gen)
        {
            return random_from_fn(self, 0, to, gen);
        }
        Tensor &random_fn(Tensor &self, std::optional<at::Generator> gen)
        {
            return random_from_fn(self, 0, std::nullopt, gen);
        }

        // native_dropout(input, p, train) -> (output, mask)
        std::tuple<Tensor, Tensor> native_dropout_fn(const Tensor &input, double p, std::optional<bool> train)
        {
            if (!gpu_dtypes_ok({input}))
                return WEBGPU_FALLBACK0(native_dropout)(input, p, train);
            if (input.numel() == 0 || !train.value_or(true) || p == 0)
                return {input.clone(), at::ones_like(input, input.options().dtype(at::kBool))};
            if (p == 1)
                return {at::zeros_like(input), at::zeros_like(input, input.options().dtype(at::kBool))};
            Tensor mask = at::empty_like(input, input.options().dtype(at::kBool), c10::MemoryFormat::Contiguous);
            bernoulli_float_fn(mask, 1.0 - p, std::nullopt);
            Tensor out = input * mask * (1.0 / (1.0 - p));
            return {out, mask};
        }

        Tensor native_dropout_backward_fn(const Tensor &grad, const Tensor &mask, double scale)
        {
            return grad * mask * scale;
        }
#undef RNG_ARGS
    }

    TORCH_LIBRARY_IMPL(aten, PrivateUse1, m)
    {
        m.impl("uniform_", TORCH_FN(uniform_fn));
        m.impl("normal_", TORCH_FN(normal_fn));
        m.impl("log_normal_", TORCH_FN(log_normal_fn));
        m.impl("exponential_", TORCH_FN(exponential_fn));
        m.impl("geometric_", TORCH_FN(geometric_fn));
        m.impl("cauchy_", TORCH_FN(cauchy_fn));
        m.impl("bernoulli_.float", TORCH_FN(bernoulli_float_fn));
        m.impl("bernoulli_.Tensor", TORCH_FN(bernoulli_tensor_fn));
        m.impl("bernoulli", TORCH_FN(bernoulli_fn));
        m.impl("random_.from", TORCH_FN(random_from_fn));
        m.impl("random_.to", TORCH_FN(random_to_fn));
        m.impl("random_", TORCH_FN(random_fn));
        m.impl("native_dropout", TORCH_FN(native_dropout_fn));
        m.impl("native_dropout_backward", TORCH_FN(native_dropout_backward_fn));
    }
}
