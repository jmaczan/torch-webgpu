#pragma once
// WGSL math library shared by pointwise kernels.
//
// WGSL's built-ins leave NaN/inf/overflow behaviour of several functions
// (min/max/clamp with NaN, tanh for large inputs, pow with negative bases)
// implementation-defined and lacks erf, erfinv, expm1 and log1p entirely.
// These helpers give PyTorch-compatible semantics with f32 accuracy.

#include <string>

namespace torch_webgpu::kernels
{
    inline const std::string &math_wgsl()
    {
        static const std::string s = R"(
fn isnan_f(x: f32) -> bool { return (bitcast<u32>(x) & 0x7fffffffu) > 0x7f800000u; }
fn isinf_f(x: f32) -> bool { return (bitcast<u32>(x) & 0x7fffffffu) == 0x7f800000u; }
fn isfinite_f(x: f32) -> bool { return (bitcast<u32>(x) & 0x7f800000u) != 0x7f800000u; }
// NaN/Inf are not representable in WGSL constant expressions; going through
// private variables makes them runtime values.
var<private> NAN_BITS: u32 = 0x7fc00000u;
var<private> INF_BITS: u32 = 0x7f800000u;
fn nan_f() -> f32 { return bitcast<f32>(NAN_BITS); }
fn inf_f() -> f32 { return bitcast<f32>(INF_BITS); }
fn signbit_f(x: f32) -> bool { return (bitcast<u32>(x) >> 31u) == 1u; }

// NaN-propagating min/max (torch.maximum / torch.minimum / clamp semantics).
fn max_nan(a: f32, b: f32) -> f32 { if (isnan_f(a) || isnan_f(b)) { return nan_f(); } return max(a, b); }
fn min_nan(a: f32, b: f32) -> f32 { if (isnan_f(a) || isnan_f(b)) { return nan_f(); } return min(a, b); }
// NaN-ignoring min/max (torch.fmax / torch.fmin).
fn fmax_f(a: f32, b: f32) -> f32 { if (isnan_f(a)) { return b; } if (isnan_f(b)) { return a; } return max(a, b); }
fn fmin_f(a: f32, b: f32) -> f32 { if (isnan_f(a)) { return b; } if (isnan_f(b)) { return a; } return min(a, b); }

fn expm1_f(x: f32) -> f32 {
    if (abs(x) < 1e-5) { return x + 0.5 * x * x; }
    let u = exp(x);
    if (u == 1.0) { return x; }
    if (isinf_f(u)) { return u; }
    let um1 = u - 1.0;
    if (um1 == -1.0) { return -1.0; }
    return um1 * x / log(u);
}

fn log1p_f(x: f32) -> f32 {
    if (isnan_f(x)) { return x; }
    if (x == -1.0) { return -inf_f(); }
    if (x < -1.0) { return nan_f(); }
    let u = 1.0 + x;
    if (u == 1.0) { return x; }
    if (isinf_f(u)) { return u; }
    return log(u) * x / (u - 1.0);
}

fn tanh_f(x: f32) -> f32 {
    let ax = abs(x);
    if (ax > 9.0) { return select(1.0, -1.0, x < 0.0); }
    if (ax < 1e-4) { return x; }
    let e = expm1_f(-2.0 * ax);
    let t = -e / (e + 2.0);
    return select(t, -t, x < 0.0);
}

fn sigmoid_f(x: f32) -> f32 { return 1.0 / (1.0 + exp(-x)); }

// erfc for z >= 0 (Numerical Recipes erfcc, fractional error < 1.2e-7).
fn erfc_pos(z: f32) -> f32 {
    let t = 1.0 / (1.0 + 0.5 * z);
    let p = -z * z - 1.26551223 + t * (1.00002368 + t * (0.37409196 + t * (0.09678418 + t * (-0.18628806 +
            t * (0.27886807 + t * (-1.13520398 + t * (1.48851587 + t * (-0.82215223 + t * 0.17087277))))))));
    return t * exp(p);
}

fn erf_f(x: f32) -> f32 {
    let ax = abs(x);
    if (ax < 0.5) {
        let x2 = x * x;
        return x * (1.1283791671 + x2 * (-0.3761263890 + x2 * (0.1128379167 + x2 * (-0.02686617064 +
               x2 * (0.005223977625 + x2 * (-0.0008548327023))))));
    }
    if (ax > 10.0) { return select(1.0, -1.0, x < 0.0); }
    let r = 1.0 - erfc_pos(ax);
    return select(r, -r, x < 0.0);
}

fn erfc_f(x: f32) -> f32 {
    if (abs(x) < 0.5) { return 1.0 - erf_f(x); }
    if (x >= 0.0) { return erfc_pos(x); }
    return 2.0 - erfc_pos(-x);
}

// Giles, "Approximating the erfinv function" (single precision).
fn erfinv_f(x: f32) -> f32 {
    let ax = abs(x);
    if (ax > 1.0 || isnan_f(x)) { return nan_f(); }
    if (ax == 1.0) { return select(inf_f(), -inf_f(), x < 0.0); }
    var w = -log((1.0 - x) * (1.0 + x));
    var p: f32;
    if (w < 5.0) {
        w = w - 2.5;
        p = 2.81022636e-08;
        p = 3.43273939e-07 + p * w;
        p = -3.5233877e-06 + p * w;
        p = -4.39150654e-06 + p * w;
        p = 0.00021858087 + p * w;
        p = -0.00125372503 + p * w;
        p = -0.00417768164 + p * w;
        p = 0.246640727 + p * w;
        p = 1.50140941 + p * w;
    } else {
        w = sqrt(w) - 3.0;
        p = -0.000200214257;
        p = 0.000100950558 + p * w;
        p = 0.00134934322 + p * w;
        p = -0.00367342844 + p * w;
        p = 0.00573950773 + p * w;
        p = -0.0076224613 + p * w;
        p = 0.00943887047 + p * w;
        p = 1.00167406 + p * w;
        p = 2.83297682 + p * w;
    }
    return p * x;
}

fn pow_abs(a: f32, b: f32) -> f32 {
    if (a == 0.0) { return select(0.0, inf_f(), b < 0.0); }
    if (isinf_f(a)) { return select(inf_f(), 0.0, b < 0.0); }
    if (isinf_f(b)) {
        if (a == 1.0) { return 1.0; }
        return select(select(0.0, inf_f(), a > 1.0), select(inf_f(), 0.0, a > 1.0), b < 0.0);
    }
    // Exact-ish small integer exponents via repeated squaring.
    if (b == trunc(b) && abs(b) <= 64.0) {
        var e = u32(abs(b));
        var base = a;
        var r = 1.0;
        loop {
            if ((e & 1u) == 1u) { r = r * base; }
            e = e >> 1u;
            if (e == 0u) { break; }
            base = base * base;
        }
        return select(r, 1.0 / r, b < 0.0);
    }
    return exp2(b * log2(a));
}

fn pow_f(a: f32, b: f32) -> f32 {
    if (b == 0.0 || a == 1.0) { return 1.0; }
    if (isnan_f(a) || isnan_f(b)) { return nan_f(); }
    if (signbit_f(a)) {
        let ab = abs(a);
        if (isinf_f(b)) { return pow_abs(ab, b); }
        if (b != trunc(b)) { return nan_f(); }
        let r = pow_abs(ab, b);
        let odd = abs(b) < 16777216.0 && (u32(abs(b)) & 1u) == 1u;
        return select(r, -r, odd);
    }
    return pow_abs(a, b);
}

fn ipow(a: i32, b: i32) -> i32 {
    if (b < 0) {
        if (a == 1) { return 1; }
        if (a == -1) { return select(1, -1, (b & 1) == 1); }
        return 0;
    }
    var e = u32(b);
    var base = a;
    var r = 1;
    loop {
        if ((e & 1u) == 1u) { r = r * base; }
        e = e >> 1u;
        if (e == 0u) { break; }
        base = base * base;
    }
    return r;
}

fn gcd_i(x: i32, y: i32) -> i32 {
    var a = abs(x);
    var b = abs(y);
    while (b != 0) { let t = a % b; a = b; b = t; }
    return a;
}

// Python-style modulo (sign of divisor), as torch.remainder.
fn remainder_f(a: f32, b: f32) -> f32 {
    var m = a - b * trunc(a / b);
    if (m != 0.0 && ((b < 0.0) != (m < 0.0))) { m = m + b; }
    return m;
}
fn remainder_i(a: i32, b: i32) -> i32 {
    if (b == 0) { return 0; }
    var m = a % b;
    if (m != 0 && ((b < 0) != (m < 0))) { m = m + b; }
    return m;
}
fn fmod_f(a: f32, b: f32) -> f32 { return a - b * trunc(a / b); }

// torch.div(..., rounding_mode='floor'), mirroring c10::div_floor_floating.
fn div_floor_f(a: f32, b: f32) -> f32 {
    if (b == 0.0) { return a / b; }
    let m = fmod_f(a, b);
    var d = (a - m) / b;
    if (m != 0.0 && ((b < 0.0) != (m < 0.0))) { d = d - 1.0; }
    var fd: f32;
    if (d != 0.0) {
        fd = floor(d);
        if (d - fd > 0.5) { fd = fd + 1.0; }
    } else {
        fd = select(0.0, -0.0, (a / b) < 0.0);
    }
    return fd;
}
fn div_floor_i(a: i32, b: i32) -> i32 {
    if (b == 0) { return 0; }
    var q = a / b;
    if ((a % b != 0) && ((a < 0) != (b < 0))) { q = q - 1; }
    return q;
}

fn copysign_f(a: f32, b: f32) -> f32 {
    return bitcast<f32>((bitcast<u32>(a) & 0x7fffffffu) | (bitcast<u32>(b) & 0x80000000u));
}

fn hypot_f(a: f32, b: f32) -> f32 {
    let x = abs(a);
    let y = abs(b);
    if (isinf_f(x) || isinf_f(y)) { return inf_f(); }
    let m = max(x, y);
    if (m == 0.0) { return 0.0; }
    let n = min(x, y) / m;
    return m * sqrt(1.0 + n * n);
}

fn logaddexp_f(a: f32, b: f32) -> f32 {
    if (isinf_f(a) && a == b) { return a; }
    let m = max_nan(a, b);
    return m + log1p_f(exp(-abs(a - b)));
}
fn logaddexp2_f(a: f32, b: f32) -> f32 {
    if (isinf_f(a) && a == b) { return a; }
    let m = max_nan(a, b);
    return m + log1p_f(exp2(-abs(a - b))) * 1.4426950408889634;
}

fn round_half_even(x: f32) -> f32 { return round(x); }

fn gelu_erf(x: f32) -> f32 { return 0.5 * x * (1.0 + erf_f(x * 0.7071067811865476)); }
fn gelu_tanh(x: f32) -> f32 {
    return 0.5 * x * (1.0 + tanh_f(0.7978845608028654 * (x + 0.044715 * x * x * x)));
}
)";
        return s;
    }
}
