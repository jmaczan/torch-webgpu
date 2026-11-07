#include "kernels/wgsl.h"

#include <c10/util/Exception.h>

#include <cstring>

namespace torch_webgpu::wgsl
{
    using c10::ScalarType;

    bool is_compute_dtype(ScalarType t)
    {
        switch (t)
        {
        case ScalarType::Float:
        case ScalarType::Half:
        case ScalarType::BFloat16:
        case ScalarType::Int:
        case ScalarType::Long:
        case ScalarType::Short:
        case ScalarType::Char:
        case ScalarType::Byte:
        case ScalarType::Bool:
            return true;
        default:
            return false;
        }
    }

    bool is_storage_dtype(ScalarType t)
    {
        if (is_compute_dtype(t))
            return true;
        size_t s = c10::elementSize(t);
        return s == 1 || s == 2 || s == 4 || s == 8 || s == 16;
    }

    CT compute_type(ScalarType t)
    {
        switch (t)
        {
        case ScalarType::Float:
        case ScalarType::Half:
        case ScalarType::BFloat16:
            return CT::F32;
        case ScalarType::Int:
        case ScalarType::Short:
        case ScalarType::Char:
        case ScalarType::Byte:
            return CT::I32;
        case ScalarType::Long:
            return CT::I64;
        case ScalarType::Bool:
            return CT::Bool;
        default:
            TORCH_CHECK(false, "torch_webgpu: dtype ", t, " is not supported by GPU kernels");
        }
    }

    const char *type_name(CT ct)
    {
        switch (ct)
        {
        case CT::F32:
            return "f32";
        case CT::I32:
            return "i32";
        case CT::Bool:
            return "bool";
        case CT::I64:
            return "i64";
        }
        return "?";
    }

    const std::string &prelude()
    {
        // Explicit handling of inf/nan/overflow: pack2x16float is
        // indeterminate for values outside the f16 range.
        static const std::string s = R"(
fn f32_to_f16_bits(v: f32) -> u32 {
    let u = bitcast<u32>(v);
    let sign = (u >> 16u) & 0x8000u;
    if ((u & 0x7f800000u) == 0x7f800000u) {
        if ((u & 0x007fffffu) != 0u) { return sign | 0x7e00u; }
        return sign | 0x7c00u;
    }
    if (abs(v) >= 65520.0) { return sign | 0x7c00u; }
    return pack2x16float(vec2<f32>(v, 0.0)) & 0xffffu;
}
// Round to nearest even, branch-free. Drivers may fold NaN tests on computed
// values away (fast math), so NaN must come out right without one: the
// magnitude saturates at 0x7fffffff, which only NaNs can reach, keeping them
// NaN (NVIDIA's canonical NaN 0x7fffffff would otherwise round to -0).
fn f32_to_bf16_bits(v: f32) -> u32 {
    let u = bitcast<u32>(v);
    let a = u & 0x7fffffffu;
    let r = min(a + 0x7fffu + ((a >> 16u) & 1u), 0x7fffffffu) >> 16u;
    return r | ((u >> 16u) & 0x8000u);
}

// ---- 64-bit integers: (lo, hi) two's complement --------------------------
alias i64 = vec2<u32>;
// (hi, lo) of the 64-bit product of two u32.
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
fn i64_from_i32(v: i32) -> i64 { return i64(bitcast<u32>(v), select(0u, 0xffffffffu, v < 0)); }
fn i64_from_u32(v: u32) -> i64 { return i64(v, 0u); }
fn i64_to_i32(a: i64) -> i32 { return bitcast<i32>(a.x); }
fn i64_is_neg(a: i64) -> bool { return (a.y >> 31u) == 1u; }
fn i64_add(a: i64, b: i64) -> i64 {
    let lo = a.x + b.x;
    return i64(lo, a.y + b.y + select(0u, 1u, lo < a.x));
}
fn i64_neg(a: i64) -> i64 { return i64_add(i64(~a.x, ~a.y), i64(1u, 0u)); }
fn i64_sub(a: i64, b: i64) -> i64 { return i64_add(a, i64_neg(b)); }
fn i64_mul(a: i64, b: i64) -> i64 {
    let p = mulhilo(a.x, b.x);
    return i64(p.y, p.x + a.x * b.y + a.y * b.x);
}
fn i64_eq(a: i64, b: i64) -> bool { return a.x == b.x && a.y == b.y; }
fn i64_lt(a: i64, b: i64) -> bool {
    let ah = bitcast<i32>(a.y);
    let bh = bitcast<i32>(b.y);
    return ah < bh || (ah == bh && a.x < b.x);
}
fn i64_le(a: i64, b: i64) -> bool { return !i64_lt(b, a); }
fn i64_gt(a: i64, b: i64) -> bool { return i64_lt(b, a); }
fn i64_ge(a: i64, b: i64) -> bool { return !i64_lt(a, b); }
fn i64_max(a: i64, b: i64) -> i64 { return select(a, b, i64_lt(a, b)); }
fn i64_min(a: i64, b: i64) -> i64 { return select(a, b, i64_lt(b, a)); }
fn i64_abs(a: i64) -> i64 { return select(a, i64_neg(a), i64_is_neg(a)); }
fn i64_sign(a: i64) -> i64 {
    if (i64_is_neg(a)) { return i64(0xffffffffu, 0xffffffffu); }
    return select(i64(1u, 0u), i64(0u, 0u), a.x == 0u && a.y == 0u);
}
fn i64_is_zero(a: i64) -> bool { return a.x == 0u && a.y == 0u; }
fn i64_shl(a: i64, n: u32) -> i64 {
    if (n >= 64u) { return i64(0u, 0u); }
    if (n >= 32u) { return i64(0u, a.x << (n - 32u)); }
    if (n == 0u) { return a; }
    return i64(a.x << n, (a.y << n) | (a.x >> (32u - n)));
}
fn i64_shr(a: i64, n: u32) -> i64 { // arithmetic
    let s = select(0u, 0xffffffffu, i64_is_neg(a));
    if (n >= 64u) { return i64(s, s); }
    if (n >= 32u) { return i64(bitcast<u32>(bitcast<i32>(a.y) >> (n - 32u)), s); }
    if (n == 0u) { return a; }
    return i64((a.x >> n) | (a.y << (32u - n)), bitcast<u32>(bitcast<i32>(a.y) >> n));
}
fn u64_lt(a: i64, b: i64) -> bool { return a.y < b.y || (a.y == b.y && a.x < b.x); }
// Unsigned long division: returns (q.lo, q.hi, r.lo, r.hi).
fn u64_divmod(n: i64, d: i64) -> vec4<u32> {
    var q = i64(0u, 0u);
    var r = i64(0u, 0u);
    for (var i = 63i; i >= 0i; i--) {
        r = i64_shl(r, 1u);
        var bit = 0u;
        if (i >= 32) { bit = (n.y >> u32(i - 32)) & 1u; } else { bit = (n.x >> u32(i)) & 1u; }
        r.x = r.x | bit;
        if (!u64_lt(r, d)) {
            r = i64_sub(r, d);
            if (i >= 32) { q.y = q.y | (1u << u32(i - 32)); } else { q.x = q.x | (1u << u32(i)); }
        }
    }
    return vec4<u32>(q, r);
}
// Truncating division / remainder (C semantics). Division by zero gives 0.
fn i64_div_trunc(a: i64, b: i64) -> i64 {
    if (i64_is_zero(b)) { return i64(0u, 0u); }
    let qr = u64_divmod(i64_abs(a), i64_abs(b));
    let q = i64(qr.x, qr.y);
    return select(q, i64_neg(q), i64_is_neg(a) != i64_is_neg(b));
}
fn i64_rem_trunc(a: i64, b: i64) -> i64 {
    if (i64_is_zero(b)) { return i64(0u, 0u); }
    let qr = u64_divmod(i64_abs(a), i64_abs(b));
    let r = i64(qr.z, qr.w);
    return select(r, i64_neg(r), i64_is_neg(a));
}
fn i64_div_floor(a: i64, b: i64) -> i64 {
    let q = i64_div_trunc(a, b);
    let r = i64_rem_trunc(a, b);
    if (!i64_is_zero(r) && (i64_is_neg(r) != i64_is_neg(b))) { return i64_sub(q, i64(1u, 0u)); }
    return q;
}
fn i64_remainder(a: i64, b: i64) -> i64 {
    let r = i64_rem_trunc(a, b);
    if (!i64_is_zero(r) && (i64_is_neg(r) != i64_is_neg(b))) { return i64_add(r, b); }
    return r;
}
fn i64_pow(a: i64, e: i64) -> i64 {
    if (i64_is_neg(e)) {
        if (i64_eq(a, i64(1u, 0u))) { return a; }
        if (i64_eq(a, i64(0xffffffffu, 0xffffffffu))) { return select(i64(1u, 0u), a, (e.x & 1u) == 1u); }
        return i64(0u, 0u);
    }
    var r = i64(1u, 0u);
    var b = a;
    var n = e;
    loop {
        if ((n.x & 1u) == 1u) { r = i64_mul(r, b); }
        n = i64_shr(n, 1u);
        if (i64_is_zero(n)) { break; }
        b = i64_mul(b, b);
    }
    return r;
}
fn i64_gcd(x: i64, y: i64) -> i64 {
    var a = i64_abs(x);
    var b = i64_abs(y);
    while (!i64_is_zero(b)) { let t = i64_rem_trunc(a, b); a = b; b = t; }
    return a;
}
fn i64_to_f32(a: i64) -> f32 {
    let neg = i64_is_neg(a);
    let m = select(a, i64_neg(a), neg);
    let f = f32(m.y) * 4294967296.0 + f32(m.x);
    return select(f, -f, neg);
}
fn i64_from_f32(v: f32) -> i64 {
    let t = trunc(v);
    let a = abs(t);
    if (a >= 9.2233720368547758e18) {
        return select(i64(0xffffffffu, 0x7fffffffu), i64(0u, 0x80000000u), t < 0.0 || t != t);
    }
    let hi = floor(a / 4294967296.0);
    let lo = a - hi * 4294967296.0;
    let r = i64(u32(lo), u32(hi));
    return select(r, i64_neg(r), t < 0.0);
}
)";
        return s;
    }

    namespace
    {
        std::string word(const std::string &var, const std::string &idx, bool atomic)
        {
            if (atomic)
                return "atomicLoad(&" + var + "[" + idx + "])";
            return var + "[" + idx + "]";
        }

        // Writes `bits` (already masked) into a sub-word lane of `var`.
        std::string subword_store(const std::string &var, size_t elem_size, const std::string &bits)
        {
            const char *shift = elem_size == 1 ? "2u" : "1u";
            const char *lane = elem_size == 1 ? "3u" : "1u";
            const char *width = elem_size == 1 ? "8u" : "16u";
            const char *mask = elem_size == 1 ? "0xffu" : "0xffffu";
            return std::string("    let w = i >> ") + shift + ";\n" +
                   "    let sh = (i & " + lane + ") * " + width + ";\n" +
                   "    let b = (" + bits + ") & " + mask + ";\n" +
                   "    atomicAnd(&" + var + "[w], ~(" + mask + " << sh));\n" +
                   "    atomicOr(&" + var + "[w], b << sh);\n";
        }

        std::string full_store(const std::string &var, const std::string &idx, const std::string &val, bool atomic)
        {
            if (atomic)
                return "    atomicStore(&" + var + "[" + idx + "], " + val + ");\n";
            return "    " + var + "[" + idx + "] = " + val + ";\n";
        }
    }

    std::string load_fn(const std::string &fname, const std::string &var, ScalarType t, bool atomic)
    {
        std::string ret = type_name(compute_type(t));
        std::string body;
        switch (t)
        {
        case ScalarType::Float:
            body = "return bitcast<f32>(" + word(var, "i", atomic) + ");";
            break;
        case ScalarType::Half:
            body = "return unpack2x16float(" + word(var, "i >> 1u", atomic) + ")[i & 1u];";
            break;
        case ScalarType::BFloat16:
            body = "return bitcast<f32>(((" + word(var, "i >> 1u", atomic) + " >> ((i & 1u) * 16u)) & 0xffffu) << 16u);";
            break;
        case ScalarType::Int:
            body = "return bitcast<i32>(" + word(var, "i", atomic) + ");";
            break;
        case ScalarType::Long:
            body = "return i64(" + word(var, "i * 2u", atomic) + ", " + word(var, "i * 2u + 1u", atomic) + ");";
            break;
        case ScalarType::Short:
            body = "return bitcast<i32>(((" + word(var, "i >> 1u", atomic) + " >> ((i & 1u) * 16u)) & 0xffffu) << 16u) >> 16u;";
            break;
        case ScalarType::Char:
            body = "return bitcast<i32>(((" + word(var, "i >> 2u", atomic) + " >> ((i & 3u) * 8u)) & 0xffu) << 24u) >> 24u;";
            break;
        case ScalarType::Byte:
            body = "return i32((" + word(var, "i >> 2u", atomic) + " >> ((i & 3u) * 8u)) & 0xffu);";
            break;
        case ScalarType::Bool:
            body = "return ((" + word(var, "i >> 2u", atomic) + " >> ((i & 3u) * 8u)) & 0xffu) != 0u;";
            break;
        default:
            TORCH_CHECK(false, "torch_webgpu: cannot load dtype ", t);
        }
        return "fn " + fname + "(i: u32) -> " + ret + " {\n    " + body + "\n}\n";
    }

    std::string store_fn(const std::string &fname, const std::string &var, ScalarType t)
    {
        std::string arg = type_name(compute_type(t));
        std::string body;
        switch (t)
        {
        case ScalarType::Float:
            body = full_store(var, "i", "bitcast<u32>(v)", false);
            break;
        case ScalarType::Int:
            body = full_store(var, "i", "bitcast<u32>(v)", false);
            break;
        case ScalarType::Long:
            body = full_store(var, "i * 2u", "v.x", false) + full_store(var, "i * 2u + 1u", "v.y", false);
            break;
        case ScalarType::Half:
            body = subword_store(var, 2, "f32_to_f16_bits(v)");
            break;
        case ScalarType::BFloat16:
            body = subword_store(var, 2, "f32_to_bf16_bits(v)");
            break;
        case ScalarType::Short:
            body = subword_store(var, 2, "bitcast<u32>(v)");
            break;
        case ScalarType::Char:
        case ScalarType::Byte:
            body = subword_store(var, 1, "bitcast<u32>(v)");
            break;
        case ScalarType::Bool:
            body = subword_store(var, 1, "select(0u, 1u, v)");
            break;
        default:
            TORCH_CHECK(false, "torch_webgpu: cannot store dtype ", t);
        }
        return "fn " + fname + "(i: u32, v: " + arg + ") {\n" + body + "}\n";
    }

    const char *raw_type(size_t elem_size)
    {
        switch (elem_size)
        {
        case 1:
        case 2:
        case 4:
            return "u32";
        case 8:
            return "vec2<u32>";
        case 16:
            return "vec4<u32>";
        default:
            TORCH_CHECK(false, "torch_webgpu: unsupported element size ", elem_size);
        }
    }

    std::string raw_load_fn(const std::string &fname, const std::string &var, size_t elem_size, bool atomic)
    {
        std::string body;
        switch (elem_size)
        {
        case 1:
            body = "return (" + word(var, "i >> 2u", atomic) + " >> ((i & 3u) * 8u)) & 0xffu;";
            break;
        case 2:
            body = "return (" + word(var, "i >> 1u", atomic) + " >> ((i & 1u) * 16u)) & 0xffffu;";
            break;
        case 4:
            body = "return " + word(var, "i", atomic) + ";";
            break;
        case 8:
            body = "return vec2<u32>(" + word(var, "i * 2u", atomic) + ", " + word(var, "i * 2u + 1u", atomic) + ");";
            break;
        case 16:
            body = "return vec4<u32>(" + word(var, "i * 4u", atomic) + ", " + word(var, "i * 4u + 1u", atomic) + ", " +
                   word(var, "i * 4u + 2u", atomic) + ", " + word(var, "i * 4u + 3u", atomic) + ");";
            break;
        default:
            TORCH_CHECK(false, "torch_webgpu: unsupported element size ", elem_size);
        }
        return "fn " + fname + "(i: u32) -> " + raw_type(elem_size) + " {\n    " + body + "\n}\n";
    }

    std::string raw_store_fn(const std::string &fname, const std::string &var, size_t elem_size)
    {
        std::string body;
        switch (elem_size)
        {
        case 1:
        case 2:
            body = subword_store(var, elem_size, "v");
            break;
        case 4:
            body = full_store(var, "i", "v", false);
            break;
        case 8:
            body = full_store(var, "i * 2u", "v.x", false) + full_store(var, "i * 2u + 1u", "v.y", false);
            break;
        case 16:
            body = full_store(var, "i * 4u", "v.x", false) + full_store(var, "i * 4u + 1u", "v.y", false) +
                   full_store(var, "i * 4u + 2u", "v.z", false) + full_store(var, "i * 4u + 3u", "v.w", false);
            break;
        default:
            TORCH_CHECK(false, "torch_webgpu: unsupported element size ", elem_size);
        }
        return "fn " + fname + "(i: u32, v: " + raw_type(elem_size) + ") {\n" + body + "}\n";
    }

    std::string convert(const std::string &expr, CT from, CT to)
    {
        if (from == to)
            return expr;
        if (from == CT::I64)
        {
            switch (to)
            {
            case CT::F32:
                return "i64_to_f32(" + expr + ")";
            case CT::I32:
                return "i64_to_i32(" + expr + ")";
            case CT::Bool:
                return "(!i64_is_zero(" + expr + "))";
            default:
                break;
            }
        }
        switch (to)
        {
        case CT::F32:
            return from == CT::Bool ? "select(0.0, 1.0, " + expr + ")" : "f32(" + expr + ")";
        case CT::I32:
            return from == CT::Bool ? "select(0, 1, " + expr + ")" : "i32(" + expr + ")";
        case CT::Bool:
            // Float -> bool tests the magnitude bits: drivers may compile `x != 0.0`
            // as an ordered comparison, making NaN false (it must be true).
            return from == CT::F32 ? "((bitcast<u32>(" + expr + ") & 0x7fffffffu) != 0u)" : "((" + expr + ") != 0)";
        case CT::I64:
            if (from == CT::F32)
                return "i64_from_f32(" + expr + ")";
            if (from == CT::Bool)
                return "select(i64(0u, 0u), i64(1u, 0u), " + expr + ")";
            return "i64_from_i32(" + expr + ")";
        }
        return expr;
    }

    std::string scalar_from_word(const std::string &w, CT ct)
    {
        switch (ct)
        {
        case CT::F32:
            return "bitcast<f32>(" + w + ")";
        case CT::I32:
            return "bitcast<i32>(" + w + ")";
        case CT::Bool:
            return "(" + w + " != 0u)";
        case CT::I64:
            TORCH_CHECK(false, "scalar_from_word: use scalar_slot_expr for int64");
        }
        return w;
    }

    void encode_scalar(const c10::Scalar &s, CT ct, uint32_t out[2])
    {
        out[0] = out[1] = 0;
        switch (ct)
        {
        case CT::F32:
        {
            float f = s.isComplex() ? static_cast<float>(s.toComplexDouble().real())
                      : s.isBoolean() ? (s.toBool() ? 1.0f : 0.0f)
                                      : static_cast<float>(s.toDouble());
            std::memcpy(&out[0], &f, 4);
            break;
        }
        case CT::I32:
        case CT::I64:
        {
            int64_t v = s.isFloatingPoint() ? static_cast<int64_t>(s.toDouble())
                        : s.isBoolean()     ? (s.toBool() ? 1 : 0)
                        : s.isComplex()     ? static_cast<int64_t>(s.toComplexDouble().real())
                                            : s.toLong();
            uint64_t u = static_cast<uint64_t>(v);
            out[0] = static_cast<uint32_t>(u);
            out[1] = ct == CT::I64 ? static_cast<uint32_t>(u >> 32) : 0u;
            break;
        }
        case CT::Bool:
            out[0] = s.isFloatingPoint() ? (s.toDouble() != 0.0)
                     : s.isBoolean()     ? s.toBool()
                     : s.isComplex()     ? (s.toComplexDouble() != c10::complex<double>(0, 0))
                                         : (s.toLong() != 0);
            break;
        }
    }

    std::string scalar_slot_expr(int slot, CT ct, const std::string &array)
    {
        auto word = [&](int w)
        { return array + "[" + std::to_string(w / 4) + "][" + std::to_string(w % 4) + "]"; };
        if (ct == CT::I64)
            return "i64(" + word(2 * slot) + ", " + word(2 * slot + 1) + ")";
        return scalar_from_word(word(2 * slot), ct);
    }

    std::string typed_ops(CT ct)
    {
        const char *T = type_name(ct);
        std::string t(T);
        auto fn = [&](const std::string &name, const std::string &args, const std::string &ret, const std::string &body)
        { return "fn " + name + "(" + args + ") -> " + ret + " { return " + body + "; }\n"; };
        std::string ab = "a: " + t + ", b: " + t;
        std::string s;
        switch (ct)
        {
        case CT::F32:
            s += fn("t_zero", "", t, "0.0") + fn("t_one", "", t, "1.0");
            s += fn("t_add", ab, t, "a + b") + fn("t_sub", ab, t, "a - b") + fn("t_mul", ab, t, "a * b");
            s += fn("t_max", ab, t, "max_nan(a, b)") + fn("t_min", ab, t, "min_nan(a, b)");
            s += fn("t_lt", ab, "bool", "a < b") + fn("t_eq", ab, "bool", "a == b");
            break;
        case CT::I32:
            s += fn("t_zero", "", t, "0") + fn("t_one", "", t, "1");
            s += fn("t_add", ab, t, "a + b") + fn("t_sub", ab, t, "a - b") + fn("t_mul", ab, t, "a * b");
            s += fn("t_max", ab, t, "max(a, b)") + fn("t_min", ab, t, "min(a, b)");
            s += fn("t_lt", ab, "bool", "a < b") + fn("t_eq", ab, "bool", "a == b");
            break;
        case CT::I64:
            s += fn("t_zero", "", t, "i64(0u, 0u)") + fn("t_one", "", t, "i64(1u, 0u)");
            s += fn("t_add", ab, t, "i64_add(a, b)") + fn("t_sub", ab, t, "i64_sub(a, b)") + fn("t_mul", ab, t, "i64_mul(a, b)");
            s += fn("t_max", ab, t, "i64_max(a, b)") + fn("t_min", ab, t, "i64_min(a, b)");
            s += fn("t_lt", ab, "bool", "i64_lt(a, b)") + fn("t_eq", ab, "bool", "i64_eq(a, b)");
            break;
        case CT::Bool:
            s += fn("t_zero", "", t, "false") + fn("t_one", "", t, "true");
            s += fn("t_add", ab, t, "a || b") + fn("t_mul", ab, t, "a && b");
            s += fn("t_max", ab, t, "a || b") + fn("t_min", ab, t, "a && b");
            s += fn("t_lt", ab, "bool", "!a && b") + fn("t_eq", ab, "bool", "a == b");
            break;
        }
        return s;
    }
}
