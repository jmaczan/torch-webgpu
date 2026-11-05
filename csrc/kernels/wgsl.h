#pragma once
// WGSL code generation helpers for PyTorch dtypes.
//
// Storage model: every tensor buffer is bound as `array<u32>` (or
// `array<atomic<u32>>` when a kernel writes sub-word elements). Elements of
// any dtype are addressed by *element index* and unpacked from 32-bit words:
//
//   dtype           storage     compute type (CT)
//   float32         1 word      f32
//   float16         1/2 word    f32   (unpack2x16float / pack2x16float)
//   bfloat16        1/2 word    f32   (bit shifts, round-to-nearest-even)
//   int32           1 word      i32
//   int64           2 words     i64   (emulated: vec2<u32> two's complement)
//   int16           1/2 word    i32
//   int8 / uint8    1/4 word    i32
//   bool            1/4 word    bool
//
// Everything else (float64, complex, uint16/32/64, float8) can be stored and
// copied bit-exactly (see raw_* helpers) but is computed on the CPU fallback.
//
// int64 is exact: kernels that have no 64-bit implementation fall back to
// the CPU rather than truncating.

#include <c10/core/Scalar.h>
#include <c10/core/ScalarType.h>

#include <string>

namespace torch_webgpu::wgsl
{
    enum class CT
    {
        F32,
        I32,
        Bool,
        I64,
    };

    // True if kernels can compute on this dtype on the GPU.
    bool is_compute_dtype(c10::ScalarType t);
    // True if the dtype can at least be stored and copied bit-exactly.
    bool is_storage_dtype(c10::ScalarType t);

    CT compute_type(c10::ScalarType t);
    const char *type_name(CT ct);

    // True if a store of this dtype touches only part of a 32-bit word.
    inline bool is_subword(c10::ScalarType t) { return c10::elementSize(t) < 4; }

    // Shared helper functions (f16/bf16 conversion). Include once per shader.
    const std::string &prelude();

    // `fn <fname>(i: u32) -> CT` reading element i of dtype `t` from `var`.
    std::string load_fn(const std::string &fname, const std::string &var, c10::ScalarType t, bool atomic);
    // `fn <fname>(i: u32, v: CT)` writing element i. Sub-word dtypes require
    // `var` to be declared as array<atomic<u32>>.
    std::string store_fn(const std::string &fname, const std::string &var, c10::ScalarType t);

    // Raw bit copies: type is u32 (elem size <= 4), vec2<u32> (8), vec4<u32> (16).
    const char *raw_type(size_t elem_size);
    std::string raw_load_fn(const std::string &fname, const std::string &var, size_t elem_size, bool atomic);
    std::string raw_store_fn(const std::string &fname, const std::string &var, size_t elem_size);

    // WGSL expression converting `expr` from compute type `from` to `to`.
    std::string convert(const std::string &expr, CT from, CT to);

    // A literal / bitcast expression for a scalar stored as 32-bit word `word_expr`.
    std::string scalar_from_word(const std::string &word_expr, CT ct);

    // Scalars occupy two 32-bit words each (slot k = words 2k, 2k+1) so that
    // 64-bit values fit. `array` is a WGSL array<vec4<u32>, N> expression.
    void encode_scalar(const c10::Scalar &s, CT ct, uint32_t out[2]);
    std::string scalar_slot_expr(int slot, CT ct, const std::string &array = "P.scalars");

    // WGSL helpers for generic code over a compute type: t_zero(), t_one(),
    // t_add, t_sub, t_mul, t_max, t_min (NaN-propagating for f32), t_lt, t_eq.
    std::string typed_ops(CT ct);
}
