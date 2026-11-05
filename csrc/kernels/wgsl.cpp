#include "kernels/wgsl.h"

#include <c10/util/Exception.h>

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
        case ScalarType::Long:
        case ScalarType::Short:
        case ScalarType::Char:
        case ScalarType::Byte:
            return CT::I32;
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
fn f32_to_bf16_bits(v: f32) -> u32 {
    let u = bitcast<u32>(v);
    if ((u & 0x7f800000u) == 0x7f800000u && (u & 0x007fffffu) != 0u) {
        return ((u >> 16u) & 0x8000u) | 0x7fc0u;
    }
    return ((u + 0x7fffu + ((u >> 16u) & 1u)) >> 16u) & 0xffffu;
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
            body = "return bitcast<i32>(" + word(var, "i * 2u", atomic) + ");";
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
            body = full_store(var, "i * 2u", "bitcast<u32>(v)", false) +
                   full_store(var, "i * 2u + 1u", "select(0u, 0xffffffffu, v < 0)", false);
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
        switch (to)
        {
        case CT::F32:
            return from == CT::Bool ? "select(0.0, 1.0, " + expr + ")" : "f32(" + expr + ")";
        case CT::I32:
            return from == CT::Bool ? "select(0, 1, " + expr + ")" : "i32(" + expr + ")";
        case CT::Bool:
            return from == CT::F32 ? "((" + expr + ") != 0.0)" : "((" + expr + ") != 0)";
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
        }
        return w;
    }
}
