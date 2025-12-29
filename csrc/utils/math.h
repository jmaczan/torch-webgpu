#pragma once
#include <cstdint>
namespace torch_webgpu
{
    static inline uint32_t ceil_div_u32(uint32_t a, uint32_t b)
    {
        return (a + b - 1) / b;
    }
}
