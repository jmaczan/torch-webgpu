#pragma once
#include <cstdint>

namespace torch_webgpu
{
    static inline uint32_t ceil_div_u32(uint32_t a, uint32_t b)
    {
        return (a + b - 1) / b;
        // explanation:
        // b / b is 1
        // b - 1 / b is 0
        // if a / b has a remainder, then the remainer is at least 1
        // if you add 1 to b - 1 / b, you get 1
        // so (a + b - 1) / b makes sure that whenever there is a
        // a = 14, b = 4, 14 / 4 = 3.x
        // (a + b - 1)/b = (14 + 4 - 1)/4 = 17/4 = 4
        // a = 13, b = 4, 13 / 4 = 3.x
        // (a + b - 1)/b = (13 + 4 - 1)/4 = 16/4 = 4
        // a = 4, b = 4, 4 / 4 = 1
        // (4 + 4 - 1)/4 = 7/4 = 1
        // a = 3, b = 4, 3 / 4 = 0
        // (a + b - 1) = (3 + 4 - 1) = 6
        // (a + b - 1)/b = 6/b = 6/4 = 1
        // a = 2, b = 4, 2 / 4 = 0
        // (a + b - 1) = 2 + 4 - 1 = 5
        // (a + b - 1)/b = 5/4 = 1
        // a = 1, b = 4, 1 / 0 = 0
        // (a + b - 1) = 1 + 4 - 1 = 4
        // (a + b - 1)/b = 4/4 = 1
        // a = 0, b = 4
        // (a + b - 1) = 0 + 1 - 1 = 3
        // (a + b - 1)/b = 3/4 = 0
        // so - 1 in formula is because we don't want to affect the output
        // when a == n * b, where n is int, then we don't want to affect the output
        // when a != 0, then we always 1 higher than regular division
    }
}
