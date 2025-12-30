#include <gtest/gtest.h>
#include "utils/math.h"

TEST(MathUtils, CeilDivU32)
{
    using torch_webgpu::ceil_div_u32;
    EXPECT_EQ(ceil_div_u32(1, 1), 1u);
    EXPECT_EQ(ceil_div_u32(7, 4), 2u);
    EXPECT_EQ(ceil_div_u32(8, 4), 2u);
    EXPECT_EQ(ceil_div_u32(9, 4), 3u);
    EXPECT_EQ(ceil_div_u32(0, 3), 0u);
}
