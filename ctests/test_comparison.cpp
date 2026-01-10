#include <gtest/gtest.h>
#include <torch/torch.h>

namespace
{

    torch::Device webgpu_device()
    {
        return torch::Device(torch::DeviceType::PrivateUse1);
    }

    torch::Tensor to_webgpu(const torch::Tensor &cpu)
    {
        return cpu.to(webgpu_device());
    }

} // namespace

// Equal tests
TEST(ComparisonOps, EqTensor)
{
    auto a_cpu = torch::tensor({1.0f, 2.0f, 3.0f, 4.0f});
    auto b_cpu = torch::tensor({1.0f, 5.0f, 3.0f, 6.0f});
    auto a = to_webgpu(a_cpu);
    auto b = to_webgpu(b_cpu);

    auto out = torch::eq(a, b);
    auto expected = torch::eq(a_cpu, b_cpu);
    ASSERT_TRUE(torch::equal(out.to(torch::kCPU), expected));
}

TEST(ComparisonOps, EqScalar)
{
    auto a_cpu = torch::tensor({1.0f, 2.0f, 3.0f, 2.0f});
    auto a = to_webgpu(a_cpu);

    auto out = torch::eq(a, 2.0f);
    auto expected = torch::eq(a_cpu, 2.0f);
    ASSERT_TRUE(torch::equal(out.to(torch::kCPU), expected));
}

// Not equal tests
TEST(ComparisonOps, NeTensor)
{
    auto a_cpu = torch::tensor({1.0f, 2.0f, 3.0f, 4.0f});
    auto b_cpu = torch::tensor({1.0f, 5.0f, 3.0f, 6.0f});
    auto a = to_webgpu(a_cpu);
    auto b = to_webgpu(b_cpu);

    auto out = torch::ne(a, b);
    auto expected = torch::ne(a_cpu, b_cpu);
    ASSERT_TRUE(torch::equal(out.to(torch::kCPU), expected));
}

TEST(ComparisonOps, NeScalar)
{
    auto a_cpu = torch::tensor({1.0f, 2.0f, 3.0f, 2.0f});
    auto a = to_webgpu(a_cpu);

    auto out = torch::ne(a, 2.0f);
    auto expected = torch::ne(a_cpu, 2.0f);
    ASSERT_TRUE(torch::equal(out.to(torch::kCPU), expected));
}

// Less than tests
TEST(ComparisonOps, LtTensor)
{
    auto a_cpu = torch::tensor({1.0f, 5.0f, 3.0f, 4.0f});
    auto b_cpu = torch::tensor({2.0f, 3.0f, 3.0f, 6.0f});
    auto a = to_webgpu(a_cpu);
    auto b = to_webgpu(b_cpu);

    auto out = torch::lt(a, b);
    auto expected = torch::lt(a_cpu, b_cpu);
    ASSERT_TRUE(torch::equal(out.to(torch::kCPU), expected));
}

TEST(ComparisonOps, LtScalar)
{
    auto a_cpu = torch::tensor({1.0f, 2.0f, 3.0f, 4.0f});
    auto a = to_webgpu(a_cpu);

    auto out = torch::lt(a, 3.0f);
    auto expected = torch::lt(a_cpu, 3.0f);
    ASSERT_TRUE(torch::equal(out.to(torch::kCPU), expected));
}

// Less than or equal tests
TEST(ComparisonOps, LeTensor)
{
    auto a_cpu = torch::tensor({1.0f, 5.0f, 3.0f, 4.0f});
    auto b_cpu = torch::tensor({2.0f, 3.0f, 3.0f, 6.0f});
    auto a = to_webgpu(a_cpu);
    auto b = to_webgpu(b_cpu);

    auto out = torch::le(a, b);
    auto expected = torch::le(a_cpu, b_cpu);
    ASSERT_TRUE(torch::equal(out.to(torch::kCPU), expected));
}

TEST(ComparisonOps, LeScalar)
{
    auto a_cpu = torch::tensor({1.0f, 2.0f, 3.0f, 4.0f});
    auto a = to_webgpu(a_cpu);

    auto out = torch::le(a, 3.0f);
    auto expected = torch::le(a_cpu, 3.0f);
    ASSERT_TRUE(torch::equal(out.to(torch::kCPU), expected));
}

// Greater than tests
TEST(ComparisonOps, GtTensor)
{
    auto a_cpu = torch::tensor({1.0f, 5.0f, 3.0f, 4.0f});
    auto b_cpu = torch::tensor({2.0f, 3.0f, 3.0f, 6.0f});
    auto a = to_webgpu(a_cpu);
    auto b = to_webgpu(b_cpu);

    auto out = torch::gt(a, b);
    auto expected = torch::gt(a_cpu, b_cpu);
    ASSERT_TRUE(torch::equal(out.to(torch::kCPU), expected));
}

TEST(ComparisonOps, GtScalar)
{
    auto a_cpu = torch::tensor({1.0f, 2.0f, 3.0f, 4.0f});
    auto a = to_webgpu(a_cpu);

    auto out = torch::gt(a, 2.0f);
    auto expected = torch::gt(a_cpu, 2.0f);
    ASSERT_TRUE(torch::equal(out.to(torch::kCPU), expected));
}

// Greater than or equal tests
TEST(ComparisonOps, GeTensor)
{
    auto a_cpu = torch::tensor({1.0f, 5.0f, 3.0f, 4.0f});
    auto b_cpu = torch::tensor({2.0f, 3.0f, 3.0f, 6.0f});
    auto a = to_webgpu(a_cpu);
    auto b = to_webgpu(b_cpu);

    auto out = torch::ge(a, b);
    auto expected = torch::ge(a_cpu, b_cpu);
    ASSERT_TRUE(torch::equal(out.to(torch::kCPU), expected));
}

TEST(ComparisonOps, GeScalar)
{
    auto a_cpu = torch::tensor({1.0f, 2.0f, 3.0f, 4.0f});
    auto a = to_webgpu(a_cpu);

    auto out = torch::ge(a, 3.0f);
    auto expected = torch::ge(a_cpu, 3.0f);
    ASSERT_TRUE(torch::equal(out.to(torch::kCPU), expected));
}

// Broadcast test
TEST(ComparisonOps, EqBroadcast)
{
    auto a_cpu = torch::randn({3, 4});
    auto b_cpu = torch::randn({4});
    auto a = to_webgpu(a_cpu);
    auto b = to_webgpu(b_cpu);

    auto out = torch::eq(a, b);
    auto expected = torch::eq(a_cpu, b_cpu);
    ASSERT_TRUE(torch::equal(out.to(torch::kCPU), expected));
}
