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

// Cat tests
TEST(ConcatenationOps, Cat1D)
{
    auto a_cpu = torch::tensor({1.0f, 2.0f, 3.0f});
    auto b_cpu = torch::tensor({4.0f, 5.0f, 6.0f});
    auto a = to_webgpu(a_cpu);
    auto b = to_webgpu(b_cpu);

    auto out = torch::cat({a, b}, 0);
    auto expected = torch::cat({a_cpu, b_cpu}, 0);
    ASSERT_EQ(out.sizes(), expected.sizes());
    ASSERT_TRUE(torch::allclose(out.to(torch::kCPU), expected, 1e-4, 1e-4));
}

TEST(ConcatenationOps, Cat2DDim0)
{
    auto a_cpu = torch::randn({2, 4});
    auto b_cpu = torch::randn({3, 4});
    auto a = to_webgpu(a_cpu);
    auto b = to_webgpu(b_cpu);

    auto out = torch::cat({a, b}, 0);
    auto expected = torch::cat({a_cpu, b_cpu}, 0);
    ASSERT_EQ(out.sizes(), expected.sizes());
    ASSERT_TRUE(torch::allclose(out.to(torch::kCPU), expected, 1e-4, 1e-4));
}

TEST(ConcatenationOps, Cat2DDim1)
{
    auto a_cpu = torch::randn({3, 2});
    auto b_cpu = torch::randn({3, 4});
    auto a = to_webgpu(a_cpu);
    auto b = to_webgpu(b_cpu);

    auto out = torch::cat({a, b}, 1);
    auto expected = torch::cat({a_cpu, b_cpu}, 1);
    ASSERT_EQ(out.sizes(), expected.sizes());
    ASSERT_TRUE(torch::allclose(out.to(torch::kCPU), expected, 1e-4, 1e-4));
}

TEST(ConcatenationOps, CatMultiple)
{
    auto a_cpu = torch::randn({2, 4});
    auto b_cpu = torch::randn({2, 4});
    auto c_cpu = torch::randn({2, 4});
    auto a = to_webgpu(a_cpu);
    auto b = to_webgpu(b_cpu);
    auto c = to_webgpu(c_cpu);

    auto out = torch::cat({a, b, c}, 0);
    auto expected = torch::cat({a_cpu, b_cpu, c_cpu}, 0);
    ASSERT_EQ(out.sizes(), expected.sizes());
    ASSERT_TRUE(torch::allclose(out.to(torch::kCPU), expected, 1e-4, 1e-4));
}

// Stack tests
TEST(ConcatenationOps, Stack1D)
{
    auto a_cpu = torch::tensor({1.0f, 2.0f, 3.0f});
    auto b_cpu = torch::tensor({4.0f, 5.0f, 6.0f});
    auto a = to_webgpu(a_cpu);
    auto b = to_webgpu(b_cpu);

    auto out = torch::stack({a, b}, 0);
    auto expected = torch::stack({a_cpu, b_cpu}, 0);
    ASSERT_EQ(out.sizes(), expected.sizes());
    ASSERT_TRUE(torch::allclose(out.to(torch::kCPU), expected, 1e-4, 1e-4));
}

TEST(ConcatenationOps, Stack2D)
{
    auto a_cpu = torch::randn({3, 4});
    auto b_cpu = torch::randn({3, 4});
    auto a = to_webgpu(a_cpu);
    auto b = to_webgpu(b_cpu);

    auto out = torch::stack({a, b}, 0);
    auto expected = torch::stack({a_cpu, b_cpu}, 0);
    ASSERT_EQ(out.sizes(), expected.sizes());
    ASSERT_TRUE(torch::allclose(out.to(torch::kCPU), expected, 1e-4, 1e-4));
}

TEST(ConcatenationOps, StackDim1)
{
    auto a_cpu = torch::randn({3, 4});
    auto b_cpu = torch::randn({3, 4});
    auto a = to_webgpu(a_cpu);
    auto b = to_webgpu(b_cpu);

    auto out = torch::stack({a, b}, 1);
    auto expected = torch::stack({a_cpu, b_cpu}, 1);
    ASSERT_EQ(out.sizes(), expected.sizes());
    ASSERT_TRUE(torch::allclose(out.to(torch::kCPU), expected, 1e-4, 1e-4));
}

// Narrow tests
TEST(ConcatenationOps, Narrow)
{
    auto a_cpu = torch::arange(20).reshape({4, 5}).to(torch::kFloat);
    auto a = to_webgpu(a_cpu);

    auto out = a.narrow(0, 1, 2);
    auto expected = a_cpu.narrow(0, 1, 2);
    ASSERT_EQ(out.sizes(), expected.sizes());
    ASSERT_TRUE(torch::allclose(out.to(torch::kCPU), expected, 1e-4, 1e-4));
}

TEST(ConcatenationOps, NarrowDim1)
{
    auto a_cpu = torch::arange(20).reshape({4, 5}).to(torch::kFloat);
    auto a = to_webgpu(a_cpu);

    auto out = a.narrow(1, 2, 3);
    auto expected = a_cpu.narrow(1, 2, 3);
    ASSERT_EQ(out.sizes(), expected.sizes());
    ASSERT_TRUE(torch::allclose(out.to(torch::kCPU), expected, 1e-4, 1e-4));
}
