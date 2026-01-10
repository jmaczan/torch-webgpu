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

// Mean tests
TEST(ReductionOps, Mean1D)
{
    auto a_cpu = torch::tensor({1.0f, 2.0f, 3.0f, 4.0f, 5.0f});
    auto a = to_webgpu(a_cpu);

    auto out = torch::mean(a);
    auto expected = torch::mean(a_cpu);
    ASSERT_TRUE(torch::allclose(out.to(torch::kCPU), expected, 1e-4, 1e-4));
}

TEST(ReductionOps, Mean2D)
{
    auto a_cpu = torch::randn({16, 32});
    auto a = to_webgpu(a_cpu);

    auto out = torch::mean(a);
    auto expected = torch::mean(a_cpu);
    ASSERT_TRUE(torch::allclose(out.to(torch::kCPU), expected, 1e-3, 1e-3));
}

TEST(ReductionOps, MeanLarge)
{
    auto a_cpu = torch::randn({64, 64});
    auto a = to_webgpu(a_cpu);

    auto out = torch::mean(a);
    auto expected = torch::mean(a_cpu);
    ASSERT_TRUE(torch::allclose(out.to(torch::kCPU), expected, 1e-3, 1e-3));
}

// Sum tests
TEST(ReductionOps, Sum1D)
{
    auto a_cpu = torch::tensor({1.0f, 2.0f, 3.0f, 4.0f, 5.0f});
    auto a = to_webgpu(a_cpu);

    auto out = torch::sum(a);
    auto expected = torch::sum(a_cpu);
    ASSERT_TRUE(torch::allclose(out.to(torch::kCPU), expected, 1e-4, 1e-4));
}

TEST(ReductionOps, Sum2D)
{
    auto a_cpu = torch::randn({16, 32});
    auto a = to_webgpu(a_cpu);

    auto out = torch::sum(a);
    auto expected = torch::sum(a_cpu);
    ASSERT_TRUE(torch::allclose(out.to(torch::kCPU), expected, 1e-2, 1e-2));
}

TEST(ReductionOps, SumLarge)
{
    auto a_cpu = torch::randn({64, 64});
    auto a = to_webgpu(a_cpu);

    auto out = torch::sum(a);
    auto expected = torch::sum(a_cpu);
    ASSERT_TRUE(torch::allclose(out.to(torch::kCPU), expected, 1e-1, 1e-1));
}

TEST(ReductionOps, SumSmallPositive)
{
    auto a_cpu = torch::ones({10, 10});
    auto a = to_webgpu(a_cpu);

    auto out = torch::sum(a);
    auto expected = torch::tensor(100.0f);
    ASSERT_TRUE(torch::allclose(out.to(torch::kCPU), expected, 1e-4, 1e-4));
}
