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

TEST(ArithmeticOps, DivBasic)
{
    auto a_cpu = torch::tensor({10.0f, 20.0f, 30.0f});
    auto b_cpu = torch::tensor({2.0f, 4.0f, 5.0f});
    auto a = to_webgpu(a_cpu);
    auto b = to_webgpu(b_cpu);

    auto out = torch::div(a, b);
    auto expected = torch::div(a_cpu, b_cpu);
    ASSERT_TRUE(torch::allclose(out.to(torch::kCPU), expected));
}

TEST(ArithmeticOps, DivFractional)
{
    auto a_cpu = torch::tensor({1.0f, 3.0f, 5.0f});
    auto b_cpu = torch::tensor({3.0f, 7.0f, 2.0f});
    auto a = to_webgpu(a_cpu);
    auto b = to_webgpu(b_cpu);

    auto out = torch::div(a, b);
    auto expected = torch::div(a_cpu, b_cpu);
    ASSERT_TRUE(torch::allclose(out.to(torch::kCPU), expected, 1e-4, 1e-4));
}

TEST(ArithmeticOps, DivNegative)
{
    auto a_cpu = torch::tensor({-10.0f, 20.0f, -30.0f});
    auto b_cpu = torch::tensor({2.0f, -4.0f, -5.0f});
    auto a = to_webgpu(a_cpu);
    auto b = to_webgpu(b_cpu);

    auto out = torch::div(a, b);
    auto expected = torch::div(a_cpu, b_cpu);
    ASSERT_TRUE(torch::allclose(out.to(torch::kCPU), expected));
}

TEST(ArithmeticOps, DivBroadcast)
{
    auto a_cpu = torch::randn({3, 4}) + 1.0f;
    auto b_cpu = torch::randn({4}).abs() + 0.1f;
    auto a = to_webgpu(a_cpu);
    auto b = to_webgpu(b_cpu);

    auto out = torch::div(a, b);
    auto expected = torch::div(a_cpu, b_cpu);
    ASSERT_TRUE(torch::allclose(out.to(torch::kCPU), expected, 1e-4, 1e-4));
}
