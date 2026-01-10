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

// Sin tests
TEST(TrigOps, Sin)
{
    auto a_cpu = torch::tensor({0.0f, 3.14159f / 2, 3.14159f, 3.14159f * 1.5f});
    auto a = to_webgpu(a_cpu);

    auto out = torch::sin(a);
    auto expected = torch::sin(a_cpu);
    ASSERT_TRUE(torch::allclose(out.to(torch::kCPU), expected, 1e-4, 1e-4));
}

TEST(TrigOps, SinRandom)
{
    auto a_cpu = torch::randn({32, 64});
    auto a = to_webgpu(a_cpu);

    auto out = torch::sin(a);
    auto expected = torch::sin(a_cpu);
    ASSERT_TRUE(torch::allclose(out.to(torch::kCPU), expected, 1e-4, 1e-4));
}

// Tanh tests
TEST(TrigOps, Tanh)
{
    auto a_cpu = torch::tensor({-2.0f, -1.0f, 0.0f, 1.0f, 2.0f});
    auto a = to_webgpu(a_cpu);

    auto out = torch::tanh(a);
    auto expected = torch::tanh(a_cpu);
    ASSERT_TRUE(torch::allclose(out.to(torch::kCPU), expected, 1e-4, 1e-4));
}

TEST(TrigOps, TanhRandom)
{
    auto a_cpu = torch::randn({32, 64});
    auto a = to_webgpu(a_cpu);

    auto out = torch::tanh(a);
    auto expected = torch::tanh(a_cpu);
    ASSERT_TRUE(torch::allclose(out.to(torch::kCPU), expected, 1e-4, 1e-4));
}

// Exp tests
TEST(TrigOps, Exp)
{
    auto a_cpu = torch::tensor({0.0f, 1.0f, 2.0f, -1.0f});
    auto a = to_webgpu(a_cpu);

    auto out = torch::exp(a);
    auto expected = torch::exp(a_cpu);
    ASSERT_TRUE(torch::allclose(out.to(torch::kCPU), expected, 1e-4, 1e-4));
}

TEST(TrigOps, ExpRandom)
{
    auto a_cpu = torch::randn({32, 64}) * 0.5f; // Keep values small to avoid overflow
    auto a = to_webgpu(a_cpu);

    auto out = torch::exp(a);
    auto expected = torch::exp(a_cpu);
    ASSERT_TRUE(torch::allclose(out.to(torch::kCPU), expected, 1e-3, 1e-3));
}

// Abs tests
TEST(UnaryOps, Abs)
{
    auto a_cpu = torch::tensor({-2.0f, -1.0f, 0.0f, 1.0f, 2.0f});
    auto a = to_webgpu(a_cpu);

    auto out = torch::abs(a);
    auto expected = torch::abs(a_cpu);
    ASSERT_TRUE(torch::allclose(out.to(torch::kCPU), expected));
}

TEST(UnaryOps, AbsRandom)
{
    auto a_cpu = torch::randn({32, 64});
    auto a = to_webgpu(a_cpu);

    auto out = torch::abs(a);
    auto expected = torch::abs(a_cpu);
    ASSERT_TRUE(torch::allclose(out.to(torch::kCPU), expected, 1e-4, 1e-4));
}

// Rsqrt tests
TEST(UnaryOps, Rsqrt)
{
    auto a_cpu = torch::tensor({1.0f, 4.0f, 9.0f, 16.0f});
    auto a = to_webgpu(a_cpu);

    auto out = torch::rsqrt(a);
    auto expected = torch::rsqrt(a_cpu);
    ASSERT_TRUE(torch::allclose(out.to(torch::kCPU), expected, 1e-4, 1e-4));
}

TEST(UnaryOps, RsqrtRandom)
{
    auto a_cpu = torch::rand({32, 64}) + 0.1f; // Positive values
    auto a = to_webgpu(a_cpu);

    auto out = torch::rsqrt(a);
    auto expected = torch::rsqrt(a_cpu);
    ASSERT_TRUE(torch::allclose(out.to(torch::kCPU), expected, 1e-4, 1e-4));
}

// Neg tests
TEST(UnaryOps, Neg)
{
    auto a_cpu = torch::tensor({-2.0f, -1.0f, 0.0f, 1.0f, 2.0f});
    auto a = to_webgpu(a_cpu);

    auto out = torch::neg(a);
    auto expected = torch::neg(a_cpu);
    ASSERT_TRUE(torch::allclose(out.to(torch::kCPU), expected));
}

TEST(UnaryOps, NegRandom)
{
    auto a_cpu = torch::randn({32, 64});
    auto a = to_webgpu(a_cpu);

    auto out = torch::neg(a);
    auto expected = torch::neg(a_cpu);
    ASSERT_TRUE(torch::allclose(out.to(torch::kCPU), expected, 1e-4, 1e-4));
}

// Pow tests
TEST(UnaryOps, PowScalar)
{
    auto a_cpu = torch::tensor({1.0f, 2.0f, 3.0f, 4.0f});
    auto a = to_webgpu(a_cpu);

    auto out = torch::pow(a, 2.0);
    auto expected = torch::pow(a_cpu, 2.0);
    ASSERT_TRUE(torch::allclose(out.to(torch::kCPU), expected, 1e-4, 1e-4));
}

TEST(UnaryOps, PowFractional)
{
    auto a_cpu = torch::tensor({1.0f, 4.0f, 9.0f, 16.0f});
    auto a = to_webgpu(a_cpu);

    auto out = torch::pow(a, 0.5);
    auto expected = torch::pow(a_cpu, 0.5);
    ASSERT_TRUE(torch::allclose(out.to(torch::kCPU), expected, 1e-4, 1e-4));
}

TEST(UnaryOps, PowRandom)
{
    auto a_cpu = torch::rand({32, 64}) + 0.1f; // Positive values
    auto a = to_webgpu(a_cpu);

    auto out = torch::pow(a, 3.0);
    auto expected = torch::pow(a_cpu, 3.0);
    ASSERT_TRUE(torch::allclose(out.to(torch::kCPU), expected, 1e-3, 1e-3));
}
