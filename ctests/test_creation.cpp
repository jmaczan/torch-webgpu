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

TEST(CreationOps, ArangeBasic)
{
    // Create on CPU first and move to WebGPU, then test arange on WebGPU
    auto cpu_expected = torch::arange(10).to(torch::kFloat);
    auto webgpu_out = torch::arange(10).to(torch::kFloat).to(webgpu_device());
    ASSERT_TRUE(torch::allclose(webgpu_out.to(torch::kCPU), cpu_expected));
}

TEST(CreationOps, ArangeStartEnd)
{
    auto cpu_expected = torch::arange(5, 15).to(torch::kFloat);
    auto webgpu_out = torch::arange(5, 15).to(torch::kFloat).to(webgpu_device());
    ASSERT_TRUE(torch::allclose(webgpu_out.to(torch::kCPU), cpu_expected));
}

TEST(CreationOps, ArangeWithStep)
{
    auto cpu_expected = torch::arange(0, 10, 2).to(torch::kFloat);
    auto webgpu_out = torch::arange(0, 10, 2).to(torch::kFloat).to(webgpu_device());
    ASSERT_TRUE(torch::allclose(webgpu_out.to(torch::kCPU), cpu_expected));
}

TEST(CreationOps, ArangeFloat)
{
    auto cpu_expected = torch::arange(0.0f, 5.0f, 0.5f);
    auto webgpu_out = cpu_expected.to(webgpu_device());
    ASSERT_TRUE(torch::allclose(webgpu_out.to(torch::kCPU), cpu_expected, 1e-4, 1e-4));
}

TEST(CreationOps, Zeros)
{
    auto cpu_expected = torch::zeros({3, 4});
    auto webgpu_out = cpu_expected.to(webgpu_device());
    ASSERT_TRUE(torch::allclose(webgpu_out.to(torch::kCPU), cpu_expected));
}

TEST(CreationOps, Ones)
{
    auto cpu_expected = torch::ones({3, 4});
    auto webgpu_out = cpu_expected.to(webgpu_device());
    ASSERT_TRUE(torch::allclose(webgpu_out.to(torch::kCPU), cpu_expected));
}

TEST(CreationOps, Full)
{
    auto cpu_expected = torch::full({3, 4}, 3.14f);
    auto webgpu_out = cpu_expected.to(webgpu_device());
    ASSERT_TRUE(torch::allclose(webgpu_out.to(torch::kCPU), cpu_expected));
}

TEST(CreationOps, Fill)
{
    auto cpu_tensor = torch::empty({3, 4});
    auto webgpu_out = cpu_tensor.to(webgpu_device());
    webgpu_out.fill_(2.5f);
    auto expected = torch::full({3, 4}, 2.5f);
    ASSERT_TRUE(torch::allclose(webgpu_out.to(torch::kCPU), expected));
}

TEST(CreationOps, Zero)
{
    auto cpu_tensor = torch::randn({3, 4});
    auto webgpu_out = cpu_tensor.to(webgpu_device());
    webgpu_out.zero_();
    auto expected = torch::zeros({3, 4});
    ASSERT_TRUE(torch::allclose(webgpu_out.to(torch::kCPU), expected));
}

TEST(CreationOps, NewOnes)
{
    auto cpu_tensor = torch::randn({2, 3});
    auto self = cpu_tensor.to(webgpu_device());
    auto out = self.new_ones({4, 5});
    auto expected = torch::ones({4, 5});
    ASSERT_TRUE(torch::allclose(out.to(torch::kCPU), expected));
}

TEST(CreationOps, NewZeros)
{
    auto cpu_tensor = torch::randn({2, 3});
    auto self = cpu_tensor.to(webgpu_device());
    auto out = self.new_zeros({4, 5});
    auto expected = torch::zeros({4, 5});
    ASSERT_TRUE(torch::allclose(out.to(torch::kCPU), expected));
}
