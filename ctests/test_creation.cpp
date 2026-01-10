#include <gtest/gtest.h>
#include <torch/torch.h>

namespace
{

    torch::Device webgpu_device()
    {
        return torch::Device(torch::DeviceType::PrivateUse1);
    }

} // namespace

TEST(CreationOps, ArangeBasic)
{
    auto out = torch::arange(10, torch::TensorOptions().device(webgpu_device()));
    auto expected = torch::arange(10);
    ASSERT_TRUE(torch::allclose(out.to(torch::kCPU), expected.to(torch::kFloat)));
}

TEST(CreationOps, ArangeStartEnd)
{
    auto out = torch::arange(5, 15, torch::TensorOptions().device(webgpu_device()));
    auto expected = torch::arange(5, 15);
    ASSERT_TRUE(torch::allclose(out.to(torch::kCPU), expected.to(torch::kFloat)));
}

TEST(CreationOps, ArangeWithStep)
{
    auto out = torch::arange(0, 10, 2, torch::TensorOptions().device(webgpu_device()));
    auto expected = torch::arange(0, 10, 2);
    ASSERT_TRUE(torch::allclose(out.to(torch::kCPU), expected.to(torch::kFloat)));
}

TEST(CreationOps, ArangeFloat)
{
    auto out = torch::arange(0.0f, 5.0f, 0.5f, torch::TensorOptions().device(webgpu_device()));
    auto expected = torch::arange(0.0f, 5.0f, 0.5f);
    ASSERT_TRUE(torch::allclose(out.to(torch::kCPU), expected, 1e-4, 1e-4));
}

TEST(CreationOps, Zeros)
{
    auto out = torch::zeros({3, 4}, torch::TensorOptions().device(webgpu_device()));
    auto expected = torch::zeros({3, 4});
    ASSERT_TRUE(torch::allclose(out.to(torch::kCPU), expected));
}

TEST(CreationOps, Ones)
{
    auto out = torch::ones({3, 4}, torch::TensorOptions().device(webgpu_device()));
    auto expected = torch::ones({3, 4});
    ASSERT_TRUE(torch::allclose(out.to(torch::kCPU), expected));
}

TEST(CreationOps, Full)
{
    auto out = torch::full({3, 4}, 3.14f, torch::TensorOptions().device(webgpu_device()));
    auto expected = torch::full({3, 4}, 3.14f);
    ASSERT_TRUE(torch::allclose(out.to(torch::kCPU), expected));
}

TEST(CreationOps, Fill)
{
    auto out = torch::empty({3, 4}, torch::TensorOptions().device(webgpu_device()));
    out.fill_(2.5f);
    auto expected = torch::full({3, 4}, 2.5f);
    ASSERT_TRUE(torch::allclose(out.to(torch::kCPU), expected));
}

TEST(CreationOps, Zero)
{
    auto cpu_tensor = torch::randn({3, 4});
    auto out = cpu_tensor.to(webgpu_device());
    out.zero_();
    auto expected = torch::zeros({3, 4});
    ASSERT_TRUE(torch::allclose(out.to(torch::kCPU), expected));
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
