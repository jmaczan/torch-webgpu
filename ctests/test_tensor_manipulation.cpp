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

// Transpose tests
TEST(TensorManipulation, Transpose2D)
{
    auto a_cpu = torch::randn({3, 4});
    auto a = to_webgpu(a_cpu);

    auto out = torch::transpose(a, 0, 1);
    auto expected = torch::transpose(a_cpu, 0, 1);
    ASSERT_EQ(out.sizes(), expected.sizes());
    ASSERT_TRUE(torch::allclose(out.to(torch::kCPU), expected, 1e-4, 1e-4));
}

TEST(TensorManipulation, Transpose3D)
{
    auto a_cpu = torch::randn({2, 3, 4});
    auto a = to_webgpu(a_cpu);

    auto out = torch::transpose(a, 0, 2);
    auto expected = torch::transpose(a_cpu, 0, 2);
    ASSERT_EQ(out.sizes(), expected.sizes());
    ASSERT_TRUE(torch::allclose(out.to(torch::kCPU).contiguous(), expected.contiguous(), 1e-4, 1e-4));
}

// Permute tests
TEST(TensorManipulation, Permute3D)
{
    auto a_cpu = torch::randn({2, 3, 4});
    auto a = to_webgpu(a_cpu);

    auto out = a.permute({2, 0, 1});
    auto expected = a_cpu.permute({2, 0, 1});
    ASSERT_EQ(out.sizes(), expected.sizes());
    ASSERT_TRUE(torch::allclose(out.to(torch::kCPU).contiguous(), expected.contiguous(), 1e-4, 1e-4));
}

TEST(TensorManipulation, Permute4D)
{
    auto a_cpu = torch::randn({2, 3, 4, 5});
    auto a = to_webgpu(a_cpu);

    auto out = a.permute({3, 1, 2, 0});
    auto expected = a_cpu.permute({3, 1, 2, 0});
    ASSERT_EQ(out.sizes(), expected.sizes());
    ASSERT_TRUE(torch::allclose(out.to(torch::kCPU).contiguous(), expected.contiguous(), 1e-4, 1e-4));
}

// Unsqueeze tests
TEST(TensorManipulation, Unsqueeze)
{
    auto a_cpu = torch::randn({3, 4});
    auto a = to_webgpu(a_cpu);

    auto out = torch::unsqueeze(a, 0);
    auto expected = torch::unsqueeze(a_cpu, 0);
    ASSERT_EQ(out.sizes(), expected.sizes());
    ASSERT_TRUE(torch::allclose(out.to(torch::kCPU), expected, 1e-4, 1e-4));
}

TEST(TensorManipulation, UnsqueezeLast)
{
    auto a_cpu = torch::randn({3, 4});
    auto a = to_webgpu(a_cpu);

    auto out = torch::unsqueeze(a, -1);
    auto expected = torch::unsqueeze(a_cpu, -1);
    ASSERT_EQ(out.sizes(), expected.sizes());
    ASSERT_TRUE(torch::allclose(out.to(torch::kCPU), expected, 1e-4, 1e-4));
}

// Squeeze tests
TEST(TensorManipulation, Squeeze)
{
    auto a_cpu = torch::randn({1, 3, 1, 4});
    auto a = to_webgpu(a_cpu);

    auto out = torch::squeeze(a);
    auto expected = torch::squeeze(a_cpu);
    ASSERT_EQ(out.sizes(), expected.sizes());
    ASSERT_TRUE(torch::allclose(out.to(torch::kCPU), expected, 1e-4, 1e-4));
}

TEST(TensorManipulation, SqueezeDim)
{
    auto a_cpu = torch::randn({1, 3, 4});
    auto a = to_webgpu(a_cpu);

    auto out = torch::squeeze(a, 0);
    auto expected = torch::squeeze(a_cpu, 0);
    ASSERT_EQ(out.sizes(), expected.sizes());
    ASSERT_TRUE(torch::allclose(out.to(torch::kCPU), expected, 1e-4, 1e-4));
}

// Expand tests
TEST(TensorManipulation, Expand)
{
    auto a_cpu = torch::randn({1, 4});
    auto a = to_webgpu(a_cpu);

    auto out = a.expand({3, 4});
    auto expected = a_cpu.expand({3, 4});
    ASSERT_EQ(out.sizes(), expected.sizes());
    ASSERT_TRUE(torch::allclose(out.to(torch::kCPU), expected, 1e-4, 1e-4));
}

TEST(TensorManipulation, ExpandAddDims)
{
    auto a_cpu = torch::randn({4});
    auto a = to_webgpu(a_cpu);

    auto out = a.expand({2, 3, 4});
    auto expected = a_cpu.expand({2, 3, 4});
    ASSERT_EQ(out.sizes(), expected.sizes());
    ASSERT_TRUE(torch::allclose(out.to(torch::kCPU), expected, 1e-4, 1e-4));
}

// Slice tests
TEST(TensorManipulation, Slice)
{
    auto a_cpu = torch::arange(20).reshape({4, 5}).to(torch::kFloat);
    auto a = to_webgpu(a_cpu);

    auto out = a.slice(0, 1, 3);
    auto expected = a_cpu.slice(0, 1, 3);
    ASSERT_EQ(out.sizes(), expected.sizes());
    ASSERT_TRUE(torch::allclose(out.to(torch::kCPU), expected, 1e-4, 1e-4));
}

TEST(TensorManipulation, SliceWithStep)
{
    auto a_cpu = torch::arange(20).reshape({4, 5}).to(torch::kFloat);
    auto a = to_webgpu(a_cpu);

    auto out = a.slice(1, 0, 5, 2);
    auto expected = a_cpu.slice(1, 0, 5, 2);
    ASSERT_EQ(out.sizes(), expected.sizes());
    ASSERT_TRUE(torch::allclose(out.to(torch::kCPU), expected, 1e-4, 1e-4));
}

// Select tests
TEST(TensorManipulation, Select)
{
    auto a_cpu = torch::arange(20).reshape({4, 5}).to(torch::kFloat);
    auto a = to_webgpu(a_cpu);

    auto out = a.select(0, 2);
    auto expected = a_cpu.select(0, 2);
    ASSERT_EQ(out.sizes(), expected.sizes());
    ASSERT_TRUE(torch::allclose(out.to(torch::kCPU), expected, 1e-4, 1e-4));
}

TEST(TensorManipulation, SelectNegative)
{
    auto a_cpu = torch::arange(20).reshape({4, 5}).to(torch::kFloat);
    auto a = to_webgpu(a_cpu);

    auto out = a.select(0, -1);
    auto expected = a_cpu.select(0, -1);
    ASSERT_EQ(out.sizes(), expected.sizes());
    ASSERT_TRUE(torch::allclose(out.to(torch::kCPU), expected, 1e-4, 1e-4));
}

// T (transpose for 2D) tests
TEST(TensorManipulation, T)
{
    auto a_cpu = torch::randn({3, 4});
    auto a = to_webgpu(a_cpu);

    auto out = a.t();
    auto expected = a_cpu.t();
    ASSERT_EQ(out.sizes(), expected.sizes());
    ASSERT_TRUE(torch::allclose(out.to(torch::kCPU), expected, 1e-4, 1e-4));
}

// Clone tests
TEST(TensorManipulation, Clone)
{
    auto a_cpu = torch::randn({3, 4});
    auto a = to_webgpu(a_cpu);

    auto out = a.clone();
    ASSERT_EQ(out.sizes(), a.sizes());
    ASSERT_TRUE(torch::allclose(out.to(torch::kCPU), a_cpu, 1e-4, 1e-4));
}

// Contiguous tests
TEST(TensorManipulation, Contiguous)
{
    auto a_cpu = torch::randn({3, 4}).transpose(0, 1);
    auto a = to_webgpu(a_cpu);

    auto out = a.contiguous();
    ASSERT_TRUE(out.is_contiguous());
    ASSERT_TRUE(torch::allclose(out.to(torch::kCPU), a_cpu.contiguous(), 1e-4, 1e-4));
}
