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

TEST(EmbeddingOps, EmbeddingBasic)
{
    auto weight_cpu = torch::randn({10, 8}); // 10 vocab, 8 dim
    auto indices_cpu = torch::tensor({0, 2, 5, 9}, torch::kInt);

    auto weight = to_webgpu(weight_cpu);
    auto indices = to_webgpu(indices_cpu);

    auto out = torch::embedding(weight, indices);
    auto expected = torch::embedding(weight_cpu, indices_cpu);
    ASSERT_EQ(out.sizes(), expected.sizes());
    ASSERT_TRUE(torch::allclose(out.to(torch::kCPU), expected, 1e-4, 1e-4));
}

TEST(EmbeddingOps, EmbeddingLongIndices)
{
    auto weight_cpu = torch::randn({100, 16});
    auto indices_cpu = torch::tensor({0, 10, 50, 99}, torch::kLong);

    auto weight = to_webgpu(weight_cpu);
    auto indices = to_webgpu(indices_cpu);

    auto out = torch::embedding(weight, indices);
    auto expected = torch::embedding(weight_cpu, indices_cpu);
    ASSERT_EQ(out.sizes(), expected.sizes());
    ASSERT_TRUE(torch::allclose(out.to(torch::kCPU), expected, 1e-4, 1e-4));
}

TEST(EmbeddingOps, Embedding2DIndices)
{
    auto weight_cpu = torch::randn({50, 32});
    auto indices_cpu = torch::tensor({{0, 1, 2}, {10, 20, 30}}, torch::kInt);

    auto weight = to_webgpu(weight_cpu);
    auto indices = to_webgpu(indices_cpu);

    auto out = torch::embedding(weight, indices);
    auto expected = torch::embedding(weight_cpu, indices_cpu);
    ASSERT_EQ(out.sizes(), expected.sizes());
    ASSERT_TRUE(torch::allclose(out.to(torch::kCPU), expected, 1e-4, 1e-4));
}

TEST(EmbeddingOps, EmbeddingLarger)
{
    auto weight_cpu = torch::randn({1000, 64});
    auto indices_cpu = torch::randint(0, 1000, {16, 32}, torch::kInt);

    auto weight = to_webgpu(weight_cpu);
    auto indices = to_webgpu(indices_cpu);

    auto out = torch::embedding(weight, indices);
    auto expected = torch::embedding(weight_cpu, indices_cpu);
    ASSERT_EQ(out.sizes(), expected.sizes());
    ASSERT_TRUE(torch::allclose(out.to(torch::kCPU), expected, 1e-4, 1e-4));
}

TEST(EmbeddingOps, EmbeddingSequential)
{
    auto weight_cpu = torch::randn({20, 8});
    auto indices_cpu = torch::arange(0, 10).to(torch::kInt);

    auto weight = to_webgpu(weight_cpu);
    auto indices = to_webgpu(indices_cpu);

    auto out = torch::embedding(weight, indices);
    auto expected = torch::embedding(weight_cpu, indices_cpu);
    ASSERT_EQ(out.sizes(), expected.sizes());
    ASSERT_TRUE(torch::allclose(out.to(torch::kCPU), expected, 1e-4, 1e-4));
}
