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

TEST(SoftmaxOps, Softmax1D)
{
    auto a_cpu = torch::tensor({1.0f, 2.0f, 3.0f, 4.0f});
    auto a = to_webgpu(a_cpu);

    auto out = torch::softmax(a, 0);
    auto expected = torch::softmax(a_cpu, 0);
    ASSERT_TRUE(torch::allclose(out.to(torch::kCPU), expected, 1e-4, 1e-4));
}

TEST(SoftmaxOps, Softmax2DLastDim)
{
    auto a_cpu = torch::randn({4, 8});
    auto a = to_webgpu(a_cpu);

    auto out = torch::softmax(a, -1);
    auto expected = torch::softmax(a_cpu, -1);
    ASSERT_TRUE(torch::allclose(out.to(torch::kCPU), expected, 1e-4, 1e-4));
}

TEST(SoftmaxOps, Softmax2DFirstDim)
{
    auto a_cpu = torch::randn({4, 8});
    auto a = to_webgpu(a_cpu);

    auto out = torch::softmax(a, 0);
    auto expected = torch::softmax(a_cpu, 0);
    ASSERT_TRUE(torch::allclose(out.to(torch::kCPU), expected, 1e-4, 1e-4));
}

TEST(SoftmaxOps, Softmax3D)
{
    auto a_cpu = torch::randn({2, 4, 8});
    auto a = to_webgpu(a_cpu);

    auto out = torch::softmax(a, -1);
    auto expected = torch::softmax(a_cpu, -1);
    ASSERT_TRUE(torch::allclose(out.to(torch::kCPU), expected, 1e-4, 1e-4));
}

TEST(SoftmaxOps, SoftmaxSumsToOne)
{
    auto a_cpu = torch::randn({4, 8});
    auto a = to_webgpu(a_cpu);

    auto out = torch::softmax(a, -1);
    auto sum = out.sum(-1).to(torch::kCPU);
    auto expected = torch::ones({4});
    ASSERT_TRUE(torch::allclose(sum, expected, 1e-4, 1e-4));
}

TEST(SoftmaxOps, LogSoftmax)
{
    auto a_cpu = torch::randn({4, 8});
    auto a = to_webgpu(a_cpu);

    auto out = torch::log_softmax(a, -1);
    auto expected = torch::log_softmax(a_cpu, -1);
    ASSERT_TRUE(torch::allclose(out.to(torch::kCPU), expected, 1e-4, 1e-4));
}
