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

TEST(ArithmeticOps, MulBasic)
{
    auto a_cpu = torch::tensor({-1.0f, 2.0f, 3.5f});
    auto b_cpu = torch::tensor({2.0f, -0.5f, 4.0f});
    auto a = to_webgpu(a_cpu);
    auto b = to_webgpu(b_cpu);

    auto out = torch::mul(a, b);
    auto expected = torch::mul(a_cpu, b_cpu);
    ASSERT_TRUE(torch::allclose(out.to(torch::kCPU), expected));

    auto out_buf = torch::zeros_like(a);
    torch::mul_out(out_buf, a, b);
    ASSERT_TRUE(torch::allclose(out_buf.to(torch::kCPU), expected));
}
