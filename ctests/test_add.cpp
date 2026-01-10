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

TEST(ArithmeticOps, AddBasicAndAlpha)
{
    auto a_cpu = torch::tensor({1.0f, 2.0f, 3.0f});
    auto b_cpu = torch::tensor({4.0f, 5.0f, 6.0f});
    auto a = to_webgpu(a_cpu);
    auto b = to_webgpu(b_cpu);

    auto out = torch::add(a, b);
    auto expected = torch::add(a_cpu, b_cpu);
    ASSERT_TRUE(torch::allclose(out.to(torch::kCPU), expected));

    auto out_alpha = torch::add(a, b, 2);
    auto expected_alpha = torch::add(a_cpu, b_cpu, 2);
    ASSERT_TRUE(torch::allclose(out_alpha.to(torch::kCPU), expected_alpha));

    auto out_buf = torch::zeros_like(a);
    torch::add_out(out_buf, a, b);
    ASSERT_TRUE(torch::allclose(out_buf.to(torch::kCPU), expected));
}
