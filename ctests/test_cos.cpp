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

TEST(ActivationOps, CosMatchesCPU)
{
    auto cpu_input = torch::tensor({-1.5f, 2.7f, 1.0f, 2.0f});
    auto webgpu_input = to_webgpu(cpu_input);

    auto out = torch::cos(webgpu_input);
    auto expected = torch::cos(cpu_input);
    ASSERT_TRUE(torch::allclose(out.to(torch::kCPU), expected, 1e-4, 1e-4));

    auto out_buf = torch::zeros_like(webgpu_input);
    auto out_ref = torch::zeros_like(cpu_input);
    torch::cos_out(out_buf, webgpu_input);
    torch::cos_out(out_ref, cpu_input);
    ASSERT_TRUE(torch::allclose(out_buf.to(torch::kCPU), out_ref, 1e-4, 1e-4));
}
