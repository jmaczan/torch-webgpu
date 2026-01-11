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

TEST(ActivationOps, GeluMatchesCPU)
{
    auto cpu_input = torch::tensor({-1.0f, 0.0f, 1.0f, 2.0f});
    auto input = to_webgpu(cpu_input);

    auto out = torch::gelu(input);
    auto expected = torch::gelu(cpu_input);
    ASSERT_TRUE(torch::allclose(out.to(torch::kCPU), expected, 1e-3, 1e-3));

    auto out_buf = torch::zeros_like(input);
    auto out_ref = torch::zeros_like(cpu_input);
    torch::gelu_out(out_buf, input, "none");
    torch::gelu_out(out_ref, cpu_input, "none");
    ASSERT_TRUE(torch::allclose(out_buf.to(torch::kCPU), out_ref, 1e-3, 1e-3));
}
