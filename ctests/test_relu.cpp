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

TEST(ActivationOps, ReluBasicAndOut)
{
    auto cpu_input = torch::tensor({-1.0f, 0.5f, 2.0f});
    auto input = to_webgpu(cpu_input);

    auto out = torch::relu(input);
    auto out_cpu = out.to(torch::kCPU);
    auto expected = torch::relu(cpu_input);
    ASSERT_TRUE(torch::allclose(out_cpu, expected));

    auto out_buf = torch::zeros_like(input);
    auto out_ref = torch::zeros_like(cpu_input);
    torch::relu_out(input, out_buf);
    torch::relu_out(cpu_input, out_ref);
    ASSERT_TRUE(torch::allclose(out_buf.to(torch::kCPU), out_ref));
}
