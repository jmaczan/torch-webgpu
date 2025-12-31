#include <gtest/gtest.h>
#include <torch/torch.h>
#include <torch/extension.h>
#include <torch/library.h>

namespace
{

    torch::Device webgpu_device()
    {
        return torch::Device(torch::DeviceType::PrivateUse1);
    }

} // namespace

TEST(CopyOps, CpuToWebgpuAndBack)
{
    auto cpu = torch::tensor({1.0f, 2.0f, 3.0f});
    auto opts = torch::TensorOptions().device(webgpu_device()).dtype(torch::kFloat);
    auto webgpu = torch::zeros(cpu.sizes(), opts);

    webgpu.copy_(cpu);
    ASSERT_TRUE(torch::allclose(webgpu.to(torch::kCPU), cpu));

    auto cpu_out = torch::zeros_like(cpu);
    cpu_out.copy_(webgpu);
    ASSERT_TRUE(torch::allclose(cpu_out, cpu));
}
