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

TEST(CopyOps, ToDeviceReturnsCpuCopy)
{
    auto webgpu = to_webgpu(torch::tensor({5.0f, 6.0f}, torch::kFloat));
    auto cpu = webgpu.to(torch::kCPU, torch::kFloat, false, false);
    ASSERT_EQ(cpu.device().type(), torch::kCPU);
    ASSERT_TRUE(torch::allclose(cpu, torch::tensor({5.0f, 6.0f})));
}
