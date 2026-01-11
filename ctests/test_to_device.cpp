#include <gtest/gtest.h>
#include <torch/torch.h>

namespace
{
    torch::Device webgpu_device()
    {
        return torch::Device(torch::DeviceType::PrivateUse1);
    }
} // namespace

TEST(ToDevice, CpuToWebgpu)
{
    auto cpu = torch::tensor({1.0f, 2.0f, 3.0f}, torch::kFloat);
    ASSERT_EQ(cpu.device().type(), torch::kCPU);

    auto webgpu = cpu.to(webgpu_device());
    ASSERT_EQ(webgpu.device().type(), torch::DeviceType::PrivateUse1);
}

TEST(ToDevice, WebgpuToCpu)
{
    auto webgpu = torch::ones({3}, torch::TensorOptions().device(webgpu_device()));
    ASSERT_EQ(webgpu.device().type(), torch::DeviceType::PrivateUse1);

    auto cpu = webgpu.to(torch::kCPU);
    ASSERT_EQ(cpu.device().type(), torch::kCPU);
    ASSERT_TRUE(torch::allclose(cpu, torch::ones({3})));
}

TEST(ToDevice, RoundTripPreservesValues)
{
    auto original = torch::tensor({1.5f, -2.7f, 3.14f, 0.0f}, torch::kFloat);

    // CPU -> WebGPU
    auto webgpu = original.to(webgpu_device());
    ASSERT_EQ(webgpu.device().type(), torch::DeviceType::PrivateUse1);

    // WebGPU -> CPU
    auto back_to_cpu = webgpu.to(torch::kCPU);
    ASSERT_EQ(back_to_cpu.device().type(), torch::kCPU);

    ASSERT_TRUE(torch::allclose(back_to_cpu, original));
}

TEST(ToDevice, WebgpuToWebgpuSameDevice)
{
    auto webgpu = torch::ones({3}, torch::TensorOptions().device(webgpu_device()));

    // Same device transfer - should preserve device and values
    auto same = webgpu.to(webgpu_device());
    ASSERT_EQ(same.device().type(), torch::DeviceType::PrivateUse1);
    // Note: _to_copy always copies, so data_ptr might differ
    // Just verify values are preserved
    ASSERT_TRUE(torch::allclose(same.to(torch::kCPU), webgpu.to(torch::kCPU)));
}

TEST(ToDevice, WebgpuToWebgpuWithCopy)
{
    auto webgpu = torch::ones({3}, torch::TensorOptions().device(webgpu_device()));

    // Same device, with copy=true -> should return clone
    auto copy = webgpu.to(webgpu_device(), webgpu.scalar_type(), false, true);
    ASSERT_EQ(copy.device().type(), torch::DeviceType::PrivateUse1);
    // Different storage but same values
    ASSERT_NE(copy.data_ptr(), webgpu.data_ptr());
}

TEST(ToDevice, LargerTensor)
{
    auto cpu = torch::randn({32, 64});

    auto webgpu = cpu.to(webgpu_device());
    ASSERT_EQ(webgpu.device().type(), torch::DeviceType::PrivateUse1);

    auto back = webgpu.to(torch::kCPU);
    ASSERT_TRUE(torch::allclose(back, cpu, 1e-5, 1e-5));
}

TEST(ToDevice, MultiDimensional)
{
    auto cpu = torch::tensor({{{1.0f, 2.0f}, {3.0f, 4.0f}}, {{5.0f, 6.0f}, {7.0f, 8.0f}}});

    auto webgpu = cpu.to(webgpu_device());
    ASSERT_EQ(webgpu.device().type(), torch::DeviceType::PrivateUse1);

    auto back = webgpu.to(torch::kCPU);
    ASSERT_TRUE(torch::allclose(back, cpu));
}
