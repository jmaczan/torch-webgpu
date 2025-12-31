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

TEST(BasicOps, ReshapeHandlesMinusOne)
{
    auto cpu = torch::arange(12, torch::kFloat).reshape({3, 4});
    auto x = to_webgpu(cpu);

    auto reshaped = x.reshape({-1, 2});
    ASSERT_EQ(reshaped.sizes(), torch::IntArrayRef({6, 2}));
    ASSERT_TRUE(torch::allclose(reshaped.to(torch::kCPU), cpu.reshape({6, 2})));
}
