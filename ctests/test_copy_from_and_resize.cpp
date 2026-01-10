#include <gtest/gtest.h>
#include <torch/torch.h>
#include <torch/extension.h>
#include <torch/library.h>
#include <ATen/ATen.h>

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

TEST(CopyOps, CopyFromAndResize)
{
    auto src = torch::tensor({1.0f, 2.0f, 3.0f, 4.0f});
    // Create dst tensor on webgpu and resize it to match src
    auto dst = to_webgpu(torch::zeros({4}, torch::kFloat));
    dst.copy_(src);
    ASSERT_EQ(dst.sizes(), torch::IntArrayRef({4}));
    ASSERT_TRUE(torch::allclose(dst.to(torch::kCPU), src));
}
