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

TEST(CopyOps, CopyFromWebgpuToCpuDst)
{
    auto cpu_dst = torch::zeros({3}, torch::kFloat);
    auto src = to_webgpu(torch::tensor({3.0f, 4.0f, 5.0f}));

    // Use public copy_ API instead of internal _copy_from
    cpu_dst.copy_(src);
    ASSERT_TRUE(torch::allclose(cpu_dst, torch::tensor({3.0f, 4.0f, 5.0f})));
}

TEST(CopyOps, CopyFromCpuToWebgpuDst)
{
    auto src = torch::tensor({6.0f, 7.0f}, torch::kFloat);
    auto dst = to_webgpu(torch::zeros_like(src));

    // Use public copy_ API
    dst.copy_(src);
    ASSERT_TRUE(torch::allclose(dst.to(torch::kCPU), src));
}
