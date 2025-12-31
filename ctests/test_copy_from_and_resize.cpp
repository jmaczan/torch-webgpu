#pragma once
#include <gtest/gtest.h>
#include <torch/torch.h>
#include <torch/extension.h>
#include <torch/library.h>
#include <ATen/ATen.h>
#include <torch/library.h>

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
    auto dst = to_webgpu(torch::zeros({2}, torch::kFloat));
    auto result = at::_copy_from_and_resize(src, dst);
    ASSERT_EQ(result.sizes(), torch::IntArrayRef({4}));
    ASSERT_TRUE(torch::allclose(result.to(torch::kCPU), src));
}
