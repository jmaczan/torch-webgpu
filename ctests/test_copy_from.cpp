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

TEST(CopyOps, CopyFromWebgpuToCpuDst)
{
    auto cpu_dst = torch::zeros({3}, torch::kFloat);
    auto src = to_webgpu(torch::tensor({3.0f, 4.0f, 5.0f}));

    auto result = at::_copy_from(src, cpu_dst, false);
    ASSERT_TRUE(torch::allclose(result, torch::tensor({3.0f, 4.0f, 5.0f})));
    ASSERT_TRUE(torch::allclose(cpu_dst, result));
}

TEST(CopyOps, CopyFromCpuToWebgpuDst)
{
    auto src = torch::tensor({6.0f, 7.0f}, torch::kFloat);
    auto dst = to_webgpu(torch::zeros_like(src));

    auto result = at::_copy_from(src, dst, false);
    ASSERT_TRUE(torch::allclose(result.to(torch::kCPU), src));
}
