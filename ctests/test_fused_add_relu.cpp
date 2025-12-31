// #pragma once
// #include <gtest/gtest.h>
// #include <torch/torch.h>
// #include <torch/extension.h>
// #include <torch/library.h>
// #include <ATen/ATen.h>
// #include <torch/library.h>

// TEST(WebgpuOps, FusedAddRelu)
// {
//     auto a_cpu = torch::tensor({-1.0f, 2.0f, -3.0f, 4.0f}).reshape({2, 2});
//     auto b_cpu = torch::tensor({1.0f, -2.0f, 3.0f, -4.0f}).reshape({2, 2});
//     auto a = a_cpu.to(torch::Device(torch::DeviceType::PrivateUse1));
//     auto b = b_cpu.to(torch::Device(torch::DeviceType::PrivateUse1));

//     auto out = torch::ops::webgpu::fused_add_relu(a, b);
//     auto expected = torch::relu(a_cpu + b_cpu);
//     ASSERT_TRUE(torch::allclose(out.to(torch::kCPU), expected));
// }
