// #pragma once
// #include <gtest/gtest.h>
// #include <torch/torch.h>
// #include <torch/extension.h>
// #include <torch/library.h>
// #include <ATen/ATen.h>
// #include <torch/library.h>

// TEST(WebgpuOps, WriteBufferCopiesData)
// {
//     std::vector<int64_t> size{2, 2};
//     std::vector<int64_t> stride{2, 1};
//     auto buf = torch::ops::webgpu::create_buffer(size, stride, torch::kFloat);

//     auto src_cpu = torch::tensor({1.0f, 2.0f, 3.0f, 4.0f}).reshape({2, 2});
//     auto src = src_cpu.to(torch::Device(torch::DeviceType::PrivateUse1));

//     auto written = torch::ops::webgpu::write_buffer(buf, src);
//     ASSERT_TRUE(torch::allclose(written.to(torch::kCPU), src_cpu));
// }
