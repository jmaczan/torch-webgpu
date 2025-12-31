// #pragma once
// #include <gtest/gtest.h>
// #include <torch/torch.h>
// #include <torch/extension.h>
// #include <torch/library.h>
// #include <ATen/ATen.h>
// #include <torch/library.h>

// TEST(WebgpuOps, CreateBufferHasShapeAndStride)
// {
//     std::vector<int64_t> size{2, 2};
//     std::vector<int64_t> stride{2, 1};
//     auto t = torch::ops::webgpu::create_buffer(size, stride, torch::kFloat);
//     ASSERT_EQ(t.device().type(), torch::DeviceType::PrivateUse1);
//     ASSERT_EQ(t.sizes(), torch::IntArrayRef(size));
//     ASSERT_EQ(t.strides(), torch::IntArrayRef(stride));
//     ASSERT_EQ(t.dtype(), torch::kFloat);
// }
