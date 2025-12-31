#include <gtest/gtest.h>
#include <torch/torch.h>

namespace
{

    torch::Device webgpu_device()
    {
        return torch::Device(torch::DeviceType::PrivateUse1);
    }

} // namespace

TEST(BasicOps, ResizeChangesShape)
{
    auto opts = torch::TensorOptions().device(webgpu_device()).dtype(torch::kFloat);
    auto t = torch::zeros({2, 2}, opts);
    t.resize_({2, 4});
    ASSERT_EQ(t.sizes(), torch::IntArrayRef({2, 4}));
}
