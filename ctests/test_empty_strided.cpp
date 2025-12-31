#include <gtest/gtest.h>
#include <torch/torch.h>

namespace
{

    torch::Device webgpu_device()
    {
        return torch::Device(torch::DeviceType::PrivateUse1);
    }

} // namespace

TEST(BasicOps, EmptyStridedRespectsStride)
{
    auto opts = torch::TensorOptions().device(webgpu_device()).dtype(torch::kFloat);
    std::vector<int64_t> size{2, 3};
    std::vector<int64_t> stride{3, 1};
    auto t = torch::empty_strided(size, stride, opts);
    ASSERT_EQ(t.device().type(), torch::DeviceType::PrivateUse1);
    ASSERT_EQ(t.sizes(), torch::IntArrayRef(size));
    ASSERT_EQ(t.strides(), torch::IntArrayRef(stride));
}
