#include <gtest/gtest.h>
#include <torch/torch.h>

namespace
{

    torch::Device webgpu_device()
    {
        return torch::Device(torch::DeviceType::PrivateUse1);
    }

} // namespace

TEST(BasicOps, EmptyMemoryFormatCreatesWebgpuTensor)
{
    auto opts = torch::TensorOptions().device(webgpu_device()).dtype(torch::kFloat).memory_format(torch::MemoryFormat::Contiguous);
    auto t = torch::empty({2, 3}, opts);
    ASSERT_EQ(t.device().type(), torch::DeviceType::PrivateUse1);
    ASSERT_EQ(t.sizes(), torch::IntArrayRef({2, 3}));
    ASSERT_EQ(t.dtype(), torch::kFloat);
}
