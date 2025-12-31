#include <gtest/gtest.h>
#include <torch/torch.h>

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

TEST(BasicOps, ViewSharesStorage)
{
    auto base_cpu = torch::arange(6, torch::kFloat).reshape({2, 3});
    auto base = to_webgpu(base_cpu);

    auto v = base.view({3, 2});
    ASSERT_EQ(v.sizes(), torch::IntArrayRef({3, 2}));
    ASSERT_EQ(v.data_ptr(), base.data_ptr());
    ASSERT_TRUE(torch::allclose(v.to(torch::kCPU), base_cpu.view({3, 2})));
}
