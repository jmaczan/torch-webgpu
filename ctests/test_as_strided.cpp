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

TEST(BasicOps, AsStridedSubview)
{
    auto base_cpu = torch::arange(10, torch::kFloat);
    auto base = to_webgpu(base_cpu);

    std::vector<int64_t> size{3, 2};
    std::vector<int64_t> stride{2, 1};
    auto view = base.as_strided(size, stride, 2);
    auto expected = base_cpu.as_strided(size, stride, 2);
    ASSERT_TRUE(torch::allclose(view.to(torch::kCPU), expected));
}
