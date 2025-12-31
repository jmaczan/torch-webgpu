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

TEST(ArithmeticOps, MmSmallMatrices)
{
    auto a_cpu = torch::tensor({{1.0f, 2.0f}, {3.0f, 4.0f}});
    auto b_cpu = torch::tensor({{5.0f, 6.0f}, {7.0f, 8.0f}});
    auto a = to_webgpu(a_cpu);
    auto b = to_webgpu(b_cpu);

    auto out = torch::mm(a, b);
    auto expected = torch::mm(a_cpu, b_cpu);
    ASSERT_TRUE(torch::allclose(out.to(torch::kCPU), expected, 1e-4, 1e-4));

    auto out_buf = torch::zeros({2, 2}, a.options());
    torch::mm_out(out_buf, a, b);
    ASSERT_TRUE(torch::allclose(out_buf.to(torch::kCPU), expected, 1e-4, 1e-4));
}
