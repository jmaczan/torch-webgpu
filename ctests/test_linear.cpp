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

// Linear tests
TEST(LinearOps, Linear2D)
{
    auto input_cpu = torch::randn({4, 8});
    auto weight_cpu = torch::randn({16, 8});
    auto bias_cpu = torch::randn({16});

    auto input = to_webgpu(input_cpu);
    auto weight = to_webgpu(weight_cpu);
    auto bias = to_webgpu(bias_cpu);

    auto out = torch::linear(input, weight, bias);
    auto expected = torch::linear(input_cpu, weight_cpu, bias_cpu);
    ASSERT_EQ(out.sizes(), expected.sizes());
    ASSERT_TRUE(torch::allclose(out.to(torch::kCPU), expected, 1e-3, 1e-3));
}

TEST(LinearOps, LinearNoBias)
{
    auto input_cpu = torch::randn({4, 8});
    auto weight_cpu = torch::randn({16, 8});

    auto input = to_webgpu(input_cpu);
    auto weight = to_webgpu(weight_cpu);

    auto out = torch::linear(input, weight);
    auto expected = torch::linear(input_cpu, weight_cpu);
    ASSERT_EQ(out.sizes(), expected.sizes());
    ASSERT_TRUE(torch::allclose(out.to(torch::kCPU), expected, 1e-3, 1e-3));
}

TEST(LinearOps, Linear3D)
{
    auto input_cpu = torch::randn({2, 4, 8});
    auto weight_cpu = torch::randn({16, 8});
    auto bias_cpu = torch::randn({16});

    auto input = to_webgpu(input_cpu);
    auto weight = to_webgpu(weight_cpu);
    auto bias = to_webgpu(bias_cpu);

    auto out = torch::linear(input, weight, bias);
    auto expected = torch::linear(input_cpu, weight_cpu, bias_cpu);
    ASSERT_EQ(out.sizes(), expected.sizes());
    ASSERT_TRUE(torch::allclose(out.to(torch::kCPU), expected, 1e-3, 1e-3));
}

// Addmm tests
TEST(LinearOps, Addmm)
{
    auto m_cpu = torch::randn({4, 16});
    auto mat1_cpu = torch::randn({4, 8});
    auto mat2_cpu = torch::randn({8, 16});

    auto m = to_webgpu(m_cpu);
    auto mat1 = to_webgpu(mat1_cpu);
    auto mat2 = to_webgpu(mat2_cpu);

    auto out = torch::addmm(m, mat1, mat2);
    auto expected = torch::addmm(m_cpu, mat1_cpu, mat2_cpu);
    ASSERT_EQ(out.sizes(), expected.sizes());
    ASSERT_TRUE(torch::allclose(out.to(torch::kCPU), expected, 1e-3, 1e-3));
}

TEST(LinearOps, AddmmWithAlphaBeta)
{
    auto m_cpu = torch::randn({4, 16});
    auto mat1_cpu = torch::randn({4, 8});
    auto mat2_cpu = torch::randn({8, 16});

    auto m = to_webgpu(m_cpu);
    auto mat1 = to_webgpu(mat1_cpu);
    auto mat2 = to_webgpu(mat2_cpu);

    auto out = torch::addmm(m, mat1, mat2, 0.5, 2.0);
    auto expected = torch::addmm(m_cpu, mat1_cpu, mat2_cpu, 0.5, 2.0);
    ASSERT_EQ(out.sizes(), expected.sizes());
    ASSERT_TRUE(torch::allclose(out.to(torch::kCPU), expected, 1e-3, 1e-3));
}

// BMM tests
TEST(LinearOps, Bmm)
{
    auto a_cpu = torch::randn({4, 8, 16});
    auto b_cpu = torch::randn({4, 16, 32});

    auto a = to_webgpu(a_cpu);
    auto b = to_webgpu(b_cpu);

    auto out = torch::bmm(a, b);
    auto expected = torch::bmm(a_cpu, b_cpu);
    ASSERT_EQ(out.sizes(), expected.sizes());
    ASSERT_TRUE(torch::allclose(out.to(torch::kCPU), expected, 1e-3, 1e-3));
}

TEST(LinearOps, BmmSmall)
{
    auto a_cpu = torch::randn({2, 3, 4});
    auto b_cpu = torch::randn({2, 4, 5});

    auto a = to_webgpu(a_cpu);
    auto b = to_webgpu(b_cpu);

    auto out = torch::bmm(a, b);
    auto expected = torch::bmm(a_cpu, b_cpu);
    ASSERT_EQ(out.sizes(), expected.sizes());
    ASSERT_TRUE(torch::allclose(out.to(torch::kCPU), expected, 1e-3, 1e-3));
}

// Matmul tests
TEST(LinearOps, MatmulVectorVector)
{
    auto a_cpu = torch::randn({8});
    auto b_cpu = torch::randn({8});

    auto a = to_webgpu(a_cpu);
    auto b = to_webgpu(b_cpu);

    auto out = torch::matmul(a, b);
    auto expected = torch::matmul(a_cpu, b_cpu);
    ASSERT_EQ(out.sizes(), expected.sizes());
    ASSERT_TRUE(torch::allclose(out.to(torch::kCPU), expected, 1e-3, 1e-3));
}

TEST(LinearOps, MatmulMatrixVector)
{
    auto a_cpu = torch::randn({4, 8});
    auto b_cpu = torch::randn({8});

    auto a = to_webgpu(a_cpu);
    auto b = to_webgpu(b_cpu);

    auto out = torch::matmul(a, b);
    auto expected = torch::matmul(a_cpu, b_cpu);
    ASSERT_EQ(out.sizes(), expected.sizes());
    ASSERT_TRUE(torch::allclose(out.to(torch::kCPU), expected, 1e-3, 1e-3));
}

TEST(LinearOps, MatmulMatrixMatrix)
{
    auto a_cpu = torch::randn({4, 8});
    auto b_cpu = torch::randn({8, 16});

    auto a = to_webgpu(a_cpu);
    auto b = to_webgpu(b_cpu);

    auto out = torch::matmul(a, b);
    auto expected = torch::matmul(a_cpu, b_cpu);
    ASSERT_EQ(out.sizes(), expected.sizes());
    ASSERT_TRUE(torch::allclose(out.to(torch::kCPU), expected, 1e-3, 1e-3));
}

TEST(LinearOps, MatmulBatched)
{
    auto a_cpu = torch::randn({2, 4, 8});
    auto b_cpu = torch::randn({2, 8, 16});

    auto a = to_webgpu(a_cpu);
    auto b = to_webgpu(b_cpu);

    auto out = torch::matmul(a, b);
    auto expected = torch::matmul(a_cpu, b_cpu);
    ASSERT_EQ(out.sizes(), expected.sizes());
    ASSERT_TRUE(torch::allclose(out.to(torch::kCPU), expected, 1e-3, 1e-3));
}

TEST(LinearOps, Matmul4D)
{
    auto a_cpu = torch::randn({2, 3, 4, 8});
    auto b_cpu = torch::randn({2, 3, 8, 16});

    auto a = to_webgpu(a_cpu);
    auto b = to_webgpu(b_cpu);

    auto out = torch::matmul(a, b);
    auto expected = torch::matmul(a_cpu, b_cpu);
    ASSERT_EQ(out.sizes(), expected.sizes());
    ASSERT_TRUE(torch::allclose(out.to(torch::kCPU), expected, 1e-3, 1e-3));
}
