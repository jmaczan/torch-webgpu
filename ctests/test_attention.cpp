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

TEST(AttentionOps, ScaledDotProductAttentionBasic)
{
    // [batch, heads, seq_len, head_dim]
    auto query_cpu = torch::randn({1, 4, 8, 16});
    auto key_cpu = torch::randn({1, 4, 8, 16});
    auto value_cpu = torch::randn({1, 4, 8, 16});

    auto query = to_webgpu(query_cpu);
    auto key = to_webgpu(key_cpu);
    auto value = to_webgpu(value_cpu);

    auto out = torch::scaled_dot_product_attention(query, key, value);
    auto expected = torch::scaled_dot_product_attention(query_cpu, key_cpu, value_cpu);

    ASSERT_EQ(out.sizes(), expected.sizes());
    ASSERT_TRUE(torch::allclose(out.to(torch::kCPU), expected, 1e-3, 1e-3));
}

TEST(AttentionOps, ScaledDotProductAttentionSmall)
{
    auto query_cpu = torch::randn({1, 2, 4, 8});
    auto key_cpu = torch::randn({1, 2, 4, 8});
    auto value_cpu = torch::randn({1, 2, 4, 8});

    auto query = to_webgpu(query_cpu);
    auto key = to_webgpu(key_cpu);
    auto value = to_webgpu(value_cpu);

    auto out = torch::scaled_dot_product_attention(query, key, value);
    auto expected = torch::scaled_dot_product_attention(query_cpu, key_cpu, value_cpu);

    ASSERT_EQ(out.sizes(), expected.sizes());
    ASSERT_TRUE(torch::allclose(out.to(torch::kCPU), expected, 1e-3, 1e-3));
}

TEST(AttentionOps, ScaledDotProductAttentionBatch)
{
    auto query_cpu = torch::randn({2, 4, 8, 16});
    auto key_cpu = torch::randn({2, 4, 8, 16});
    auto value_cpu = torch::randn({2, 4, 8, 16});

    auto query = to_webgpu(query_cpu);
    auto key = to_webgpu(key_cpu);
    auto value = to_webgpu(value_cpu);

    auto out = torch::scaled_dot_product_attention(query, key, value);
    auto expected = torch::scaled_dot_product_attention(query_cpu, key_cpu, value_cpu);

    ASSERT_EQ(out.sizes(), expected.sizes());
    ASSERT_TRUE(torch::allclose(out.to(torch::kCPU), expected, 1e-3, 1e-3));
}

TEST(AttentionOps, ScaledDotProductAttentionCausal)
{
    auto query_cpu = torch::randn({1, 4, 8, 16});
    auto key_cpu = torch::randn({1, 4, 8, 16});
    auto value_cpu = torch::randn({1, 4, 8, 16});

    auto query = to_webgpu(query_cpu);
    auto key = to_webgpu(key_cpu);
    auto value = to_webgpu(value_cpu);

    // is_causal = true
    auto out = torch::scaled_dot_product_attention(query, key, value, {}, 0.0, true);
    auto expected = torch::scaled_dot_product_attention(query_cpu, key_cpu, value_cpu, {}, 0.0, true);

    ASSERT_EQ(out.sizes(), expected.sizes());
    ASSERT_TRUE(torch::allclose(out.to(torch::kCPU), expected, 1e-3, 1e-3));
}

TEST(AttentionOps, ScaledDotProductAttentionCustomScale)
{
    auto query_cpu = torch::randn({1, 4, 8, 16});
    auto key_cpu = torch::randn({1, 4, 8, 16});
    auto value_cpu = torch::randn({1, 4, 8, 16});

    auto query = to_webgpu(query_cpu);
    auto key = to_webgpu(key_cpu);
    auto value = to_webgpu(value_cpu);

    // Custom scale
    double scale = 0.5;
    auto out = torch::scaled_dot_product_attention(query, key, value, {}, 0.0, false, scale);
    auto expected = torch::scaled_dot_product_attention(query_cpu, key_cpu, value_cpu, {}, 0.0, false, scale);

    ASSERT_EQ(out.sizes(), expected.sizes());
    ASSERT_TRUE(torch::allclose(out.to(torch::kCPU), expected, 1e-3, 1e-3));
}
