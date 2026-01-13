/**
 * Tests for MoE (Mixture of Experts) operations.
 * These ops are essential for running MoE models like Llama-4-Scout.
 */

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

// =====================================================
// topk tests
// =====================================================

TEST(MoEOps, TopkBasic)
{
    auto cpu_input = torch::tensor({3.0f, 1.0f, 4.0f, 1.0f, 5.0f, 9.0f, 2.0f, 6.0f});
    auto input = to_webgpu(cpu_input);

    auto [values, indices] = torch::topk(input, 3);
    auto [expected_values, expected_indices] = torch::topk(cpu_input, 3);

    ASSERT_TRUE(torch::allclose(values.to(torch::kCPU), expected_values));
    ASSERT_TRUE(torch::equal(indices.to(torch::kCPU), expected_indices));
}

TEST(MoEOps, TopkWithDim)
{
    auto cpu_input = torch::tensor({{3.0f, 1.0f, 4.0f}, {1.0f, 5.0f, 9.0f}});
    auto input = to_webgpu(cpu_input);

    auto [values, indices] = torch::topk(input, 2, /*dim=*/1);
    auto [expected_values, expected_indices] = torch::topk(cpu_input, 2, /*dim=*/1);

    ASSERT_TRUE(torch::allclose(values.to(torch::kCPU), expected_values));
    ASSERT_TRUE(torch::equal(indices.to(torch::kCPU), expected_indices));
}

TEST(MoEOps, TopkSmallest)
{
    auto cpu_input = torch::tensor({3.0f, 1.0f, 4.0f, 1.0f, 5.0f, 9.0f});
    auto input = to_webgpu(cpu_input);

    auto [values, indices] = torch::topk(input, 2, /*dim=*/-1, /*largest=*/false);
    auto [expected_values, expected_indices] = torch::topk(cpu_input, 2, /*dim=*/-1, /*largest=*/false);

    ASSERT_TRUE(torch::allclose(values.to(torch::kCPU), expected_values));
    ASSERT_TRUE(torch::equal(indices.to(torch::kCPU), expected_indices));
}

// =====================================================
// scatter tests
// =====================================================

TEST(MoEOps, ScatterSrc)
{
    auto cpu_self = torch::zeros({3, 5});
    auto cpu_index = torch::tensor({{0, 1, 2, 0, 0}, {2, 0, 0, 1, 2}}, torch::kLong);
    auto cpu_src = torch::tensor({{1.0f, 2.0f, 3.0f, 4.0f, 5.0f}, {6.0f, 7.0f, 8.0f, 9.0f, 10.0f}});

    auto self = to_webgpu(cpu_self);
    auto index = to_webgpu(cpu_index);
    auto src = to_webgpu(cpu_src);

    auto out = self.scatter(0, index, src);
    auto expected = cpu_self.scatter(0, cpu_index, cpu_src);

    ASSERT_TRUE(torch::allclose(out.to(torch::kCPU), expected));
}

TEST(MoEOps, ScatterValue)
{
    auto cpu_self = torch::zeros({3, 5});
    auto cpu_index = torch::tensor({{0, 1, 2, 0, 0}, {2, 0, 0, 1, 2}}, torch::kLong);

    auto self = to_webgpu(cpu_self);
    auto index = to_webgpu(cpu_index);

    auto out = self.scatter(0, index, 1.0);
    auto expected = cpu_self.scatter(0, cpu_index, 1.0);

    ASSERT_TRUE(torch::allclose(out.to(torch::kCPU), expected));
}

// =====================================================
// scatter_add tests
// =====================================================

TEST(MoEOps, ScatterAdd)
{
    auto cpu_self = torch::zeros({3, 5});
    auto cpu_index = torch::tensor({{0, 1, 2, 0, 0}, {2, 0, 0, 1, 2}}, torch::kLong);
    auto cpu_src = torch::ones({2, 5});

    auto self = to_webgpu(cpu_self);
    auto index = to_webgpu(cpu_index);
    auto src = to_webgpu(cpu_src);

    auto out = self.scatter_add(0, index, src);
    auto expected = cpu_self.scatter_add(0, cpu_index, cpu_src);

    ASSERT_TRUE(torch::allclose(out.to(torch::kCPU), expected));
}

// =====================================================
// any tests
// =====================================================

TEST(MoEOps, AnyAll)
{
    auto cpu_input = torch::tensor({false, false, true, false});
    auto input = to_webgpu(cpu_input);

    auto out = torch::any(input);
    auto expected = torch::any(cpu_input);

    ASSERT_TRUE(torch::equal(out.to(torch::kCPU), expected));
}

TEST(MoEOps, AnyDim)
{
    auto cpu_input = torch::tensor({{false, true, false}, {false, false, false}});
    auto input = to_webgpu(cpu_input);

    auto out = torch::any(input, 1);
    auto expected = torch::any(cpu_input, 1);

    ASSERT_TRUE(torch::equal(out.to(torch::kCPU), expected));
}

TEST(MoEOps, AnyNone)
{
    auto cpu_input = torch::tensor({false, false, false});
    auto input = to_webgpu(cpu_input);

    auto out = torch::any(input);
    auto expected = torch::any(cpu_input);

    ASSERT_TRUE(torch::equal(out.to(torch::kCPU), expected));
}

// =====================================================
// all tests
// =====================================================

TEST(MoEOps, AllTrue)
{
    auto cpu_input = torch::tensor({true, true, true, true});
    auto input = to_webgpu(cpu_input);

    auto out = torch::all(input);
    auto expected = torch::all(cpu_input);

    ASSERT_TRUE(torch::equal(out.to(torch::kCPU), expected));
}

TEST(MoEOps, AllFalse)
{
    auto cpu_input = torch::tensor({true, true, false, true});
    auto input = to_webgpu(cpu_input);

    auto out = torch::all(input);
    auto expected = torch::all(cpu_input);

    ASSERT_TRUE(torch::equal(out.to(torch::kCPU), expected));
}

TEST(MoEOps, AllDim)
{
    auto cpu_input = torch::tensor({{true, true, true}, {true, false, true}});
    auto input = to_webgpu(cpu_input);

    auto out = torch::all(input, 1);
    auto expected = torch::all(cpu_input, 1);

    ASSERT_TRUE(torch::equal(out.to(torch::kCPU), expected));
}

// =====================================================
// nonzero tests
// =====================================================

TEST(MoEOps, Nonzero)
{
    auto cpu_input = torch::tensor({0.0f, 1.0f, 0.0f, 2.0f, 0.0f});
    auto input = to_webgpu(cpu_input);

    auto out = torch::nonzero(input);
    auto expected = torch::nonzero(cpu_input);

    ASSERT_TRUE(torch::equal(out.to(torch::kCPU), expected));
}

TEST(MoEOps, Nonzero2D)
{
    auto cpu_input = torch::tensor({{0.0f, 1.0f, 0.0f}, {2.0f, 0.0f, 3.0f}});
    auto input = to_webgpu(cpu_input);

    auto out = torch::nonzero(input);
    auto expected = torch::nonzero(cpu_input);

    ASSERT_TRUE(torch::equal(out.to(torch::kCPU), expected));
}

// =====================================================
// masked_select tests
// =====================================================

TEST(MoEOps, MaskedSelect)
{
    auto cpu_input = torch::tensor({1.0f, 2.0f, 3.0f, 4.0f, 5.0f});
    auto cpu_mask = torch::tensor({true, false, true, false, true});

    auto input = to_webgpu(cpu_input);
    auto mask = to_webgpu(cpu_mask);

    auto out = torch::masked_select(input, mask);
    auto expected = torch::masked_select(cpu_input, cpu_mask);

    ASSERT_TRUE(torch::allclose(out.to(torch::kCPU), expected));
}

TEST(MoEOps, MaskedSelect2D)
{
    auto cpu_input = torch::tensor({{1.0f, 2.0f, 3.0f}, {4.0f, 5.0f, 6.0f}});
    auto cpu_mask = torch::tensor({{true, false, true}, {false, true, false}});

    auto input = to_webgpu(cpu_input);
    auto mask = to_webgpu(cpu_mask);

    auto out = torch::masked_select(input, mask);
    auto expected = torch::masked_select(cpu_input, cpu_mask);

    ASSERT_TRUE(torch::allclose(out.to(torch::kCPU), expected));
}

// =====================================================
// MoE routing simulation test
// =====================================================

TEST(MoEOps, MoERoutingSimulation)
{
    // Simulate basic MoE routing:
    // 1. Router produces logits
    // 2. topk selects top experts
    // 3. scatter routes tokens

    // Router logits for 4 tokens across 4 experts
    auto cpu_logits = torch::tensor({
        {0.1f, 0.9f, 0.3f, 0.2f},  // Token 0: expert 1 best
        {0.8f, 0.1f, 0.2f, 0.3f},  // Token 1: expert 0 best
        {0.2f, 0.3f, 0.1f, 0.9f},  // Token 2: expert 3 best
        {0.4f, 0.1f, 0.8f, 0.2f},  // Token 3: expert 2 best
    });
    auto logits = to_webgpu(cpu_logits);

    // Select top-2 experts per token
    auto [values, indices] = torch::topk(logits, 2, /*dim=*/1);

    auto cpu_values = values.to(torch::kCPU);
    auto cpu_indices = indices.to(torch::kCPU);

    // Verify shapes
    ASSERT_EQ(cpu_values.size(0), 4);
    ASSERT_EQ(cpu_values.size(1), 2);
    ASSERT_EQ(cpu_indices.size(0), 4);
    ASSERT_EQ(cpu_indices.size(1), 2);

    // Verify that values are the correct top-k values
    auto [expected_values, expected_indices] = torch::topk(cpu_logits, 2, /*dim=*/1);
    ASSERT_TRUE(torch::allclose(cpu_values, expected_values));
    ASSERT_TRUE(torch::equal(cpu_indices, expected_indices));
}
