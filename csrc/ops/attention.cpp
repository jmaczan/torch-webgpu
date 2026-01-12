#include <ATen/ATen.h>
#include <torch/library.h>
#include <webgpu/webgpu_cpp.h>
#include <cmath>
#include "core/webgpu_context.h"
#include "core/webgpu_allocator.h"

namespace torch_webgpu
{
    namespace ops
    {
        // Scaled Dot-Product Attention
        // query: [batch, heads, seq_len_q, head_dim]
        // key:   [batch, heads, seq_len_k, head_dim]
        // value: [batch, heads, seq_len_k, head_dim]
        // attn_mask: optional [batch, heads, seq_len_q, seq_len_k] or broadcastable
        // output: [batch, heads, seq_len_q, head_dim]
        at::Tensor scaled_dot_product_attention(
            const at::Tensor &query,
            const at::Tensor &key,
            const at::Tensor &value,
            const c10::optional<at::Tensor> &attn_mask,
            double dropout_p,
            bool is_causal,
            c10::optional<double> scale,
            bool enable_gqa)
        {
            TORCH_CHECK(query.device().type() == c10::DeviceType::PrivateUse1);
            TORCH_CHECK(key.device().type() == c10::DeviceType::PrivateUse1);
            TORCH_CHECK(value.device().type() == c10::DeviceType::PrivateUse1);
            TORCH_CHECK(dropout_p == 0.0, "sdpa: dropout not supported on WebGPU yet");

            // Handle GQA (Grouped Query Attention) - expand key/value heads to match query heads
            // Qwen uses 14 query heads and 2 key-value heads
            auto num_heads_q = query.size(1);
            auto num_heads_kv = key.size(1);
            at::Tensor k = key;
            at::Tensor v = value;

            if (num_heads_q != num_heads_kv)
            {
                TORCH_CHECK(num_heads_q % num_heads_kv == 0,
                            "sdpa: query heads must be divisible by key/value heads for GQA");
                auto n_rep = num_heads_q / num_heads_kv;
                auto batch = key.size(0);
                auto seq_len = key.size(2);
                auto head_dim_size = key.size(3);

                // Expand key: [batch, kv_heads, seq, dim] -> [batch, kv_heads, n_rep, seq, dim] -> [batch, q_heads, seq, dim]
                // Note: expand creates non-contiguous tensor, need contiguous() before reshape
                k = key.unsqueeze(2).expand({batch, num_heads_kv, n_rep, seq_len, head_dim_size}).contiguous().reshape({batch, num_heads_q, seq_len, head_dim_size});
                v = value.unsqueeze(2).expand({batch, num_heads_kv, n_rep, seq_len, head_dim_size}).contiguous().reshape({batch, num_heads_q, seq_len, head_dim_size});
            }

            // Compute attention scale
            double head_dim = static_cast<double>(query.size(-1));
            float attn_scale_f = static_cast<float>(scale.value_or(1.0 / std::sqrt(head_dim)));

            // Q @ K^T -> [batch, heads, seq_len_q, seq_len_k]
            auto key_t = k.transpose(-2, -1);
            auto attn_weights = at::matmul(query, key_t);

            // Scale with matching dtype
            auto scale_tensor = at::scalar_tensor(attn_scale_f, attn_weights.options());
            attn_weights = attn_weights * scale_tensor;

            // Apply causal mask if requested
            if (is_causal)
            {
                auto seq_len_q = query.size(-2);
                auto seq_len_k = k.size(-2);

                // Create causal mask on CPU and move to device
                auto mask = at::ones({seq_len_q, seq_len_k}, at::TensorOptions().dtype(at::kFloat));
                mask = at::triu(mask, 1); // Upper triangular excluding diagonal

                // Multiply by large negative value with correct dtype
                auto neg_inf = at::scalar_tensor(-1e9f, mask.options());
                mask = mask * neg_inf;

                // Move to device
                mask = mask.to(query.device());

                attn_weights = attn_weights + mask;
            }

            // Apply attention mask if provided
            if (attn_mask.has_value() && attn_mask->defined())
            {
                auto mask = attn_mask.value();
                if (mask.scalar_type() == at::kBool)
                {
                    // Boolean mask: True means masked (should be -inf)
                    auto float_mask = mask.to(at::kFloat);
                    auto one_tensor = at::scalar_tensor(1.0f, float_mask.options());
                    auto neg_inf = at::scalar_tensor(-1e9f, float_mask.options());
                    float_mask = (one_tensor - float_mask) * neg_inf;
                    attn_weights = attn_weights + float_mask;
                }
                else
                {
                    // Additive mask
                    attn_weights = attn_weights + mask;
                }
            }

            // Softmax over last dimension
            attn_weights = at::softmax(attn_weights, -1, c10::nullopt);

            // Attention weights @ V -> [batch, heads, seq_len_q, head_dim]
            auto output = at::matmul(attn_weights, v);

            return output;
        }
    }

    TORCH_LIBRARY_IMPL(aten, PrivateUse1, m)
    {
        m.impl("scaled_dot_product_attention", TORCH_FN(ops::scaled_dot_product_attention));
    }
}
