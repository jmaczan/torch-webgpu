// torch.ops.webgpu.* custom operators used by the torch_webgpu compiler.

#include <ATen/ATen.h>
#include <ATen/TensorIterator.h>
#include <torch/library.h>

#include "kernels/elementwise.h"
#include "kernels/math.h"
#include "kernels/matmul.h"
#include "ops/common.h"

namespace torch_webgpu::ops
{
    namespace
    {
        at::Tensor create_buffer(c10::IntArrayRef size, c10::IntArrayRef stride, std::optional<at::ScalarType> dtype)
        {
            return at::empty_strided(size, stride,
                                     at::TensorOptions().dtype(dtype).device(c10::DeviceType::PrivateUse1));
        }

        at::Tensor &write_buffer(at::Tensor &self, const at::Tensor &src)
        {
            self.copy_(src);
            return self;
        }

        at::Tensor fused_add_relu(const at::Tensor &self, const at::Tensor &other)
        {
            at::Tensor out;
            auto iter = at::TensorIterator::binary_op(out, self, other);
            kernels::ElementwiseOp op;
            op.name = "fused_add_relu";
            op.in_ct = kernels::common_ct(iter);
            op.out_ct = op.in_ct;
            TORCH_CHECK(op.in_ct != kernels::CT::Bool, "fused_add_relu does not support bool tensors");
            op.code = op.in_ct == kernels::CT::F32   ? "select(max(x0 + x1, 0.0), x0 + x1, isnan_f(x0 + x1))"
                      : op.in_ct == kernels::CT::I64 ? "i64_max(i64_add(x0, x1), i64(0u, 0u))"
                                                     : "max(x0 + x1, 0)";
            op.math = true;
            kernels::run_elementwise(iter, op);
            return iter.output();
        }

        // Reference / fallback: one linear per weight into its column block.
        at::Tensor multi_linear_ref(const at::Tensor &x, at::TensorList weights,
                                    const c10::List<std::optional<at::Tensor>> &biases)
        {
            std::vector<at::Tensor> outs;
            for (size_t j = 0; j < weights.size(); ++j)
            {
                std::optional<at::Tensor> b = biases.get(j);
                outs.push_back(at::linear(x, weights[j], b));
            }
            return at::cat(outs, -1);
        }

        at::Tensor multi_linear(const at::Tensor &x, at::TensorList weights,
                                const c10::List<std::optional<at::Tensor>> &biases)
        {
            TORCH_CHECK(weights.size() == biases.size(), "multi_linear: one bias (or None) per weight");
            int64_t total = 0;
            for (const auto &w : weights)
                total += w.size(0);
            if (x.dim() == 2)
            {
                at::Tensor out = at::empty({x.size(0), total}, x.options());
                std::vector<at::Tensor> ws(weights.begin(), weights.end()), bs;
                for (size_t j = 0; j < biases.size(); ++j)
                {
                    std::optional<at::Tensor> b = biases.get(j);
                    bs.push_back(b.has_value() ? *b : at::Tensor());
                }
                if (kernels::run_multi_gemv(x, ws, bs, out))
                    return out;
            }
            return multi_linear_ref(x, weights, biases);
        }

        // Decode fusion (see _inductor/fx_passes.py): RMSNorm of a (+ b)
        // feeding linears, SiLU(gate) * up feeding linears. The references
        // round intermediates to the input dtype like the unfused graphs.
        std::tuple<at::Tensor, at::Tensor> rms_norm_multi_linear_ref(const at::Tensor &a, const std::optional<at::Tensor> &b,
                                                                     const at::Tensor &norm_weight, double eps,
                                                                     at::TensorList weights,
                                                                     const c10::List<std::optional<at::Tensor>> &biases)
        {
            at::Tensor h = b.has_value() ? at::add(a, *b) : a;
            at::Tensor hf = h.to(at::kFloat);
            at::Tensor xn = (hf * at::rsqrt(hf.pow(2).mean({-1}, true) + eps)).to(a.scalar_type());
            at::Tensor y = multi_linear_ref(at::mul(norm_weight, xn), weights, biases);
            return {y, b.has_value() ? h : at::empty({0}, a.options())};
        }

        at::Tensor silu_mul_multi_linear_ref(const at::Tensor &gate, const at::Tensor &up, at::TensorList weights,
                                             const c10::List<std::optional<at::Tensor>> &biases)
        {
            at::Tensor x = at::mul(at::silu(gate.to(at::kFloat)).to(gate.scalar_type()), up);
            return multi_linear_ref(x, weights, biases);
        }

        std::vector<at::Tensor> unpack(at::TensorList weights, const c10::List<std::optional<at::Tensor>> &biases,
                                       std::vector<at::Tensor> &bs)
        {
            for (size_t j = 0; j < biases.size(); ++j)
            {
                std::optional<at::Tensor> b = biases.get(j);
                bs.push_back(b.has_value() ? *b : at::Tensor());
            }
            return std::vector<at::Tensor>(weights.begin(), weights.end());
        }

        int64_t total_rows(at::TensorList weights)
        {
            int64_t total = 0;
            for (const auto &w : weights)
                total += w.size(0);
            return total;
        }

        std::tuple<at::Tensor, at::Tensor> rms_norm_multi_linear(const at::Tensor &a, const std::optional<at::Tensor> &b,
                                                                 const at::Tensor &norm_weight, double eps,
                                                                 at::TensorList weights,
                                                                 const c10::List<std::optional<at::Tensor>> &biases)
        {
            TORCH_CHECK(weights.size() == biases.size(), "rms_norm_multi_linear: one bias (or None) per weight");
            if (a.dim() == 2)
            {
                kernels::GemvPrologue pro;
                pro.kind = 1;
                pro.a = a;
                pro.b = b.has_value() ? *b : at::Tensor();
                pro.norm_weight = norm_weight;
                pro.eps = eps;
                pro.h = b.has_value() ? at::empty(a.sizes(), a.options()) : at::empty({0}, a.options());
                at::Tensor out = at::empty({a.size(0), total_rows(weights)}, a.options());
                std::vector<at::Tensor> bs;
                auto ws = unpack(weights, biases, bs);
                if (kernels::run_multi_gemv_prologue(pro, ws, bs, out))
                    return {out, pro.h};
            }
            return rms_norm_multi_linear_ref(a, b, norm_weight, eps, weights, biases);
        }

        at::Tensor silu_mul_multi_linear(const at::Tensor &gate, const at::Tensor &up, at::TensorList weights,
                                         const c10::List<std::optional<at::Tensor>> &biases)
        {
            TORCH_CHECK(weights.size() == biases.size(), "silu_mul_multi_linear: one bias (or None) per weight");
            if (gate.dim() == 2)
            {
                kernels::GemvPrologue pro;
                pro.kind = 2;
                pro.a = gate;
                pro.b = up;
                at::Tensor out = at::empty({gate.size(0), total_rows(weights)}, gate.options());
                std::vector<at::Tensor> bs;
                auto ws = unpack(weights, biases, bs);
                if (kernels::run_multi_gemv_prologue(pro, ws, bs, out))
                    return out;
            }
            return silu_mul_multi_linear_ref(gate, up, weights, biases);
        }
    }

    TORCH_LIBRARY(webgpu, m)
    {
        m.def("create_buffer(int[] size, int[] stride, ScalarType? dtype=None) -> Tensor");
        m.def("write_buffer(Tensor(a!) self, Tensor src) -> Tensor(a!)");
        m.def("fused_add_relu(Tensor self, Tensor other) -> Tensor");
        // Several linears of one input, outputs concatenated along the last
        // dim (one dispatch for decode-size inputs).
        m.def("multi_linear(Tensor x, Tensor[] weights, Tensor?[] biases) -> Tensor");
        // (y, a + b); the second output is empty without b.
        m.def("rms_norm_multi_linear(Tensor a, Tensor? b, Tensor norm_weight, float eps, Tensor[] weights, "
              "Tensor?[] biases) -> (Tensor, Tensor)");
        m.def("silu_mul_multi_linear(Tensor gate, Tensor up, Tensor[] weights, Tensor?[] biases) -> Tensor");
    }

    TORCH_LIBRARY_IMPL(webgpu, PrivateUse1, m)
    {
        m.impl("write_buffer", TORCH_FN(write_buffer));
        m.impl("fused_add_relu", TORCH_FN(fused_add_relu));
        m.impl("multi_linear", TORCH_FN(multi_linear));
        m.impl("rms_norm_multi_linear", TORCH_FN(rms_norm_multi_linear));
        m.impl("silu_mul_multi_linear", TORCH_FN(silu_mul_multi_linear));
    }

    // create_buffer has no tensor arguments, so it can't dispatch on device.
    TORCH_LIBRARY_IMPL(webgpu, CompositeExplicitAutograd, m)
    {
        m.impl("create_buffer", TORCH_FN(create_buffer));
        m.impl("multi_linear", TORCH_FN(multi_linear_ref));
        m.impl("rms_norm_multi_linear", TORCH_FN(rms_norm_multi_linear_ref));
        m.impl("silu_mul_multi_linear", TORCH_FN(silu_mul_multi_linear_ref));
    }
}
