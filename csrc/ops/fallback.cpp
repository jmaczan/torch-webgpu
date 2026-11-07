// Boxed CPU fallback for ATen ops without a native WebGPU kernel.
//
// Behaviour is controlled by environment variables:
//   TORCH_WEBGPU_DISABLE_CPU_FALLBACK=1  raise instead of falling back
//   TORCH_WEBGPU_FALLBACK_WARN=0         don't warn (default: warn once per op)

#include <ATen/native/CPUFallback.h>
#include <torch/library.h>

#include "ops/common.h"

#include <cstdlib>
#include <mutex>
#include <string>
#include <unordered_set>

namespace torch_webgpu::ops
{
    namespace
    {
        bool env_flag(const char *name, bool def)
        {
            const char *v = std::getenv(name);
            if (!v || !*v)
                return def;
            return std::string(v) != "0";
        }

        std::mutex &seen_mutex()
        {
            static std::mutex m;
            return m;
        }
        std::unordered_set<std::string> &seen()
        {
            static auto *s = new std::unordered_set<std::string>();
            return *s;
        }
    }

    namespace
    {
        // A dtype WebGPU hardware can't compute in (float64, complex), if
        // any tensor argument has one.
        std::optional<at::ScalarType> unsupported_dtype(const c10::OperatorHandle &op, const torch::jit::Stack *stack)
        {
            const auto n = op.schema().arguments().size();
            auto check = [](const at::Tensor &t) -> std::optional<at::ScalarType>
            {
                if (t.defined() && (t.scalar_type() == at::kDouble || t.is_complex()))
                    return t.scalar_type();
                return std::nullopt;
            };
            for (size_t i = stack->size() - n; i < stack->size(); ++i)
            {
                const auto &v = (*stack)[i];
                if (v.isTensor())
                {
                    if (auto r = check(v.toTensor()))
                        return r;
                }
                else if (v.isTensorList())
                {
                    for (const auto &t : v.toTensorVector())
                        if (auto r = check(t))
                            return r;
                }
            }
            return std::nullopt;
        }
    }

    void cpu_fallback(const c10::OperatorHandle &op, torch::jit::Stack *stack)
    {
        const auto &name = op.schema().operator_name();
        const auto bad_dtype = unsupported_dtype(op, stack);
        const std::string reason = bad_dtype ? std::string(" (") + c10::toString(*bad_dtype) +
                                                   " is not supported by WebGPU hardware; use float32)"
                                             : std::string();
        static const bool disabled = env_flag("TORCH_WEBGPU_DISABLE_CPU_FALLBACK", false);
        TORCH_CHECK_NOT_IMPLEMENTED(!disabled, "torch_webgpu: '", name,
                                    "' is not implemented for the webgpu device", reason,
                                    " and the CPU fallback is disabled (TORCH_WEBGPU_DISABLE_CPU_FALLBACK=1)");
        static const bool warn = env_flag("TORCH_WEBGPU_FALLBACK_WARN", true);
        {
            std::string key = name.name + (name.overload_name.empty() ? "" : "." + name.overload_name);
            if (bad_dtype)
                key += std::string("[") + c10::toString(*bad_dtype) + "]";
            std::lock_guard<std::mutex> g(seen_mutex());
            if (seen().insert(key).second && warn)
            {
                if (bad_dtype)
                {
                    TORCH_WARN("torch_webgpu: '", name, "' runs on the CPU", reason,
                               ". Set TORCH_WEBGPU_FALLBACK_WARN=0 to silence.");
                }
                else
                {
                    TORCH_WARN("torch_webgpu: '", key,
                               "' is not natively supported on webgpu and falls back to the CPU. "
                               "This may hurt performance. Set TORCH_WEBGPU_FALLBACK_WARN=0 to silence.");
                }
            }
        }
        at::native::cpu_fallback(op, stack);
    }

    std::vector<std::string> fallback_ops_seen()
    {
        std::lock_guard<std::mutex> g(seen_mutex());
        return {seen().begin(), seen().end()};
    }

    TORCH_LIBRARY_IMPL(_, PrivateUse1, m)
    {
        m.fallback(torch::CppFunction::makeFromBoxedFunction<&cpu_fallback>());
    }
}
