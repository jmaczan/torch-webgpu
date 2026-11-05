#pragma once
// Shared infrastructure for ATen op registrations.
//
// Structured kernels: PyTorch exports the "meta" half of every structured op
// (at::meta::structured_<op>), which performs argument checking, type
// promotion, broadcasting and output allocation/resizing. We mirror the code
// torchgen generates for in-tree backends (RegisterCPU.cpp): wrap the meta
// class, let it allocate outputs on our device, then run a WebGPU kernel.
// For TensorIterator-based metas the meta object *is* the iterator.

#include <ATen/ATen.h>
#include <ATen/TensorIterator.h>
#include <ATen/Operators.h>
#include <ATen/native/CPUFallback.h>
#include <ATen/native/Resize.h>
#include <torch/library.h>

#include "kernels/wgsl.h"

#include <array>
#include <functional>
#include <optional>
#include <type_traits>

namespace torch_webgpu::ops
{
    // Boxed CPU fallback used for every op without a native WebGPU kernel.
    void cpu_fallback(const c10::OperatorHandle &op, torch::jit::Stack *stack);

    // Calls the CPU fallback for a specific (unboxed) ATen op, e.g.
    //   return WEBGPU_FALLBACK(add, Tensor)(self, other, alpha);
#define WEBGPU_FALLBACK(op, overload) \
    at::native::call_fallback_fn<&::torch_webgpu::ops::cpu_fallback, ATEN_OP2(op, overload)>::call
#define WEBGPU_FALLBACK0(op) \
    at::native::call_fallback_fn<&::torch_webgpu::ops::cpu_fallback, ATEN_OP(op)>::call

    inline bool is_webgpu(const at::TensorBase &t)
    {
        return t.defined() && t.device().type() == c10::DeviceType::PrivateUse1;
    }

    // True if every defined WebGPU tensor argument has a GPU compute dtype.
    // CPU tensors (e.g. wrapped Python scalars) are ignored.
    inline bool gpu_dtypes_ok(std::initializer_list<std::reference_wrapper<const at::Tensor>> ts)
    {
        for (const at::Tensor &t : ts)
            if (is_webgpu(t) && !wgsl::is_compute_dtype(t.scalar_type()))
                return false;
        return true;
    }
    inline bool gpu_dtypes_ok(const at::TensorList ts)
    {
        for (const at::Tensor &t : ts)
            if (is_webgpu(t) && !wgsl::is_compute_dtype(t.scalar_type()))
                return false;
        return true;
    }

    inline at::Tensor empty_webgpu(at::IntArrayRef sizes, const at::TensorOptions &options)
    {
        return at::empty(sizes, options.device(c10::DeviceType::PrivateUse1));
    }

    namespace detail
    {
        inline at::Tensor create_out(at::IntArrayRef sizes, at::IntArrayRef strides, const at::TensorOptions &options)
        {
            if (strides.empty())
                return at::empty(sizes, options);
            return at::empty_strided(sizes, strides, options);
        }

        inline void resize_out(const at::Tensor &out, at::IntArrayRef sizes, at::IntArrayRef strides,
                               const at::TensorOptions &options)
        {
            TORCH_CHECK(options.dtype() == out.dtype(), "Expected out tensor to have dtype ", options.dtype(),
                        ", but got ", out.dtype(), " instead");
            TORCH_CHECK(options.device() == out.device(), "Expected out tensor to have device ", options.device(),
                        ", but got ", out.device(), " instead");
            const bool resized = at::native::resize_output(out, sizes);
            if (resized && !strides.empty())
            {
                TORCH_INTERNAL_ASSERT(!options.memory_format_opt().has_value());
                out.as_strided_(sizes, strides);
            }
            else if (options.memory_format_opt().has_value())
            {
                out.unsafeGetTensorImpl()->empty_tensor_restride(*options.memory_format_opt());
            }
        }

        inline void check_inplace(const at::Tensor &self, at::IntArrayRef sizes, const at::TensorOptions &options)
        {
            TORCH_CHECK(options.dtype() == self.dtype(), "Bad in-place call: input tensor dtype ", self.dtype(),
                        " and output tensor dtype ", options.dtype(), " should match");
            TORCH_CHECK(options.device() == self.device(), "Bad in-place call: input tensor device ",
                        self.device(), " and output tensor device ", options.device(), " should match");
            TORCH_CHECK(sizes == self.sizes(), "Bad in-place call: input tensor size ", self.sizes(),
                        " and output tensor size ", sizes, " should match");
        }

        inline std::optional<at::Tensor> maybe_create_proxy(const at::Tensor &out, at::IntArrayRef sizes,
                                                            at::IntArrayRef strides, const at::TensorOptions &options)
        {
            if (out.strides() != strides)
                return at::empty_strided(sizes, strides, options);
            return std::nullopt;
        }

        template <class Meta>
        constexpr bool is_iter = std::is_base_of_v<at::TensorIteratorBase, Meta>;
    }

    template <class Meta, int N = 1>
    struct Functional final : Meta
    {
        void set_output_strided(int64_t i, at::IntArrayRef sizes, at::IntArrayRef strides,
                                at::TensorOptions options) override
        {
            outputs_[i] = detail::create_out(sizes, strides, options);
            if constexpr (detail::is_iter<Meta>)
                Meta::set_output_raw_strided(i, sizes, strides, options);
        }
        void set_output_raw_strided(int64_t i, at::IntArrayRef sizes, at::IntArrayRef strides,
                                    at::TensorOptions options) override
        {
            outputs_[i] = detail::create_out(sizes, strides, options);
            if constexpr (detail::is_iter<Meta>)
                Meta::set_output_raw_strided(i, sizes, strides, options);
        }
        const at::Tensor &maybe_get_output(int64_t i) override { return outputs_[i]; }
        std::array<at::Tensor, N> outputs_;
    };

    template <class Meta, int N = 1>
    struct Out final : Meta
    {
        template <class... Ts>
        explicit Out(Ts &...outs) : outputs_{std::ref(outs)...} {}

        void set_output_strided(int64_t i, at::IntArrayRef sizes, at::IntArrayRef strides,
                                at::TensorOptions options) override
        {
            const auto &out = outputs_[i].get();
            detail::resize_out(out, sizes, strides, options);
            auto proxy = detail::maybe_create_proxy(out, sizes, strides, options);
            if (proxy)
                proxies_[i] = std::move(*proxy);
            if constexpr (detail::is_iter<Meta>)
                Meta::set_output_raw_strided(i, sizes, strides, options);
        }
        void set_output_raw_strided(int64_t i, at::IntArrayRef sizes, at::IntArrayRef strides,
                                    at::TensorOptions options) override
        {
            detail::resize_out(outputs_[i].get(), sizes, strides, options);
            if constexpr (detail::is_iter<Meta>)
                Meta::set_output_raw_strided(i, sizes, strides, options);
        }
        const at::Tensor &maybe_get_output(int64_t i) override
        {
            return proxies_[i].has_value() ? *proxies_[i] : outputs_[i].get();
        }
        // Copies proxy outputs back into the user-provided tensors.
        void finish()
        {
            for (int i = 0; i < N; ++i)
                if (proxies_[i].has_value())
                    outputs_[i].get().copy_(*proxies_[i]);
        }
        std::array<std::reference_wrapper<at::Tensor>, N> outputs_;
        std::array<std::optional<at::Tensor>, N> proxies_;
    };

    template <class Meta, int N = 1>
    struct Inplace final : Meta
    {
        template <class... Ts>
        explicit Inplace(Ts &...outs) : outputs_{std::ref(outs)...} {}

        void set_output_strided(int64_t i, at::IntArrayRef sizes, at::IntArrayRef strides,
                                at::TensorOptions options) override
        {
            const auto &out = outputs_[i].get();
            detail::check_inplace(out, sizes, options);
            auto proxy = detail::maybe_create_proxy(out, sizes, strides, options);
            if (proxy)
                proxies_[i] = std::move(*proxy);
            if constexpr (detail::is_iter<Meta>)
                Meta::set_output_raw_strided(i, sizes, strides, options);
        }
        void set_output_raw_strided(int64_t i, at::IntArrayRef sizes, at::IntArrayRef strides,
                                    at::TensorOptions options) override
        {
            detail::check_inplace(outputs_[i].get(), sizes, options);
            if constexpr (detail::is_iter<Meta>)
                Meta::set_output_raw_strided(i, sizes, strides, options);
        }
        const at::Tensor &maybe_get_output(int64_t i) override
        {
            return proxies_[i].has_value() ? *proxies_[i] : outputs_[i].get();
        }
        void finish()
        {
            for (int i = 0; i < N; ++i)
                if (proxies_[i].has_value())
                    outputs_[i].get().copy_(*proxies_[i]);
        }
        std::array<std::reference_wrapper<at::Tensor>, N> outputs_;
        std::array<std::optional<at::Tensor>, N> proxies_;
    };
}
