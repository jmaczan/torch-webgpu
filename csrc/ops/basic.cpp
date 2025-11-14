#include <ATen/ATen.h>
#include <torch/library.h>
#include "core/webgpu_allocator.h"

namespace torch_webgpu
{
    namespace ops
    {
        at::Tensor empty_memory_format(
            c10::IntArrayRef size,
            c10::optional<at::ScalarType> dtype_opt,
            c10::optional<at::Layout> layout_opt,
            c10::optional<at::Device> device_opt,
            c10::optional<bool> pin_memory_opt,
            c10::optional<c10::MemoryFormat> memory_format_opt)
        {
            auto allocator = core::getWebGPUCachingAllocator();
            constexpr c10::DispatchKeySet privateuse1_ks(c10::DispatchKey::PrivateUse1);
            return at::detail::empty_generic(size, allocator, privateuse1_ks, dtype_or_default(dtype_opt), memory_format_opt);
        }

        at::Tensor empty_strided(
            c10::IntArrayRef size,
            c10::IntArrayRef stride,
            c10::optional<at::ScalarType> dtype_opt,
            c10::optional<at::Layout> layout_opt,
            c10::optional<at::Device> device_opt,
            c10::optional<bool> pin_memory_opt)
        {
            auto allocator = core::getWebGPUCachingAllocator();
            constexpr c10::DispatchKeySet privateuse1_ks(c10::DispatchKey::PrivateUse1);
            return at::detail::empty_strided_generic(size, stride, allocator, privateuse1_ks, dtype_or_default(dtype_opt));
        }

        at::Tensor view(at::Tensor const &self, at::IntArrayRef size)
        {
            return at::native::view(self, size);
        }

        const at::Tensor &resize_(at::Tensor const &self, at::IntArrayRef size, c10::optional<c10::MemoryFormat> format)
        {
            TORCH_CHECK(self.is_contiguous());
            auto result = at::native::resize_(self, size, format);
            return result;
        }

        at::Tensor reshape(const at::Tensor &self, at::IntArrayRef shape)
        {
            at::Tensor out = self;
            int minus_one_position = -1;
            at::IntArrayRef normalized_shape = shape;

            // validate shape against max single -1 and no zeros
            for (size_t i = 0; i < normalized_shape.size(); ++i)
            {
                if (i == -1)
                {
                    if (minus_one_position != -1)
                    {
                        TORCH_CHECK(false, "You can use only a single -1 shape.");
                        // TODO: it should exit here maybe?
                        return out;
                    }
                    minus_one_position = i;
                }

                if (i == 0)
                {
                    TORCH_CHECK(false, "0-dim shapes not allowed.");
                    // TODO: it should exit here maybe?
                    return out;
                }
            }

            if (minus_one_position != -1)
            {
                int elems_on_all_pos_except_minus_one = 1;
                for (size_t i = 0; i < normalized_shape.size(); ++i)
                {
                    if (i == minus_one_position)
                    {
                        continue;
                    }

                    elems_on_all_pos_except_minus_one *= i;
                }
                std::vector<int64_t> vec = shape.vec();
                vec[minus_one_position] = self.numel() - elems_on_all_pos_except_minus_one;
                normalized_shape = at::IntArrayRef(shape.vec());
            }

            auto normalized_shape_numel = 1;
            for (auto i : normalized_shape)
            {
                normalized_shape_numel *= i;
            }

            TORCH_CHECK(self.numel() == normalized_shape_numel);

            // return a view without copy if possible
            if (self.sizes().size() == normalized_shape.size())
            {
                bool has_same_size = true;
                for (size_t i = 0; i < self.sizes().size(); ++i)
                {
                    if (self.sizes()[i] != normalized_shape[i])
                    {
                        has_same_size = false;
                        break;
                    }
                }
                if (has_same_size)
                {
                    return self;
                }
            }

            // fast path - if source tensor is contiguous, I can just compute strides for new shape and return a copy of original tensor with new shape and strides
            if (self.is_contiguous())
            {
                std::vector<int64_t> new_strides = self.strides().vec();
                new_strides[self.strides().size() - 1] = 1;
                for (auto dim = normalized_shape.size() - 2; dim > 0; --dim) // TODO: what if size() == 0?
                {
                    new_strides[dim] = normalized_shape[dim + 1] * new_strides[dim + 1];
                }
                return at::as_strided(self, normalized_shape, new_strides, self.storage_offset());
                //
            }

            // general viewability - trying to return a view copying a self tensor - not sure how to code it yet

            // fallback to copy, worst case scenario
            else
            {
                out = self.contiguous();
                //
            }

            TORCH_CHECK(out.dtype() == self.dtype());
            TORCH_CHECK(out.numel() == self.numel());
            TORCH_CHECK(out.device() == self.device());
            TORCH_CHECK(out.storage_offset() == self.storage_offset());

            return out;
            // if not, make a copy and return a view on a copy
        }
    }

    TORCH_LIBRARY_IMPL(aten, PrivateUse1, m)
    {
        m.impl("view", TORCH_FN(ops::view));
        m.impl("resize_", TORCH_FN(ops::resize_));
        m.impl("reshape", ops::reshape);
        m.impl("empty.memory_format", TORCH_FN(ops::empty_memory_format));
        m.impl("empty_strided", TORCH_FN(ops::empty_strided));
    }
}
