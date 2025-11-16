#include <algorithm>
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

        struct MemoryBlock
        {
            int block_size;
            int block_stride;
            int new_dim_start;
            int new_dim_end;
        };

        at::Tensor reshape(const at::Tensor &self, at::SymIntArrayRef shape)
        {
            at::Tensor out = self;
            int minus_one_position = -1;
            auto new_shape_opt = c10::asIntArrayRefSlowOpt(shape);
            if (new_shape_opt == std::nullopt)
            {
                TORCH_CHECK("Incorrect input shape");
                return out;
            }
            auto new_shape = new_shape_opt.value();

            // validate shape against max single -1 and no zeros
            for (size_t i = 0; i < new_shape.size(); ++i)
            {
                if (new_shape[i] == -1)
                {
                    if (minus_one_position != -1)
                    {
                        TORCH_CHECK(false, "You can use only a single -1 shape.");
                        // TODO: it should exit here maybe?
                        return out;
                    }
                    minus_one_position = i;
                }

                if (new_shape[i] == 0) // TODO - make it work with zeros too once everything else works
                {
                    TORCH_CHECK(false, "0-dim shapes not supported, yet.");
                    // TODO: it should exit here maybe?
                    return out;
                }
            }

            if (minus_one_position != -1)
            {
                int elems_on_all_pos_except_minus_one = 1;
                for (size_t i = 0; i < new_shape.size(); ++i)
                {
                    if (i == minus_one_position)
                    {
                        continue;
                    }

                    elems_on_all_pos_except_minus_one *= new_shape[i];
                }
                std::vector<int64_t> vec = new_shape.vec();
                TORCH_CHECK(self.numel() % elems_on_all_pos_except_minus_one == 0);
                vec[minus_one_position] = self.numel() / elems_on_all_pos_except_minus_one;
                new_shape = at::IntArrayRef(vec);
            }

            int64_t normalized_shape_numel = 1;
            for (auto i : new_shape)
            {
                normalized_shape_numel *= i;
            }

            TORCH_CHECK(self.numel() == normalized_shape_numel);

            // return a view without copy if possible
            if (self.sizes().size() == new_shape.size())
            {
                bool has_same_size = true;
                for (size_t i = 0; i < self.sizes().size(); ++i)
                {
                    if (self.sizes()[i] != new_shape[i])
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
                std::vector<int64_t> new_strides(new_shape.size());
                new_strides[new_shape.size() - 1] = 1;
                for (int dim = static_cast<int>(new_shape.size()) - 2; dim >= 0; --dim) // TODO: what if size() == 0?
                {
                    new_strides[dim] = new_shape[dim + 1] * new_strides[dim + 1];
                }
                return at::as_strided(self, new_shape, new_strides, self.storage_offset());
            }

            // general path - trying to return a view copying a self tensor
            MemoryBlock block;
            block.block_size = self.sizes()[self.sizes().size() - 1];
            block.block_stride = self.strides()[self.strides().size() - 1];

            // split memory to blocks
            std::vector<MemoryBlock> blocks = {};
            blocks.push_back(block);
            for (int dim = static_cast<int>(self.sizes().size()) - 2; dim >= 0; --dim)
            {
                if (self.strides()[dim] == blocks.back().block_size * blocks.back().block_stride)
                {
                    // can merge
                    blocks.back().block_size *= self.sizes()[dim];
                }
                else
                {
                    MemoryBlock new_block;
                    new_block.block_size = self.sizes()[dim];
                    new_block.block_stride = self.strides()[dim];
                    blocks.push_back(new_block);
                }
            }

            std::reverse(blocks.begin(), blocks.end());

            // check if new shape can fit these blocks
            bool do_new_shapes_match_blocks = true;
            int current_block_index = 0;
            for (auto i = 0; i < new_shape.size(); ++i)
            {
                if (new_shape[i] == blocks[current_block_index].block_size)
                {
                    blocks[current_block_index].new_dim_start = i;
                    blocks[current_block_index].new_dim_end = i;

                    current_block_index += 1;
                }
                else
                {
                    if (current_block_index < blocks.size() - 2)
                    {
                        int multiple_block_size = blocks[current_block_index].block_size;
                        bool multiplied_block_matches_shape = false;
                        blocks[current_block_index].new_dim_start = i;
                        for (auto j = current_block_index + 1; j < blocks.size(); ++j)
                        {
                            multiple_block_size *= blocks[j].block_size;
                            if (new_shape[i] == multiple_block_size)
                            {
                                multiplied_block_matches_shape = true;
                                blocks[current_block_index].new_dim_end = j;
                                current_block_index = j;
                                break;
                            }
                        }

                        if (multiplied_block_matches_shape)
                        {
                            continue;
                        }
                        do_new_shapes_match_blocks = false;
                        break;
                    }

                    do_new_shapes_match_blocks = false;
                    break;
                }
            }

            if (do_new_shapes_match_blocks)
            {
                std::vector<int64_t> new_strides;

                for (auto i = blocks.size() - 1; i > 0; --i)
                {
                    for (auto j = blocks[i].new_dim_end; j >= blocks[i].new_dim_start; --j)
                    {
                        if (j == blocks.size() - 1)
                        {
                            new_strides.push_back(blocks[i].block_stride);
                        }
                        else
                        {
                            new_strides.push_back(blocks[i].block_stride * blocks[i].block_size);
                        }
                    }
                }

                return at::as_strided(self, new_shape, new_strides, self.storage_offset());
            }

            // fallback to copy, worst case scenario
            out = self.contiguous();

            TORCH_CHECK(out.dtype() == self.dtype());
            TORCH_CHECK(out.numel() == self.numel());
            TORCH_CHECK(out.device() == self.device());
            TORCH_CHECK(out.storage_offset() == self.storage_offset());

            return out;
        }
    }

    TORCH_LIBRARY_IMPL(aten, PrivateUse1, m)
    {
        m.impl("view", TORCH_FN(ops::view));
        m.impl("resize_", TORCH_FN(ops::resize_));
        m.impl("reshape", TORCH_FN(ops::reshape));
        m.impl("empty.memory_format", TORCH_FN(ops::empty_memory_format));
        m.impl("empty_strided", TORCH_FN(ops::empty_strided));
    }

    TORCH_LIBRARY_IMPL(aten, AutogradPrivateUse1, m)
    {
        m.impl("reshape", TORCH_FN(torch_webgpu::ops::reshape));
    }
}
