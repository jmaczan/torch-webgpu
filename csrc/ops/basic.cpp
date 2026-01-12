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
                TORCH_CHECK(false, "Incorrect input shape");
                return out;
            }
            std::vector<int64_t> new_shape_vec(new_shape_opt->begin(), new_shape_opt->end());
            at::IntArrayRef new_shape(new_shape_vec);

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
                TORCH_CHECK(self.numel() % elems_on_all_pos_except_minus_one == 0);
                new_shape_vec[minus_one_position] = self.numel() / elems_on_all_pos_except_minus_one;
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

        at::Tensor as_strided(
            const at::Tensor &self,
            c10::IntArrayRef size,
            c10::IntArrayRef stride,
            c10::optional<int64_t> storage_offset_opt)
        {
            TORCH_CHECK(size.size() == stride.size(), "Shape and size should have the same amount of elements");

            auto storage_offset = storage_offset_opt.value_or(self.storage_offset());
            auto storage_size = self.storage().nbytes() / self.element_size();

            int64_t new_numel = 1;
            for (auto dim : size)
            {
                new_numel *= dim;
            }

            if (new_numel == 0)
            {
                TORCH_CHECK(storage_offset >= 0 && storage_offset <= storage_size, "Storage offset is negative or bigger than storage size");
            }
            else
            {
                int64_t min_index = storage_offset;
                int64_t max_index = storage_offset;
                for (auto dim = 0; dim < size.size(); ++dim)
                {
                    if (stride[dim] > 0)
                    {
                        max_index += stride[dim] * (size[dim] - 1);
                    }
                    else if (stride[dim] < 0)
                    {
                        min_index += stride[dim] * (size[dim] - 1);
                    }
                }

                TORCH_CHECK(min_index >= 0 && max_index < storage_size, "New tensor size needs to fit into the storage.");
            }
            at::Tensor out = at::detail::make_tensor<at::TensorImpl>(c10::TensorImpl::VIEW, c10::Storage(self.storage()), self.key_set(), self.dtype());
            out.unsafeGetTensorImpl()->set_storage_offset(storage_offset);
            out.unsafeGetTensorImpl()->set_sizes_and_strides(size, stride);
            return out;
        }

        // Transpose - swap two dimensions
        at::Tensor transpose(const at::Tensor &self, int64_t dim0, int64_t dim1)
        {
            auto ndim = self.dim();
            if (dim0 < 0)
                dim0 += ndim;
            if (dim1 < 0)
                dim1 += ndim;

            TORCH_CHECK(dim0 >= 0 && dim0 < ndim, "dim0 out of range");
            TORCH_CHECK(dim1 >= 0 && dim1 < ndim, "dim1 out of range");

            if (dim0 == dim1)
            {
                return self;
            }

            std::vector<int64_t> new_sizes(self.sizes().begin(), self.sizes().end());
            std::vector<int64_t> new_strides(self.strides().begin(), self.strides().end());

            std::swap(new_sizes[dim0], new_sizes[dim1]);
            std::swap(new_strides[dim0], new_strides[dim1]);

            return at::as_strided(self, new_sizes, new_strides, self.storage_offset());
        }

        // Permute - rearrange dimensions
        at::Tensor permute(const at::Tensor &self, at::IntArrayRef dims)
        {
            auto ndim = self.dim();
            TORCH_CHECK(dims.size() == static_cast<size_t>(ndim), "permute: dims size must match tensor dimensions");

            std::vector<bool> seen(ndim, false);
            for (auto d : dims)
            {
                auto dim = d < 0 ? d + ndim : d;
                TORCH_CHECK(dim >= 0 && dim < ndim, "permute: dimension out of range");
                TORCH_CHECK(!seen[dim], "permute: duplicate dimension");
                seen[dim] = true;
            }

            std::vector<int64_t> new_sizes(ndim);
            std::vector<int64_t> new_strides(ndim);

            for (int64_t i = 0; i < ndim; ++i)
            {
                auto old_dim = dims[i] < 0 ? dims[i] + ndim : dims[i];
                new_sizes[i] = self.size(old_dim);
                new_strides[i] = self.stride(old_dim);
            }

            return at::as_strided(self, new_sizes, new_strides, self.storage_offset());
        }

        // Unsqueeze - add dimension of size 1
        at::Tensor unsqueeze(const at::Tensor &self, int64_t dim)
        {
            auto ndim = self.dim();
            if (dim < 0)
                dim += ndim + 1;

            TORCH_CHECK(dim >= 0 && dim <= ndim, "unsqueeze: dimension out of range");

            std::vector<int64_t> new_sizes;
            std::vector<int64_t> new_strides;

            for (int64_t i = 0; i < ndim + 1; ++i)
            {
                if (i == dim)
                {
                    new_sizes.push_back(1);
                    // Stride for size-1 dimension can be anything, use 1 for simplicity
                    new_strides.push_back(i < ndim ? self.stride(i) : 1);
                }
                else
                {
                    auto old_dim = i < dim ? i : i - 1;
                    new_sizes.push_back(self.size(old_dim));
                    new_strides.push_back(self.stride(old_dim));
                }
            }

            return at::as_strided(self, new_sizes, new_strides, self.storage_offset());
        }

        // Squeeze - remove dimensions of size 1
        at::Tensor squeeze(const at::Tensor &self)
        {
            std::vector<int64_t> new_sizes;
            std::vector<int64_t> new_strides;

            for (int64_t i = 0; i < self.dim(); ++i)
            {
                if (self.size(i) != 1)
                {
                    new_sizes.push_back(self.size(i));
                    new_strides.push_back(self.stride(i));
                }
            }

            if (new_sizes.empty())
            {
                new_sizes.push_back(1);
                new_strides.push_back(1);
            }

            return at::as_strided(self, new_sizes, new_strides, self.storage_offset());
        }

        at::Tensor squeeze_dim(const at::Tensor &self, int64_t dim)
        {
            auto ndim = self.dim();
            if (dim < 0)
                dim += ndim;

            TORCH_CHECK(dim >= 0 && dim < ndim, "squeeze: dimension out of range");

            if (self.size(dim) != 1)
            {
                return self;
            }

            std::vector<int64_t> new_sizes;
            std::vector<int64_t> new_strides;

            for (int64_t i = 0; i < ndim; ++i)
            {
                if (i != dim)
                {
                    new_sizes.push_back(self.size(i));
                    new_strides.push_back(self.stride(i));
                }
            }

            if (new_sizes.empty())
            {
                new_sizes.push_back(1);
                new_strides.push_back(1);
            }

            return at::as_strided(self, new_sizes, new_strides, self.storage_offset());
        }

        // Expand - broadcast tensor to new shape
        at::Tensor expand(const at::Tensor &self, c10::IntArrayRef size, bool implicit)
        {
            auto ndim = static_cast<int64_t>(size.size());
            auto self_ndim = self.dim();

            TORCH_CHECK(ndim >= self_ndim, "expand: target dimensions must be at least as large as input dimensions");

            std::vector<int64_t> new_sizes(ndim);
            std::vector<int64_t> new_strides(ndim);

            auto dim_offset = ndim - self_ndim;

            for (int64_t i = 0; i < ndim; ++i)
            {
                if (i < dim_offset)
                {
                    // New leading dimensions
                    TORCH_CHECK(size[i] >= 0, "expand: size cannot be negative");
                    new_sizes[i] = size[i];
                    new_strides[i] = 0; // Broadcasting stride
                }
                else
                {
                    auto self_dim = i - dim_offset;
                    auto self_size = self.size(self_dim);
                    auto target_size = size[i];

                    if (target_size == -1)
                    {
                        // Keep original size
                        new_sizes[i] = self_size;
                        new_strides[i] = self.stride(self_dim);
                    }
                    else if (self_size == 1)
                    {
                        // Can broadcast
                        new_sizes[i] = target_size;
                        new_strides[i] = 0; // Broadcasting stride
                    }
                    else if (self_size == target_size)
                    {
                        // Same size, no broadcast
                        new_sizes[i] = self_size;
                        new_strides[i] = self.stride(self_dim);
                    }
                    else
                    {
                        TORCH_CHECK(false, "expand: sizes must be compatible (", self_size, " vs ", target_size, ")");
                    }
                }
            }

            at::Tensor out = at::detail::make_tensor<at::TensorImpl>(c10::TensorImpl::VIEW, c10::Storage(self.storage()), self.key_set(), self.dtype());
            out.unsafeGetTensorImpl()->set_storage_offset(self.storage_offset());
            out.unsafeGetTensorImpl()->set_sizes_and_strides(new_sizes, new_strides);
            return out;
        }

        // Contiguous - make tensor contiguous in memory (may copy)
        at::Tensor contiguous(const at::Tensor &self, c10::MemoryFormat memory_format)
        {
            if (self.is_contiguous(memory_format))
            {
                return self;
            }

            // Need to make a copy
            at::Tensor out = at::empty(self.sizes(), self.options());
            out.copy_(self);
            return out;
        }

        // Clone - create a copy of the tensor
        at::Tensor clone(const at::Tensor &self, c10::optional<c10::MemoryFormat> memory_format)
        {
            auto format = memory_format.value_or(c10::MemoryFormat::Contiguous);
            at::Tensor out = at::empty(self.sizes(), self.options(), format);
            out.copy_(self);
            return out;
        }

        // Slice - select a range along a dimension
        at::Tensor slice(const at::Tensor &self, int64_t dim, c10::optional<c10::SymInt> start_opt, c10::optional<c10::SymInt> end_opt, c10::SymInt step)
        {
            auto ndim = self.dim();
            if (dim < 0)
                dim += ndim;

            TORCH_CHECK(dim >= 0 && dim < ndim, "slice: dimension out of range");

            auto dim_size = self.size(dim);
            auto step_val = step.expect_int();
            TORCH_CHECK(step_val > 0, "slice: step must be positive");

            int64_t start = start_opt.has_value() ? start_opt->expect_int() : 0;
            int64_t end = end_opt.has_value() ? end_opt->expect_int() : dim_size;

            if (start < 0)
                start += dim_size;
            if (end < 0)
                end += dim_size;

            start = std::max<int64_t>(0, std::min(start, dim_size));
            end = std::max<int64_t>(0, std::min(end, dim_size));

            if (start >= end)
            {
                // Empty result
                std::vector<int64_t> new_sizes(self.sizes().begin(), self.sizes().end());
                new_sizes[dim] = 0;
                return at::empty(new_sizes, self.options());
            }

            auto new_size = (end - start + step_val - 1) / step_val;

            std::vector<int64_t> new_sizes(self.sizes().begin(), self.sizes().end());
            std::vector<int64_t> new_strides(self.strides().begin(), self.strides().end());

            new_sizes[dim] = new_size;
            new_strides[dim] = self.stride(dim) * step_val;

            auto new_offset = self.storage_offset() + start * self.stride(dim);

            return at::as_strided(self, new_sizes, new_strides, new_offset);
        }

        // Select - select a single index along a dimension (reduces dimension)
        at::Tensor select(const at::Tensor &self, int64_t dim, c10::SymInt index_sym)
        {
            auto ndim = self.dim();
            if (dim < 0)
                dim += ndim;

            TORCH_CHECK(dim >= 0 && dim < ndim, "select: dimension out of range");

            auto index = index_sym.expect_int();
            auto dim_size = self.size(dim);

            if (index < 0)
                index += dim_size;

            TORCH_CHECK(index >= 0 && index < dim_size, "select: index out of range");

            std::vector<int64_t> new_sizes;
            std::vector<int64_t> new_strides;

            for (int64_t i = 0; i < ndim; ++i)
            {
                if (i != dim)
                {
                    new_sizes.push_back(self.size(i));
                    new_strides.push_back(self.stride(i));
                }
            }

            auto new_offset = self.storage_offset() + index * self.stride(dim);

            if (new_sizes.empty())
            {
                // 0-dim tensor
                return at::as_strided(self, {}, {}, new_offset);
            }

            return at::as_strided(self, new_sizes, new_strides, new_offset);
        }

        // T - transpose for 2D tensors (alias for transpose(0, 1))
        at::Tensor t(const at::Tensor &self)
        {
            TORCH_CHECK(self.dim() <= 2, "t() expects a tensor with <= 2 dimensions, but got ", self.dim());
            if (self.dim() < 2)
            {
                return self;
            }
            return torch_webgpu::ops::transpose(self, 0, 1);
        }

        // _local_scalar_dense - convert single-element tensor to scalar
        at::Scalar _local_scalar_dense(const at::Tensor &self)
        {
            TORCH_CHECK(self.numel() == 1, "_local_scalar_dense requires a single-element tensor");
            // Move to CPU and get the scalar value
            auto cpu_tensor = self.to(at::kCPU);
            return cpu_tensor.item();
        }

        // masked_fill_ - in-place fill where mask is true
        at::Tensor &masked_fill_scalar(at::Tensor &self, const at::Tensor &mask, const at::Scalar &value)
        {
            // CPU fallback for now
            auto self_cpu = self.to(at::kCPU);
            auto mask_cpu = mask.to(at::kCPU);
            self_cpu.masked_fill_(mask_cpu, value);
            self.copy_(self_cpu);
            return self;
        }

        at::Tensor &masked_fill_tensor(at::Tensor &self, const at::Tensor &mask, const at::Tensor &value)
        {
            // CPU fallback for now
            auto self_cpu = self.to(at::kCPU);
            auto mask_cpu = mask.to(at::kCPU);
            auto value_cpu = value.to(at::kCPU);
            self_cpu.masked_fill_(mask_cpu, value_cpu);
            self.copy_(self_cpu);
            return self;
        }

        // gather - gather values along an axis
        at::Tensor gather_impl(const at::Tensor &self, int64_t dim, const at::Tensor &index, bool sparse_grad)
        {
            // CPU fallback for now
            auto self_cpu = self.to(at::kCPU);
            auto index_cpu = index.to(at::kCPU);
            auto result = at::gather(self_cpu, dim, index_cpu, sparse_grad);
            return result.to(self.device());
        }

        at::Tensor &gather_out_impl(const at::Tensor &self, int64_t dim, const at::Tensor &index, bool sparse_grad, at::Tensor &out)
        {
            auto result = gather_impl(self, dim, index, sparse_grad);
            out.copy_(result);
            return out;
        }

        // where - select elements based on condition
        at::Tensor where_self(const at::Tensor &condition, const at::Tensor &self, const at::Tensor &other)
        {
            // CPU fallback for now
            auto condition_cpu = condition.to(at::kCPU);
            auto self_cpu = self.to(at::kCPU);
            auto other_cpu = other.to(at::kCPU);
            auto result = at::where(condition_cpu, self_cpu, other_cpu);
            return result.to(self.device());
        }

        at::Tensor &where_self_out(const at::Tensor &condition, const at::Tensor &self, const at::Tensor &other, at::Tensor &out)
        {
            auto result = where_self(condition, self, other);
            out.copy_(result);
            return out;
        }

        // scatter - scatter values into tensor at indices
        at::Tensor scatter_src(const at::Tensor &self, int64_t dim, const at::Tensor &index, const at::Tensor &src)
        {
            // CPU fallback for now
            auto self_cpu = self.to(at::kCPU);
            auto index_cpu = index.to(at::kCPU);
            auto src_cpu = src.to(at::kCPU);
            auto result = self_cpu.scatter(dim, index_cpu, src_cpu);
            return result.to(self.device());
        }

        at::Tensor &scatter_src_out(const at::Tensor &self, int64_t dim, const at::Tensor &index, const at::Tensor &src, at::Tensor &out)
        {
            auto result = scatter_src(self, dim, index, src);
            out.copy_(result);
            return out;
        }

        // argmax - find index of maximum value
        at::Tensor argmax_impl(const at::Tensor &self, c10::optional<int64_t> dim, bool keepdim)
        {
            // CPU fallback for now
            auto self_cpu = self.to(at::kCPU);
            at::Tensor result;
            if (dim.has_value())
            {
                result = at::argmax(self_cpu, dim.value(), keepdim);
            }
            else
            {
                result = at::argmax(self_cpu);
            }
            return result.to(self.device());
        }

        at::Tensor &argmax_out_impl(const at::Tensor &self, c10::optional<int64_t> dim, bool keepdim, at::Tensor &out)
        {
            auto result = argmax_impl(self, dim, keepdim);
            out.copy_(result);
            return out;
        }

        // index.Tensor - advanced indexing with list of optional tensors
        at::Tensor index_tensor(const at::Tensor &self, const c10::List<c10::optional<at::Tensor>> &indices)
        {
            // CPU fallback for now
            auto self_cpu = self.to(at::kCPU);

            // Convert indices to CPU
            c10::List<c10::optional<at::Tensor>> indices_cpu;
            for (size_t i = 0; i < indices.size(); ++i) {
                c10::optional<at::Tensor> idx = indices.get(i);
                if (idx.has_value()) {
                    indices_cpu.push_back(idx->to(at::kCPU));
                } else {
                    indices_cpu.push_back(c10::nullopt);
                }
            }

            auto result = at::index(self_cpu, indices_cpu);
            // Move result to WebGPU
            return result.to(c10::DeviceType::PrivateUse1);
        }

        at::Tensor &index_tensor_out(const at::Tensor &self, const c10::List<c10::optional<at::Tensor>> &indices, at::Tensor &out)
        {
            auto result = index_tensor(self, indices);
            out.copy_(result);
            return out;
        }

        // index_select - select along a dimension using index tensor
        at::Tensor index_select_impl(const at::Tensor &self, int64_t dim, const at::Tensor &index)
        {
            // CPU fallback for now
            auto self_cpu = self.to(at::kCPU);
            auto index_cpu = index.to(at::kCPU);
            auto result = at::index_select(self_cpu, dim, index_cpu);
            return result.to(c10::DeviceType::PrivateUse1);
        }

        at::Tensor &index_select_out_impl(const at::Tensor &self, int64_t dim, const at::Tensor &index, at::Tensor &out)
        {
            auto result = index_select_impl(self, dim, index);
            out.copy_(result);
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
        m.impl("as_strided", TORCH_FN(ops::as_strided));
        m.impl("transpose.int", TORCH_FN(ops::transpose));
        m.impl("permute", TORCH_FN(ops::permute));
        m.impl("unsqueeze", TORCH_FN(ops::unsqueeze));
        m.impl("squeeze", TORCH_FN(ops::squeeze));
        m.impl("squeeze.dim", TORCH_FN(ops::squeeze_dim));
        m.impl("expand", TORCH_FN(ops::expand));
        m.impl("contiguous", TORCH_FN(ops::contiguous));
        m.impl("clone", TORCH_FN(ops::clone));
        m.impl("slice.Tensor", TORCH_FN(ops::slice));
        m.impl("select.int", TORCH_FN(ops::select));
        m.impl("t", TORCH_FN(ops::t));
        m.impl("_local_scalar_dense", TORCH_FN(ops::_local_scalar_dense));
        m.impl("masked_fill_.Scalar", TORCH_FN(ops::masked_fill_scalar));
        m.impl("masked_fill_.Tensor", TORCH_FN(ops::masked_fill_tensor));
        m.impl("gather", TORCH_FN(ops::gather_impl));
        m.impl("gather.out", TORCH_FN(ops::gather_out_impl));
        m.impl("where.self", TORCH_FN(ops::where_self));
        m.impl("where.self_out", TORCH_FN(ops::where_self_out));
        m.impl("scatter.src", TORCH_FN(ops::scatter_src));
        m.impl("scatter.src_out", TORCH_FN(ops::scatter_src_out));
        m.impl("argmax", TORCH_FN(ops::argmax_impl));
        m.impl("argmax.out", TORCH_FN(ops::argmax_out_impl));
        m.impl("index.Tensor", TORCH_FN(ops::index_tensor));
        m.impl("index.Tensor_out", TORCH_FN(ops::index_tensor_out));
        m.impl("index_select", TORCH_FN(ops::index_select_impl));
        m.impl("index_select.out", TORCH_FN(ops::index_select_out_impl));
    }

    TORCH_LIBRARY_IMPL(aten, AutogradPrivateUse1, m)
    {
        m.impl("reshape", TORCH_FN(torch_webgpu::ops::reshape));
    }
}
