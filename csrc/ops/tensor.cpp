// Tensor creation, storage management, views, copies and fills.

#include <ATen/ATen.h>
#include <ATen/EmptyTensor.h>
#include <ATen/TensorIterator.h>
#include <ATen/native/Resize.h>
#include <ATen/native/TensorFactories.h>
#include <ATen/ops/as_strided_native.h>
#include <ATen/ops/view_native.h>
#include <ATen/ops/_reshape_alias_native.h>
#include <ATen/ops/unfold_native.h>
#include <ATen/ops/view_as_real_native.h>
#include <ATen/ops/view_as_complex_native.h>
#include <ATen/ops/is_set_to_native.h>
#include <torch/library.h>

#include "core/allocator.h"
#include "core/context.h"
#include "core/stream.h"
#include "kernels/elementwise.h"
#include "ops/common.h"

#include <cmath>
#include <cstring>
#include <limits>

namespace torch_webgpu::core
{
    // Grows (never shrinks) a storage, preserving its contents.
    void resize_storage_bytes(c10::StorageImpl *storage, size_t new_size)
    {
        TORCH_CHECK(storage->resizable(), "Trying to resize storage that is not resizable");
        const size_t old_size = storage->nbytes();
        if (new_size == old_size)
            return;
        at::DataPtr new_data = get_allocator()->allocate(new_size);
        const size_t n = std::min(old_size, new_size);
        if (n > 0)
        {
            auto *src = static_cast<Allocation *>(storage->data_ptr().get_context());
            auto *dst = static_cast<Allocation *>(new_data.get_context());
            // Capacities are multiples of 4 >= the rounded sizes, so rounding
            // the copy up stays in bounds of both buffers.
            uint64_t bytes = (n + 3) / 4 * 4;
            Stream::get().copy_buffer(src->buffer, 0, dst->buffer, 0, bytes);
        }
        storage->set_data_ptr_noswap(std::move(new_data));
        storage->set_nbytes(new_size);
    }
}

namespace torch_webgpu::ops
{
    using kernels::CT;
    using kernels::ElementwiseOp;

    namespace
    {
        constexpr c10::DispatchKeySet kWebGPUKeys{c10::DispatchKey::PrivateUse1};

        // ---------------------------------------------------------- empty
        at::Tensor empty_memory_format(c10::IntArrayRef size, std::optional<at::ScalarType> dtype,
                                       std::optional<at::Layout> layout, std::optional<at::Device> device,
                                       std::optional<bool> pin_memory,
                                       std::optional<c10::MemoryFormat> memory_format)
        {
            TORCH_CHECK(!pin_memory.value_or(false), "torch_webgpu: pin_memory is not supported for webgpu tensors");
            TORCH_CHECK(layout.value_or(at::kStrided) == at::kStrided, "torch_webgpu: only strided layout is supported");
            auto dt = c10::dtype_or_default(dtype);
            TORCH_CHECK(wgsl::is_storage_dtype(dt), "torch_webgpu: dtype ", dt, " is not supported");
            return at::detail::empty_generic(size, core::get_allocator(), kWebGPUKeys, dt, memory_format);
        }

        at::Tensor empty_strided(c10::IntArrayRef size, c10::IntArrayRef stride, std::optional<at::ScalarType> dtype,
                                 std::optional<at::Layout> layout, std::optional<at::Device> device,
                                 std::optional<bool> pin_memory)
        {
            TORCH_CHECK(!pin_memory.value_or(false), "torch_webgpu: pin_memory is not supported for webgpu tensors");
            auto dt = c10::dtype_or_default(dtype);
            TORCH_CHECK(wgsl::is_storage_dtype(dt), "torch_webgpu: dtype ", dt, " is not supported");
            return at::detail::empty_strided_generic(size, stride, core::get_allocator(), kWebGPUKeys, dt);
        }

        // --------------------------------------------------------- resize
        void maybe_resize_storage(c10::TensorImpl *self, size_t new_size_bytes)
        {
            if (self->numel() == 0)
                return;
            auto *storage = self->unsafe_storage().unsafeGetStorageImpl();
            if (!storage)
            {
                auto s = c10::make_intrusive<c10::StorageImpl>(c10::StorageImpl::use_byte_size_t(), new_size_bytes,
                                                               core::get_allocator(), true);
                self->set_storage_keep_dtype(c10::Storage(std::move(s)));
            }
            else if (new_size_bytes > storage->nbytes())
            {
                core::resize_storage_bytes(storage, new_size_bytes);
            }
        }

        const at::Tensor &resize_(const at::Tensor &self, c10::IntArrayRef size,
                                  std::optional<c10::MemoryFormat> memory_format)
        {
            auto *impl = self.unsafeGetTensorImpl();
            if (impl->sizes() != size)
            {
                impl->set_sizes_contiguous(size);
                const auto itemsize = impl->dtype().itemsize();
                const auto storage_offset = impl->storage_offset();
                size_t bytes = at::detail::computeStorageNbytesContiguous(size, itemsize, storage_offset);
                maybe_resize_storage(impl, bytes);
            }
            if (memory_format.has_value())
            {
                auto fmt = *memory_format;
                TORCH_CHECK(fmt != c10::MemoryFormat::Preserve, "Unsupported memory format", fmt);
                impl->empty_tensor_restride(fmt);
            }
            return self;
        }

        // ---------------------------------------------------------- set_
        at::Tensor &set_storage(at::Tensor &self, c10::Storage source, int64_t storage_offset,
                                c10::IntArrayRef size, c10::IntArrayRef stride)
        {
            at::native::checkSetStorage(self, source, storage_offset, size, stride);
            self.unsafeGetTensorImpl()->set_storage_offset(storage_offset);
            std::optional<c10::IntArrayRef> stride_opt =
                stride.data() != nullptr ? std::optional<c10::IntArrayRef>(stride) : std::nullopt;
            auto *impl = self.unsafeGetTensorImpl();
            if (stride_opt)
                impl->set_sizes_and_strides(size, *stride_opt);
            else
                impl->set_sizes_contiguous(size);
            size_t bytes = stride_opt
                               ? at::detail::computeStorageNbytes(size, *stride_opt, impl->dtype().itemsize(), storage_offset)
                               : at::detail::computeStorageNbytesContiguous(size, impl->dtype().itemsize(), storage_offset);
            maybe_resize_storage(impl, bytes);
            return self;
        }

        at::Tensor &set_source_storage(at::Tensor &self, c10::Storage source)
        {
            int64_t new_size = static_cast<int64_t>(source.nbytes() / self.dtype().itemsize());
            return set_storage(self, source, 0, {new_size}, {});
        }

        at::Tensor &set_source_tensor(at::Tensor &self, const at::Tensor &src)
        {
            if (self.is_same(src))
                return self;
            return set_storage(self, src.storage(), src.storage_offset(), src.sizes(), src.strides());
        }

        at::Tensor &set_empty(at::Tensor &self)
        {
            caffe2::TypeMeta dtype = self.dtype();
            auto storage = c10::make_intrusive<c10::StorageImpl>(c10::StorageImpl::use_byte_size_t(), 0,
                                                                 core::get_allocator(), true);
            self.set_(c10::Storage(std::move(storage)), 0, {0}, {});
            TORCH_INTERNAL_ASSERT(dtype == self.dtype());
            return self;
        }

        // --------------------------------------------------------- copies
        bool aligned4(uint64_t x) { return x % 4 == 0; }

        // Device -> device copy of same-shape tensors (any dtypes/strides).
        void copy_device_to_device(const at::Tensor &dst, const at::Tensor &src)
        {
            if (dst.numel() == 0)
                return;
            const auto esize = dst.element_size();
            if (dst.scalar_type() == src.scalar_type() && dst.is_contiguous() && src.is_contiguous() &&
                dst.sizes() == src.sizes())
            {
                uint64_t so = core::byte_offset_of(src), dof = core::byte_offset_of(dst);
                uint64_t n = static_cast<uint64_t>(dst.numel()) * esize;
                if (aligned4(so) && aligned4(dof) && aligned4(n))
                {
                    core::Stream::get().copy_buffer(core::buffer_of(src), so, core::buffer_of(dst), dof, n);
                    return;
                }
            }

            auto iter = at::TensorIteratorConfig()
                            .add_output(dst)
                            .add_const_input(src)
                            .resize_outputs(false)
                            .check_all_same_dtype(false)
                            .check_all_same_device(false)
                            .set_check_mem_overlap(true)
                            .build();

            ElementwiseOp op;
            if (dst.scalar_type() == src.scalar_type())
            {
                op.name = "copy_raw";
                op.raw_copy = true;
            }
            else if (wgsl::is_compute_dtype(src.scalar_type()) && wgsl::is_compute_dtype(dst.scalar_type()))
            {
                op.name = "copy_cast";
                op.in_ct = wgsl::compute_type(src.scalar_type());
                op.out_ct = op.in_ct;
                op.code = "x0";
            }
            else
            {
                // e.g. float64 -> float32: convert on the CPU (exact), then upload.
                at::Tensor tmp = src.cpu().to(dst.scalar_type());
                dst.copy_(tmp);
                return;
            }
            kernels::run_elementwise(iter, op);
        }

        void copy_host_to_device(const at::Tensor &dst, const at::Tensor &src_cpu)
        {
            // Convert dtype / broadcast / layout on the CPU, where it is exact.
            at::Tensor src = src_cpu.to(dst.scalar_type()).expand_as(dst).contiguous();
            const uint64_t nbytes = static_cast<uint64_t>(src.numel()) * src.element_size();
            const uint64_t off = core::byte_offset_of(dst);
            if (dst.is_contiguous() && aligned4(off) && aligned4(nbytes))
            {
                core::Stream::get().write_buffer(core::buffer_of(dst), off, src.data_ptr(), nbytes);
                return;
            }
            // Upload into a fresh contiguous buffer (offset 0, padded to 4
            // bytes) and scatter it on the device.
            at::Tensor staging = at::empty(dst.sizes(), dst.options().memory_format(c10::MemoryFormat::Contiguous));
            uint64_t padded = (nbytes + 3) / 4 * 4;
            if (padded == nbytes)
            {
                core::Stream::get().write_buffer(core::buffer_of(staging), 0, src.data_ptr(), nbytes);
            }
            else
            {
                std::vector<uint8_t> tmp(padded, 0);
                std::memcpy(tmp.data(), src.data_ptr(), nbytes);
                core::Stream::get().write_buffer(core::buffer_of(staging), 0, tmp.data(), padded);
            }
            copy_device_to_device(dst, staging);
        }

        void copy_device_to_host(const at::Tensor &dst_cpu, const at::Tensor &src)
        {
            at::Tensor s = src;
            if (!s.is_contiguous())
            {
                at::Tensor c = at::empty(s.sizes(), s.options().memory_format(c10::MemoryFormat::Contiguous));
                copy_device_to_device(c, s);
                s = c;
            }
            const uint64_t nbytes = static_cast<uint64_t>(s.numel()) * s.element_size();
            const bool direct = dst_cpu.is_contiguous() && dst_cpu.scalar_type() == s.scalar_type() &&
                                dst_cpu.sizes() == s.sizes();
            if (direct)
            {
                core::Stream::get().read_buffer(core::buffer_of(s), core::byte_offset_of(s), dst_cpu.data_ptr(), nbytes);
                return;
            }
            at::Tensor tmp = at::empty(s.sizes(), s.options().device(at::kCPU));
            core::Stream::get().read_buffer(core::buffer_of(s), core::byte_offset_of(s), tmp.data_ptr(), nbytes);
            dst_cpu.copy_(tmp);
        }

        // _copy_from(self=src, dst): reached from the composite copy_ for any
        // copy involving a webgpu tensor.
        at::Tensor copy_from(const at::Tensor &src, const at::Tensor &dst, bool non_blocking)
        {
            if (dst.numel() == 0 || src.is_same(dst))
                return dst;
            const bool src_gpu = is_webgpu(src), dst_gpu = is_webgpu(dst);
            if (dst_gpu && src_gpu)
            {
                at::Tensor s = src;
                if (s.sizes() != dst.sizes())
                    s = s.expand_as(dst);
                copy_device_to_device(dst, s);
            }
            else if (dst_gpu)
            {
                at::Tensor s = src.device().is_cpu() ? src : src.cpu();
                copy_host_to_device(dst, s);
            }
            else if (src_gpu)
            {
                if (dst.device().is_cpu())
                {
                    if (src.sizes() != dst.sizes())
                    {
                        at::Tensor tmp = at::empty(src.sizes(), src.options().device(at::kCPU));
                        copy_device_to_host(tmp, src);
                        dst.copy_(tmp);
                    }
                    else
                    {
                        copy_device_to_host(dst, src);
                    }
                }
                else
                {
                    dst.copy_(src.cpu());
                }
            }
            else
            {
                TORCH_CHECK(false, "torch_webgpu: _copy_from called without a webgpu tensor");
            }
            return dst;
        }

        at::Tensor _copy_from_and_resize(const at::Tensor &self, const at::Tensor &dst)
        {
            at::native::resize_output(dst, self.sizes());
            copy_from(self, dst, false);
            return dst;
        }

        at::Scalar _local_scalar_dense(const at::Tensor &self)
        {
            TORCH_CHECK(self.numel() == 1, "a Tensor with ", self.numel(),
                        " elements cannot be converted to Scalar");
            at::Tensor cpu = at::empty({}, self.options().device(at::kCPU));
            core::Stream::get().read_buffer(core::buffer_of(self), core::byte_offset_of(self), cpu.data_ptr(),
                                            self.element_size());
            return cpu.item();
        }

        // ---------------------------------------------------------- fills
        at::Tensor &fill_scalar(at::Tensor &self, const at::Scalar &value)
        {
            if (self.numel() == 0)
                return self;
            // Encode the value bit-exactly in the destination dtype on the CPU.
            at::Tensor bits = at::scalar_tensor(value, at::TensorOptions().dtype(self.scalar_type()));
            const size_t es = self.element_size();
            uint32_t words[4] = {0, 0, 0, 0};
            std::memcpy(words, bits.data_ptr(), es);

            auto iter = at::TensorIteratorConfig()
                            .add_output(self)
                            .resize_outputs(false)
                            .check_all_same_dtype(false)
                            .set_check_mem_overlap(false)
                            .build();
            ElementwiseOp op;
            op.name = "fill";
            op.raw_copy = true; // raw store of scalar words
            op.code = "fill";
            for (size_t w = 0; w < (es + 3) / 4; ++w)
                op.scalars.push_back({c10::Scalar(static_cast<int64_t>(static_cast<int32_t>(words[w]))), CT::I32});
            kernels::run_elementwise(iter, op);
            return self;
        }

        at::Tensor &fill_tensor(at::Tensor &self, const at::Tensor &value)
        {
            TORCH_CHECK(value.dim() == 0, "fill_ only supports 0-dimension value tensor but got tensor with ",
                        value.dim(), " dimensions.");
            if (is_webgpu(value))
            {
                // Avoid a host round-trip: broadcast-copy the 0-dim tensor.
                if (self.numel() == 0)
                    return self;
                copy_device_to_device(self, value.expand_as(self));
                return self;
            }
            return fill_scalar(self, value.item());
        }

        at::Tensor &zero_(at::Tensor &self) { return fill_scalar(self, 0); }

        // ----------------------------------------------------- ranges
        void affine_fill(const at::Tensor &out, const char *name, const char *code_f, const char *code_i,
                         std::vector<std::pair<at::Scalar, CT>> scalars)
        {
            if (out.numel() == 0)
                return;
            at::Tensor o = out.is_contiguous() ? out : at::empty(out.sizes(), out.options());
            auto iter = at::TensorIteratorConfig().add_output(o).resize_outputs(false).check_all_same_dtype(false).build();
            ElementwiseOp op;
            op.name = name;
            op.in_ct = wgsl::compute_type(out.scalar_type());
            op.out_ct = op.in_ct == CT::Bool ? CT::I32 : op.in_ct;
            op.code = op.out_ct == CT::F32 ? code_f : code_i;
            op.pass_index = true;
            op.scalars = std::move(scalars);
            kernels::run_elementwise(iter, op);
            if (!o.is_same(out))
                out.copy_(o);
        }

        at::Tensor &arange_out(const at::Scalar &start, const at::Scalar &end, const at::Scalar &step, at::Tensor &out)
        {
            if (!wgsl::is_compute_dtype(out.scalar_type()) || out.scalar_type() == at::kBool)
                return WEBGPU_FALLBACK(arange, start_out)(start, end, step, out);
            const bool fp = at::isFloatingType(out.scalar_type());
            double xs = start.toDouble(), xe = end.toDouble(), xst = step.toDouble();
            TORCH_CHECK(xst > 0 || xst < 0, "step must be nonzero");
            TORCH_CHECK(std::isfinite(xs) && std::isfinite(xe), "unsupported range: ", xs, " -> ", xe);
            TORCH_CHECK(((xst > 0) && (xe >= xs)) || ((xst < 0) && (xe <= xs)),
                        "upper bound and larger bound inconsistent with step sign");
            double size_d = fp ? std::ceil((xe - xs) / xst)
                               : std::ceil(static_cast<double>(end.toLong() - start.toLong()) / step.toLong());
            TORCH_CHECK(size_d >= 0 && size_d <= static_cast<double>(std::numeric_limits<int64_t>::max()),
                        "invalid size, possible overflow?");
            int64_t size = static_cast<int64_t>(size_d);
            if (out.numel() != size)
                out.resize_({size});
            CT ct = fp ? CT::F32 : CT::I32;
            affine_fill(out, "arange", "s0 + f32(idx) * s1", "s0 + i32(idx) * s1", {{start, ct}, {step, ct}});
            return out;
        }

        at::Tensor &linspace_out(const at::Scalar &start, const at::Scalar &end, int64_t steps, at::Tensor &out)
        {
            TORCH_CHECK(steps >= 0, "number of steps must be non-negative");
            if (!wgsl::is_compute_dtype(out.scalar_type()) || out.scalar_type() == at::kBool || out.is_complex())
                return WEBGPU_FALLBACK(linspace, out)(start, end, steps, out);
            if (out.numel() != steps)
                out.resize_({steps});
            if (steps == 0)
                return out;
            if (steps == 1)
                return out.fill_(start);
            double s = start.toDouble(), e = end.toDouble();
            double step = (e - s) / static_cast<double>(steps - 1);
            const int64_t half = steps / 2;
            // Symmetric evaluation, as in PyTorch's linspace kernel.
            affine_fill(out, "linspace",
                        "select(s1 - s2 * f32(u32(s3) - 1u - idx), s0 + s2 * f32(idx), idx < u32(s4))",
                        "i32(select(s1 - s2 * f32(u32(s3) - 1u - idx), s0 + s2 * f32(idx), idx < u32(s4)))",
                        {{s, CT::F32}, {e, CT::F32}, {step, CT::F32}, {at::Scalar(steps), CT::I32}, {at::Scalar(half), CT::I32}});
            return out;
        }
    }

    TORCH_LIBRARY_IMPL(aten, PrivateUse1, m)
    {
        m.impl("empty.memory_format", TORCH_FN(empty_memory_format));
        m.impl("empty_strided", TORCH_FN(empty_strided));
        m.impl("resize_", TORCH_FN(resize_));
        m.impl("set_.source_Storage", TORCH_FN(set_source_storage));
        m.impl("set_.source_Storage_storage_offset", TORCH_FN(set_storage));
        m.impl("set_.source_Tensor", TORCH_FN(set_source_tensor));
        m.impl("set_", TORCH_FN(set_empty));
        m.impl("is_set_to", TORCH_FN(at::native::is_set_to));

        // Views: the generic TensorImpl-based implementations work for any device.
        m.impl("as_strided", TORCH_FN(at::native::as_strided_tensorimpl));
        m.impl("view", TORCH_FN(at::native::view));
        m.impl("_reshape_alias", TORCH_FN(at::native::_reshape_alias));
        m.impl("unfold", TORCH_FN(at::native::unfold));
        m.impl("view_as_real", TORCH_FN(at::native::view_as_real));
        m.impl("view_as_complex", TORCH_FN(at::native::view_as_complex));

        m.impl("_copy_from", TORCH_FN(copy_from));
        m.impl("_copy_from_and_resize", TORCH_FN(_copy_from_and_resize));
        m.impl("_local_scalar_dense", TORCH_FN(_local_scalar_dense));

        m.impl("fill_.Scalar", TORCH_FN(fill_scalar));
        m.impl("fill_.Tensor", TORCH_FN(fill_tensor));
        m.impl("zero_", TORCH_FN(zero_));
        m.impl("arange.start_out", TORCH_FN(arange_out));
        m.impl("linspace.out", TORCH_FN(linspace_out));
    }
}
