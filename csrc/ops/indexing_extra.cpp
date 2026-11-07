// Ops composed from native webgpu kernels: take/put, masked_scatter,
// nonzero_static, repeat_interleave, trace, vdot, addbmm, legacy norm,
// renorm, embedding_renorm_, and reflection / replication padding.

#include <ATen/ATen.h>
#include <ATen/native/Resize.h>
#include <ATen/ops/norm_meta.h>
#include <ATen/ops/reflection_pad1d_backward_meta.h>
#include <ATen/ops/reflection_pad1d_meta.h>
#include <ATen/ops/reflection_pad3d_backward_meta.h>
#include <ATen/ops/reflection_pad3d_meta.h>
#include <ATen/ops/renorm_meta.h>
#include <ATen/ops/replication_pad1d_backward_meta.h>
#include <ATen/ops/replication_pad1d_meta.h>
#include <ATen/ops/replication_pad2d_meta.h>
#include <ATen/ops/replication_pad3d_meta.h>
#include <ATen/ops/channel_shuffle_native.h>
#include <ATen/ops/vdot_native.h>
#include <torch/library.h>

#include "ops/common.h"

namespace torch_webgpu::ops
{
    using at::Scalar;
    using at::Tensor;

    namespace
    {
        // ------------------------------------------------------ take / put
        // Linear (row-major) indices into self; negatives wrap.
        Tensor &take_out_fn(const Tensor &self, const Tensor &index, Tensor &out)
        {
            TORCH_CHECK(index.scalar_type() == at::kLong, "take(): Expected a long tensor for index, but got ",
                        index.scalar_type());
            TORCH_CHECK(!(self.numel() == 0 && index.numel() != 0), "take(): tried to take from an empty tensor");
            at::native::resize_output(out, index.sizes());
            if (index.numel() == 0)
                return out;
            out.copy_(at::index(self.reshape({-1}), {index}));
            return out;
        }
        Tensor take_fn(const Tensor &self, const Tensor &index)
        {
            Tensor out = at::empty({0}, self.options());
            return take_out_fn(self, index, out);
        }

        Tensor &put_inplace_fn(Tensor &self, const Tensor &index, const Tensor &source, bool accumulate)
        {
            TORCH_CHECK(index.scalar_type() == at::kLong, "put_(): Expected a long tensor for index, but got ",
                        index.scalar_type());
            TORCH_CHECK(self.scalar_type() == source.scalar_type(),
                        "put_(): self and source expected to have the same dtype, but got self.dtype = ",
                        self.scalar_type(), " and source.dtype = ", source.scalar_type());
            TORCH_CHECK(index.numel() == source.numel(), "put_(): Expected source and index to have the same number of elements, but got source.numel() = ",
                        source.numel(), ", index.numel() = ", index.numel());
            TORCH_CHECK(!(self.numel() == 0 && index.numel() != 0), "put_(): Tried to put elements into an empty tensor");
            if (index.numel() == 0)
                return self;
            Tensor target = self.is_contiguous() ? self : self.contiguous();
            target.view({-1}).index_put_({index.reshape({-1})}, source.reshape({-1}), accumulate);
            if (!target.is_same(self))
                self.copy_(target);
            return self;
        }

        // -------------------------------------------------- masked_scatter
        // self[mask] = source[:mask.sum()] in row-major order.
        Tensor &masked_scatter_inplace_fn(Tensor &self, const Tensor &mask, const Tensor &source)
        {
            TORCH_CHECK(mask.scalar_type() == at::kBool, "masked_scatter_ only supports boolean masks, but got mask with dtype ",
                        mask.scalar_type());
            TORCH_CHECK(self.scalar_type() == source.scalar_type(),
                        "masked_scatter_: expected self and source to have same dtypes but got", self.scalar_type(),
                        " and ", source.scalar_type());
            Tensor m = mask.expand(self.sizes());
            if (self.numel() == 0)
                return self;
            Tensor positions = m.reshape({-1}).nonzero().view({-1});
            const int64_t count = positions.size(0);
            TORCH_CHECK(source.numel() >= count,
                        "masked_scatter: expected source to have at least as many elements as ones in mask");
            if (count == 0)
                return self;
            Tensor values = source.reshape({-1}).narrow(0, 0, count);
            Tensor target = self.is_contiguous() ? self : self.contiguous();
            target.view({-1}).index_put_({positions}, values);
            if (!target.is_same(self))
                self.copy_(target);
            return self;
        }

        // -------------------------------------------------- nonzero_static
        Tensor &nonzero_static_out_fn(const Tensor &self, int64_t size, int64_t fill_value, Tensor &out)
        {
            TORCH_CHECK(size >= 0, "nonzero_static: 'size' must be an non-negative integer");
            Tensor nz = at::nonzero(self);
            const int64_t ndim = self.dim();
            at::native::resize_output(out, {size, ndim});
            if (size == 0 || ndim == 0)
                return out;
            out.fill_(fill_value);
            const int64_t n = std::min<int64_t>(size, nz.size(0));
            if (n > 0)
                out.narrow(0, 0, n).copy_(nz.narrow(0, 0, n));
            return out;
        }
        Tensor nonzero_static_fn(const Tensor &self, int64_t size, int64_t fill_value)
        {
            Tensor out = at::empty({0}, self.options().dtype(at::kLong));
            return nonzero_static_out_fn(self, size, fill_value, out);
        }

        // ------------------------------------------------ repeat_interleave
        // repeats [r0, r1, ...] -> [0]*r0 ++ [1]*r1 ++ ...: for each output
        // position, the number of cumulative counts <= it.
        Tensor repeat_interleave_fn(const Tensor &repeats, std::optional<c10::SymInt> output_size)
        {
            TORCH_CHECK(repeats.dim() == 1, "repeat_interleave only accept 1D vector as repeat");
            TORCH_CHECK(repeats.scalar_type() == at::kLong || repeats.scalar_type() == at::kInt,
                        "repeats has to be Long or Int tensor");
            if (repeats.numel() == 0)
                return at::empty({0}, repeats.options());
            TORCH_CHECK(repeats.min().item<int64_t>() >= 0, "repeats can not be negative");
            Tensor cum = repeats.cumsum(0);
            const int64_t total = output_size ? output_size->expect_int() : cum[-1].item<int64_t>();
            TORCH_CHECK(!output_size || cum[-1].item<int64_t>() == total, "allocated size does not match required size");
            Tensor pos = at::arange(total, repeats.options());
            return at::searchsorted(cum, pos, /*out_int32=*/repeats.scalar_type() == at::kInt, /*right=*/true);
        }

        // ------------------------------------------------- trace / addbmm
        Tensor trace_fn(const Tensor &self)
        {
            TORCH_CHECK(self.dim() == 2, "trace: expected a matrix, but got tensor with dim ", self.dim());
            // Integral inputs accumulate in int64, as on CPU.
            return at::isIntegralType(self.scalar_type(), /*includeBool=*/true)
                       ? self.diagonal().sum(at::kLong)
                       : self.diagonal().sum();
        }

        // addbmm: beta * self + alpha * sum_b batch1[b] @ batch2[b], as one
        // GEMM over the concatenated reduction dimension.
        Tensor &addbmm_out_fn(const Tensor &self, const Tensor &batch1, const Tensor &batch2, const Scalar &beta,
                              const Scalar &alpha, Tensor &out)
        {
            TORCH_CHECK(batch1.dim() == 3, "batch1 must be a 3D tensor");
            TORCH_CHECK(batch2.dim() == 3, "batch2 must be a 3D tensor");
            TORCH_CHECK(batch1.size(0) == batch2.size(0), "batch1 and batch2 must have same number of batches, got ",
                        batch1.size(0), " and ", batch2.size(0));
            TORCH_CHECK(batch1.size(2) == batch2.size(1), "Incompatible matrix sizes for bmm (", batch1.size(1), "x",
                        batch1.size(2), " and ", batch2.size(1), "x", batch2.size(2), ")");
            const int64_t B = batch1.size(0), n = batch1.size(1), k = batch1.size(2), m = batch2.size(2);
            Tensor a = batch1.permute({1, 0, 2}).reshape({n, B * k});
            Tensor w = batch2.reshape({B * k, m});
            at::addmm_out(out, self.expand({n, m}), a, w, beta, alpha);
            return out;
        }
        Tensor addbmm_fn(const Tensor &self, const Tensor &batch1, const Tensor &batch2, const Scalar &beta,
                         const Scalar &alpha)
        {
            Tensor out = at::empty({0}, self.options());
            return addbmm_out_fn(self, batch1, batch2, beta, alpha, out);
        }
        Tensor &addbmm_inplace_fn(Tensor &self, const Tensor &batch1, const Tensor &batch2, const Scalar &beta,
                                  const Scalar &alpha)
        {
            Tensor tmp = addbmm_fn(self, batch1, batch2, beta, alpha);
            TORCH_CHECK(tmp.sizes() == self.sizes(), "addbmm_: output shape ", tmp.sizes(),
                        " doesn't match self ", self.sizes());
            return self.copy_(tmp);
        }

        // ---------------------------------------------- legacy norm.out
        Tensor &norm_out_fn(const Tensor &self, const std::optional<Scalar> &p, at::IntArrayRef dim, bool keepdim,
                            Tensor &out)
        {
            return at::linalg_vector_norm_out(out, self, p.value_or(2.0), dim, keepdim, std::nullopt);
        }
        Tensor &norm_dtype_out_fn(const Tensor &self, const std::optional<Scalar> &p, at::IntArrayRef dim, bool keepdim,
                                  at::ScalarType dtype, Tensor &out)
        {
            return at::linalg_vector_norm_out(out, self, p.value_or(2.0), dim, keepdim, dtype);
        }

        // ---------------------------------------------------------- renorm
        // Sub-tensors along `dim` whose p-norm exceeds maxnorm are scaled to
        // maxnorm / (norm + 1e-7), as CPU/CUDA.
        void renorm_impl(const Tensor &self, const Scalar &p, int64_t dim, const Scalar &maxnorm, const Tensor &out)
        {
            dim = at::maybe_wrap_dim(dim, self.dim());
            std::vector<int64_t> reduce_dims;
            for (int64_t d = 0; d < self.dim(); ++d)
                if (d != dim)
                    reduce_dims.push_back(d);
            Tensor norm = at::linalg_vector_norm(self, p, reduce_dims, /*keepdim=*/true, at::kFloat);
            const double mx = maxnorm.toDouble();
            Tensor factor = at::where(norm > mx, mx / (norm + 1e-7), at::ones_like(norm));
            at::mul_out(const_cast<Tensor &>(out), self, factor.to(self.scalar_type()));
        }
        Tensor renorm_fn(const Tensor &self, const Scalar &p, int64_t dim, const Scalar &maxnorm)
        {
            Functional<at::meta::structured_renorm> op;
            op.meta(self, p, dim, maxnorm);
            renorm_impl(self, p, dim, maxnorm, op.outputs_[0]);
            return std::move(op.outputs_[0]);
        }
        Tensor &renorm_out_fn(const Tensor &self, const Scalar &p, int64_t dim, const Scalar &maxnorm, Tensor &out)
        {
            Out<at::meta::structured_renorm> op(out);
            op.meta(self, p, dim, maxnorm);
            renorm_impl(self, p, dim, maxnorm, op.maybe_get_output(0));
            op.finish();
            return out;
        }
        Tensor &renorm_inplace_fn(Tensor &self, const Scalar &p, int64_t dim, const Scalar &maxnorm)
        {
            Inplace<at::meta::structured_renorm> op(self);
            op.meta(self, p, dim, maxnorm);
            Tensor tmp = at::empty_like(self);
            renorm_impl(self, p, dim, maxnorm, tmp);
            self.copy_(tmp);
            return self;
        }

        // ------------------------------------------------ embedding_renorm_
        Tensor &embedding_renorm_fn(Tensor &self, const Tensor &indices, double max_norm, double norm_type)
        {
            TORCH_CHECK(self.dim() == 2, "'self' must be 2-D, but got ", self.dim(), "-D");
            TORCH_CHECK(indices.scalar_type() == at::kLong || indices.scalar_type() == at::kInt,
                        "embedding_renorm_(): Expected indices to be Long or Int");
            if (indices.numel() == 0)
                return self;
            Tensor rows = std::get<0>(at::_unique2(indices.reshape({-1}).to(at::kLong), /*sorted=*/true, false, false));
            Tensor sel = self.index_select(0, rows);
            Tensor norm = at::linalg_vector_norm(sel, norm_type, {1}, /*keepdim=*/true, at::kFloat);
            Tensor scale = at::where(norm > max_norm, max_norm / (norm + 1e-7), at::ones_like(norm));
            self.index_copy_(0, rows, sel * scale.to(self.scalar_type()));
            return self;
        }

        // ------------------------------------------- triu / tril indices
        // Pure index generation: computed on the host, uploaded once.
        Tensor tri_indices(bool upper, int64_t row, int64_t col, int64_t offset, std::optional<at::ScalarType> dtype,
                           std::optional<at::Layout> layout, std::optional<at::Device> device, std::optional<bool> pin)
        {
            auto cpu = upper ? at::triu_indices(row, col, offset, dtype, layout, at::kCPU, pin)
                             : at::tril_indices(row, col, offset, dtype, layout, at::kCPU, pin);
            return cpu.to(device.value_or(at::Device(c10::DeviceType::PrivateUse1)));
        }
        Tensor triu_indices_fn(int64_t row, int64_t col, int64_t offset, std::optional<at::ScalarType> dtype,
                               std::optional<at::Layout> layout, std::optional<at::Device> device, std::optional<bool> pin)
        {
            return tri_indices(true, row, col, offset, dtype, layout, device, pin);
        }
        Tensor tril_indices_fn(int64_t row, int64_t col, int64_t offset, std::optional<at::ScalarType> dtype,
                               std::optional<at::Layout> layout, std::optional<at::Device> device, std::optional<bool> pin)
        {
            return tri_indices(false, row, col, offset, dtype, layout, device, pin);
        }

        // ---------------------------------------------------------- padding
        // Padding is separable: per padded dim, out = in.index_select(d, idx)
        // with idx mapping output to input positions (reflect / clamp);
        // backward accumulates with index_add in reverse order.
        Tensor pad_indices(int64_t n, int64_t before, int64_t after, bool reflect, const at::TensorOptions &opt)
        {
            const int64_t out_n = n + before + after;
            std::vector<int64_t> idx(out_n);
            for (int64_t i = 0; i < out_n; ++i)
            {
                int64_t j = i - before;
                if (reflect)
                {
                    if (j < 0)
                        j = -j;
                    if (j >= n)
                        j = 2 * (n - 1) - j;
                }
                else
                    j = std::min(std::max<int64_t>(j, 0), n - 1);
                idx[i] = j;
            }
            return at::tensor(idx, at::TensorOptions().dtype(at::kLong)).to(opt.device());
        }

        // padding = (last-dim before, last-dim after, second-last before, ...)
        Tensor pad_forward(const Tensor &self, at::IntArrayRef padding, bool reflect)
        {
            const int64_t k = static_cast<int64_t>(padding.size()) / 2;
            Tensor x = self;
            for (int64_t i = 0; i < k; ++i)
            {
                const int64_t d = self.dim() - 1 - i;
                x = x.index_select(d, pad_indices(self.size(d), padding[2 * i], padding[2 * i + 1], reflect, self.options()));
            }
            return x;
        }
        Tensor pad_backward(const Tensor &grad_output, const Tensor &self, at::IntArrayRef padding, bool reflect)
        {
            const int64_t k = static_cast<int64_t>(padding.size()) / 2;
            Tensor g = grad_output;
            for (int64_t i = k - 1; i >= 0; --i)
            {
                const int64_t d = self.dim() - 1 - i;
                auto sizes = g.sizes().vec();
                sizes[d] = self.size(d);
                Tensor acc = at::zeros(sizes, grad_output.options());
                acc.index_add_(d, pad_indices(self.size(d), padding[2 * i], padding[2 * i + 1], reflect, self.options()), g);
                g = acc;
            }
            return g;
        }

        void check_pad_dims(const Tensor &self, int64_t k, const char *name)
        {
            const int64_t dim_w = self.dim();
            TORCH_CHECK(dim_w == k + 1 || dim_w == k + 2, name, ": expected ", k + 1, "D or ", k + 2,
                        "D input tensor, but got ", self.sizes());
        }

#define WG_PAD_STRUCTURED(NAME, REFLECT)                                                              \
    Tensor NAME##_fn(const Tensor &self, at::IntArrayRef padding)                                     \
    {                                                                                                 \
        Functional<at::meta::structured_##NAME> op;                                                   \
        op.meta(self, padding);                                                                       \
        op.outputs_[0].copy_(pad_forward(self, padding, REFLECT));                                    \
        return std::move(op.outputs_[0]);                                                             \
    }                                                                                                 \
    Tensor &NAME##_out_fn(const Tensor &self, at::IntArrayRef padding, Tensor &out)                   \
    {                                                                                                 \
        Out<at::meta::structured_##NAME> op(out);                                                     \
        op.meta(self, padding);                                                                       \
        op.maybe_get_output(0).copy_(pad_forward(self, padding, REFLECT));                            \
        op.finish();                                                                                  \
        return out;                                                                                   \
    }
        WG_PAD_STRUCTURED(reflection_pad1d, true)
        WG_PAD_STRUCTURED(reflection_pad3d, true)
        WG_PAD_STRUCTURED(replication_pad1d, false)
        WG_PAD_STRUCTURED(replication_pad2d, false)
        WG_PAD_STRUCTURED(replication_pad3d, false)
#undef WG_PAD_STRUCTURED

#define WG_PAD_BACKWARD_STRUCTURED(NAME, REFLECT)                                                                 \
    Tensor NAME##_fn(const Tensor &grad_output, const Tensor &self, at::IntArrayRef padding)                      \
    {                                                                                                             \
        Functional<at::meta::structured_##NAME> op;                                                               \
        op.meta(grad_output, self, padding);                                                                      \
        op.outputs_[0].copy_(pad_backward(grad_output, self, padding, REFLECT));                                  \
        return std::move(op.outputs_[0]);                                                                         \
    }                                                                                                             \
    Tensor &NAME##_out_fn(const Tensor &grad_output, const Tensor &self, at::IntArrayRef padding, Tensor &out)    \
    {                                                                                                             \
        Out<at::meta::structured_##NAME> op(out);                                                                 \
        op.meta(grad_output, self, padding);                                                                      \
        op.maybe_get_output(0).copy_(pad_backward(grad_output, self, padding, REFLECT));                          \
        op.finish();                                                                                              \
        return out;                                                                                               \
    }
        WG_PAD_BACKWARD_STRUCTURED(reflection_pad1d_backward, true)
        WG_PAD_BACKWARD_STRUCTURED(reflection_pad3d_backward, true)
        WG_PAD_BACKWARD_STRUCTURED(replication_pad1d_backward, false)
#undef WG_PAD_BACKWARD_STRUCTURED

        // Unstructured variants: reflection_pad2d and some backward ops.
        Tensor &reflection_pad2d_out_fn(const Tensor &self, at::IntArrayRef padding, Tensor &out)
        {
            TORCH_CHECK(padding.size() == 4, "padding size is expected to be 4");
            check_pad_dims(self, 2, "reflection_pad2d");
            const int64_t W = self.size(-1), H = self.size(-2);
            TORCH_CHECK(padding[0] < W && padding[1] < W,
                        "Argument #4: Padding size should be less than the corresponding input dimension, but got: padding (",
                        padding[0], ", ", padding[1], ") at dimension ", self.dim() - 1, " of input ", self.sizes());
            TORCH_CHECK(padding[2] < H && padding[3] < H,
                        "Argument #6: Padding size should be less than the corresponding input dimension, but got: padding (",
                        padding[2], ", ", padding[3], ") at dimension ", self.dim() - 2, " of input ", self.sizes());
            const int64_t oh = H + padding[2] + padding[3], ow = W + padding[0] + padding[1];
            TORCH_CHECK(ow >= 1 || oh >= 1, "input (H: ", H, ", W: ", W, ") is too small. Calculated output H: ", oh,
                        " W: ", ow);
            Tensor r = pad_forward(self, padding, true);
            at::native::resize_output(out, r.sizes());
            out.copy_(r);
            return out;
        }
        Tensor reflection_pad2d_fn(const Tensor &self, at::IntArrayRef padding)
        {
            Tensor out = at::empty({0}, self.options());
            return reflection_pad2d_out_fn(self, padding, out);
        }

#define WG_PAD_BACKWARD(NAME, REFLECT)                                                                         \
    Tensor &NAME##_out_fn(const Tensor &grad_output, const Tensor &self, at::IntArrayRef padding, Tensor &out) \
    {                                                                                                          \
        Tensor r = pad_backward(grad_output, self, padding, REFLECT);                                          \
        at::native::resize_output(out, self.sizes());                                                          \
        out.copy_(r);                                                                                          \
        return out;                                                                                            \
    }                                                                                                          \
    Tensor NAME##_fn(const Tensor &grad_output, const Tensor &self, at::IntArrayRef padding)                   \
    {                                                                                                          \
        return pad_backward(grad_output, self, padding, REFLECT);                                              \
    }
        WG_PAD_BACKWARD(reflection_pad2d_backward, true)
        WG_PAD_BACKWARD(replication_pad2d_backward, false)
        WG_PAD_BACKWARD(replication_pad3d_backward, false)
#undef WG_PAD_BACKWARD
    }

    TORCH_LIBRARY_IMPL(aten, PrivateUse1, m)
    {
        m.impl("triu_indices", TORCH_FN(triu_indices_fn));
        m.impl("tril_indices", TORCH_FN(tril_indices_fn));
        m.impl("take", TORCH_FN(take_fn));
        m.impl("take.out", TORCH_FN(take_out_fn));
        m.impl("put_", TORCH_FN(put_inplace_fn));
        m.impl("masked_scatter_", TORCH_FN(masked_scatter_inplace_fn));
        m.impl("nonzero_static", TORCH_FN(nonzero_static_fn));
        m.impl("nonzero_static.out", TORCH_FN(nonzero_static_out_fn));
        m.impl("repeat_interleave.Tensor", TORCH_FN(repeat_interleave_fn));
        m.impl("trace", TORCH_FN(trace_fn));
        m.impl("vdot", TORCH_FN(at::native::vdot));
        m.impl("channel_shuffle", TORCH_FN(at::native::channel_shuffle));
        m.impl("addbmm", TORCH_FN(addbmm_fn));
        m.impl("addbmm.out", TORCH_FN(addbmm_out_fn));
        m.impl("addbmm_", TORCH_FN(addbmm_inplace_fn));
        m.impl("norm.out", TORCH_FN(norm_out_fn));
        m.impl("norm.dtype_out", TORCH_FN(norm_dtype_out_fn));
        m.impl("renorm", TORCH_FN(renorm_fn));
        m.impl("renorm.out", TORCH_FN(renorm_out_fn));
        m.impl("renorm_", TORCH_FN(renorm_inplace_fn));
        m.impl("embedding_renorm_", TORCH_FN(embedding_renorm_fn));
        m.impl("reflection_pad1d", TORCH_FN(reflection_pad1d_fn));
        m.impl("reflection_pad1d.out", TORCH_FN(reflection_pad1d_out_fn));
        m.impl("reflection_pad2d", TORCH_FN(reflection_pad2d_fn));
        m.impl("reflection_pad2d.out", TORCH_FN(reflection_pad2d_out_fn));
        m.impl("reflection_pad3d", TORCH_FN(reflection_pad3d_fn));
        m.impl("reflection_pad3d.out", TORCH_FN(reflection_pad3d_out_fn));
        m.impl("replication_pad1d", TORCH_FN(replication_pad1d_fn));
        m.impl("replication_pad1d.out", TORCH_FN(replication_pad1d_out_fn));
        m.impl("replication_pad2d", TORCH_FN(replication_pad2d_fn));
        m.impl("replication_pad2d.out", TORCH_FN(replication_pad2d_out_fn));
        m.impl("replication_pad3d", TORCH_FN(replication_pad3d_fn));
        m.impl("replication_pad3d.out", TORCH_FN(replication_pad3d_out_fn));
        m.impl("reflection_pad1d_backward", TORCH_FN(reflection_pad1d_backward_fn));
        m.impl("reflection_pad1d_backward.grad_input", TORCH_FN(reflection_pad1d_backward_out_fn));
        m.impl("reflection_pad2d_backward", TORCH_FN(reflection_pad2d_backward_fn));
        m.impl("reflection_pad2d_backward.grad_input", TORCH_FN(reflection_pad2d_backward_out_fn));
        m.impl("reflection_pad3d_backward", TORCH_FN(reflection_pad3d_backward_fn));
        m.impl("reflection_pad3d_backward.grad_input", TORCH_FN(reflection_pad3d_backward_out_fn));
        m.impl("replication_pad1d_backward", TORCH_FN(replication_pad1d_backward_fn));
        m.impl("replication_pad1d_backward.grad_input", TORCH_FN(replication_pad1d_backward_out_fn));
        m.impl("replication_pad2d_backward", TORCH_FN(replication_pad2d_backward_fn));
        m.impl("replication_pad2d_backward.grad_input", TORCH_FN(replication_pad2d_backward_out_fn));
        m.impl("replication_pad3d_backward", TORCH_FN(replication_pad3d_backward_fn));
        m.impl("replication_pad3d_backward.grad_input", TORCH_FN(replication_pad3d_backward_out_fn));
    }
}
