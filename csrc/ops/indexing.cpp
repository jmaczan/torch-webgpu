// Indexing ops: gather/scatter family, advanced indexing, index_select/add/
// copy/fill, embedding backward, nonzero/masked_select, cat, flip, triu/tril.

#include <ATen/ATen.h>
#include <ATen/native/IndexingUtils.h>
#include <ATen/ExpandUtils.h>
#include <ATen/native/Resize.h>
#include <ATen/ops/cat_meta.h>
#include <ATen/ops/gather_meta.h>
#include <ATen/ops/index_add_meta.h>
#include <ATen/ops/index_copy_meta.h>
#include <ATen/ops/index_meta.h>
#include <ATen/ops/scatter_add_meta.h>
#include <ATen/ops/scatter_meta.h>
#include <ATen/ops/scatter_reduce_meta.h>
#include <ATen/ops/tril_meta.h>
#include <ATen/ops/triu_meta.h>
#include <torch/library.h>

#include "kernels/elementwise.h"
#include "kernels/index.h"
#include "ops/common.h"

namespace torch_webgpu::ops
{
    using at::Scalar;
    using at::Tensor;
    using kernels::ScatterMode;

    namespace
    {
        at::TensorIteratorConfig index_config()
        {
            at::TensorIteratorConfig c;
            c.check_all_same_dtype(false).set_check_mem_overlap(false).resize_outputs(false);
            return c;
        }

        // `t` viewed with `shape` and stride 0 along `dim`.
        Tensor restride(const Tensor &t, int64_t dim, at::IntArrayRef shape)
        {
            if (t.dim() == 0)
                return t.as_strided(shape, std::vector<int64_t>(shape.size(), 0), t.storage_offset());
            auto st = t.strides().vec();
            st[dim] = 0;
            return t.as_strided(shape, st, t.storage_offset());
        }

        // Port of at::native::make_info / AdvancedIndex (whose constructor is
        // not exported from libtorch): expands masks, broadcasts indices,
        // moves indexed dims to the front when they are not adjacent and
        // restrides `src` so the indexed subspace has stride 0.
        struct IndexInfo
        {
            Tensor src;
            std::vector<Tensor> indices;
            std::vector<int64_t> indexed_sizes;
            std::vector<int64_t> indexed_strides; // elements
        };

        IndexInfo make_index_info(Tensor self, const c10::List<std::optional<Tensor>> &orig)
        {
            at::native::checkIndexTensorTypes(orig, /*allow_int=*/true);
            auto indices = at::native::expandTensors(self, orig, /*ensure_same_device=*/true);
            try
            {
                indices = at::expand_outplace(indices);
            }
            catch (std::exception &)
            {
                TORCH_CHECK_INDEX(false, "shape mismatch: indexing tensors could not be broadcast together");
            }
            while (indices.size() < static_cast<size_t>(self.dim()))
                indices.emplace_back();
            if (!at::native::hasContiguousSubspace(indices))
                std::tie(self, indices) = at::native::transposeToFront(self, indices);

            IndexInfo info;
            int64_t dims_before = 0, dims_after = 0, dims_indexed = 0;
            at::IntArrayRef replacement_shape;
            for (size_t d = 0; d < indices.size(); ++d)
            {
                if (!indices[d].defined())
                {
                    (dims_indexed == 0 ? dims_before : dims_after)++;
                    continue;
                }
                dims_indexed++;
                replacement_shape = indices[d].sizes();
                info.indexed_sizes.push_back(self.size(d));
                info.indexed_strides.push_back(self.stride(d));
            }
            if (std::find(info.indexed_sizes.begin(), info.indexed_sizes.end(), 0) != info.indexed_sizes.end() &&
                std::find(replacement_shape.begin(), replacement_shape.end(), 0) == replacement_shape.end())
                TORCH_CHECK_INDEX(false, "index is out of bounds for dimension with size 0");

            auto shape = self.sizes().vec();
            auto strides = self.strides().vec();
            shape.erase(shape.begin() + dims_before, shape.begin() + dims_before + dims_indexed);
            strides.erase(strides.begin() + dims_before, strides.begin() + dims_before + dims_indexed);
            shape.insert(shape.begin() + dims_before, replacement_shape.begin(), replacement_shape.end());
            strides.insert(strides.begin() + dims_before, replacement_shape.size(), 0);
            info.src = self.as_strided(shape, strides, self.storage_offset());

            for (auto &index : indices)
            {
                if (!index.defined())
                    continue;
                std::vector<int64_t> ishape(dims_before, 1);
                ishape.insert(ishape.end(), index.sizes().begin(), index.sizes().end());
                ishape.insert(ishape.end(), dims_after, 1);
                Tensor idx = index.scalar_type() == at::kInt ? index.to(at::kLong) : index;
                info.indices.push_back(idx.reshape(ishape));
            }
            return info;
        }

        int64_t size_or_1(const Tensor &t, int64_t dim) { return t.dim() == 0 ? 1 : t.size(dim); }
        int64_t stride_or_1(const Tensor &t, int64_t dim) { return t.dim() == 0 ? 1 : t.stride(dim); }

        // ----------------------------------------------------- gather
        void gather_impl(const Tensor &self, int64_t dim, const Tensor &index, const Tensor &out)
        {
            if (index.numel() == 0)
                return;
            dim = at::maybe_wrap_dim(dim, std::max<int64_t>(self.dim(), 1));
            Tensor src = restride(self, dim, index.sizes());
            auto iter = index_config().add_output(out).add_const_input(src).add_const_input(index).build();
            kernels::run_gather(iter, {size_or_1(self, dim)}, {stride_or_1(self, dim)});
        }
        Tensor gather_fn(const Tensor &self, int64_t dim, const Tensor &index, bool sparse_grad)
        {
            Functional<at::meta::structured_gather> op;
            op.meta(self, dim, index, sparse_grad);
            gather_impl(self, dim, index, op.outputs_[0]);
            return std::move(op.outputs_[0]);
        }
        Tensor &gather_out_fn(const Tensor &self, int64_t dim, const Tensor &index, bool sparse_grad, Tensor &out)
        {
            Out<at::meta::structured_gather> op(out);
            op.meta(self, dim, index, sparse_grad);
            gather_impl(self, dim, index, op.maybe_get_output(0));
            op.finish();
            return out;
        }

        // ---------------------------------------------------- scatter
        // dst[..index[i]..] (op)= src[i] along `dim`, dst already holds self.
        void scatter_impl(const Tensor &dst, int64_t dim, const Tensor &index, const Tensor &src, ScatterMode mode)
        {
            if (index.numel() == 0)
                return;
            dim = at::maybe_wrap_dim(dim, std::max<int64_t>(dst.dim(), 1));
            Tensor d = restride(dst, dim, index.sizes());
            Tensor s = src.dim() == 0 ? src.expand(index.sizes()) : src.as_strided(index.sizes(), src.strides(), src.storage_offset());
            auto iter = index_config().add_output(d).add_const_input(s).add_const_input(index).build();
            kernels::run_scatter(iter, {size_or_1(dst, dim)}, {stride_or_1(dst, dim)}, mode);
        }

        ScatterMode reduce_mode(c10::string_view reduce)
        {
            if (reduce == "add" || reduce == "sum")
                return ScatterMode::Add;
            if (reduce == "multiply" || reduce == "prod")
                return ScatterMode::Mul;
            if (reduce == "amax")
                return ScatterMode::Max;
            if (reduce == "amin")
                return ScatterMode::Min;
            TORCH_CHECK(false, "torch_webgpu: unsupported scatter reduction '", reduce, "'");
        }

        bool scatter_supported(const Tensor &self, ScatterMode mode)
        {
            return mode == ScatterMode::Assign ? wgsl::is_storage_dtype(self.scalar_type())
                                               : (wgsl::is_compute_dtype(self.scalar_type()) && self.element_size() <= 4);
        }

        Tensor value_tensor(const Tensor &self, const Scalar &value)
        {
            return at::scalar_tensor(value, self.options());
        }

        // Generates functional / out / inplace variants of a scatter op.
#define WG_SCATTER(ID, META, PARAMS, ARGS, SRC_EXPR, MODE_EXPR, FB_F, FB_O, FB_I)                         \
    Tensor ID##_fn(const Tensor &self, int64_t dim, const Tensor &index PARAMS)                          \
    {                                                                                                    \
        auto mode = MODE_EXPR;                                                                           \
        if (!scatter_supported(self, mode))                                                              \
            return FB_F(self, dim, index ARGS);                                                          \
        Functional<at::meta::META> op;                                                                   \
        op.meta(self, dim, index ARGS);                                                                  \
        Tensor &out = op.outputs_[0];                                                                    \
        out.copy_(self);                                                                                 \
        scatter_impl(out, dim, index, SRC_EXPR, mode);                                                   \
        return std::move(out);                                                                           \
    }                                                                                                    \
    Tensor &ID##_out_fn(const Tensor &self, int64_t dim, const Tensor &index PARAMS, Tensor &out)        \
    {                                                                                                    \
        auto mode = MODE_EXPR;                                                                           \
        if (!scatter_supported(self, mode))                                                              \
            return FB_O(self, dim, index ARGS, out);                                                     \
        Out<at::meta::META> op(out);                                                                     \
        op.meta(self, dim, index ARGS);                                                                  \
        const Tensor &o = op.maybe_get_output(0);                                                        \
        if (!o.is_same(self))                                                                            \
            o.copy_(self);                                                                               \
        scatter_impl(o, dim, index, SRC_EXPR, mode);                                                     \
        op.finish();                                                                                     \
        return out;                                                                                      \
    }                                                                                                    \
    Tensor &ID##_inplace_fn(Tensor &self, int64_t dim, const Tensor &index PARAMS)                       \
    {                                                                                                    \
        auto mode = MODE_EXPR;                                                                           \
        if (!scatter_supported(self, mode))                                                              \
            return FB_I(self, dim, index ARGS);                                                          \
        Inplace<at::meta::META> op(self);                                                                \
        op.meta(self, dim, index ARGS);                                                                  \
        scatter_impl(op.maybe_get_output(0), dim, index, SRC_EXPR, mode);                                \
        op.finish();                                                                                     \
        return self;                                                                                     \
    }
#define SP1(T, A) , T A
#define SA1(A) , A
#define SP2(T1, A1, T2, A2) , T1 A1, T2 A2
#define SA2(A1, A2) , A1, A2
        WG_SCATTER(scatter_src, structured_scatter_src, SP1(const Tensor &, src), SA1(src), src, ScatterMode::Assign,
                   WEBGPU_FALLBACK(scatter, src), WEBGPU_FALLBACK(scatter, src_out), WEBGPU_FALLBACK(scatter_, src))
        WG_SCATTER(scatter_value, structured_scatter_value, SP1(const Scalar &, value), SA1(value),
                   value_tensor(self, value), ScatterMode::Assign, WEBGPU_FALLBACK(scatter, value),
                   WEBGPU_FALLBACK(scatter, value_out), WEBGPU_FALLBACK(scatter_, value))
        WG_SCATTER(scatter_add, structured_scatter_add, SP1(const Tensor &, src), SA1(src), src, ScatterMode::Add,
                   WEBGPU_FALLBACK0(scatter_add), WEBGPU_FALLBACK(scatter_add, out), WEBGPU_FALLBACK0(scatter_add_))
        WG_SCATTER(scatter_reduce, structured_scatter_reduce, SP2(const Tensor &, src, c10::string_view, reduce),
                   SA2(src, reduce), src, reduce_mode(reduce), WEBGPU_FALLBACK(scatter, reduce),
                   WEBGPU_FALLBACK(scatter, reduce_out), WEBGPU_FALLBACK(scatter_, reduce))
        WG_SCATTER(scatter_value_reduce, structured_scatter_value_reduce, SP2(const Scalar &, value, c10::string_view, reduce),
                   SA2(value, reduce), value_tensor(self, value), reduce_mode(reduce),
                   WEBGPU_FALLBACK(scatter, value_reduce), WEBGPU_FALLBACK(scatter, value_reduce_out),
                   WEBGPU_FALLBACK(scatter_, value_reduce))

        // scatter_reduce.two(self, dim, index, src, reduce, include_self)
        Tensor scatter_reduce_two_fn(const Tensor &self, int64_t dim, const Tensor &index, const Tensor &src,
                                     c10::string_view reduce, bool include_self)
        {
            if (reduce == "mean" || !include_self || !scatter_supported(self, ScatterMode::Add))
                return WEBGPU_FALLBACK(scatter_reduce, two)(self, dim, index, src, reduce, include_self);
            Functional<at::meta::structured_scatter_reduce_two> op;
            op.meta(self, dim, index, src, reduce, include_self);
            Tensor &out = op.outputs_[0];
            out.copy_(self);
            scatter_impl(out, dim, index, src, reduce_mode(reduce));
            return std::move(out);
        }

        // --------------------------------------------- advanced indexing
        Tensor index_fn(const Tensor &self, const c10::List<std::optional<Tensor>> &indices)
        {
            if (!wgsl::is_storage_dtype(self.scalar_type()))
                return WEBGPU_FALLBACK(index, Tensor)(self, indices);
            Functional<at::meta::structured_index_Tensor> op;
            auto pre = op.meta(self, indices);
            std::vector<int64_t> strides;
            for (auto s : pre.strides)
                strides.push_back(s / static_cast<int64_t>(self.element_size()));
            kernels::run_gather(op, pre.sizes, strides, /*wrap_negative=*/true);
            return std::move(op.outputs_[0]);
        }
        Tensor &index_out_fn(const Tensor &self, const c10::List<std::optional<Tensor>> &indices, Tensor &out)
        {
            Out<at::meta::structured_index_Tensor> op(out);
            auto pre = op.meta(self, indices);
            std::vector<int64_t> strides;
            for (auto s : pre.strides)
                strides.push_back(s / static_cast<int64_t>(self.element_size()));
            kernels::run_gather(op, pre.sizes, strides, /*wrap_negative=*/true);
            op.finish();
            return out;
        }

        Tensor &index_put_impl_fn(Tensor &self, const c10::List<std::optional<Tensor>> &indices, const Tensor &values,
                                  bool accumulate, bool unsafe)
        {
            auto mode = accumulate ? ScatterMode::Add : ScatterMode::Assign;
            if (!scatter_supported(self, mode))
                return WEBGPU_FALLBACK0(_index_put_impl_)(self, indices, values, accumulate, unsafe);
            TORCH_CHECK_INDEX(static_cast<int64_t>(indices.size()) <= self.dim(), "too many indices for tensor of dimension ",
                              self.dim(), " (got ", indices.size(), ")");
            if (self.numel() == 0)
                return self;
            Tensor value = values.to(self.device(), self.scalar_type());
            auto info = make_index_info(self, indices);
            auto config = index_config();
            config.add_output(info.src).add_const_input(value);
            for (auto &idx : info.indices)
                config.add_const_input(idx);
            auto iter = config.build();
            kernels::run_scatter(iter, info.indexed_sizes, info.indexed_strides, mode, /*wrap_negative=*/true);
            return self;
        }

        // --------------------------------------------------- index_select
        Tensor &index_select_out_fn(const Tensor &self, int64_t dim, const Tensor &index, Tensor &out)
        {
            TORCH_CHECK_INDEX(index.dim() <= 1, "index_select(): Index is supposed to be a vector");
            TORCH_CHECK(index.scalar_type() == at::kLong || index.scalar_type() == at::kInt,
                        "index_select(): Expected dtype int32 or int64 for index");
            TORCH_CHECK(self.scalar_type() == out.scalar_type(), "index_select(): self and result must have the same scalar type");
            dim = at::maybe_wrap_dim(dim, std::max<int64_t>(self.dim(), 1));
            Tensor idx = index.reshape({-1});
            std::vector<int64_t> shape = self.sizes().vec();
            if (self.dim() == 0)
            {
                TORCH_CHECK_INDEX(idx.numel() == 1, "index_select(): Index to scalar can have only 1 value, got ",
                                  idx.numel(), " value(s)");
                at::native::resize_output(out, {});
                Tensor src = self.view({1}).as_strided({1}, {0}, self.storage_offset());
                Tensor o = out.view({1});
                auto iter = index_config().add_output(o).add_const_input(src).add_const_input(idx).build();
                kernels::run_gather(iter, {1}, {1});
                return out;
            }
            shape[dim] = idx.numel();
            at::native::resize_output(out, shape);
            if (out.numel() == 0)
                return out;
            Tensor src = restride(self, dim, shape);
            std::vector<int64_t> ist(shape.size(), 0);
            ist[dim] = idx.stride(0);
            Tensor idx_r = idx.as_strided(shape, ist, idx.storage_offset());
            auto iter = index_config().add_output(out).add_const_input(src).add_const_input(idx_r).build();
            kernels::run_gather(iter, {self.size(dim)}, {self.stride(dim)});
            return out;
        }
        Tensor index_select_fn(const Tensor &self, int64_t dim, const Tensor &index)
        {
            Tensor out = at::empty({0}, self.options());
            return index_select_out_fn(self, dim, index, out);
        }

        // Writes `source` slices into dst along dim at `index` (dst is the result).
        void index_scatter_dim(const Tensor &dst, int64_t dim, const Tensor &index, const Tensor &source, ScatterMode mode)
        {
            if (index.numel() == 0)
                return;
            dim = at::maybe_wrap_dim(dim, std::max<int64_t>(dst.dim(), 1));
            Tensor idx = index.reshape({-1});
            if (dst.dim() == 0)
            {
                Tensor d = dst.view({1}).as_strided({1}, {0}, dst.storage_offset());
                Tensor src1 = source.reshape({1});
                auto iter = index_config().add_output(d).add_const_input(src1).add_const_input(idx).build();
                kernels::run_scatter(iter, {1}, {1}, mode);
                return;
            }
            std::vector<int64_t> shape = dst.sizes().vec();
            shape[dim] = idx.numel();
            Tensor src = source.dim() == 0 ? source.expand(shape) : source.expand(shape);
            Tensor d = restride(dst, dim, shape);
            std::vector<int64_t> ist(shape.size(), 0);
            ist[dim] = idx.stride(0);
            Tensor idx_r = idx.as_strided(shape, ist, idx.storage_offset());
            auto iter = index_config().add_output(d).add_const_input(src).add_const_input(idx_r).build();
            kernels::run_scatter(iter, {dst.size(dim)}, {dst.stride(dim)}, mode);
        }

        Tensor &index_add_out_fn(const Tensor &self, int64_t dim, const Tensor &index, const Tensor &source,
                                 const Scalar &alpha, Tensor &out)
        {
            if (!scatter_supported(self, ScatterMode::Add))
                return WEBGPU_FALLBACK(index_add, out)(self, dim, index, source, alpha, out);
            Out<at::meta::structured_index_add> op(out);
            op.meta(self, dim, index, source, alpha);
            const Tensor &o = op.maybe_get_output(0);
            if (!o.is_same(self))
                o.copy_(self);
            Tensor src = (alpha.isIntegral(true) && alpha.toLong() == 1) ? source : source * alpha;
            index_scatter_dim(o, dim, index, src, ScatterMode::Add);
            op.finish();
            return out;
        }
        Tensor index_add_fn(const Tensor &self, int64_t dim, const Tensor &index, const Tensor &source, const Scalar &alpha)
        {
            if (!scatter_supported(self, ScatterMode::Add))
                return WEBGPU_FALLBACK0(index_add)(self, dim, index, source, alpha);
            Functional<at::meta::structured_index_add> op;
            op.meta(self, dim, index, source, alpha);
            Tensor &o = op.outputs_[0];
            o.copy_(self);
            Tensor src = (alpha.isIntegral(true) && alpha.toLong() == 1) ? source : source * alpha;
            index_scatter_dim(o, dim, index, src, ScatterMode::Add);
            return std::move(o);
        }
        Tensor &index_add_inplace_fn(Tensor &self, int64_t dim, const Tensor &index, const Tensor &source, const Scalar &alpha)
        {
            if (!scatter_supported(self, ScatterMode::Add))
                return WEBGPU_FALLBACK0(index_add_)(self, dim, index, source, alpha);
            Inplace<at::meta::structured_index_add> op(self);
            op.meta(self, dim, index, source, alpha);
            Tensor src = (alpha.isIntegral(true) && alpha.toLong() == 1) ? source : source * alpha;
            index_scatter_dim(op.maybe_get_output(0), dim, index, src, ScatterMode::Add);
            op.finish();
            return self;
        }

        Tensor &index_copy_out_fn(const Tensor &self, int64_t dim, const Tensor &index, const Tensor &source, Tensor &out)
        {
            Out<at::meta::structured_index_copy> op(out);
            op.meta(self, dim, index, source);
            const Tensor &o = op.maybe_get_output(0);
            if (!o.is_same(self))
                o.copy_(self);
            index_scatter_dim(o, dim, index, source, ScatterMode::Assign);
            op.finish();
            return out;
        }

        Tensor &index_fill_scalar_fn(Tensor &self, int64_t dim, const Tensor &index, const Scalar &value)
        {
            TORCH_CHECK_INDEX(index.dim() <= 1, "index_fill_(): Index is supposed to be a vector");
            index_scatter_dim(self, dim, index, at::scalar_tensor(value, self.options()), ScatterMode::Assign);
            return self;
        }
        Tensor &index_fill_tensor_fn(Tensor &self, int64_t dim, const Tensor &index, const Tensor &value)
        {
            TORCH_CHECK(value.dim() == 0, "index_fill_ only supports a 0-dimensional value tensor, but got tensor with ",
                        value.dim(), " dimension(s).");
            index_scatter_dim(self, dim, index, value.to(self.device(), self.scalar_type()), ScatterMode::Assign);
            return self;
        }

        // --------------------------------------------- embedding backward
        Tensor embedding_dense_backward_fn(const Tensor &grad_output, const Tensor &indices, c10::SymInt num_weights_sym,
                                           c10::SymInt padding_idx_sym, bool scale_grad_by_freq)
        {
            int64_t num_weights = num_weights_sym.expect_int(), padding_idx = padding_idx_sym.expect_int();
            if (scale_grad_by_freq || !scatter_supported(grad_output, ScatterMode::Add))
                return WEBGPU_FALLBACK0(embedding_dense_backward)(grad_output, indices, num_weights, padding_idx,
                                                                  scale_grad_by_freq);
            const int64_t D = grad_output.size(-1);
            Tensor grad_weight = at::zeros({num_weights, D}, grad_output.options());
            const int64_t N = indices.numel();
            if (N == 0)
                return grad_weight;
            Tensor idx = indices.reshape({N});
            Tensor g = grad_output.reshape({N, D});
            if (padding_idx >= 0)
                g = at::where((idx != padding_idx).unsqueeze(1), g, at::zeros({}, g.options()));
            Tensor dst = grad_weight.as_strided({N, D}, {0, grad_weight.stride(1)});
            Tensor idx_r = idx.as_strided({N, D}, {idx.stride(0), 0}, idx.storage_offset());
            auto iter = index_config().add_output(dst).add_const_input(g).add_const_input(idx_r).build();
            kernels::run_scatter(iter, {num_weights}, {grad_weight.stride(0)}, ScatterMode::Add);
            return grad_weight;
        }

        // ----------------------------------------- nonzero / masked_select
        Tensor &nonzero_out_fn(const Tensor &self, Tensor &out)
        {
            TORCH_CHECK(out.scalar_type() == at::kLong, "nonzero: Expected out tensor to have scalar type Long");
            if (!gpu_dtypes_ok({self}))
                return WEBGPU_FALLBACK(nonzero, out)(self, out);
            const int64_t nd = self.dim();
            Tensor m = self.scalar_type() == at::kBool ? self : (self != 0);
            Tensor flat = m.reshape({-1}).contiguous();
            if (flat.numel() == 0)
            {
                at::native::resize_output(out, {0, nd});
                return out;
            }
            Tensor pos = flat.to(at::kInt).cumsum(0, at::kInt);
            int64_t count = pos[-1].item<int64_t>();
            at::native::resize_output(out, {count, nd});
            if (count == 0 || nd == 0)
                return out;
            Tensor o = out.is_contiguous() ? out : at::empty({count, nd}, out.options());
            kernels::run_nonzero(flat, pos, o, self.sizes());
            if (!o.is_same(out))
                out.copy_(o);
            return out;
        }
        Tensor nonzero_fn(const Tensor &self)
        {
            Tensor out = at::empty({0}, self.options().dtype(at::kLong));
            return nonzero_out_fn(self, out);
        }

        Tensor masked_select_fn(const Tensor &self, const Tensor &mask)
        {
            TORCH_CHECK(mask.scalar_type() == at::kBool, "masked_select: expected BoolTensor for mask");
            auto shapes = at::infer_size(self.sizes(), mask.sizes());
            Tensor s = self.expand(shapes), m = mask.expand(shapes);
            if (shapes.empty())
            {
                s = s.view({1});
                m = m.view({1});
            }
            c10::List<std::optional<Tensor>> idx;
            idx.push_back(m);
            return at::index(s, idx);
        }
        Tensor &masked_select_out_fn(const Tensor &self, const Tensor &mask, Tensor &out)
        {
            Tensor r = masked_select_fn(self, mask);
            at::native::resize_output(out, r.sizes());
            out.copy_(r);
            return out;
        }

        // ------------------------------------------------------------ cat
        void cat_impl(const at::ITensorListRef &tensors, int64_t dim, const Tensor &out)
        {
            int64_t offset = 0;
            for (const Tensor &t : tensors)
            {
                if (t.dim() == 1 && t.numel() == 0 && out.dim() != 1)
                    continue; // legacy empty tensor
                int64_t n = t.size(dim);
                if (n == 0)
                    continue;
                out.narrow(dim, offset, n).copy_(t);
                offset += n;
            }
        }
        Tensor cat_fn(const at::ITensorListRef &tensors, int64_t dim)
        {
            Functional<at::meta::structured_cat> op;
            auto pre = op.meta(tensors, dim);
            cat_impl(tensors, pre.dim, op.outputs_[0]);
            return std::move(op.outputs_[0]);
        }
        Tensor &cat_out_fn(const at::ITensorListRef &tensors, int64_t dim, Tensor &out)
        {
            Out<at::meta::structured_cat> op(out);
            auto pre = op.meta(tensors, dim);
            cat_impl(tensors, pre.dim, op.maybe_get_output(0));
            op.finish();
            return out;
        }

        // ----------------------------------------------------------- flip
        Tensor flip_fn(const Tensor &self, at::IntArrayRef dims)
        {
            const int64_t nd = self.dim();
            std::vector<bool> flip(std::max<int64_t>(nd, 1), false);
            for (auto d : dims)
            {
                d = at::maybe_wrap_dim(d, std::max<int64_t>(nd, 1));
                TORCH_CHECK(!flip[d], "dims has duplicates, original flip dims size=", dims.size());
                flip[d] = true;
            }
            Tensor out = at::empty(self.sizes(), self.options().memory_format(c10::MemoryFormat::Contiguous));
            if (self.numel() == 0 || nd == 0)
                return out.copy_(self);
            auto st = self.strides().vec();
            std::vector<int64_t> sizes, strides;
            std::vector<Tensor> idx;
            for (int64_t d = 0; d < nd; ++d)
            {
                if (!flip[d] || self.size(d) == 1)
                    continue;
                st[d] = 0;
                std::vector<int64_t> ist(nd, 0);
                Tensor r = at::arange(self.size(d) - 1, -1, -1, self.options().dtype(at::kLong));
                ist[d] = 1;
                idx.push_back(r.as_strided(self.sizes(), ist));
                sizes.push_back(self.size(d));
                strides.push_back(self.stride(d));
            }
            Tensor src = self.as_strided(self.sizes(), st, self.storage_offset());
            auto config = index_config();
            config.add_output(out).add_const_input(src);
            for (auto &i : idx)
                config.add_const_input(i);
            auto iter = config.build();
            kernels::run_gather(iter, sizes, strides);
            return out;
        }

        // ----------------------------------------------------------- roll
        Tensor roll_fn(const Tensor &self, c10::SymIntArrayRef shifts_sym, at::IntArrayRef dims)
        {
            auto shifts = C10_AS_INTARRAYREF_SLOW(shifts_sym);
            TORCH_CHECK(!shifts.empty(), "`shifts` required");
            if (dims.empty() && shifts.size() == 1)
                return roll_fn(self.flatten(), shifts_sym, {0}).view(self.sizes());
            TORCH_CHECK(shifts.size() == dims.size(), "shifts and dimensions must align. shifts: ", shifts.size(),
                        ", dims:", dims.size());
            Tensor r = self;
            for (size_t k = 0; k < dims.size(); ++k)
            {
                int64_t d = at::maybe_wrap_dim(dims[k], r.dim());
                int64_t n = r.size(d);
                if (n == 0)
                    continue;
                int64_t s = ((shifts[k] % n) + n) % n;
                if (s == 0)
                {
                    r = r.clone();
                    continue;
                }
                r = at::cat({r.narrow(d, n - s, s), r.narrow(d, 0, n - s)}, d);
            }
            return r;
        }

        Tensor &eye_out_fn(int64_t n, int64_t m, Tensor &out)
        {
            TORCH_CHECK(n >= 0 && m >= 0, "n and m must be non-negative");
            out.resize_({n, m});
            out.zero_();
            if (n > 0 && m > 0)
                out.diagonal().fill_(1);
            return out;
        }
        Tensor &eye_n_out_fn(int64_t n, Tensor &out) { return eye_out_fn(n, n, out); }

        // ---------------------------------------------------- triu / tril
        void tri_impl(const Tensor &self, int64_t k, const Tensor &out, bool upper)
        {
            TORCH_CHECK(self.dim() >= 2, upper ? "triu" : "tril", ": input tensor must have at least 2 dimensions");
            const int64_t M = self.size(-2), N = self.size(-1);
            auto lopts = self.options().dtype(at::kLong);
            Tensor rows = at::arange(M, lopts).view({M, 1});
            Tensor cols = at::arange(N, lopts).view({1, N});
            Tensor keep = upper ? (cols - rows >= k) : (cols - rows <= k);
            Tensor r = at::where(keep, self, at::zeros({}, self.options()));
            if (!r.is_same(out))
                out.copy_(r);
        }
#define WG_TRI(NAME, UPPER)                                                                   \
    Tensor NAME##_fn(const Tensor &self, int64_t k)                                           \
    {                                                                                         \
        Functional<at::meta::structured_##NAME> op;                                           \
        op.meta(self, k);                                                                     \
        tri_impl(self, k, op.outputs_[0], UPPER);                                             \
        return std::move(op.outputs_[0]);                                                     \
    }                                                                                         \
    Tensor &NAME##_out_fn(const Tensor &self, int64_t k, Tensor &out)                         \
    {                                                                                         \
        Out<at::meta::structured_##NAME> op(out);                                             \
        op.meta(self, k);                                                                     \
        tri_impl(self, k, op.maybe_get_output(0), UPPER);                                     \
        op.finish();                                                                          \
        return out;                                                                           \
    }                                                                                         \
    Tensor &NAME##_inplace_fn(Tensor &self, int64_t k)                                        \
    {                                                                                         \
        Inplace<at::meta::structured_##NAME> op(self);                                        \
        op.meta(self, k);                                                                     \
        tri_impl(self, k, op.maybe_get_output(0), UPPER);                                     \
        op.finish();                                                                          \
        return self;                                                                          \
    }
        WG_TRI(triu, true)
        WG_TRI(tril, false)
    }

    TORCH_LIBRARY_IMPL(aten, PrivateUse1, m)
    {
        m.impl("gather", TORCH_FN(gather_fn));
        m.impl("gather.out", TORCH_FN(gather_out_fn));
        m.impl("scatter.src", TORCH_FN(scatter_src_fn));
        m.impl("scatter.src_out", TORCH_FN(scatter_src_out_fn));
        m.impl("scatter_.src", TORCH_FN(scatter_src_inplace_fn));
        m.impl("scatter.value", TORCH_FN(scatter_value_fn));
        m.impl("scatter.value_out", TORCH_FN(scatter_value_out_fn));
        m.impl("scatter_.value", TORCH_FN(scatter_value_inplace_fn));
        m.impl("scatter.reduce", TORCH_FN(scatter_reduce_fn));
        m.impl("scatter.reduce_out", TORCH_FN(scatter_reduce_out_fn));
        m.impl("scatter_.reduce", TORCH_FN(scatter_reduce_inplace_fn));
        m.impl("scatter.value_reduce", TORCH_FN(scatter_value_reduce_fn));
        m.impl("scatter.value_reduce_out", TORCH_FN(scatter_value_reduce_out_fn));
        m.impl("scatter_.value_reduce", TORCH_FN(scatter_value_reduce_inplace_fn));
        m.impl("scatter_add", TORCH_FN(scatter_add_fn));
        m.impl("scatter_add.out", TORCH_FN(scatter_add_out_fn));
        m.impl("scatter_add_", TORCH_FN(scatter_add_inplace_fn));
        m.impl("scatter_reduce.two", TORCH_FN(scatter_reduce_two_fn));
        m.impl("index.Tensor", TORCH_FN(index_fn));
        m.impl("index.Tensor_out", TORCH_FN(index_out_fn));
        m.impl("_index_put_impl_", TORCH_FN(index_put_impl_fn));
        m.impl("index_select", TORCH_FN(index_select_fn));
        m.impl("index_select.out", TORCH_FN(index_select_out_fn));
        m.impl("index_add", TORCH_FN(index_add_fn));
        m.impl("index_add.out", TORCH_FN(index_add_out_fn));
        m.impl("index_add_", TORCH_FN(index_add_inplace_fn));
        m.impl("index_copy.out", TORCH_FN(index_copy_out_fn));
        m.impl("index_fill_.int_Scalar", TORCH_FN(index_fill_scalar_fn));
        m.impl("index_fill_.int_Tensor", TORCH_FN(index_fill_tensor_fn));
        m.impl("embedding_dense_backward", TORCH_FN(embedding_dense_backward_fn));
        m.impl("nonzero", TORCH_FN(nonzero_fn));
        m.impl("nonzero.out", TORCH_FN(nonzero_out_fn));
        m.impl("masked_select", TORCH_FN(masked_select_fn));
        m.impl("masked_select.out", TORCH_FN(masked_select_out_fn));
        m.impl("cat", TORCH_FN(cat_fn));
        m.impl("cat.out", TORCH_FN(cat_out_fn));
        m.impl("flip", TORCH_FN(flip_fn));
        m.impl("roll", TORCH_FN(roll_fn));
        m.impl("eye.m_out", TORCH_FN(eye_out_fn));
        m.impl("eye.out", TORCH_FN(eye_n_out_fn));
        m.impl("triu", TORCH_FN(triu_fn));
        m.impl("triu.out", TORCH_FN(triu_out_fn));
        m.impl("triu_", TORCH_FN(triu_inplace_fn));
        m.impl("tril", TORCH_FN(tril_fn));
        m.impl("tril.out", TORCH_FN(tril_out_fn));
        m.impl("tril_", TORCH_FN(tril_inplace_fn));
    }
}
