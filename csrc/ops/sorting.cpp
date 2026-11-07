// Sorting and searching: sort, topk, kthvalue, median, searchsorted,
// bucketize, isin, unique, plus a few small reductions (equal, aminmax,
// count_nonzero).

#include <ATen/ATen.h>
#include <ATen/native/Resize.h>
#include <ATen/ops/aminmax_meta.h>
#include <ATen/ops/isin_meta.h>
#include <ATen/ops/sort_meta.h>
#include <ATen/ops/topk_meta.h>
#include <torch/library.h>

#include "kernels/sort.h"
#include "ops/common.h"

namespace torch_webgpu::ops
{
    using at::Scalar;
    using at::Tensor;

    namespace
    {
        // Sort keys are 32/64-bit: f16/bf16 -> f32, small ints/bool -> i32.
        std::optional<at::ScalarType> key_dtype(at::ScalarType t)
        {
            switch (t)
            {
            case at::kFloat:
            case at::kHalf:
            case at::kBFloat16:
                return at::kFloat;
            case at::kLong:
                return at::kLong;
            case at::kInt:
            case at::kShort:
            case at::kChar:
            case at::kByte:
            case at::kBool:
                return at::kInt;
            default:
                return std::nullopt;
            }
        }

        bool sortable(const Tensor &t) { return key_dtype(t.scalar_type()).has_value(); }

        // Sorts `self` along `dim`; writes values/indices (any strides).
        void sort_impl(const Tensor &self, int64_t dim, bool descending, const Tensor &values, const Tensor &indices)
        {
            if (self.numel() == 0)
                return;
            if (self.dim() == 0)
            {
                values.copy_(self);
                indices.zero_();
                return;
            }
            dim = at::maybe_wrap_dim(dim, self.dim());
            Tensor moved = self.movedim(dim, -1);
            const int64_t n = moved.size(-1), R = moved.numel() / n;
            Tensor keys = moved.reshape({R, n}).to(*key_dtype(self.scalar_type()), /*non_blocking=*/false, /*copy=*/true)
                              .contiguous();
            Tensor idx = at::empty({R, n}, keys.options().dtype(at::kInt));
            kernels::run_sort_rows(keys, idx, descending);
            auto shape = moved.sizes();
            values.copy_(keys.view(shape).movedim(-1, dim));
            indices.copy_(idx.view(shape).movedim(-1, dim));
        }

        std::tuple<Tensor, Tensor> sort_stable_fn(const Tensor &self, std::optional<bool> stable, int64_t dim, bool descending)
        {
            if (!sortable(self))
                return WEBGPU_FALLBACK(sort, stable)(self, stable, dim, descending);
            Functional<at::meta::structured_sort_stable, 2> op;
            op.meta(self, stable, dim, descending);
            sort_impl(self, dim, descending, op.outputs_[0], op.outputs_[1]);
            return {std::move(op.outputs_[0]), std::move(op.outputs_[1])};
        }
        std::tuple<Tensor &, Tensor &> sort_stable_out_fn(const Tensor &self, std::optional<bool> stable, int64_t dim,
                                                          bool descending, Tensor &values, Tensor &indices)
        {
            if (!sortable(self))
                return WEBGPU_FALLBACK(sort, values_stable)(self, stable, dim, descending, values, indices);
            Out<at::meta::structured_sort_stable, 2> op(values, indices);
            op.meta(self, stable, dim, descending);
            sort_impl(self, dim, descending, op.maybe_get_output(0), op.maybe_get_output(1));
            op.finish();
            return {values, indices};
        }

        // topk: a full stable sort, then the first k (sorted output is a
        // valid result for sorted=False as well).
        void topk_impl(const Tensor &self, int64_t k, int64_t dim, bool largest, const Tensor &values, const Tensor &indices)
        {
            if (self.dim() == 0)
            {
                values.copy_(self);
                indices.zero_();
                return;
            }
            dim = at::maybe_wrap_dim(dim, self.dim());
            if (k == 0 || self.numel() == 0)
                return;
            Tensor v = at::empty_like(self), i = at::empty_like(self, self.options().dtype(at::kLong));
            sort_impl(self, dim, largest, v, i);
            values.copy_(v.narrow(dim, 0, k));
            indices.copy_(i.narrow(dim, 0, k));
        }
        std::tuple<Tensor, Tensor> topk_fn(const Tensor &self, c10::SymInt k_s, int64_t dim, bool largest, bool sorted)
        {
            int64_t k = k_s.expect_int();
            if (!sortable(self))
                return WEBGPU_FALLBACK0(topk)(self, k, dim, largest, sorted);
            Functional<at::meta::structured_topk, 2> op;
            op.meta(self, k, dim, largest, sorted);
            topk_impl(self, k, dim, largest, op.outputs_[0], op.outputs_[1]);
            return {std::move(op.outputs_[0]), std::move(op.outputs_[1])};
        }
        std::tuple<Tensor &, Tensor &> topk_out_fn(const Tensor &self, c10::SymInt k_s, int64_t dim, bool largest, bool sorted,
                                                   Tensor &values, Tensor &indices)
        {
            int64_t k = k_s.expect_int();
            if (!sortable(self))
                return WEBGPU_FALLBACK(topk, values)(self, k, dim, largest, sorted, values, indices);
            Out<at::meta::structured_topk, 2> op(values, indices);
            op.meta(self, k, dim, largest, sorted);
            topk_impl(self, k, dim, largest, op.maybe_get_output(0), op.maybe_get_output(1));
            op.finish();
            return {values, indices};
        }

        // ---------------------------------------- kthvalue / median
        // Picks position pos[...] (a [R, 1] long tensor over rows) of each
        // ascending-sorted row along dim.
        std::tuple<Tensor, Tensor> pick_sorted(const Tensor &self, int64_t dim, bool keepdim, const Tensor &pos_rows)
        {
            dim = at::maybe_wrap_dim(dim, std::max<int64_t>(self.dim(), 1));
            Tensor x = self.dim() == 0 ? self.reshape({1}) : self;
            Tensor v = at::empty_like(x), i = at::empty_like(x, x.options().dtype(at::kLong));
            sort_impl(x, dim, false, v, i);
            Tensor pos = pos_rows;
            Tensor vals = v.gather(dim, pos), idxs = i.gather(dim, pos);
            if (!keepdim || self.dim() == 0)
            {
                vals = vals.squeeze(dim);
                idxs = idxs.squeeze(dim);
            }
            if (self.dim() == 0)
            {
                vals = vals.reshape(at::IntArrayRef{});
                idxs = idxs.reshape(at::IntArrayRef{});
            }
            return {vals, idxs};
        }

        std::tuple<Tensor &, Tensor &> kthvalue_out_fn(const Tensor &self, c10::SymInt k_s, int64_t dim, bool keepdim,
                                                       Tensor &values, Tensor &indices)
        {
            int64_t k = k_s.expect_int();
            if (!sortable(self))
                return WEBGPU_FALLBACK(kthvalue, values)(self, k, dim, keepdim, values, indices);
            int64_t d = at::maybe_wrap_dim(dim, std::max<int64_t>(self.dim(), 1));
            int64_t n = self.dim() == 0 ? 1 : self.size(d);
            TORCH_CHECK(k >= 1 && k <= n, "kthvalue(): selected number k out of range for dimension ", d);
            auto shape = self.dim() == 0 ? std::vector<int64_t>{1} : self.sizes().vec();
            shape[d] = 1;
            Tensor pos = at::full(shape, k - 1, self.options().dtype(at::kLong));
            auto [v, i] = pick_sorted(self, dim, keepdim, pos);
            at::native::resize_output(values, v.sizes());
            at::native::resize_output(indices, i.sizes());
            values.copy_(v);
            indices.copy_(i);
            return {values, indices};
        }

        // median (lower median); NaN propagates (NaN sorts last, so a row with
        // NaN picks its first NaN). nanmedian ignores NaNs.
        Tensor median_position(const Tensor &x, int64_t d, bool ignore_nan)
        {
            auto shape = x.sizes().vec();
            shape[d] = 1;
            const int64_t n = x.size(d);
            if (!x.is_floating_point())
                return at::full(shape, (n - 1) / 2, x.options().dtype(at::kLong));
            Tensor non_nan = (~x.isnan()).sum(d, true); // long
            if (ignore_nan)
                return at::floor_divide((non_nan - 1).clamp_min(0), 2);
            Tensor has_nan = non_nan < n;
            return at::where(has_nan, non_nan, at::full({}, (n - 1) / 2, non_nan.options()));
        }

        std::tuple<Tensor &, Tensor &> median_dim_impl(const Tensor &self, int64_t dim, bool keepdim, Tensor &values,
                                                       Tensor &indices, bool ignore_nan)
        {
            int64_t d = at::maybe_wrap_dim(dim, std::max<int64_t>(self.dim(), 1));
            Tensor x = self.dim() == 0 ? self.reshape({1}) : self;
            TORCH_CHECK(x.size(d) > 0, "median(): Expected reduction dim ", d, " to have non-zero size.");
            auto [v, i] = pick_sorted(self, dim, keepdim, median_position(x, d, ignore_nan));
            at::native::resize_output(values, v.sizes());
            at::native::resize_output(indices, i.sizes());
            values.copy_(v);
            indices.copy_(i);
            return {values, indices};
        }
        std::tuple<Tensor &, Tensor &> median_dim_out_fn(const Tensor &self, int64_t dim, bool keepdim, Tensor &values,
                                                         Tensor &indices)
        {
            if (!sortable(self))
                return WEBGPU_FALLBACK(median, dim_values)(self, dim, keepdim, values, indices);
            return median_dim_impl(self, dim, keepdim, values, indices, false);
        }
        std::tuple<Tensor &, Tensor &> nanmedian_dim_out_fn(const Tensor &self, int64_t dim, bool keepdim, Tensor &values,
                                                            Tensor &indices)
        {
            if (!sortable(self))
                return WEBGPU_FALLBACK(nanmedian, dim_values)(self, dim, keepdim, values, indices);
            return median_dim_impl(self, dim, keepdim, values, indices, true);
        }
        Tensor median_all_impl(const Tensor &self, bool ignore_nan)
        {
            TORCH_CHECK(self.numel() > 0, "median() input tensor cannot be empty");
            Tensor v = at::empty({0}, self.options()), i = at::empty({0}, self.options().dtype(at::kLong));
            median_dim_impl(self.reshape({-1}), 0, false, v, i, ignore_nan);
            return v;
        }
        Tensor median_fn(const Tensor &self)
        {
            if (!sortable(self))
                return WEBGPU_FALLBACK0(median)(self);
            return median_all_impl(self, false);
        }
        Tensor nanmedian_fn(const Tensor &self)
        {
            if (!sortable(self))
                return WEBGPU_FALLBACK0(nanmedian)(self);
            return median_all_impl(self, true);
        }

        // ----------------------------------------------- searchsorted
        Tensor &searchsorted_impl(const Tensor &seq_in, const Tensor &self, bool out_int32, bool right,
                                  std::optional<c10::string_view> side, const std::optional<Tensor> &sorter, Tensor &out)
        {
            if (side.has_value())
            {
                TORCH_CHECK(*side == "left" || *side == "right", "torch.searchsorted(): side can only be 'left' or 'right' but got ", *side);
                TORCH_CHECK(!(right && *side == "left"), "torch.searchsorted(): side and right can't be set to opposites");
                right = *side == "right";
            }
            Tensor seq = seq_in;
            if (sorter.has_value() && sorter->defined())
                seq = seq.gather(-1, *sorter);
            auto out_dtype = out_int32 ? at::kInt : at::kLong;
            at::native::resize_output(out, self.sizes());
            TORCH_CHECK(out.scalar_type() == out_dtype, "torch.searchsorted(): output tensor's dtype is wrong");
            if (self.numel() == 0)
                return out;
            auto common = at::promote_types(seq.scalar_type(), self.scalar_type());
            auto kd = key_dtype(common);
            TORCH_CHECK(kd.has_value(), "torch_webgpu: searchsorted dtype not supported");
            const int64_t n = seq.size(-1);
            Tensor s2, v2;
            if (seq.dim() == 1)
            {
                s2 = seq.to(*kd).contiguous().view({1, n});
                v2 = self.to(*kd).contiguous().view({1, -1});
            }
            else
            {
                TORCH_CHECK(seq.sizes().slice(0, seq.dim() - 1).equals(self.sizes().slice(0, self.dim() - 1)),
                            "torch.searchsorted(): boundaries tensor should be 1 dimension or the first N-1 dimensions of "
                            "boundaries tensor and input value tensor must match");
                s2 = seq.to(*kd).contiguous().view({-1, n});
                v2 = self.to(*kd).contiguous().view({s2.size(0), -1});
            }
            Tensor o = at::empty(v2.sizes(), out.options());
            if (n == 0)
                o.zero_();
            else
                kernels::run_searchsorted(s2, v2, o, right);
            out.copy_(o.view(self.sizes()));
            return out;
        }
        Tensor searchsorted_fn(const Tensor &seq, const Tensor &self, bool out_int32, bool right,
                               std::optional<c10::string_view> side, const std::optional<Tensor> &sorter)
        {
            Tensor out = at::empty({0}, self.options().dtype(out_int32 ? at::kInt : at::kLong));
            return searchsorted_impl(seq, self, out_int32, right, side, sorter, out);
        }
        Tensor &searchsorted_out_fn(const Tensor &seq, const Tensor &self, bool out_int32, bool right,
                                    std::optional<c10::string_view> side, const std::optional<Tensor> &sorter, Tensor &out)
        {
            return searchsorted_impl(seq, self, out_int32, right, side, sorter, out);
        }
        Tensor searchsorted_scalar_fn(const Tensor &seq, const Scalar &self, bool out_int32, bool right,
                                      std::optional<c10::string_view> side, const std::optional<Tensor> &sorter)
        {
            return searchsorted_fn(seq, at::scalar_tensor(self, seq.options()), out_int32, right, side, sorter);
        }
        Tensor bucketize_fn(const Tensor &self, const Tensor &boundaries, bool out_int32, bool right)
        {
            TORCH_CHECK(boundaries.dim() == 1, "boundaries tensor must be 1 dimension, but got dim(", boundaries.dim(), ")");
            return searchsorted_fn(boundaries, self, out_int32, right, std::nullopt, std::nullopt);
        }
        Tensor &bucketize_out_fn(const Tensor &self, const Tensor &boundaries, bool out_int32, bool right, Tensor &out)
        {
            TORCH_CHECK(boundaries.dim() == 1, "boundaries tensor must be 1 dimension, but got dim(", boundaries.dim(), ")");
            return searchsorted_impl(boundaries, self, out_int32, right, std::nullopt, std::nullopt, out);
        }
        Tensor bucketize_scalar_fn(const Scalar &self, const Tensor &boundaries, bool out_int32, bool right)
        {
            return bucketize_fn(at::scalar_tensor(self, boundaries.options()), boundaries, out_int32, right);
        }

        // ------------------------------------------------------- isin
        void isin_impl(const Tensor &elements, const Tensor &test, bool invert, const Tensor &out)
        {
            if (elements.numel() == 0)
                return;
            if (test.numel() == 0)
            {
                out.fill_(invert);
                return;
            }
            auto common = at::promote_types(elements.scalar_type(), test.scalar_type());
            Tensor t = std::get<0>(at::sort(test.flatten().to(common)));
            Tensor e = elements.to(common);
            Tensor pos = at::searchsorted(t, e).clamp_max(t.numel() - 1);
            Tensor found = t.index_select(0, pos.flatten()).view(e.sizes()) == e;
            out.copy_(invert ? found.logical_not() : found);
        }
        bool isin_ok(const Tensor &a, const Tensor &b)
        {
            return key_dtype(at::promote_types(a.scalar_type(), b.scalar_type())).has_value() && gpu_dtypes_ok({a, b});
        }
        Tensor &isin_tt_out_fn(const Tensor &elements, const Tensor &test, bool assume_unique, bool invert, Tensor &out)
        {
            if (!isin_ok(elements, test))
                return WEBGPU_FALLBACK(isin, Tensor_Tensor_out)(elements, test, assume_unique, invert, out);
            Out<at::meta::structured_isin_Tensor_Tensor> op(out);
            op.meta(elements, test, assume_unique, invert);
            isin_impl(elements, test, invert, op.maybe_get_output(0));
            op.finish();
            return out;
        }
        Tensor isin_tt_fn(const Tensor &elements, const Tensor &test, bool assume_unique, bool invert)
        {
            if (!isin_ok(elements, test))
                return WEBGPU_FALLBACK(isin, Tensor_Tensor)(elements, test, assume_unique, invert);
            Functional<at::meta::structured_isin_Tensor_Tensor> op;
            op.meta(elements, test, assume_unique, invert);
            isin_impl(elements, test, invert, op.outputs_[0]);
            return std::move(op.outputs_[0]);
        }
        Tensor isin_ts_fn(const Tensor &elements, const Scalar &test, bool assume_unique, bool invert)
        {
            return invert ? elements.ne(test) : elements.eq(test);
        }
        Tensor isin_st_fn(const Scalar &element, const Tensor &test, bool assume_unique, bool invert)
        {
            Tensor r = (test == element).any();
            return invert ? r.logical_not() : r;
        }

        // ------------------------------------------------------- unique
        std::tuple<Tensor, Tensor, Tensor> unique2_fn(const Tensor &self, bool sorted, bool return_inverse, bool return_counts)
        {
            if (!sortable(self))
                return WEBGPU_FALLBACK0(_unique2)(self, sorted, return_inverse, return_counts);
            auto lopt = self.options().dtype(at::kLong);
            const int64_t n = self.numel();
            if (n == 0)
                return {at::empty({0}, self.options()), at::empty(self.sizes(), lopt), at::empty({0}, lopt)};
            Tensor flat = self.reshape({-1});
            Tensor sv = at::empty_like(flat), perm = at::empty_like(flat, lopt);
            sort_impl(flat, 0, false, sv, perm);
            Tensor is_start = at::ones({n}, self.options().dtype(at::kBool));
            if (n > 1)
                is_start.narrow(0, 1, n - 1).copy_(sv.narrow(0, 1, n - 1) != sv.narrow(0, 0, n - 1));
            Tensor starts = is_start.nonzero().view({-1});
            Tensor values = sv.index_select(0, starts);
            Tensor inverse = at::empty({0}, lopt), counts = at::empty({0}, lopt);
            if (return_inverse)
            {
                Tensor group = is_start.to(at::kLong).cumsum(0) - 1;
                inverse = at::empty({n}, lopt).scatter_(0, perm, group).view(self.sizes());
            }
            if (return_counts)
            {
                Tensor ends = at::cat({starts.narrow(0, 1, starts.numel() - 1), at::full({1}, n, lopt)});
                counts = ends - starts;
            }
            return {values, inverse, counts};
        }

        // ------------------------------------------- unique_consecutive / dim
        // Groups of equal consecutive rows of x2 [n, m] (m == 1 for elements):
        // returns is_start [n] bool.
        Tensor group_starts(const Tensor &x2)
        {
            const int64_t n = x2.size(0);
            Tensor is_start = at::ones({n}, x2.options().dtype(at::kBool));
            if (n > 1)
            {
                Tensor diff = x2.narrow(0, 1, n - 1) != x2.narrow(0, 0, n - 1);
                is_start.narrow(0, 1, n - 1).copy_(x2.size(1) == 1 ? diff.squeeze(1) : diff.any(1));
            }
            return is_start;
        }
        std::tuple<Tensor, Tensor> inverse_and_counts(const Tensor &is_start, const Tensor &starts, bool ri, bool rc)
        {
            auto lopt = is_start.options().dtype(at::kLong);
            const int64_t n = is_start.size(0);
            Tensor inverse = ri ? is_start.to(at::kLong).cumsum(0) - 1 : at::empty({0}, lopt);
            Tensor counts = at::empty({0}, lopt);
            if (rc)
            {
                Tensor ends = at::cat({starts.narrow(0, 1, starts.numel() - 1), at::full({1}, n, lopt)});
                counts = ends - starts;
            }
            return {inverse, counts};
        }

        // unique along a dim of an empty tensor: only the empty dim itself
        // may be selected (as CPU).
        std::tuple<Tensor, Tensor, Tensor> unique_dim_empty(const Tensor &self, int64_t d)
        {
            TORCH_CHECK(self.size(d) == 0,
                        "There are 0 sized dimensions, and they aren't selected, so unique cannot be applied");
            auto lopt = self.options().dtype(at::kLong);
            return {at::empty(self.sizes(), self.options()), at::empty({0}, lopt), at::empty({0}, lopt)};
        }

        std::tuple<Tensor, Tensor, Tensor> unique_consecutive_fn(const Tensor &self, bool return_inverse,
                                                                 bool return_counts, std::optional<int64_t> dim)
        {
            if (!gpu_dtypes_ok({self}))
                return WEBGPU_FALLBACK0(unique_consecutive)(self, return_inverse, return_counts, dim);
            if (dim.has_value() && self.numel() == 0)
                return unique_dim_empty(self, at::maybe_wrap_dim(*dim, self.dim()));
            auto lopt = self.options().dtype(at::kLong);
            if (!dim.has_value())
            {
                const int64_t n = self.numel();
                if (n == 0)
                    return {at::empty({0}, self.options()), at::empty(self.sizes(), lopt), at::empty({0}, lopt)};
                Tensor flat = self.reshape({n, 1});
                Tensor is_start = group_starts(flat);
                Tensor starts = is_start.nonzero().view({-1});
                auto [inv, counts] = inverse_and_counts(is_start, starts, return_inverse, return_counts);
                return {flat.view({-1}).index_select(0, starts), return_inverse ? inv.view(self.sizes()) : inv, counts};
            }
            const int64_t d = at::maybe_wrap_dim(*dim, self.dim());
            const int64_t n = self.size(d);
            Tensor x2 = self.movedim(d, 0).reshape({n, self.numel() / n});
            Tensor is_start = group_starts(x2);
            Tensor starts = is_start.nonzero().view({-1});
            auto [inv, counts] = inverse_and_counts(is_start, starts, return_inverse, return_counts);
            return {self.index_select(d, starts), inv, counts};
        }

        std::tuple<Tensor, Tensor, Tensor> unique_dim_fn(const Tensor &self, int64_t dim, bool sorted, bool return_inverse,
                                                         bool return_counts)
        {
            if (!sortable(self))
                return WEBGPU_FALLBACK0(unique_dim)(self, dim, sorted, return_inverse, return_counts);
            if (self.numel() == 0)
                return unique_dim_empty(self, at::maybe_wrap_dim(dim, self.dim()));
            const int64_t d = at::maybe_wrap_dim(dim, self.dim());
            const int64_t n = self.size(d), m = self.numel() / n;
            auto lopt = self.options().dtype(at::kLong);
            Tensor x2 = self.movedim(d, 0).reshape({n, m});
            // Lexicographic order of rows: stable sorts from the last column.
            Tensor perm = at::arange(n, lopt);
            for (int64_t c = m - 1; c >= 0; --c)
            {
                Tensor keys = x2.select(1, c).index_select(0, perm).contiguous();
                Tensor sv = at::empty_like(keys), p = at::empty_like(keys, lopt);
                sort_impl(keys, 0, false, sv, p);
                perm = perm.index_select(0, p);
            }
            Tensor is_start = group_starts(x2.index_select(0, perm));
            Tensor starts = is_start.nonzero().view({-1});
            auto [group, counts] = inverse_and_counts(is_start, starts, true, return_counts);
            Tensor inverse = return_inverse ? at::empty({n}, lopt).scatter_(0, perm, group) : at::empty({0}, lopt);
            return {self.index_select(d, perm.index_select(0, starts)), inverse, counts};
        }

        // ------------------------------------------------------------ mode
        // Most frequent value per row (ties: smallest value) and the largest
        // index at which it occurs, as CPU.
        std::tuple<Tensor &, Tensor &> mode_out_fn(const Tensor &self, int64_t dim, bool keepdim, Tensor &values,
                                                   Tensor &indices)
        {
            if (!sortable(self) || self.numel() == 0)
                return WEBGPU_FALLBACK(mode, values)(self, dim, keepdim, values, indices);
            const int64_t d = at::maybe_wrap_dim(dim, std::max<int64_t>(self.dim(), 1));
            Tensor x = self.dim() == 0 ? self.reshape({1}) : self;
            const int64_t n = x.size(d);
            Tensor rows = x.movedim(d, -1).reshape({-1, n}).contiguous();
            const int64_t R = rows.size(0);
            auto lopt = self.options().dtype(at::kLong);
            Tensor sv = at::empty_like(rows), si = at::empty_like(rows, lopt);
            sort_impl(rows, 1, false, sv, si);
            Tensor pos = at::arange(n, lopt).expand({R, n});
            Tensor is_end = at::ones({R, n}, self.options().dtype(at::kBool));
            if (n > 1)
                is_end.narrow(1, 0, n - 1).copy_(sv.narrow(1, 0, n - 1) != sv.narrow(1, 1, n - 1));
            // Position of the previous run end (-1 if none): exclusive running max.
            Tensor end_pos = at::where(is_end, pos, at::full_like(pos, -1));
            Tensor prev = at::full({R, n}, -1, lopt);
            if (n > 1)
                prev.narrow(1, 1, n - 1).copy_(std::get<0>(at::cummax(end_pos.narrow(1, 0, n - 1), 1)));
            Tensor freq = at::where(is_end, pos - prev, at::zeros_like(pos));
            Tensor best = freq.argmax(1, /*keepdim=*/true);
            std::vector<int64_t> out_shape = x.sizes().vec();
            out_shape.erase(out_shape.begin() + d);
            Tensor v = sv.gather(1, best).view(out_shape), i = si.gather(1, best).view(out_shape);
            if (keepdim && self.dim() > 0)
            {
                v = v.unsqueeze(d);
                i = i.unsqueeze(d);
            }
            at::native::resize_output(values, v.sizes());
            at::native::resize_output(indices, i.sizes());
            values.copy_(v);
            indices.copy_(i);
            return {values, indices};
        }
        std::tuple<Tensor, Tensor> mode_fn(const Tensor &self, int64_t dim, bool keepdim)
        {
            Tensor values = at::empty({0}, self.options()), indices = at::empty({0}, self.options().dtype(at::kLong));
            mode_out_fn(self, dim, keepdim, values, indices);
            return {values, indices};
        }

        // ------------------------------------------------ small reductions
        bool equal_fn(const Tensor &self, const Tensor &other)
        {
            if (!gpu_dtypes_ok({self, other}) || !is_webgpu(other))
                return WEBGPU_FALLBACK0(equal)(self, other);
            if (self.sizes() != other.sizes() || self.scalar_type() != other.scalar_type())
                return false;
            if (self.numel() == 0)
                return true;
            // NaN != NaN, as in torch.equal.
            return (self == other).all().item<bool>();
        }

        std::tuple<Tensor &, Tensor &> aminmax_out_fn(const Tensor &self, std::optional<int64_t> dim, bool keepdim,
                                                      Tensor &min, Tensor &max)
        {
            if (!gpu_dtypes_ok({self}))
                return WEBGPU_FALLBACK(aminmax, out)(self, dim, keepdim, min, max);
            Out<at::meta::structured_aminmax, 2> op(min, max);
            op.meta(self, dim, keepdim);
            Tensor x = self.dim() == 0 ? self.reshape({1}) : self;
            Tensor mn, mx;
            if (dim.has_value())
            {
                mn = x.amin({*dim}, keepdim);
                mx = x.amax({*dim}, keepdim);
            }
            else
            {
                mn = x.amin(at::IntArrayRef{}, keepdim);
                mx = x.amax(at::IntArrayRef{}, keepdim);
            }
            op.maybe_get_output(0).copy_(mn.view(op.maybe_get_output(0).sizes()));
            op.maybe_get_output(1).copy_(mx.view(op.maybe_get_output(1).sizes()));
            op.finish();
            return {min, max};
        }

        Tensor count_nonzero_fn(const Tensor &self, at::IntArrayRef dims)
        {
            if (!gpu_dtypes_ok({self}))
                return WEBGPU_FALLBACK(count_nonzero, dim_IntList)(self, dims);
            return (self != 0).sum(dims);
        }
    }

    TORCH_LIBRARY_IMPL(aten, PrivateUse1, m)
    {
        m.impl("sort.stable", TORCH_FN(sort_stable_fn));
        m.impl("sort.values_stable", TORCH_FN(sort_stable_out_fn));
        m.impl("topk", TORCH_FN(topk_fn));
        m.impl("topk.values", TORCH_FN(topk_out_fn));
        m.impl("kthvalue.values", TORCH_FN(kthvalue_out_fn));
        m.impl("median", TORCH_FN(median_fn));
        m.impl("median.dim_values", TORCH_FN(median_dim_out_fn));
        m.impl("nanmedian", TORCH_FN(nanmedian_fn));
        m.impl("nanmedian.dim_values", TORCH_FN(nanmedian_dim_out_fn));
        m.impl("searchsorted.Tensor", TORCH_FN(searchsorted_fn));
        m.impl("searchsorted.Tensor_out", TORCH_FN(searchsorted_out_fn));
        m.impl("searchsorted.Scalar", TORCH_FN(searchsorted_scalar_fn));
        m.impl("bucketize.Tensor", TORCH_FN(bucketize_fn));
        m.impl("bucketize.Tensor_out", TORCH_FN(bucketize_out_fn));
        m.impl("bucketize.Scalar", TORCH_FN(bucketize_scalar_fn));
        m.impl("isin.Tensor_Tensor", TORCH_FN(isin_tt_fn));
        m.impl("isin.Tensor_Tensor_out", TORCH_FN(isin_tt_out_fn));
        m.impl("isin.Tensor_Scalar", TORCH_FN(isin_ts_fn));
        m.impl("isin.Scalar_Tensor", TORCH_FN(isin_st_fn));
        m.impl("_unique2", TORCH_FN(unique2_fn));
        m.impl("unique_consecutive", TORCH_FN(unique_consecutive_fn));
        m.impl("unique_dim", TORCH_FN(unique_dim_fn));
        m.impl("mode", TORCH_FN(mode_fn));
        m.impl("mode.values", TORCH_FN(mode_out_fn));
        m.impl("equal", TORCH_FN(equal_fn));
        m.impl("aminmax.out", TORCH_FN(aminmax_out_fn));
        m.impl("count_nonzero.dim_IntList", TORCH_FN(count_nonzero_fn));
    }
}
