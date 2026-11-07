// histc, histogram and the _histogramdd_* family. Bin indices are computed
// with elementwise ops / searchsorted and accumulated with index_add, using
// the binning algorithms of ATen's HistogramKernel:
//   LINEAR (histc): pos = trunc((x - left) * nbins / (right - left))
//   LINEAR_LOCAL (histogram with a bin count): LINEAR, then corrected by an
//     upper_bound search over the neighbouring edges
//   BINARY (explicit edges): pos = upper_bound(edges, x) - 1
// Elements outside [left, right] (and NaNs) are skipped; the last bin
// includes its right edge.

#include <ATen/ATen.h>
#include <ATen/native/Resize.h>
#include <torch/library.h>

#include <cmath>
#include <optional>

#include "ops/common.h"

namespace torch_webgpu::ops
{
    using at::Tensor;

    namespace
    {
        enum class Algo
        {
            Linear,
            LinearLocal,
            Binary
        };

        // x2: [N, D]; edges[d]: 1-D increasing, same dtype as x2.
        Tensor histdd(const Tensor &x2, const std::vector<Tensor> &edges, const std::optional<Tensor> &weight,
                      bool density, Algo algo)
        {
            const int64_t N = x2.size(0), D = x2.size(1);
            std::vector<int64_t> nbins;
            for (const auto &e : edges)
                nbins.push_back(e.numel() - 1);
            Tensor hist = at::zeros(nbins, x2.options());
            if (N > 0)
            {
                auto lopt = x2.options().dtype(at::kLong);
                Tensor valid = at::ones({N}, x2.options().dtype(at::kBool));
                Tensor flat = at::zeros({N}, lopt);
                for (int64_t d = 0; d < D; ++d)
                {
                    const Tensor &e = edges[d];
                    const int64_t nb = nbins[d];
                    Tensor x = x2.select(1, d);
                    Tensor left = e.narrow(0, 0, 1).squeeze(0), right = e.narrow(0, nb, 1).squeeze(0);
                    Tensor in_range = (x >= left) & (x <= right);
                    valid = valid & in_range;
                    Tensor xx = at::where(in_range, x, left);
                    Tensor pos;
                    if (algo == Algo::Binary)
                        pos = at::searchsorted(e, xx, /*out_int32=*/false, /*right=*/true) - 1;
                    else
                    {
                        pos = ((xx - left) * nb / (right - left)).trunc().to(at::kLong);
                        if (algo == Algo::LinearLocal)
                        {
                            Tensor ub = at::searchsorted(e, xx, false, true);
                            Tensor lo = (pos - 1).clamp_min(0), hi = (pos + 2).clamp_max(nb + 1);
                            pos = at::clamp(ub, lo, hi) - 1;
                        }
                    }
                    pos = at::where(pos == nb, pos - 1, pos).clamp(0, nb - 1);
                    flat = flat * nb + pos;
                }
                Tensor w = weight.has_value() && weight->defined() ? weight->reshape({-1}).to(x2.scalar_type())
                                                                   : at::ones({N}, x2.options());
                w = at::where(valid, w, at::zeros_like(w));
                hist.view({-1}).index_add_(0, flat, w);
            }
            if (density)
            {
                hist.div_(hist.sum());
                for (int64_t d = 0; d < D; ++d)
                {
                    std::vector<int64_t> shape(D, 1);
                    shape[d] = nbins[d];
                    hist.div_(edges[d].diff().reshape(shape));
                }
            }
            return hist;
        }

        // Outer edges per dim: the given range, else the data's min / max
        // (degenerate ranges widened by 0.5 each way), as select_outer_bin_edges.
        std::vector<std::pair<double, double>> outer_edges(const Tensor &x2, std::optional<at::ArrayRef<double>> range)
        {
            const int64_t D = x2.size(1);
            std::vector<std::pair<double, double>> r(D, {0.0, 1.0});
            if (range.has_value())
            {
                TORCH_CHECK(static_cast<int64_t>(range->size()) == 2 * D, "torch.histogramdd: for a ", D,
                            "-dimensional histogram range should have ", 2 * D, " elements, but got ", range->size());
                for (int64_t d = 0; d < D; ++d)
                    r[d] = {(*range)[2 * d], (*range)[2 * d + 1]};
            }
            else if (x2.numel() > 0)
            {
                auto [mn, mx] = at::aminmax(x2, 0);
                Tensor both = at::stack({mn, mx}).to(at::kDouble).cpu();
                for (int64_t d = 0; d < D; ++d)
                    r[d] = {both[0][d].item<double>(), both[1][d].item<double>()};
            }
            for (int64_t d = 0; d < D; ++d)
            {
                auto &[lo, hi] = r[d];
                TORCH_CHECK(std::isfinite(lo) && std::isfinite(hi), "torch.histogramdd: dimension ", d,
                            "'s range [", lo, ", ", hi, "] is not finite");
                TORCH_CHECK(lo <= hi, "torch.histogramdd: min should not exceed max, but got min ", lo, " max ", hi,
                            " for dimension ", d);
                if (lo == hi)
                {
                    lo -= 0.5;
                    hi += 0.5;
                }
            }
            return r;
        }

        Tensor as_2d(const Tensor &self)
        {
            TORCH_CHECK(self.dim() >= 1, "torch.histogramdd: input tensor should have at least 1 dimension");
            return self.dim() == 1 ? self.reshape({-1, 1}) : self.reshape({-1, self.size(-1)});
        }

        std::vector<Tensor> linear_edges(const Tensor &x2, at::IntArrayRef bins, std::optional<at::ArrayRef<double>> range)
        {
            TORCH_CHECK(static_cast<int64_t>(bins.size()) == x2.size(1), "torch.histogramdd: expected ", x2.size(1),
                        " sequences of bin edges for a ", x2.size(1), "-dimensional histogram but got ", bins.size());
            auto outer = outer_edges(x2, range);
            std::vector<Tensor> edges;
            for (size_t d = 0; d < bins.size(); ++d)
            {
                TORCH_CHECK(bins[d] > 0, "torch.histogramdd(): bins must be > 0, but got ", bins[d], " for dimension ", d);
                edges.push_back(at::linspace(outer[d].first, outer[d].second, bins[d] + 1, x2.options()));
            }
            return edges;
        }

        void check_float(const Tensor &self, const char *name)
        {
            TORCH_CHECK(at::isFloatingType(self.scalar_type()), name, ": input tensor and bins tensors should have the same floating-point dtype, but got input ",
                        self.scalar_type());
        }

        // ---------------------------------------------------------- histc
        Tensor &histc_out_fn(const Tensor &self, int64_t bins, const at::Scalar &min, const at::Scalar &max, Tensor &out)
        {
            check_float(self, "torch.histc");
            TORCH_CHECK(bins > 0, "torch.histc(): bins must be > 0, but got ", bins, " for dimension 0");
            double lo = min.toDouble(), hi = max.toDouble();
            if (lo == hi && self.numel() > 0)
            {
                auto [mn, mx] = at::aminmax(self);
                lo = mn.item<double>();
                hi = mx.item<double>();
            }
            if (lo == hi)
            {
                lo -= 1;
                hi += 1;
            }
            TORCH_CHECK(std::isfinite(lo) && std::isfinite(hi), "torch.histc: range of [", lo, ", ", hi, "] is not finite");
            TORCH_CHECK(lo < hi, "torch.histc: max must be larger than min");
            Tensor x2 = self.reshape({-1, 1});
            Tensor edges = at::linspace(lo, hi, bins + 1, self.options());
            Tensor h = histdd(x2, {edges}, std::nullopt, false, Algo::Linear);
            at::native::resize_output(out, h.sizes());
            return out.copy_(h);
        }
        Tensor histc_fn(const Tensor &self, int64_t bins, const at::Scalar &min, const at::Scalar &max)
        {
            Tensor out = at::empty({0}, self.options());
            return histc_out_fn(self, bins, min, max, out);
        }

        // ------------------------------------------------------ histogram
        std::tuple<Tensor &, Tensor &> histogram_ct_out_fn(const Tensor &self, int64_t bins,
                                                           std::optional<at::ArrayRef<double>> range,
                                                           const std::optional<Tensor> &weight, bool density, Tensor &hist,
                                                           Tensor &bin_edges)
        {
            check_float(self, "torch.histogram");
            Tensor x2 = self.reshape({-1, 1});
            auto edges = linear_edges(x2, {bins}, range);
            Tensor h = histdd(x2, edges, weight, density, Algo::LinearLocal);
            at::native::resize_output(hist, h.sizes());
            at::native::resize_output(bin_edges, edges[0].sizes());
            hist.copy_(h);
            bin_edges.copy_(edges[0]);
            return {hist, bin_edges};
        }
        std::tuple<Tensor, Tensor> histogram_ct_fn(const Tensor &self, int64_t bins, std::optional<at::ArrayRef<double>> range,
                                                   const std::optional<Tensor> &weight, bool density)
        {
            Tensor hist = at::empty({0}, self.options()), edges = at::empty({0}, self.options());
            histogram_ct_out_fn(self, bins, range, weight, density, hist, edges);
            return {hist, edges};
        }
        std::tuple<Tensor &, Tensor &> histogram_t_out_fn(const Tensor &self, const Tensor &bins,
                                                          const std::optional<Tensor> &weight, bool density, Tensor &hist,
                                                          Tensor &bin_edges)
        {
            check_float(self, "torch.histogram");
            TORCH_CHECK(bins.dim() == 1, "torch.histogram: bins tensor should have one dimension, but got ", bins.dim());
            TORCH_CHECK(bins.numel() > 0, "torch.histogram: bins tensor should have at least 1 element, but got ",
                        bins.numel(), " elements");
            Tensor e = bins.to(self.scalar_type()).contiguous();
            Tensor h = histdd(self.reshape({-1, 1}), {e}, weight, density, Algo::Binary);
            at::native::resize_output(hist, h.sizes());
            at::native::resize_output(bin_edges, e.sizes());
            hist.copy_(h);
            bin_edges.copy_(e);
            return {hist, bin_edges};
        }
        std::tuple<Tensor, Tensor> histogram_t_fn(const Tensor &self, const Tensor &bins, const std::optional<Tensor> &weight,
                                                  bool density)
        {
            Tensor hist = at::empty({0}, self.options()), edges = at::empty({0}, self.options());
            histogram_t_out_fn(self, bins, weight, density, hist, edges);
            return {hist, edges};
        }

        // ------------------------------------------------- _histogramdd_*
        std::vector<Tensor> histogramdd_bin_edges_fn(const Tensor &self, at::IntArrayRef bins,
                                                     std::optional<at::ArrayRef<double>> range,
                                                     const std::optional<Tensor> &, bool)
        {
            check_float(self, "torch.histogramdd");
            return linear_edges(as_2d(self), bins, range);
        }
        Tensor histogramdd_from_bin_cts_fn(const Tensor &self, at::IntArrayRef bins,
                                           std::optional<at::ArrayRef<double>> range, const std::optional<Tensor> &weight,
                                           bool density)
        {
            check_float(self, "torch.histogramdd");
            Tensor x2 = as_2d(self);
            return histdd(x2, linear_edges(x2, bins, range), weight, density, Algo::LinearLocal);
        }
        Tensor histogramdd_from_bin_tensors_fn(const Tensor &self, at::TensorList bins, const std::optional<Tensor> &weight,
                                               bool density)
        {
            check_float(self, "torch.histogramdd");
            Tensor x2 = as_2d(self);
            TORCH_CHECK(static_cast<int64_t>(bins.size()) == x2.size(1), "torch.histogramdd: expected ", x2.size(1),
                        " sequences of bin edges for a ", x2.size(1), "-dimensional histogram but got ", bins.size());
            std::vector<Tensor> edges;
            for (const auto &b : bins)
                edges.push_back(b.to(self.scalar_type()).contiguous());
            return histdd(x2, edges, weight, density, Algo::Binary);
        }
    }

    TORCH_LIBRARY_IMPL(aten, PrivateUse1, m)
    {
        m.impl("histc", TORCH_FN(histc_fn));
        m.impl("histc.out", TORCH_FN(histc_out_fn));
        m.impl("histogram.bin_ct", TORCH_FN(histogram_ct_fn));
        m.impl("histogram.bin_ct_out", TORCH_FN(histogram_ct_out_fn));
        m.impl("histogram.bins_tensor", TORCH_FN(histogram_t_fn));
        m.impl("histogram.bins_tensor_out", TORCH_FN(histogram_t_out_fn));
        m.impl("_histogramdd_bin_edges", TORCH_FN(histogramdd_bin_edges_fn));
        m.impl("_histogramdd_from_bin_cts", TORCH_FN(histogramdd_from_bin_cts_fn));
        m.impl("_histogramdd_from_bin_tensors", TORCH_FN(histogramdd_from_bin_tensors_fn));
    }
}
