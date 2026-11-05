// Matrix products: mm, addmm, bmm, baddbmm, addmv, dot.

#include <ATen/ATen.h>
#include <ATen/MemoryOverlap.h>
#include <ATen/ops/addmm_meta.h>
#include <ATen/ops/addmv_meta.h>
#include <ATen/ops/baddbmm_meta.h>
#include <ATen/ops/bmm_meta.h>
#include <ATen/ops/mm_meta.h>
#include <torch/library.h>

#include "kernels/matmul.h"
#include "ops/common.h"

namespace torch_webgpu::ops
{
    using at::Scalar;
    using at::Tensor;

    namespace
    {
        // GEMM supports f32/f16/bf16 and 32-bit-or-narrower ints; int64 and
        // others use the CPU fallback.
        bool mm_dtypes_ok(std::initializer_list<std::reference_wrapper<const Tensor>> ts)
        {
            for (const Tensor &t : ts)
                if (is_webgpu(t) && (!wgsl::is_compute_dtype(t.scalar_type()) || t.scalar_type() == at::kLong ||
                                     t.scalar_type() == at::kBool))
                    return false;
            return true;
        }

        bool overlaps(const Tensor &out, std::initializer_list<std::reference_wrapper<const Tensor>> ins)
        {
            for (const Tensor &t : ins)
                if (t.defined() && t.numel() > 0 && out.numel() > 0 && at::get_overlap_status(out, t) != at::MemOverlapStatus::No)
                    return true;
            return false;
        }

        // Runs out = beta * bias + alpha * a @ w on 3-D views, through a
        // temporary when the output aliases an input.
        void gemm(const Tensor &a, const Tensor &w, const Tensor &out, const Tensor &bias, const Scalar &beta,
                  const Scalar &alpha)
        {
            if (overlaps(out, {a, w, bias}))
            {
                Tensor tmp = at::empty(out.sizes(), out.options());
                kernels::run_matmul(a, w, tmp, bias, beta, alpha);
                out.copy_(tmp);
                return;
            }
            kernels::run_matmul(a, w, out, bias, beta, alpha);
        }

        void mm_impl(const Tensor &self, const Tensor &mat2, const Tensor &out)
        {
            gemm(self.unsqueeze(0), mat2.unsqueeze(0), out.unsqueeze(0), Tensor(), 0, 1);
        }

        Tensor mm_fn(const Tensor &self, const Tensor &mat2)
        {
            Functional<at::meta::structured_mm> op;
            op.meta(self, mat2);
            if (!mm_dtypes_ok({self, mat2}))
                return WEBGPU_FALLBACK0(mm)(self, mat2);
            mm_impl(self, mat2, op.outputs_[0]);
            return std::move(op.outputs_[0]);
        }
        Tensor &mm_out_fn(const Tensor &self, const Tensor &mat2, Tensor &out)
        {
            Out<at::meta::structured_mm> op(out);
            op.meta(self, mat2);
            if (!mm_dtypes_ok({self, mat2, out}))
                return WEBGPU_FALLBACK(mm, out)(self, mat2, out);
            mm_impl(self, mat2, op.maybe_get_output(0));
            op.finish();
            return out;
        }

        void addmm_impl(const Tensor &self, const Tensor &m1, const Tensor &m2, const Scalar &beta, const Scalar &alpha,
                        const Tensor &out)
        {
            Tensor bias = self.expand({m1.size(0), m2.size(1)}).unsqueeze(0);
            gemm(m1.unsqueeze(0), m2.unsqueeze(0), out.unsqueeze(0), bias, beta, alpha);
        }
        Tensor addmm_fn(const Tensor &self, const Tensor &m1, const Tensor &m2, const Scalar &beta, const Scalar &alpha)
        {
            Functional<at::meta::structured_addmm> op;
            op.meta(self, m1, m2, beta, alpha);
            if (!mm_dtypes_ok({self, m1, m2}))
                return WEBGPU_FALLBACK0(addmm)(self, m1, m2, beta, alpha);
            addmm_impl(self, m1, m2, beta, alpha, op.outputs_[0]);
            return std::move(op.outputs_[0]);
        }
        Tensor &addmm_out_fn(const Tensor &self, const Tensor &m1, const Tensor &m2, const Scalar &beta,
                             const Scalar &alpha, Tensor &out)
        {
            Out<at::meta::structured_addmm> op(out);
            op.meta(self, m1, m2, beta, alpha);
            if (!mm_dtypes_ok({self, m1, m2, out}))
                return WEBGPU_FALLBACK(addmm, out)(self, m1, m2, beta, alpha, out);
            addmm_impl(self, m1, m2, beta, alpha, op.maybe_get_output(0));
            op.finish();
            return out;
        }
        Tensor &addmm_inplace_fn(Tensor &self, const Tensor &m1, const Tensor &m2, const Scalar &beta, const Scalar &alpha)
        {
            Inplace<at::meta::structured_addmm> op(self);
            op.meta(self, m1, m2, beta, alpha);
            if (!mm_dtypes_ok({self, m1, m2}))
                return WEBGPU_FALLBACK0(addmm_)(self, m1, m2, beta, alpha);
            addmm_impl(self, m1, m2, beta, alpha, op.maybe_get_output(0));
            op.finish();
            return self;
        }

        Tensor bmm_fn(const Tensor &self, const Tensor &mat2)
        {
            Functional<at::meta::structured_bmm> op;
            op.meta(self, mat2);
            if (!mm_dtypes_ok({self, mat2}))
                return WEBGPU_FALLBACK0(bmm)(self, mat2);
            gemm(self, mat2, op.outputs_[0], Tensor(), 0, 1);
            return std::move(op.outputs_[0]);
        }
        Tensor &bmm_out_fn(const Tensor &self, const Tensor &mat2, Tensor &out)
        {
            Out<at::meta::structured_bmm> op(out);
            op.meta(self, mat2);
            if (!mm_dtypes_ok({self, mat2, out}))
                return WEBGPU_FALLBACK(bmm, out)(self, mat2, out);
            gemm(self, mat2, op.maybe_get_output(0), Tensor(), 0, 1);
            op.finish();
            return out;
        }

        void baddbmm_impl(const Tensor &self, const Tensor &b1, const Tensor &b2, const Scalar &beta, const Scalar &alpha,
                          const Tensor &out)
        {
            Tensor bias = self.expand({b1.size(0), b1.size(1), b2.size(2)});
            gemm(b1, b2, out, bias, beta, alpha);
        }
        Tensor baddbmm_fn(const Tensor &self, const Tensor &b1, const Tensor &b2, const Scalar &beta, const Scalar &alpha)
        {
            Functional<at::meta::structured_baddbmm> op;
            op.meta(self, b1, b2, beta, alpha);
            if (!mm_dtypes_ok({self, b1, b2}))
                return WEBGPU_FALLBACK0(baddbmm)(self, b1, b2, beta, alpha);
            baddbmm_impl(self, b1, b2, beta, alpha, op.outputs_[0]);
            return std::move(op.outputs_[0]);
        }
        Tensor &baddbmm_out_fn(const Tensor &self, const Tensor &b1, const Tensor &b2, const Scalar &beta,
                               const Scalar &alpha, Tensor &out)
        {
            Out<at::meta::structured_baddbmm> op(out);
            op.meta(self, b1, b2, beta, alpha);
            if (!mm_dtypes_ok({self, b1, b2, out}))
                return WEBGPU_FALLBACK(baddbmm, out)(self, b1, b2, beta, alpha, out);
            baddbmm_impl(self, b1, b2, beta, alpha, op.maybe_get_output(0));
            op.finish();
            return out;
        }
        Tensor &baddbmm_inplace_fn(Tensor &self, const Tensor &b1, const Tensor &b2, const Scalar &beta,
                                   const Scalar &alpha)
        {
            Inplace<at::meta::structured_baddbmm> op(self);
            op.meta(self, b1, b2, beta, alpha);
            if (!mm_dtypes_ok({self, b1, b2}))
                return WEBGPU_FALLBACK0(baddbmm_)(self, b1, b2, beta, alpha);
            baddbmm_impl(self, b1, b2, beta, alpha, op.maybe_get_output(0));
            op.finish();
            return self;
        }

        void addmv_impl(const Tensor &self, const Tensor &mat, const Tensor &vec, const Scalar &beta, const Scalar &alpha,
                        const Tensor &out)
        {
            Tensor bias = self.expand({mat.size(0)}).unsqueeze(1).unsqueeze(0);
            gemm(mat.unsqueeze(0), vec.unsqueeze(1).unsqueeze(0), out.unsqueeze(1).unsqueeze(0), bias, beta, alpha);
        }
        Tensor addmv_fn(const Tensor &self, const Tensor &mat, const Tensor &vec, const Scalar &beta, const Scalar &alpha)
        {
            Functional<at::meta::structured_addmv> op;
            op.meta(self, mat, vec, beta, alpha);
            if (!mm_dtypes_ok({self, mat, vec}))
                return WEBGPU_FALLBACK0(addmv)(self, mat, vec, beta, alpha);
            addmv_impl(self, mat, vec, beta, alpha, op.outputs_[0]);
            return std::move(op.outputs_[0]);
        }
        Tensor &addmv_out_fn(const Tensor &self, const Tensor &mat, const Tensor &vec, const Scalar &beta,
                             const Scalar &alpha, Tensor &out)
        {
            Out<at::meta::structured_addmv> op(out);
            op.meta(self, mat, vec, beta, alpha);
            if (!mm_dtypes_ok({self, mat, vec, out}))
                return WEBGPU_FALLBACK(addmv, out)(self, mat, vec, beta, alpha, out);
            addmv_impl(self, mat, vec, beta, alpha, op.maybe_get_output(0));
            op.finish();
            return out;
        }

        Tensor dot_fn(const Tensor &self, const Tensor &other)
        {
            TORCH_CHECK(self.dim() == 1 && other.dim() == 1, "1D tensors expected, but got ", self.dim(), "D and ",
                        other.dim(), "D tensors");
            TORCH_CHECK(self.scalar_type() == other.scalar_type(),
                        "dot : expected both vectors to have same dtype, but found ", self.scalar_type(), " and ",
                        other.scalar_type());
            TORCH_CHECK(self.numel() == other.numel(), "inconsistent tensor size, expected tensor [", self.numel(),
                        "] and src [", other.numel(), "] to have the same number of elements, but got ", self.numel(),
                        " and ", other.numel(), " elements respectively");
            if (!mm_dtypes_ok({self, other}))
                return WEBGPU_FALLBACK0(dot)(self, other);
            Tensor out = at::empty({}, self.options());
            gemm(self.unsqueeze(0).unsqueeze(0), other.unsqueeze(1).unsqueeze(0), out.view({1, 1, 1}), Tensor(), 0, 1);
            return out;
        }
    }

    TORCH_LIBRARY_IMPL(aten, PrivateUse1, m)
    {
        m.impl("mm", TORCH_FN(mm_fn));
        m.impl("mm.out", TORCH_FN(mm_out_fn));
        m.impl("addmm", TORCH_FN(addmm_fn));
        m.impl("addmm.out", TORCH_FN(addmm_out_fn));
        m.impl("addmm_", TORCH_FN(addmm_inplace_fn));
        m.impl("bmm", TORCH_FN(bmm_fn));
        m.impl("bmm.out", TORCH_FN(bmm_out_fn));
        m.impl("baddbmm", TORCH_FN(baddbmm_fn));
        m.impl("baddbmm.out", TORCH_FN(baddbmm_out_fn));
        m.impl("baddbmm_", TORCH_FN(baddbmm_inplace_fn));
        m.impl("addmv", TORCH_FN(addmv_fn));
        m.impl("addmv.out", TORCH_FN(addmv_out_fn));
        m.impl("dot", TORCH_FN(dot_fn));
    }
}
