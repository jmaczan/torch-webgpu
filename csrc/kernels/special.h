#pragma once
// WGSL special functions (gamma family, Bessel, normal distribution, ...),
// ported from PyTorch's float implementations (ATen/native/Math.h, mostly
// Cephes). Kept out of math_wgsl() so ordinary kernels stay small: ops
// request the functions they use and get them with their dependencies.
//
// Polynomial and Chebyshev tables are emitted as unrolled code with inline
// constants (WGSL can't index constant arrays dynamically everywhere, and
// unrolled evaluation is faster anyway).

#include <string>
#include <vector>

namespace torch_webgpu::kernels
{
    // Returns WGSL defining the named functions (and their dependencies), to
    // be appended after math_wgsl(). Known names:
    //   lgamma_f digamma_f trigamma_f zeta_f(x, q) polygamma_f(n+1, (-1)^(n+1) n!, x) i0_f i0e_f i1_f i1e_f
    //   ndtri_f erfcx_f log_ndtr_f entr_f xlog1py_f(x, y) frexp_f (-> vec2<u32>(mantissa bits, exponent))
    std::string special_wgsl(const std::vector<std::string> &names);
}
