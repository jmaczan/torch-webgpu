#pragma once
// A tiled GEMM whose operand loaders and output store are WGSL snippets:
//   C[z, m, n] = sum_k A(z, m, k) * B(z, k, n)
// for "implicit" GEMMs whose operands are never materialized (e.g.
// convolutions reading input patches on the fly).
//
// Snippets see the bound tensors as ld_<name>(i) / st_<name>(i, v) (element
// index into the binding window, with the tensor's window offset available
// as off_<name>) and the user parameters as P.v[i] (u32). They must define:
//   load_a: body of fn load_a(z: u32, m: u32, k: u32) -> f32
//   load_b: body of fn load_b(z: u32, k: u32, n: u32) -> f32
//   store:  body of fn store_c(z: u32, m: u32, n: u32, v: f32)
// Loads are only called for in-range (m, k) / (k, n).

#include <ATen/ATen.h>

#include <string>
#include <vector>

namespace torch_webgpu::kernels
{
    struct GemmOperand
    {
        std::string name;
        at::Tensor tensor;
        bool written = false;
    };

    struct CustomGemm
    {
        std::string name;  // pipeline label / cache key prefix
        std::string key;   // identifies the snippets and their parameter layout
        int64_t M = 0, N = 0, K = 0, Z = 1;
        std::vector<GemmOperand> operands;
        std::vector<uint32_t> params; // P.v[0..]
        std::string load_a, load_b, store;
        // Tile load mappings: consecutive threads along k for A (else along m),
        // along n for B (else along k).
        bool a_k_fast = true, b_n_fast = true;
    };

    void run_custom_gemm(const CustomGemm &g);

    // Split-K: the K range is cut into `slices` inside one dispatch; partial
    // products go to `partial` ([slices, Z, M, N] float, contiguous) instead
    // of `store`. The caller reduces them.
    void run_custom_gemm_split(const CustomGemm &g, int64_t slices, const at::Tensor &partial);
}
