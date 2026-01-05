#include <benchmark/benchmark.h>
#include <ATen/ATen.h>
#include <cstdint>
#include "core/webgpu_context.h"

namespace torch_webgpu
{
    namespace ops
    {
        void mm_kernel_webgpu(const at::Tensor &self, const at::Tensor &mat2, at::Tensor &out);
    }
}

using torch_webgpu::core::getWebGPUContext;

static std::vector<int64_t> make_contiguous_strides(const std::vector<int64_t> &sizes)
{
    std::vector<int64_t> strides(sizes.size(), 1);
    for (int64_t i = static_cast<int64_t>(sizes.size()) - 2; i >= 0; --i)
    {
        strides[i] = strides[i + 1] * sizes[i + 1];
    }
    return strides;
}

static at::Tensor make_webgpu_tensor(const std::vector<int64_t> &sizes)
{
    auto opts = at::TensorOptions().dtype(at::kFloat).device(c10::Device(c10::kPrivateUse1));
    auto strides = make_contiguous_strides(sizes);
    auto gpu = at::empty_strided(sizes, strides, opts);
    auto cpu = at::randn(sizes, at::kFloat);
    gpu.copy_(cpu);
    return gpu;
}

static void wait_for_queue()
{
    auto &ctx = getWebGPUContext();
    bool ok = true;
    auto fut = ctx.getQueue().OnSubmittedWorkDone(
        wgpu::CallbackMode::WaitAnyOnly,
        [&ok](wgpu::QueueWorkDoneStatus status, wgpu::StringView)
        {
            ok = (status == wgpu::QueueWorkDoneStatus::Success);
        });
    ctx.getInstance().WaitAny(fut, UINT64_MAX);
    TORCH_CHECK(ok, "WebGPU queue reported failure in benchmark");
}

static void BM_MM(benchmark::State &state)
{
    const int64_t M = state.range(0);
    const int64_t N = state.range(1);
    const int64_t K = state.range(2);

    auto a = make_webgpu_tensor({M, N});
    auto b = make_webgpu_tensor({N, K});
    auto out_strides = make_contiguous_strides({M, K});
    auto out = at::empty_strided({M, K}, out_strides, at::TensorOptions().dtype(at::kFloat).device(c10::Device(c10::kPrivateUse1)));
    // Warmups to exclude lazy init / pipeline compilation from measured iterations.
    for (int i = 0; i < 2; ++i)
    {
        torch_webgpu::ops::mm_kernel_webgpu(a, b, out);
        wait_for_queue();
    }

    const double flops_per_call = 2.0 * static_cast<double>(M) * static_cast<double>(N) * static_cast<double>(K);
    const double bytes_per_call = static_cast<double>(a.nbytes() + b.nbytes() + out.nbytes());

    for (auto _ : state)
    {
        auto t0 = std::chrono::high_resolution_clock::now();
        torch_webgpu::ops::mm_kernel_webgpu(a, b, out);
        wait_for_queue();
        auto t1 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> duration = t1 - t0;
        state.SetIterationTime(duration.count());
        benchmark::DoNotOptimize(out);
        benchmark::ClobberMemory();
    }

    const auto flags = benchmark::Counter::kIsRate | benchmark::Counter::kIsIterationInvariantRate;
    state.counters["gflops"] = benchmark::Counter{flops_per_call / 1e9, flags};
    state.counters["bytes"] = benchmark::Counter{bytes_per_call, flags};
}

// Register a few common shapes; tweak as needed.
BENCHMARK(BM_MM)
    ->Args({32, 32, 32})
    ->Args({64, 64, 64})
    ->Args({128, 128, 128})
    ->Args({256, 256, 256})
    ->Args({512, 512, 512})
    ->Args({1024, 1024, 1024})
    ->Args({2048, 2048, 2048})
    ->Args({4096, 4096, 4096})
    ->UseManualTime()
    ->Iterations(2)
    ->Unit(benchmark::kMillisecond);

BENCHMARK_MAIN();
