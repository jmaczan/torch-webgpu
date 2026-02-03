#include "dispatch_profiler.h"
#include <sstream>
#include <iomanip>
#include <chrono>

namespace torch_webgpu {
namespace core {

DispatchProfiler& getDispatchProfiler() {
    static DispatchProfiler instance;
    return instance;
}

std::string DispatchProfiler::getSummary() const {
    std::ostringstream ss;
    ss << std::fixed << std::setprecision(2);

    size_t dispatches = getDispatchCount();
    size_t submissions = getSubmissionCount();

    ss << "=== WebGPU Dispatch Profiler Summary ===\n";
    ss << "Dispatch count: " << dispatches << "\n";
    ss << "Submission count: " << submissions << "\n";
    ss << "\n";

    ss << "--- Timing Breakdown (microseconds) ---\n";
    ss << "Total encoder creation:    " << getTotalEncoderTimeUs() << " us\n";
    ss << "Total bind group creation: " << getTotalBindGroupTimeUs() << " us\n";
    ss << "Total submission:          " << getTotalSubmissionTimeUs() << " us\n";
    ss << "Total GPU sync:            " << getTotalGpuSyncTimeUs() << " us\n";
    ss << "\n";

    ss << "--- Per-Dispatch Averages ---\n";
    ss << "Avg encoder creation:      " << getAvgEncoderTimeUs() << " us\n";
    ss << "Avg bind group creation:   " << getAvgBindGroupTimeUs() << " us\n";
    ss << "Avg submission:            " << getAvgSubmissionTimeUs() << " us\n";
    ss << "Total per-dispatch overhead: " << getAvgDispatchOverheadUs() << " us\n";
    ss << "\n";

    // Calculate total overhead as percentage of hypothetical compute time
    double total_overhead_ms = (getTotalEncoderTimeUs() + getTotalBindGroupTimeUs() +
                                getTotalSubmissionTimeUs()) / 1000.0;
    ss << "--- Summary ---\n";
    ss << "Total dispatch overhead: " << total_overhead_ms << " ms\n";
    ss << "Dispatches per submission: " << (submissions > 0 ? dispatches / (double)submissions : 0) << "\n";

    return ss.str();
}

ScopedTimer::ScopedTimer()
    : start_(std::chrono::high_resolution_clock::now()) {}

double ScopedTimer::elapsedUs() const {
    auto end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::micro>(end - start_).count();
}

} // namespace core
} // namespace torch_webgpu
