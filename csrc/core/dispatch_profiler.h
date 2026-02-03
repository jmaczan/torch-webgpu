#pragma once

#include <atomic>
#include <string>

namespace torch_webgpu {
namespace core {

/**
 * DispatchProfiler - Direct measurement of WebGPU dispatch overhead.
 *
 * This profiler measures actual time spent in WebGPU operations:
 * - Encoder creation time
 * - Bind group creation time
 * - Command submission time
 * - GPU synchronization time
 *
 * Addresses reviewer concern about "derived not measured" overhead claims.
 */
class DispatchProfiler {
public:
    DispatchProfiler() = default;

    // Enable/disable profiling (disabled by default for performance)
    void setEnabled(bool enabled) { enabled_ = enabled; }
    bool isEnabled() const { return enabled_; }

    // Record a dispatch event
    void recordDispatch() {
        if (enabled_) dispatch_count_.fetch_add(1, std::memory_order_relaxed);
    }

    // Record timing measurements (in microseconds)
    void recordEncoderCreation(double elapsed_us) {
        if (enabled_) {
            encoder_time_us_.fetch_add(static_cast<int64_t>(elapsed_us * 1000),
                                       std::memory_order_relaxed);
        }
    }

    void recordBindGroupCreation(double elapsed_us) {
        if (enabled_) {
            bind_group_time_us_.fetch_add(static_cast<int64_t>(elapsed_us * 1000),
                                          std::memory_order_relaxed);
        }
    }

    void recordSubmission(double elapsed_us) {
        if (enabled_) {
            submission_count_.fetch_add(1, std::memory_order_relaxed);
            submission_time_us_.fetch_add(static_cast<int64_t>(elapsed_us * 1000),
                                          std::memory_order_relaxed);
        }
    }

    void recordGpuSync(double elapsed_us) {
        if (enabled_) {
            gpu_sync_time_us_.fetch_add(static_cast<int64_t>(elapsed_us * 1000),
                                        std::memory_order_relaxed);
        }
    }

    // Get statistics
    size_t getDispatchCount() const {
        return dispatch_count_.load(std::memory_order_relaxed);
    }

    size_t getSubmissionCount() const {
        return submission_count_.load(std::memory_order_relaxed);
    }

    double getTotalEncoderTimeUs() const {
        return encoder_time_us_.load(std::memory_order_relaxed) / 1000.0;
    }

    double getTotalBindGroupTimeUs() const {
        return bind_group_time_us_.load(std::memory_order_relaxed) / 1000.0;
    }

    double getTotalSubmissionTimeUs() const {
        return submission_time_us_.load(std::memory_order_relaxed) / 1000.0;
    }

    double getTotalGpuSyncTimeUs() const {
        return gpu_sync_time_us_.load(std::memory_order_relaxed) / 1000.0;
    }

    // Average per-dispatch overhead
    double getAvgEncoderTimeUs() const {
        size_t count = dispatch_count_.load(std::memory_order_relaxed);
        return count > 0 ? getTotalEncoderTimeUs() / count : 0.0;
    }

    double getAvgBindGroupTimeUs() const {
        size_t count = dispatch_count_.load(std::memory_order_relaxed);
        return count > 0 ? getTotalBindGroupTimeUs() / count : 0.0;
    }

    double getAvgSubmissionTimeUs() const {
        size_t count = submission_count_.load(std::memory_order_relaxed);
        return count > 0 ? getTotalSubmissionTimeUs() / count : 0.0;
    }

    // Total dispatch overhead (encoder + bind group creation)
    double getAvgDispatchOverheadUs() const {
        return getAvgEncoderTimeUs() + getAvgBindGroupTimeUs();
    }

    // Reset all counters
    void reset() {
        dispatch_count_.store(0, std::memory_order_relaxed);
        submission_count_.store(0, std::memory_order_relaxed);
        encoder_time_us_.store(0, std::memory_order_relaxed);
        bind_group_time_us_.store(0, std::memory_order_relaxed);
        submission_time_us_.store(0, std::memory_order_relaxed);
        gpu_sync_time_us_.store(0, std::memory_order_relaxed);
    }

    // Get human-readable summary
    std::string getSummary() const;

private:
    bool enabled_ = false;
    std::atomic<size_t> dispatch_count_{0};
    std::atomic<size_t> submission_count_{0};
    // Store as int64_t * 1000 to avoid floating-point atomics
    std::atomic<int64_t> encoder_time_us_{0};
    std::atomic<int64_t> bind_group_time_us_{0};
    std::atomic<int64_t> submission_time_us_{0};
    std::atomic<int64_t> gpu_sync_time_us_{0};
};

// Singleton accessor
DispatchProfiler& getDispatchProfiler();

// RAII timer helper
class ScopedTimer {
public:
    ScopedTimer();
    double elapsedUs() const;
private:
    std::chrono::high_resolution_clock::time_point start_;
};

} // namespace core
} // namespace torch_webgpu
