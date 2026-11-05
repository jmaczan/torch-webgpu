#pragma once
// WebGPU device context: instance/adapter/device/queue, limits and features.
//
// There is a single process-wide context (WebGPU exposes one logical device
// per adapter; we pick the high-performance adapter). All GPU work is
// serialized through the context mutex — see stream.h.

#include <webgpu/webgpu_cpp.h>

#include <atomic>
#include <cstdint>
#include <mutex>
#include <string>

namespace torch_webgpu::core
{
    struct DeviceInfo
    {
        std::string vendor;
        std::string architecture;
        std::string device;
        std::string description;
        std::string backend; // "Vulkan", "Metal", "D3D12", ...
        std::string adapter_type; // "discrete", "integrated", "cpu", ...
    };

    class Context
    {
    public:
        static Context &get();

        const wgpu::Instance &instance() const { return instance_; }
        const wgpu::Adapter &adapter() const { return adapter_; }
        const wgpu::Device &device() const { return device_; }
        const wgpu::Queue &queue() const { return queue_; }
        const wgpu::Limits &limits() const { return limits_; }
        const DeviceInfo &info() const { return info_; }

        bool has_f16() const { return has_f16_; }
        bool has_subgroups() const { return has_subgroups_; }
        bool has_timestamp_query() const { return has_timestamp_query_; }

        // Global lock for everything that touches the device/queue/encoder.
        // Recursive because ops call other ops.
        std::recursive_mutex &mutex() { return mutex_; }

        // Blocks until `future` completes.
        void wait(wgpu::Future future);

        // Errors raised asynchronously by Dawn (uncaptured validation errors,
        // device loss) are recorded here and re-thrown on the calling thread
        // at the next synchronization point.
        void record_error(std::string msg);
        void check_errors();

        bool device_lost() const { return device_lost_.load(); }

    private:
        Context();
        Context(const Context &) = delete;
        Context &operator=(const Context &) = delete;

        wgpu::Instance instance_;
        wgpu::Adapter adapter_;
        wgpu::Device device_;
        wgpu::Queue queue_;
        wgpu::Limits limits_{};
        DeviceInfo info_;
        bool has_f16_ = false;
        bool has_subgroups_ = false;
        bool has_timestamp_query_ = false;

        std::recursive_mutex mutex_;
        std::mutex error_mutex_;
        std::string pending_error_;
        std::atomic<bool> has_error_{false};
        std::atomic<bool> device_lost_{false};
    };

    // True when the context can be (or has been) created. Never throws.
    bool is_available();

    // Debug logging, enabled with TORCH_WEBGPU_DEBUG=1.
    bool debug_enabled();
}
