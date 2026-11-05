#include "core/context.h"

#include <c10/util/Exception.h>

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <sstream>
#include <vector>

namespace torch_webgpu::core
{
    namespace
    {
        std::string to_string(wgpu::StringView sv)
        {
            if (sv.data == nullptr)
                return {};
            if (sv.length == WGPU_STRLEN)
                return std::string(sv.data);
            return std::string(sv.data, sv.length);
        }

        std::string lower(std::string s)
        {
            std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c)
                           { return static_cast<char>(std::tolower(c)); });
            return s;
        }

        std::vector<std::string> split_csv(const char *s)
        {
            std::vector<std::string> out;
            if (!s)
                return out;
            std::stringstream ss(s);
            std::string item;
            while (std::getline(ss, item, ','))
            {
                if (!item.empty())
                    out.push_back(item);
            }
            return out;
        }

        const char *backend_name(wgpu::BackendType t)
        {
            switch (t)
            {
            case wgpu::BackendType::Null:
                return "Null";
            case wgpu::BackendType::WebGPU:
                return "WebGPU";
            case wgpu::BackendType::D3D11:
                return "D3D11";
            case wgpu::BackendType::D3D12:
                return "D3D12";
            case wgpu::BackendType::Metal:
                return "Metal";
            case wgpu::BackendType::Vulkan:
                return "Vulkan";
            case wgpu::BackendType::OpenGL:
                return "OpenGL";
            case wgpu::BackendType::OpenGLES:
                return "OpenGLES";
            default:
                return "Undefined";
            }
        }

        const char *adapter_type_name(wgpu::AdapterType t)
        {
            switch (t)
            {
            case wgpu::AdapterType::DiscreteGPU:
                return "discrete";
            case wgpu::AdapterType::IntegratedGPU:
                return "integrated";
            case wgpu::AdapterType::CPU:
                return "cpu";
            default:
                return "unknown";
            }
        }

        wgpu::BackendType parse_backend(const std::string &name)
        {
            auto n = lower(name);
            if (n == "vulkan")
                return wgpu::BackendType::Vulkan;
            if (n == "metal")
                return wgpu::BackendType::Metal;
            if (n == "d3d12")
                return wgpu::BackendType::D3D12;
            if (n == "d3d11")
                return wgpu::BackendType::D3D11;
            if (n == "opengl" || n == "gl")
                return wgpu::BackendType::OpenGL;
            if (n == "opengles" || n == "gles")
                return wgpu::BackendType::OpenGLES;
            TORCH_CHECK(false, "TORCH_WEBGPU_BACKEND: unknown backend '", name,
                        "' (expected vulkan, metal, d3d12, d3d11, opengl, opengles)");
        }
    }

    bool debug_enabled()
    {
        static const bool enabled = []
        {
            const char *v = std::getenv("TORCH_WEBGPU_DEBUG");
            return v && std::string(v) != "0" && std::string(v) != "";
        }();
        return enabled;
    }

    Context &Context::get()
    {
        // Intentionally leaked: Dawn objects must not be destroyed during
        // static destruction (the driver may already be gone at that point).
        static Context *ctx = new Context();
        return *ctx;
    }

    bool is_available()
    {
        try
        {
            Context::get();
            return true;
        }
        catch (const std::exception &)
        {
            return false;
        }
    }

    Context::Context()
    {
        static const auto k_timed_wait_any = wgpu::InstanceFeatureName::TimedWaitAny;
        wgpu::InstanceDescriptor instance_desc{};
        instance_desc.requiredFeatureCount = 1;
        instance_desc.requiredFeatures = &k_timed_wait_any;
        instance_ = wgpu::CreateInstance(&instance_desc);
        TORCH_CHECK(instance_, "torch_webgpu: failed to create a WebGPU instance");

        // ---- Adapter -------------------------------------------------------
        wgpu::RequestAdapterOptions adapter_opts{};
        adapter_opts.powerPreference = wgpu::PowerPreference::HighPerformance;
        if (const char *b = std::getenv("TORCH_WEBGPU_BACKEND"); b && *b)
            adapter_opts.backendType = parse_backend(b);
        if (const char *f = std::getenv("TORCH_WEBGPU_FORCE_FALLBACK_ADAPTER"); f && std::string(f) == "1")
            adapter_opts.forceFallbackAdapter = true;

        std::string adapter_error;
        wait(instance_.RequestAdapter(
            &adapter_opts, wgpu::CallbackMode::WaitAnyOnly,
            [&](wgpu::RequestAdapterStatus status, wgpu::Adapter a, wgpu::StringView message)
            {
                if (status == wgpu::RequestAdapterStatus::Success)
                    adapter_ = std::move(a);
                else
                    adapter_error = to_string(message);
            }));
        TORCH_CHECK(adapter_, "torch_webgpu: no WebGPU adapter available",
                    adapter_error.empty() ? "" : ": ", adapter_error);

        wgpu::AdapterInfo ainfo{};
        if (adapter_.GetInfo(&ainfo) == wgpu::Status::Success)
        {
            info_.vendor = to_string(ainfo.vendor);
            info_.architecture = to_string(ainfo.architecture);
            info_.device = to_string(ainfo.device);
            info_.description = to_string(ainfo.description);
            info_.backend = backend_name(ainfo.backendType);
            info_.adapter_type = adapter_type_name(ainfo.adapterType);
        }

        wgpu::Limits adapter_limits{};
        TORCH_CHECK(adapter_.GetLimits(&adapter_limits) == wgpu::Status::Success,
                    "torch_webgpu: failed to query adapter limits");

        // ---- Features ------------------------------------------------------
        std::vector<wgpu::FeatureName> features;
        auto want = [&](wgpu::FeatureName f) -> bool
        {
            if (adapter_.HasFeature(f))
            {
                features.push_back(f);
                return true;
            }
            return false;
        };
        const bool disable_f16 = std::getenv("TORCH_WEBGPU_DISABLE_F16") != nullptr;
        has_f16_ = !disable_f16 && want(wgpu::FeatureName::ShaderF16);
        has_subgroups_ = want(wgpu::FeatureName::Subgroups);
        has_timestamp_query_ = want(wgpu::FeatureName::TimestampQuery);

        // ---- Toggles (opt-in, native only) --------------------------------
        // e.g. TORCH_WEBGPU_DAWN_TOGGLES=disable_robustness,skip_validation
        //      TORCH_WEBGPU_DAWN_DISABLED_TOGGLES=lazy_clear_resource_on_first_use
        std::vector<std::string> toggle_storage = split_csv(std::getenv("TORCH_WEBGPU_DAWN_TOGGLES"));
        std::vector<std::string> disabled_storage = split_csv(std::getenv("TORCH_WEBGPU_DAWN_DISABLED_TOGGLES"));
        std::vector<const char *> toggles, disabled;
        for (auto &t : toggle_storage)
            toggles.push_back(t.c_str());
        // Dawn zero-initializes buffers lazily on first use by default. With
        // the Dawn build we ship (Apr 2026) on NVIDIA/Vulkan the lazy clear can
        // race with a partial copy into a fresh buffer, leaving zeros in the
        // copied range (reproducible with large uploads after other work).
        // Clearing eagerly at creation keeps WebGPU's zero-init guarantee
        // without the race. TORCH_WEBGPU_LAZY_CLEAR=1 restores Dawn's default.
        const char *lazy = std::getenv("TORCH_WEBGPU_LAZY_CLEAR");
        if (!(lazy && std::string(lazy) == "1"))
            disabled_storage.push_back("lazy_clear_resource_on_first_use");
        for (auto &t : disabled_storage)
            disabled.push_back(t.c_str());
        wgpu::DawnTogglesDescriptor toggles_desc{};
        toggles_desc.enabledToggleCount = toggles.size();
        toggles_desc.enabledToggles = toggles.data();
        toggles_desc.disabledToggleCount = disabled.size();
        toggles_desc.disabledToggles = disabled.data();

        // ---- Device --------------------------------------------------------
        wgpu::DeviceDescriptor device_desc{};
        device_desc.label = "torch_webgpu device";
        device_desc.requiredLimits = &adapter_limits; // ask for everything the adapter offers
        device_desc.requiredFeatureCount = features.size();
        device_desc.requiredFeatures = features.data();
        if (!toggles.empty() || !disabled.empty())
            device_desc.nextInChain = &toggles_desc;

        device_desc.SetUncapturedErrorCallback(
            [](const wgpu::Device &, wgpu::ErrorType type, wgpu::StringView message)
            {
                std::string kind = type == wgpu::ErrorType::OutOfMemory ? "out of memory" : type == wgpu::ErrorType::Validation ? "validation error"
                                                                                                                                  : "internal error";
                Context::get().record_error("WebGPU " + kind + ": " + to_string(message));
            });
        device_desc.SetDeviceLostCallback(
            wgpu::CallbackMode::AllowSpontaneous,
            [](const wgpu::Device &, wgpu::DeviceLostReason reason, wgpu::StringView message)
            {
                if (reason == wgpu::DeviceLostReason::Destroyed ||
                    reason == wgpu::DeviceLostReason::CallbackCancelled)
                    return;
                auto &ctx = Context::get();
                ctx.device_lost_.store(true);
                ctx.record_error("WebGPU device lost: " + to_string(message));
            });

        std::string device_error;
        wait(adapter_.RequestDevice(
            &device_desc, wgpu::CallbackMode::WaitAnyOnly,
            [&](wgpu::RequestDeviceStatus status, wgpu::Device d, wgpu::StringView message)
            {
                if (status == wgpu::RequestDeviceStatus::Success)
                    device_ = std::move(d);
                else
                    device_error = to_string(message);
            }));
        TORCH_CHECK(device_, "torch_webgpu: failed to create a WebGPU device: ", device_error);

        queue_ = device_.GetQueue();
        TORCH_CHECK(device_.GetLimits(&limits_) == wgpu::Status::Success,
                    "torch_webgpu: failed to query device limits");

        if (debug_enabled())
        {
            std::cerr << "[torch_webgpu] adapter: " << info_.description << " (" << info_.vendor << ", "
                      << info_.architecture << ", " << info_.backend << ", " << info_.adapter_type << ")"
                      << " f16=" << has_f16_ << " subgroups=" << has_subgroups_
                      << " maxStorageBufferBindingSize=" << limits_.maxStorageBufferBindingSize
                      << " maxBufferSize=" << limits_.maxBufferSize << "\n";
        }
    }

    void Context::wait(wgpu::Future future)
    {
        auto status = instance_.WaitAny(future, UINT64_MAX);
        TORCH_CHECK(status == wgpu::WaitStatus::Success, "torch_webgpu: WaitAny failed with status ",
                    static_cast<int>(status));
    }

    void Context::record_error(std::string msg)
    {
        if (debug_enabled())
            std::cerr << "[torch_webgpu] " << msg << "\n";
        std::lock_guard<std::mutex> g(error_mutex_);
        if (pending_error_.empty())
            pending_error_ = std::move(msg);
        has_error_.store(true);
    }

    void Context::check_errors()
    {
        if (!has_error_.load(std::memory_order_relaxed))
            return;
        std::string msg;
        {
            std::lock_guard<std::mutex> g(error_mutex_);
            msg.swap(pending_error_);
            has_error_.store(device_lost_.load());
        }
        if (device_lost_.load() && msg.empty())
            msg = "WebGPU device lost";
        TORCH_CHECK(msg.empty(), "torch_webgpu: ", msg);
    }
}
