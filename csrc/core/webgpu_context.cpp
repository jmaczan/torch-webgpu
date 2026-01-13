#include <webgpu/webgpu_cpp.h>
#include <iostream>
#include "webgpu_context.h"

namespace torch_webgpu
{
    namespace core
    {
        WebGPUContext::WebGPUContext()
        {
            static const auto k_timed_wait_any = wgpu::InstanceFeatureName::TimedWaitAny;
            wgpu::InstanceDescriptor instance_descriptor{};
            instance_descriptor.requiredFeatureCount = 1;
            instance_descriptor.requiredFeatures = &k_timed_wait_any;
            instance = wgpu::CreateInstance(&instance_descriptor);

            wgpu::RequestAdapterOptions adapter_options{};
            adapter_options.powerPreference = wgpu::PowerPreference::HighPerformance;
            bool acquired_high_perf_adapter = false;
            wgpu::Future adapter_future = instance.RequestAdapter(
                &adapter_options, wgpu::CallbackMode::WaitAnyOnly,
                [this, &acquired_high_perf_adapter](wgpu::RequestAdapterStatus status, wgpu::Adapter a, wgpu::StringView message)
                {
                    if (status != wgpu::RequestAdapterStatus::Success)
                    {
                        std::cout << "Failed to load High Performance WebGPU Adapter. Trying to get a regular one..." << "\n";
                        return;
                    }
                    this->adapter = std::move(a);
                    acquired_high_perf_adapter = true;
                    std::cout << "Chosen high performance WebGPU Adapter" << "\n";
                    wgpu::AdapterInfo info{};
                    if (adapter.GetInfo(&info) == wgpu::Status::Success)
                    {
                        std::cout << "High performance WebGPU Adapter details:" << info.vendor.data << " " << info.architecture.data
                                  << " " << info.description.data << "\n";
                    }
                });
            instance.WaitAny(adapter_future, UINT64_MAX);

            if (!acquired_high_perf_adapter)
            {
                wgpu::Future adapter_future = instance.RequestAdapter(
                    nullptr, wgpu::CallbackMode::WaitAnyOnly,
                    [this](wgpu::RequestAdapterStatus status, wgpu::Adapter a, wgpu::StringView message)
                    {
                        if (status != wgpu::RequestAdapterStatus::Success)
                        {
                            std::cout << "Failed to load WebGPU adapter \n";
                            exit(1);
                        }
                        this->adapter = std::move(a);
                        std::cout << "Chosen a regular WebGPU Adapter (not high performance)" << "\n";
                        wgpu::AdapterInfo info{};
                        if (adapter.GetInfo(&info) == wgpu::Status::Success)
                        {
                            std::cout << "Chosen WebGPU Adapter details:" << info.vendor.data << " " << info.architecture.data
                                      << " " << info.description.data << "\n";
                        }
                    });
                instance.WaitAny(adapter_future, UINT64_MAX);
            }

            wgpu::Limits adapter_limits{};
            if (adapter.GetLimits(&adapter_limits) != wgpu::Status::Success)
            {
                std::cout << "Failed to query WebGPU adapter limits" << "\n";
                exit(1);
            }

            wgpu::DeviceDescriptor device_descriptor{};
            device_descriptor.requiredLimits = &adapter_limits;
            device_descriptor.SetUncapturedErrorCallback([](const wgpu::Device &, wgpu::ErrorType errorType, wgpu::StringView message)
                                                         { std::cout << "Error in device descriptor" << static_cast<int>(errorType) << std::string(message.data, message.length) << "\n"; });

            wgpu::Future device_future = adapter.RequestDevice(
                &device_descriptor, wgpu::CallbackMode::WaitAnyOnly,
                [this](wgpu::RequestDeviceStatus status, wgpu::Device d, wgpu::StringView message)
                {
                    if (status != wgpu::RequestDeviceStatus::Success)
                    {
                        std::cout << "Request WebGPU device failed" << "\n";
                        exit(1);
                    }
                    this->device = std::move(d);
                    this->queue = device.GetQueue();
                });
            instance.WaitAny(device_future, UINT64_MAX);
        }

        wgpu::Instance WebGPUContext::getInstance()
        {
            return instance;
        }

        wgpu::Device WebGPUContext::getDevice()
        {
            return device;
        }

        wgpu::Queue WebGPUContext::getQueue()
        {
            return queue;
        }

        WebGPUContext &getWebGPUContext()
        {
            static WebGPUContext webgpu_context;
            return webgpu_context;
        }
    }
}