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
            wgpu::InstanceDescriptor instance_descriptor{
                .requiredFeatureCount = 1,
                .requiredFeatures = &k_timed_wait_any};
            instance = wgpu::CreateInstance(&instance_descriptor);

            wgpu::Future adapter_future = instance.RequestAdapter(
                nullptr, wgpu::CallbackMode::WaitAnyOnly,
                [this](wgpu::RequestAdapterStatus status, wgpu::Adapter a, wgpu::StringView message)
                {
                    if (status != wgpu::RequestAdapterStatus::Success)
                    {
                        std::cout << "Failed to load WebGPU Adapter" << "\n";
                        exit(1);
                    }
                    this->adapter = std::move(a);
                });
            instance.WaitAny(adapter_future, UINT64_MAX);

            wgpu::DeviceDescriptor device_descriptor;
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