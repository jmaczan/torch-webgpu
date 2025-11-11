#pragma once
#include <torch/library.h>

namespace torch_webgpu
{
    namespace core
    {
        static thread_local int current_webgpu_device = 0;
        static int webgpu_device_count = 1;

        struct WebGPUGuardImpl final : public c10::impl::DeviceGuardImplInterface
        {
            static constexpr c10::DeviceType static_type = c10::DeviceType::PrivateUse1;

            WebGPUGuardImpl() {}

            c10::DeviceType type() const;
            c10::Device exchangeDevice(c10::Device d) const;
            c10::Device getDevice() const;
            void setDevice(c10::Device d) const;
            void uncheckedSetDevice(c10::Device d) const noexcept;
            c10::DeviceIndex deviceCount() const noexcept;
            c10::Stream getStream(c10::Device d) const noexcept;
            c10::Stream exchangeStream(c10::Stream stream) const noexcept;
        };
    }
}