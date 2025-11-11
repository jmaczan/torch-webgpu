#include <torch/library.h>
#include "webgpu_device_guard.h"

namespace torch_webgpu
{
    namespace core
    {

        c10::DeviceType WebGPUGuardImpl::type() const
        {
            return c10::DeviceType::PrivateUse1;
        }

        c10::Device WebGPUGuardImpl::exchangeDevice(c10::Device d) const
        {
            c10::Device old_device = getDevice();
            setDevice(d);
            return old_device;
        }

        c10::Device WebGPUGuardImpl::getDevice() const
        {
            return c10::Device(c10::DeviceType::PrivateUse1, current_webgpu_device);
        }

        void WebGPUGuardImpl::setDevice(c10::Device d) const
        {
            current_webgpu_device = d.index();
        }

        void WebGPUGuardImpl::uncheckedSetDevice(c10::Device d) const noexcept
        {
            setDevice(d);
        }

        c10::DeviceIndex WebGPUGuardImpl::deviceCount() const noexcept
        {
            return webgpu_device_count;
        }

        c10::Stream WebGPUGuardImpl::getStream(c10::Device d) const noexcept
        {
            return c10::Stream(c10::Stream::DEFAULT, getDevice());
        }

        c10::Stream WebGPUGuardImpl::exchangeStream(c10::Stream stream) const noexcept
        {
            return getStream(getDevice());
        }
    };
}
