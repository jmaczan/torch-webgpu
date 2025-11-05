#include <ATen/ATen.h>
#include <torch/library.h>
#include <ATen/EmptyTensor.h>
#include <ATen/native/CPUFallback.h>
#include <ATen/native/TensorFactories.h>
#include <c10/core/CPUAllocator.h>
#include <Python.h>
#include <vector>
#include <cstdlib>
#include <webgpu/webgpu.h>
#include <webgpu/webgpu_cpp.h>
#include <iostream>

struct WebGPUContext
{
    wgpu::Instance instance;
    wgpu::Adapter adapter;
    wgpu::Device device;
    wgpu::Queue queue;

    WebGPUContext()
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
                                                     { std::cout << "Error in device descriptor" << "\n"; });

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

    wgpu::Device getDevice()
    {
        return device;
    }

    wgpu::Queue getQueue()
    {
        return queue;
    }
};

static WebGPUContext &getWebGPUContext()
{
    static WebGPUContext webgpu_context;
    return webgpu_context;
}

struct WebGPUAllocation
{
    wgpu::Buffer buffer;
    explicit WebGPUAllocation(wgpu::Buffer &&b) : buffer(std::move(b)) {}
};

struct WebGPUAllocator
{
    void allocate(void **ptr, size_t size)
    {
        wgpu::BufferDescriptor buffer_desc;
        buffer_desc.label = "WebGPU buffer";
        buffer_desc.usage = wgpu::BufferUsage::CopyDst | wgpu::BufferUsage::CopySrc;
        buffer_desc.size = size;
        buffer_desc.mappedAtCreation = false;
        *ptr = new WebGPUAllocation(getWebGPUContext().getDevice().CreateBuffer(&buffer_desc));
    }
};

static WebGPUAllocator &getWebGPUAllocator()
{
    static WebGPUAllocator webgpu_allocator;
    return webgpu_allocator;
}

static void WebGPUCachingHostDeleter(void *ptr)
{
    delete static_cast<WebGPUAllocation *>(ptr);
}

static thread_local int current_webgpu_device = 0;
static int webgpu_device_count = 1;
static int webgpu_stream_count = 1;

struct WebGPUGuardImpl final : public c10::impl::DeviceGuardImplInterface
{
    static constexpr c10::DeviceType static_type = c10::DeviceType::PrivateUse1;

    WebGPUGuardImpl() {}

    c10::DeviceType type() const
    {
        return c10::DeviceType::PrivateUse1;
    }

    c10::Device exchangeDevice(c10::Device d) const
    {
        c10::Device old_device = getDevice();
        setDevice(d);
        return old_device;
    }

    c10::Device getDevice() const
    {
        return c10::Device(c10::DeviceType::PrivateUse1, current_webgpu_device);
    }

    void setDevice(c10::Device d) const
    {
        current_webgpu_device = d.index();
    }

    void uncheckedSetDevice(c10::Device d) const noexcept
    {
        setDevice(d);
    }

    c10::DeviceIndex deviceCount() const noexcept
    {
        return webgpu_device_count;
    }

    c10::Stream getStream(c10::Device d) const noexcept
    {
        return c10::Stream(c10::Stream::DEFAULT, getDevice());
    }

    c10::Stream exchangeStream(c10::Stream stream) const noexcept
    {
        return getStream(getDevice());
    }
};

C10_REGISTER_GUARD_IMPL(PrivateUse1, WebGPUGuardImpl);

// not really caching, yet
struct WebGPUCachingAllocator final : public c10::Allocator
{
    at::DataPtr allocate(size_t size) override
    {
        void *ptr = nullptr;
        if (size > 0)
        {
            getWebGPUAllocator().allocate(&ptr, size);
        }
        return {ptr, ptr, &WebGPUCachingHostDeleter, at::DeviceType::PrivateUse1};
    }

    at::DeleterFnPtr raw_deleter() const override
    {
        return &WebGPUCachingHostDeleter;
    }

    void copy_data(void *dest, const void *src, std::size_t count) const
    {
        TORCH_CHECK_NOT_IMPLEMENTED(false, "copy_data not implemented in WebGPUCachingAllocator");
    }
};

at::Allocator *getWebGPUCachingAllocator()
{
    static WebGPUCachingAllocator webgpu_caching_allocator;
    return &webgpu_caching_allocator;
}

at::Tensor empty_memory_format(
    c10::IntArrayRef size,
    c10::optional<at::ScalarType> dtype_opt,
    c10::optional<at::Layout> layout_opt,
    c10::optional<at::Device> device_opt,
    c10::optional<bool> pin_memory_opt,
    c10::optional<c10::MemoryFormat> memory_format_opt)
{
    auto allocator = getWebGPUCachingAllocator();
    constexpr c10::DispatchKeySet cpu_ks(c10::DispatchKey::PrivateUse1);
    return at::detail::empty_generic(size, allocator, cpu_ks, dtype_or_default(dtype_opt), memory_format_opt);
}

at::Tensor empty_strided(
    c10::IntArrayRef size,
    c10::IntArrayRef stride,
    c10::optional<at::ScalarType> dtype_opt,
    c10::optional<at::Layout> layout_opt,
    c10::optional<at::Device> device_opt,
    c10::optional<bool> pin_memory_opt)
{
    auto allocator = getWebGPUCachingAllocator();
    constexpr c10::DispatchKeySet cpu_ks(c10::DispatchKey::PrivateUse1);
    return at::detail::empty_strided_generic(size, stride, allocator, cpu_ks, dtype_or_default(dtype_opt));
}

at::Tensor &copy_(
    at::Tensor &self, at::Tensor const &src, bool non_blocking = false)
{
    // TODO: take non_blocking into consideration

    TORCH_CHECK(src.is_contiguous());
    TORCH_CHECK(self.is_contiguous());

    auto src_size = src.numel() * at::elementSize(src.dtype().toScalarType()); // TODO: probably some size check here?
    auto self_size = self.numel() * at::elementSize(self.dtype().toScalarType());
    TORCH_CHECK(self_size == src_size);

    if (src.device().is_cpu() && self.device().is_privateuseone())
    {
        TORCH_CHECK(src.dtype() == self.dtype());
        TORCH_CHECK(src.numel() == self.numel());
        auto src_data = src.data_ptr();
        auto self_data = static_cast<WebGPUAllocation *>(self.data_ptr());

        TORCH_CHECK(self_data->buffer.GetSize() >= src_size);

        getWebGPUContext().getQueue().WriteBuffer(self_data->buffer, 0, src_data, src_size);
    }
    else if (src.device().is_privateuseone() && self.device().is_privateuseone())
    {
        TORCH_CHECK(src.dtype() == self.dtype());
        TORCH_CHECK(src.numel() == self.numel());

        auto src_data = static_cast<WebGPUAllocation *>(src.data_ptr());
        auto self_data = static_cast<WebGPUAllocation *>(self.data_ptr());

        TORCH_CHECK(src_data->buffer.GetSize() >= src_size);
        TORCH_CHECK(self_data->buffer.GetSize() >= src_size);

        wgpu::CommandEncoder encoder = getWebGPUContext().getDevice().CreateCommandEncoder();
        encoder.CopyBufferToBuffer(src_data->buffer, 0, self_data->buffer, 0, src_size);
        wgpu::CommandBuffer command = encoder.Finish();

        getWebGPUContext().getQueue().Submit(1, &command); // TODO: Submit is async, handle it correctly
    }
    else
    {
        TORCH_CHECK(false, "copy_ to WebGPU is not supported for this source device");
    }

    return self;
}

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m)
{
    m.impl("empty.memory_format", TORCH_FN(empty_memory_format));
    m.impl("empty_strided", TORCH_FN(empty_strided));
    m.impl("copy_", TORCH_FN(copy_));
}

PyMODINIT_FUNC PyInit__C(void)
{
    static std::vector<PyMethodDef> methods;
    static const int python_api_version = 1013;
    static struct PyModuleDef module_def = {
        PyModuleDef_HEAD_INIT,
        "torch_webgpu._C",
        nullptr,
        -1,
        methods.data()};
    PyAPI_FUNC(PyObject *) module = PyModule_Create2(&module_def, python_api_version);
    return module;
}