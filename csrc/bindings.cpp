#include <ATen/ATen.h>
#include <ATen/RedispatchFunctions.h>
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
#include <ATen/native/CPUFallback.h>
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

    wgpu::Instance getInstance()
    {
        return instance;
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
        buffer_desc.usage = wgpu::BufferUsage::CopyDst | wgpu::BufferUsage::CopySrc | wgpu::BufferUsage::Storage;
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
    constexpr c10::DispatchKeySet privateuse1_ks(c10::DispatchKey::PrivateUse1);
    return at::detail::empty_generic(size, allocator, privateuse1_ks, dtype_or_default(dtype_opt), memory_format_opt);
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
    constexpr c10::DispatchKeySet privateuse1_ks(c10::DispatchKey::PrivateUse1);
    return at::detail::empty_strided_generic(size, stride, allocator, privateuse1_ks, dtype_or_default(dtype_opt));
}

struct BufferCopyContext
{
    bool ready;
    wgpu::Buffer buffer;
};

at::Tensor &cpu_copy_with_webgpu(
    at::Tensor &self, at::Tensor const &src, bool non_blocking = false)
{
    // TODO: take non_blocking into consideration
    if (src.device().is_privateuseone() && self.device().is_cpu())
    {
        auto src_size = src.numel() * at::elementSize(src.dtype().toScalarType()); // TODO: probably some size check here?
        auto self_size = self.numel() * at::elementSize(self.dtype().toScalarType());

        TORCH_CHECK(src.dtype() == self.dtype());
        TORCH_CHECK(src.numel() == self.numel());
        TORCH_CHECK(src.is_contiguous());
        TORCH_CHECK(self.is_contiguous());
        TORCH_CHECK(self_size == src_size);
        auto src_data = static_cast<WebGPUAllocation *>(src.data_ptr());
        auto self_data = self.data_ptr();
        TORCH_CHECK(src_data->buffer.GetSize() >= src_size);

        wgpu::BufferDescriptor buffer_desc;
        buffer_desc.label = "WebGPU temp buffer";
        buffer_desc.usage = wgpu::BufferUsage::CopyDst | wgpu::BufferUsage::MapRead;
        buffer_desc.size = src_size;
        buffer_desc.mappedAtCreation = false;

        WebGPUContext &ctx = getWebGPUContext();
        wgpu::Buffer tmp = ctx.getDevice().CreateBuffer(&buffer_desc);

        wgpu::CommandEncoder encoder = ctx.getDevice().CreateCommandEncoder();
        encoder.CopyBufferToBuffer(src_data->buffer, 0, tmp, 0, src_size);
        wgpu::CommandBuffer command = encoder.Finish();

        ctx.getQueue().Submit(1, &command);

        auto noop = [](wgpu::MapAsyncStatus, wgpu::StringView) {};

        wgpu::Future map_async_future = tmp.MapAsync(wgpu::MapMode::Read, 0, src_size, wgpu::CallbackMode::WaitAnyOnly, noop);

        ctx.getInstance().WaitAny(map_async_future, UINT64_MAX);

        const void *mapped = tmp.GetConstMappedRange(0, src_size);
        std::memcpy(self_data, mapped, src_size);
        tmp.Unmap();
        return self;
    }
    else
    {
        return at::native::copy_(self, src, non_blocking);
    }
}

at::Tensor &copy_(
    at::Tensor &self, at::Tensor const &src, bool non_blocking = false)
{
    // TODO: take non_blocking into consideration

    auto src_size = src.numel() * at::elementSize(src.dtype().toScalarType()); // TODO: probably some size check here?
    auto self_size = self.numel() * at::elementSize(self.dtype().toScalarType());

    if (src.device().is_cpu() && self.device().is_privateuseone())
    {
        TORCH_CHECK(src.dtype() == self.dtype());
        TORCH_CHECK(src.numel() == self.numel());
        TORCH_CHECK(src.is_contiguous());
        TORCH_CHECK(self.is_contiguous());
        TORCH_CHECK(self_size == src_size);
        auto src_data = src.data_ptr();
        auto self_data = static_cast<WebGPUAllocation *>(self.data_ptr());

        TORCH_CHECK(self_data->buffer.GetSize() >= src_size);

        getWebGPUContext().getQueue().WriteBuffer(self_data->buffer, 0, src_data, src_size);
        return self;
    }
    else if (src.device().is_privateuseone() && self.device().is_privateuseone())
    {
        TORCH_CHECK(src.dtype() == self.dtype());
        TORCH_CHECK(src.numel() == self.numel());
        TORCH_CHECK(src.is_contiguous());
        TORCH_CHECK(self.is_contiguous());
        TORCH_CHECK(self_size == src_size);
        auto src_data = static_cast<WebGPUAllocation *>(src.data_ptr());
        auto self_data = static_cast<WebGPUAllocation *>(self.data_ptr());

        TORCH_CHECK(src_data->buffer.GetSize() >= src_size);
        TORCH_CHECK(self_data->buffer.GetSize() >= src_size);

        wgpu::CommandEncoder encoder = getWebGPUContext().getDevice().CreateCommandEncoder();
        encoder.CopyBufferToBuffer(src_data->buffer, 0, self_data->buffer, 0, src_size);
        wgpu::CommandBuffer command = encoder.Finish();

        getWebGPUContext().getQueue().Submit(1, &command); // TODO: Submit is async, handle it correctly
        return self;
    }
    else
    {
        return at::native::copy_(self, src, non_blocking);
    }
}

at::Tensor _copy_from(at::Tensor const &self, at::Tensor const &dst, bool non_blocking = false)
{
    auto &dst_non_const = const_cast<at::Tensor &>(dst);

    c10::DispatchKeySet ks;

    if (dst.device().is_privateuseone())
    {
        ks = c10::DispatchKeySet(c10::DispatchKey::PrivateUse1);
    }
    else if (dst.device().is_cpu())
    {
        ks = c10::DispatchKeySet(c10::DispatchKey::CPU);
    }
    else
    {
        TORCH_CHECK(false, "Unsupported destination device in _copy_from");
    }

    at::redispatch::copy_(ks, dst_non_const, self, non_blocking);
    return dst;
}

at::Tensor add(at::Tensor const &self, at::Tensor const &other, at::Scalar const &alpha = 1)
{
    WebGPUContext &ctx = getWebGPUContext();
    constexpr const char *addWGSL = R"wgsl(
    struct Params {
        length: u32,
    };

    @group(0) @binding(0)
    var<storage, read> selfBuffer: array<f32>;

    @group(0) @binding(1)
    var<storage, read> otherBuffer: array<f32>;

    @group(0) @binding(2)
    var<storage, read_write> outBuffer: array<f32>;

    @group(0) @binding(3)
    var<uniform> params: Params;

    @compute @workgroup_size(64)
    fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
        let i = gid.x;
        if (i >= params.length) { return; }
        outBuffer[i] = selfBuffer[i] + otherBuffer[i];
    }
    )wgsl";

    wgpu::ShaderSourceWGSL shader_source{
        wgpu::ShaderSourceWGSL::Init{
            nullptr,
            wgpu::StringView{addWGSL, std::strlen(addWGSL)},
        }};

    wgpu::ShaderModuleDescriptor shader_descriptor{};
    shader_descriptor.nextInChain = &shader_source;
    shader_descriptor.label = "at::Tensor add shader";
    wgpu::ShaderModule shader_module = ctx.getDevice().CreateShaderModule(&shader_descriptor);

    wgpu::BindGroupLayoutEntry bindings[4]{};

    bindings[0].binding = 0;
    bindings[0].visibility = wgpu::ShaderStage::Compute;
    bindings[0].buffer.type = wgpu::BufferBindingType::ReadOnlyStorage;
    bindings[0].buffer.hasDynamicOffset = false;
    bindings[0].buffer.minBindingSize = 0;

    bindings[1].binding = 1;
    bindings[1].visibility = wgpu::ShaderStage::Compute;
    bindings[1].buffer.type = wgpu::BufferBindingType::ReadOnlyStorage;
    bindings[1].buffer.hasDynamicOffset = false;
    bindings[1].buffer.minBindingSize = 0;

    bindings[2].binding = 2;
    bindings[2].visibility = wgpu::ShaderStage::Compute;
    bindings[2].buffer.type = wgpu::BufferBindingType::Storage;
    bindings[2].buffer.hasDynamicOffset = false;
    bindings[2].buffer.minBindingSize = 0;

    bindings[3].binding = 3;
    bindings[3].visibility = wgpu::ShaderStage::Compute;
    bindings[3].buffer.type = wgpu::BufferBindingType::Uniform;
    bindings[3].buffer.hasDynamicOffset = false;
    bindings[3].buffer.minBindingSize = 0;

    wgpu::BindGroupLayoutDescriptor layout_descriptor{};
    layout_descriptor.entryCount = 4;
    layout_descriptor.entries = bindings;

    wgpu::BindGroupLayout bind_group_layout = ctx.getDevice().CreateBindGroupLayout(&layout_descriptor);

    wgpu::PipelineLayoutDescriptor pipeline_layout_descriptor{};
    pipeline_layout_descriptor.bindGroupLayoutCount = 1;
    pipeline_layout_descriptor.bindGroupLayouts = &bind_group_layout;

    wgpu::PipelineLayout pipeline_layout = ctx.getDevice().CreatePipelineLayout(&pipeline_layout_descriptor);

    wgpu::ComputePipelineDescriptor pipeline_descriptor{};
    pipeline_descriptor.layout = pipeline_layout;
    pipeline_descriptor.compute.module = shader_module;
    pipeline_descriptor.compute.entryPoint = wgpu::StringView{"main", 4};

    wgpu::ComputePipeline pipeline = ctx.getDevice().CreateComputePipeline(&pipeline_descriptor);

    // everything above should be cached
    TORCH_CHECK(self.dtype() == other.dtype());
    TORCH_CHECK(self.is_contiguous());
    TORCH_CHECK(other.is_contiguous());
    TORCH_CHECK(self.sizes().equals(other.sizes()));

    at::Tensor out = at::empty_like(self, self.options().device(self.device()));

    WebGPUAllocation *self_allocation = static_cast<WebGPUAllocation *>(self.data_ptr());
    WebGPUAllocation *other_allocation = static_cast<WebGPUAllocation *>(other.data_ptr());
    WebGPUAllocation *out_allocation = static_cast<WebGPUAllocation *>(out.data_ptr());

    wgpu::Buffer self_buffer = self_allocation->buffer;
    wgpu::Buffer other_buffer = other_allocation->buffer;
    wgpu::Buffer out_buffer = out_allocation->buffer;

    uint32_t numel = static_cast<uint32_t>(self.numel());
    size_t bytes = numel * at::elementSize(self.scalar_type());

    struct Params
    {
        uint32_t length;
    };
    Params params{numel};

    wgpu::BufferDescriptor uniform_descriptor{};
    uniform_descriptor.label = "Params";
    uniform_descriptor.usage = wgpu::BufferUsage::Uniform | wgpu::BufferUsage::CopyDst;
    uniform_descriptor.size = sizeof(Params);
    uniform_descriptor.mappedAtCreation = false;

    wgpu::Buffer params_buffer = ctx.getDevice().CreateBuffer(&uniform_descriptor);
    ctx.getQueue().WriteBuffer(params_buffer, 0, &params, sizeof(Params));

    wgpu::BindGroupEntry bind_group_entries[4]{};
    bind_group_entries[0].binding = 0;
    bind_group_entries[0].buffer = self_buffer;
    bind_group_entries[0].offset = 0;
    bind_group_entries[0].size = bytes;

    bind_group_entries[1].binding = 1;
    bind_group_entries[1].buffer = other_buffer;
    bind_group_entries[1].offset = 0;
    bind_group_entries[1].size = bytes;

    bind_group_entries[2].binding = 2;
    bind_group_entries[2].buffer = out_buffer;
    bind_group_entries[2].offset = 0;
    bind_group_entries[2].size = bytes;

    bind_group_entries[3].binding = 3;
    bind_group_entries[3].buffer = params_buffer;
    bind_group_entries[3].offset = 0;
    bind_group_entries[3].size = sizeof(Params);

    wgpu::BindGroupDescriptor bind_group_descriptor{};
    bind_group_descriptor.layout = bind_group_layout;
    bind_group_descriptor.entryCount = 4;
    bind_group_descriptor.entries = bind_group_entries;

    wgpu::BindGroup bind_group = ctx.getDevice().CreateBindGroup(&bind_group_descriptor);

    wgpu::CommandEncoder encoder = ctx.getDevice().CreateCommandEncoder();
    wgpu::ComputePassDescriptor pass_descriptor;
    wgpu::ComputePassEncoder pass_encoder = encoder.BeginComputePass(&pass_descriptor);
    pass_encoder.SetPipeline(pipeline);
    pass_encoder.SetBindGroup(0, bind_group);

    const uint32_t workgroup_size = 64;
    uint32_t num_workgroups = (numel + workgroup_size - 1) / workgroup_size;

    pass_encoder.DispatchWorkgroups(num_workgroups);
    pass_encoder.End();

    wgpu::CommandBuffer command_buffer = encoder.Finish();
    ctx.getQueue().Submit(1, &command_buffer);

    return out;
}

at::Tensor view(at::Tensor const &self, at::IntArrayRef size)
{
    return at::native::view(self, size);
}

static void webgpu_cpu_fallback_boxed(const c10::OperatorHandle &op, torch::jit::Stack *stack)
{
    at::native::cpu_fallback(op, stack);
}

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m)
{
    m.impl("empty.memory_format", TORCH_FN(empty_memory_format));
    m.impl("empty_strided", TORCH_FN(empty_strided));
    m.impl("copy_", TORCH_FN(copy_));
    m.impl("_copy_from", TORCH_FN(_copy_from));
    m.impl("add.Tensor", TORCH_FN(add));
    m.impl("view", TORCH_FN(view));

    m.impl("abs", torch::CppFunction::makeFromBoxedFunction<&webgpu_cpu_fallback_boxed>());
}

TORCH_LIBRARY_IMPL(aten, CPU, m)
{
    m.impl("copy_", TORCH_FN(cpu_copy_with_webgpu));
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