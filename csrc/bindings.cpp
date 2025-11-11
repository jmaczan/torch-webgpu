#include <ATen/ATen.h>
#include <ATen/RedispatchFunctions.h>
#include <ATen/TensorIterator.h>
#include <ATen/native/DispatchStub.h>
#include <ATen/native/BinaryOps.h>
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
        TORCH_CHECK(src.dtype() == at::ScalarType::Float);
        TORCH_CHECK(src.numel() == self.numel());
        TORCH_CHECK(src.is_contiguous());
        TORCH_CHECK(self.is_contiguous());
        TORCH_CHECK(self_size == src_size);
        auto src_data = static_cast<WebGPUAllocation *>(src.storage().data_ptr().get());

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

struct UnaryKernel
{
    wgpu::BindGroupLayout bind_group_layout;
    wgpu::ComputePipeline pipeline;
};

enum class UnaryOp
{
    Copy,
    ReLU
};

struct CacheHash
{
    template <typename T>
    std::size_t operator()(T t) const noexcept
    {
        return static_cast<std::size_t>(t);
    }
};

const std::string unary_shader_template = R"wgsl(
const MAX_DIMS: u32 = 8u;

struct Params {
    length: u32,
    ndim: u32,
    _pad: u32,

    out_offset: u32,
    self_offset: u32,
    _pad2: u32,

    out_strides: array<u32, MAX_DIMS>,
    self_strides: array<u32, MAX_DIMS>,
    shape: array<u32, MAX_DIMS>,
};

@group(0) @binding(0)
var<storage, read> selfBuffer: array<f32>;

@group(0) @binding(1)
var<storage, read_write> outBuffer: array<f32>;

@group(0) @binding(2)
var<uniform> params: Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.length) { return; }

    var remaining = i;
    var coord: array<u32, MAX_DIMS>;

    for (var d: i32 = i32(params.ndim) - 1; d >= 0; d--) {
        let ud = u32(d);
        let s = params.shape[ud];
        coord[ud] = remaining % s;
        remaining = remaining / s;
    }

    var idx_out: u32 = 0u;
    var idx_self: u32 = 0u;

    for (var d: u32 = 0u; d < params.ndim; d++) {
        let c = coord[d];
        idx_out += c * params.out_strides[d];
        idx_self += c * params.self_strides[d];
    }

    idx_out += params.out_offset;
    idx_self += params.self_offset;

    outBuffer[idx_out] = __UNARY_OP__;
}
)wgsl";

inline void replace_string(std::string &src, const std::string &from, const std::string &to)
{
    size_t pos = 0;
    while ((pos = src.find(from, pos)) != std::string::npos)
    {
        src.replace(pos, from.size(), to);
        pos += to.size();
    }
}

std::string get_unary_shader(UnaryOp unary_op)
{
    std::string shader = unary_shader_template;
    std::string op_impl;
    switch (unary_op)
    {
    case UnaryOp::Copy:
        op_impl = "selfBuffer[idx_self]";
        break;
    case UnaryOp::ReLU:
        op_impl = "max(0.0, selfBuffer[idx_self])";
        break;
    default:
        TORCH_CHECK(false, "Unsupported unary op, can't produce a WGSL shader");
    }

    replace_string(shader, "__UNARY_OP__", op_impl);

    return shader;
}

UnaryKernel &get_unary_kernel(UnaryOp unary_op)
{
    static std::unordered_map<UnaryOp, UnaryKernel, CacheHash> kernel_cache;
    auto cached_kernel = kernel_cache.find(unary_op);
    if (cached_kernel != kernel_cache.end())
    {
        return cached_kernel->second;
    }

    std::string shader = get_unary_shader(unary_op);

    wgpu::ShaderSourceWGSL shader_source{
        wgpu::ShaderSourceWGSL::Init{
            nullptr,
            wgpu::StringView{shader.c_str(), shader.size()},
        }};

    wgpu::ShaderModuleDescriptor shader_descriptor{};
    shader_descriptor.nextInChain = &shader_source;
    shader_descriptor.label = "Unary kernel";
    WebGPUContext &ctx = getWebGPUContext();
    wgpu::ShaderModule shader_module = ctx.getDevice().CreateShaderModule(&shader_descriptor);

    wgpu::BindGroupLayoutEntry bindings[3]{};

    bindings[0].binding = 0;
    bindings[0].visibility = wgpu::ShaderStage::Compute;
    bindings[0].buffer.type = wgpu::BufferBindingType::ReadOnlyStorage;
    bindings[0].buffer.hasDynamicOffset = false;
    bindings[0].buffer.minBindingSize = 0;

    bindings[1].binding = 1;
    bindings[1].visibility = wgpu::ShaderStage::Compute;
    bindings[1].buffer.type = wgpu::BufferBindingType::Storage;
    bindings[1].buffer.hasDynamicOffset = false;
    bindings[1].buffer.minBindingSize = 0;

    bindings[2].binding = 2;
    bindings[2].visibility = wgpu::ShaderStage::Compute;
    bindings[2].buffer.type = wgpu::BufferBindingType::Uniform;
    bindings[2].buffer.hasDynamicOffset = false;
    bindings[2].buffer.minBindingSize = 0;

    wgpu::BindGroupLayoutDescriptor layout_descriptor{};
    layout_descriptor.entryCount = 3;
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
    auto [iter, inserted] = kernel_cache.emplace(unary_op, UnaryKernel{bind_group_layout, pipeline});
    TORCH_CHECK(inserted, "Failed to insert a kernel to the cache");
    return iter->second;
}

void copy_kernel_webgpu(at::TensorIteratorBase &iter)
{
    TORCH_CHECK(iter.ntensors() == 2);
    TORCH_CHECK(iter.common_dtype() == at::ScalarType::Float);
    TORCH_CHECK(iter.device_type() == c10::DeviceType::PrivateUse1);

    UnaryKernel &kernel = get_unary_kernel(UnaryOp::Copy);
    auto out = iter.tensor(0);
    auto self = iter.tensor(1);
    auto ndim = static_cast<uint32_t>(iter.ndim());
    auto shape = iter.shape();
    auto out_strides_bytes = iter.strides(0);
    auto self_strides_bytes = iter.strides(1);

    auto element_size = iter.element_size(0);

    std::vector<int64_t> out_strides(ndim);
    std::vector<int64_t> self_strides(ndim);

    for (int64_t i = 0; i < ndim; ++i)
    {
        int64_t out_bytes = out_strides_bytes[i];
        if (out_bytes == 0)
        {
            out_strides[i] = 0;
        }
        else
        {
            TORCH_CHECK(out_bytes % element_size == 0);
            out_strides[i] = out_bytes / element_size;
        }

        int64_t self_bytes = self_strides_bytes[i];
        if (self_bytes == 0)
        {
            self_strides[i] = 0;
        }
        else
        {
            TORCH_CHECK(self_bytes % element_size == 0);
            self_strides[i] = self_bytes / element_size;
        }
    }

    auto length = iter.numel();

    WebGPUAllocation *out_allocation = static_cast<WebGPUAllocation *>(out.storage().data_ptr().get());
    WebGPUAllocation *self_allocation = static_cast<WebGPUAllocation *>(self.storage().data_ptr().get());

    wgpu::Buffer self_buffer = self_allocation->buffer;
    wgpu::Buffer out_buffer = out_allocation->buffer;

    auto out_offset = out.storage_offset();
    auto self_offset = self.storage_offset();

    constexpr uint32_t MAX_DIMS = 8;
    TORCH_CHECK(ndim <= MAX_DIMS);

    struct Params
    {
        uint32_t length;
        uint32_t ndim;
        uint32_t _pad; // allegedly, it's a padding we need for webgpu

        uint32_t out_offset;
        uint32_t self_offset;
        uint32_t _pad2;

        uint32_t out_strides[MAX_DIMS];
        uint32_t self_strides[MAX_DIMS];
        uint32_t shape[MAX_DIMS];
    };

    Params params{};
    params.length = static_cast<uint32_t>(iter.numel());
    params.ndim = ndim;
    params._pad = 0;

    params.out_offset = static_cast<uint32_t>(out_offset);
    params.self_offset = static_cast<uint32_t>(self_offset);
    params._pad2 = 0;

    for (uint32_t d = 0; d < MAX_DIMS; ++d)
    {
        params.out_strides[d] = 0;
        params.self_strides[d] = 0;
        params.shape[d] = 1;
    }

    for (int64_t i = 0; i < ndim; ++i)
    {
        auto dim_size = shape[i];
        TORCH_CHECK(dim_size >= 0 && dim_size <= std::numeric_limits<uint32_t>::max());
        params.shape[i] = static_cast<uint32_t>(dim_size);

        auto out_stride = out_strides[i];
        auto self_stride = self_strides[i];

        TORCH_CHECK(out_stride >= 0 && out_stride <= std::numeric_limits<uint32_t>::max());
        TORCH_CHECK(self_stride >= 0 && self_stride <= std::numeric_limits<uint32_t>::max());

        params.out_strides[i] = static_cast<uint32_t>(out_stride);
        params.self_strides[i] = static_cast<uint32_t>(self_stride);
    }

    wgpu::BufferDescriptor uniform_descriptor{};
    uniform_descriptor.label = "Params";
    uniform_descriptor.usage = wgpu::BufferUsage::Uniform | wgpu::BufferUsage::CopyDst;
    uniform_descriptor.size = sizeof(Params);
    uniform_descriptor.mappedAtCreation = false;

    WebGPUContext &ctx = getWebGPUContext();
    wgpu::Buffer params_buffer = ctx.getDevice().CreateBuffer(&uniform_descriptor);
    ctx.getQueue().WriteBuffer(params_buffer, 0, &params, sizeof(Params));

    wgpu::BindGroupEntry bind_group_entries[3]{};
    bind_group_entries[0].binding = 0;
    bind_group_entries[0].buffer = self_buffer;
    bind_group_entries[0].offset = 0;
    bind_group_entries[0].size = self_buffer.GetSize();

    bind_group_entries[1].binding = 1;
    bind_group_entries[1].buffer = out_buffer;
    bind_group_entries[1].offset = 0;
    bind_group_entries[1].size = out_buffer.GetSize();

    bind_group_entries[2].binding = 2;
    bind_group_entries[2].buffer = params_buffer;
    bind_group_entries[2].offset = 0;
    bind_group_entries[2].size = sizeof(Params);

    wgpu::BindGroupDescriptor bind_group_descriptor{};
    bind_group_descriptor.layout = kernel.bind_group_layout;
    bind_group_descriptor.entryCount = 3;
    bind_group_descriptor.entries = bind_group_entries;

    wgpu::BindGroup bind_group = ctx.getDevice().CreateBindGroup(&bind_group_descriptor);

    wgpu::CommandEncoder encoder = ctx.getDevice().CreateCommandEncoder();
    wgpu::ComputePassDescriptor pass_descriptor;
    wgpu::ComputePassEncoder pass_encoder = encoder.BeginComputePass(&pass_descriptor);
    pass_encoder.SetPipeline(kernel.pipeline);
    pass_encoder.SetBindGroup(0, bind_group);

    const uint32_t workgroup_size = 64;
    uint32_t num_workgroups = (length + workgroup_size - 1) / workgroup_size;

    pass_encoder.DispatchWorkgroups(num_workgroups);
    pass_encoder.End();

    wgpu::CommandBuffer command_buffer = encoder.Finish();
    ctx.getQueue().Submit(1, &command_buffer);
}

at::Tensor &copy_(
    at::Tensor &self, at::Tensor const &src, bool non_blocking = false)
{
    // TODO: take non_blocking into consideration

    if (src.device().is_cpu() && self.device().is_privateuseone())
    {
        TORCH_CHECK(src.dtype() == self.dtype());
        TORCH_CHECK(src.numel() == self.numel());
        TORCH_CHECK(self.is_contiguous(), "WebGPU doesn't support copying from CPU to non-contiguous WebGPU tensor, yet");

        at::Tensor src_contiguous = src.is_contiguous() ? src : src.contiguous();
        uint64_t write_nbytes = static_cast<u_int64_t>(src_contiguous.numel()) * static_cast<uint64_t>(at::elementSize(src_contiguous.scalar_type()));

        auto self_data = static_cast<WebGPUAllocation *>(self.storage().data_ptr().get());
        auto self_storage_offset = self.storage_offset();
        TORCH_CHECK(self_storage_offset >= 0, "WebGPU doesn't support negative offset yet");
        uint64_t buffer_offset = static_cast<uint64_t>(self_storage_offset) * static_cast<uint64_t>(at::elementSize(self.scalar_type()));

        TORCH_CHECK(self_data->buffer.GetSize() >= buffer_offset + write_nbytes);

        getWebGPUContext().getQueue().WriteBuffer(self_data->buffer, buffer_offset, src_contiguous.data_ptr(), write_nbytes);
        return self;
    }
    else if (src.device().is_privateuseone() && self.device().is_privateuseone())
    {
        if (src.is_contiguous() and self.is_contiguous())
        {
            // TODO: handle a scenario when src and self share the storage and their memory ranges overlap

            TORCH_CHECK(src.dtype() == self.dtype());
            TORCH_CHECK(src.numel() == self.numel());

            auto src_data = static_cast<WebGPUAllocation *>(src.storage().data_ptr().get());
            auto src_storage_offset = src.storage_offset();
            TORCH_CHECK(src_storage_offset >= 0, "WebGPU doesn't support negative offset yet");
            uint64_t src_buffer_offset = static_cast<uint64_t>(src_storage_offset) * static_cast<uint64_t>(at::elementSize(src.scalar_type()));

            auto self_data = static_cast<WebGPUAllocation *>(self.storage().data_ptr().get());
            auto self_storage_offset = self.storage_offset();
            TORCH_CHECK(self_storage_offset >= 0, "WebGPU doesn't support negative offset yet");
            uint64_t self_buffer_offset = static_cast<uint64_t>(self_storage_offset) * static_cast<uint64_t>(at::elementSize(self.scalar_type()));

            uint64_t write_nbytes = static_cast<uint64_t>(src.numel()) * at::elementSize(src.dtype().toScalarType());
            TORCH_CHECK(src_data->buffer.GetSize() >= src_buffer_offset + write_nbytes);
            TORCH_CHECK(self_data->buffer.GetSize() >= self_buffer_offset + write_nbytes);

            wgpu::CommandEncoder encoder = getWebGPUContext().getDevice().CreateCommandEncoder();
            encoder.CopyBufferToBuffer(src_data->buffer, src_buffer_offset, self_data->buffer, self_buffer_offset, write_nbytes);
            wgpu::CommandBuffer command = encoder.Finish();

            getWebGPUContext().getQueue().Submit(1, &command); // TODO: Submit is async, handle it correctly
            return self;
        }
        else
        {
            at::TensorIteratorConfig config;
            config.set_check_mem_overlap(true);
            config.add_output(self);
            config.add_input(src);
            config.check_all_same_dtype(true);
            auto iter = config.build();

            copy_kernel_webgpu(iter);
            return self;
        }
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

at::Tensor view(at::Tensor const &self, at::IntArrayRef size)
{
    return at::native::view(self, size);
}

const at::Tensor &resize_(at::Tensor const &self, at::IntArrayRef size, c10::optional<c10::MemoryFormat> format)
{
    TORCH_CHECK(self.is_contiguous());
    auto result = at::native::resize_(self, size, format);
    return result;
}

at::Tensor _copy_from_and_resize(at::Tensor const &self, at::Tensor const &dst)
{
    TORCH_CHECK(self.is_contiguous());
    dst.resize_(self.sizes());
    return dst.copy_(self);
}

static void webgpu_cpu_fallback_boxed(const c10::OperatorHandle &op, torch::jit::Stack *stack)
{
    at::native::cpu_fallback(op, stack);
}
using at::native::add_stub;

namespace at
{
    namespace native
    {

        void add_kernel_webgpu(TensorIteratorBase &iter, const Scalar &alpha)
        {
            TORCH_CHECK(iter.ntensors() == 3);
            TORCH_CHECK(iter.common_dtype() == at::ScalarType::Float);
            TORCH_CHECK(iter.device_type() == c10::DeviceType::PrivateUse1);
            TORCH_CHECK(iter.dtype(0) == iter.dtype(1));
            TORCH_CHECK(iter.dtype(1) == iter.dtype(2));

            WebGPUContext &ctx = getWebGPUContext();
            constexpr const char *addWGSL = R"wgsl(
const MAX_DIMS: u32 = 8u;

struct Params {
    length: u32,
    ndim: u32,
    alpha: f32,
    _pad: u32,

    out_offset: u32,
    self_offset: u32,
    other_offset: u32,
    _pad2: u32,

    out_strides: array<u32, MAX_DIMS>,
    self_strides: array<u32, MAX_DIMS>,
    other_strides: array<u32, MAX_DIMS>,
    shape: array<u32, MAX_DIMS>,
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

    var remaining = i;
    var coord: array<u32, MAX_DIMS>;

    for (var d: i32 = i32(params.ndim) - 1; d >= 0; d--) {
        let ud = u32(d);
        let s = params.shape[ud];
        coord[ud] = remaining % s;
        remaining = remaining / s;
    }

    var idx_out: u32 = 0u;
    var idx_self: u32 = 0u;
    var idx_other: u32 = 0u;

    for (var d: u32 = 0u; d < params.ndim; d++) {
        let c = coord[d];
        idx_out += c * params.out_strides[d];
        idx_self += c * params.self_strides[d];
        idx_other += c * params.other_strides[d];
    }

    idx_out += params.out_offset;
    idx_self += params.self_offset;
    idx_other += params.other_offset;

    outBuffer[idx_out] = selfBuffer[idx_self] + params.alpha * otherBuffer[idx_other];
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
            // everything above should be cached, uniform (params) perhaps too?
            auto out = iter.tensor(0);
            auto self = iter.tensor(1);
            auto other = iter.tensor(2);
            auto ndim = static_cast<uint32_t>(iter.ndim());
            auto shape = iter.shape();
            auto out_strides_bytes = iter.strides(0);
            auto self_strides_bytes = iter.strides(1);
            auto other_strides_bytes = iter.strides(2);

            auto element_size = iter.element_size(0);

            std::vector<int64_t> out_strides(ndim);
            std::vector<int64_t> self_strides(ndim);
            std::vector<int64_t> other_strides(ndim);

            for (int64_t i = 0; i < ndim; ++i)
            {
                int64_t out_bytes = out_strides_bytes[i];
                if (out_bytes == 0)
                {
                    out_strides[i] = 0;
                }
                else
                {
                    TORCH_CHECK(out_bytes % element_size == 0);
                    out_strides[i] = out_bytes / element_size;
                }

                int64_t self_bytes = self_strides_bytes[i];
                if (self_bytes == 0)
                {
                    self_strides[i] = 0;
                }
                else
                {
                    TORCH_CHECK(self_bytes % element_size == 0);
                    self_strides[i] = self_bytes / element_size;
                }

                int64_t other_bytes = other_strides_bytes[i];
                if (other_bytes == 0)
                {
                    other_strides[i] = 0;
                }
                else
                {
                    TORCH_CHECK(other_bytes % element_size == 0);
                    other_strides[i] = other_bytes / element_size;
                }
            }

            auto length = iter.numel();

            WebGPUAllocation *out_allocation = static_cast<WebGPUAllocation *>(out.storage().data_ptr().get());
            WebGPUAllocation *self_allocation = static_cast<WebGPUAllocation *>(self.storage().data_ptr().get());
            WebGPUAllocation *other_allocation = static_cast<WebGPUAllocation *>(other.storage().data_ptr().get());

            wgpu::Buffer self_buffer = self_allocation->buffer;
            wgpu::Buffer other_buffer = other_allocation->buffer;
            wgpu::Buffer out_buffer = out_allocation->buffer;

            auto out_offset = out.storage_offset();
            auto self_offset = self.storage_offset();
            auto other_offset = other.storage_offset();

            constexpr uint32_t MAX_DIMS = 8;
            TORCH_CHECK(ndim <= MAX_DIMS);

            struct Params
            {
                uint32_t length;
                uint32_t ndim;
                float alpha;
                uint32_t _pad; // allegedly, it's a padding we need for webgpu

                uint32_t out_offset;
                uint32_t self_offset;
                uint32_t other_offset;
                uint32_t _pad2;

                uint32_t out_strides[MAX_DIMS];
                uint32_t self_strides[MAX_DIMS];
                uint32_t other_strides[MAX_DIMS];
                uint32_t shape[MAX_DIMS];
            };

            Params params{};
            params.length = static_cast<uint32_t>(iter.numel());
            params.ndim = ndim;
            params.alpha = alpha.to<float>();
            params._pad = 0;

            params.out_offset = static_cast<uint32_t>(out_offset);
            params.self_offset = static_cast<uint32_t>(self_offset);
            params.other_offset = static_cast<uint32_t>(other_offset);
            params._pad2 = 0;

            for (uint32_t d = 0; d < MAX_DIMS; ++d)
            {
                params.out_strides[d] = 0;
                params.self_strides[d] = 0;
                params.other_strides[d] = 0;
                params.shape[d] = 1;
            }

            for (int64_t i = 0; i < ndim; ++i)
            {
                auto dim_size = shape[i];
                TORCH_CHECK(dim_size >= 0 && dim_size <= std::numeric_limits<uint32_t>::max());
                params.shape[i] = static_cast<uint32_t>(dim_size);

                auto out_stride = out_strides[i];
                auto self_stride = self_strides[i];
                auto other_stride = other_strides[i];

                TORCH_CHECK(out_stride >= 0 && out_stride <= std::numeric_limits<uint32_t>::max());
                TORCH_CHECK(self_stride >= 0 && self_stride <= std::numeric_limits<uint32_t>::max());
                TORCH_CHECK(other_stride >= 0 && other_stride <= std::numeric_limits<uint32_t>::max());

                params.out_strides[i] = static_cast<uint32_t>(out_stride);
                params.self_strides[i] = static_cast<uint32_t>(self_stride);
                params.other_strides[i] = static_cast<uint32_t>(other_stride);
            }

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
            bind_group_entries[0].size = self_buffer.GetSize();

            bind_group_entries[1].binding = 1;
            bind_group_entries[1].buffer = other_buffer;
            bind_group_entries[1].offset = 0;
            bind_group_entries[1].size = other_buffer.GetSize();

            bind_group_entries[2].binding = 2;
            bind_group_entries[2].buffer = out_buffer;
            bind_group_entries[2].offset = 0;
            bind_group_entries[2].size = out_buffer.GetSize();

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
            uint32_t num_workgroups = (length + workgroup_size - 1) / workgroup_size;

            pass_encoder.DispatchWorkgroups(num_workgroups);
            pass_encoder.End();

            wgpu::CommandBuffer command_buffer = encoder.Finish();
            ctx.getQueue().Submit(1, &command_buffer);
        }

        REGISTER_PRIVATEUSE1_DISPATCH(add_stub, &add_kernel_webgpu);

    }
}

at::Tensor &add_out_webgpu(
    const at::Tensor &self,
    const at::Tensor &other,
    const at::Scalar &alpha,
    at::Tensor &out)
{
    at::TensorIteratorConfig config;
    config.set_check_mem_overlap(true);
    config.add_output(out);
    config.add_input(self);
    config.add_input(other);
    config.promote_inputs_to_common_dtype(true);
    config.cast_common_dtype_to_outputs(true);
    config.check_all_same_device(false);
    auto iter = config.build();

    at::native::add_kernel_webgpu(iter, alpha);

    return out;
}

void relu_kernel_webgpu(at::TensorIteratorBase &iter)
{
    TORCH_CHECK(iter.ntensors() == 2);
    TORCH_CHECK(iter.common_dtype() == at::ScalarType::Float);
    TORCH_CHECK(iter.device_type() == c10::DeviceType::PrivateUse1);
    UnaryKernel &kernel = get_unary_kernel(UnaryOp::ReLU);
    auto out = iter.tensor(0);
    auto self = iter.tensor(1);
    auto ndim = static_cast<uint32_t>(iter.ndim());
    auto shape = iter.shape();
    auto out_strides_bytes = iter.strides(0);
    auto self_strides_bytes = iter.strides(1);

    auto element_size = iter.element_size(0);

    std::vector<int64_t> out_strides(ndim);
    std::vector<int64_t> self_strides(ndim);

    for (int64_t i = 0; i < ndim; ++i)
    {
        int64_t out_bytes = out_strides_bytes[i];
        if (out_bytes == 0)
        {
            out_strides[i] = 0;
        }
        else
        {
            TORCH_CHECK(out_bytes % element_size == 0);
            out_strides[i] = out_bytes / element_size;
        }

        int64_t self_bytes = self_strides_bytes[i];
        if (self_bytes == 0)
        {
            self_strides[i] = 0;
        }
        else
        {
            TORCH_CHECK(self_bytes % element_size == 0);
            self_strides[i] = self_bytes / element_size;
        }
    }

    auto length = static_cast<uint32_t>(iter.numel());

    WebGPUAllocation *out_allocation = static_cast<WebGPUAllocation *>(out.storage().data_ptr().get());
    WebGPUAllocation *self_allocation = static_cast<WebGPUAllocation *>(self.storage().data_ptr().get());

    wgpu::Buffer self_buffer = self_allocation->buffer;
    wgpu::Buffer out_buffer = out_allocation->buffer;

    auto out_offset = out.storage_offset();
    auto self_offset = self.storage_offset();

    constexpr uint32_t MAX_DIMS = 8;
    TORCH_CHECK(ndim <= MAX_DIMS);

    struct Params
    {
        uint32_t length;
        uint32_t ndim;
        uint32_t _pad; // allegedly, it's a padding we need for webgpu

        uint32_t out_offset;
        uint32_t self_offset;
        uint32_t _pad2;

        uint32_t out_strides[MAX_DIMS];
        uint32_t self_strides[MAX_DIMS];
        uint32_t shape[MAX_DIMS];
    };

    Params params{};
    params.length = length;
    params.ndim = ndim;
    params._pad = 0;

    params.out_offset = static_cast<uint32_t>(out_offset);
    params.self_offset = static_cast<uint32_t>(self_offset);
    params._pad2 = 0;

    for (uint32_t d = 0; d < MAX_DIMS; ++d)
    {
        params.out_strides[d] = 0;
        params.self_strides[d] = 0;
        params.shape[d] = 1;
    }

    for (int64_t i = 0; i < ndim; ++i)
    {
        auto dim_size = shape[i];
        TORCH_CHECK(dim_size >= 0 && dim_size <= std::numeric_limits<uint32_t>::max());
        params.shape[i] = static_cast<uint32_t>(dim_size);

        auto out_stride = out_strides[i];
        auto self_stride = self_strides[i];

        TORCH_CHECK(out_stride >= 0 && out_stride <= std::numeric_limits<uint32_t>::max());
        TORCH_CHECK(self_stride >= 0 && self_stride <= std::numeric_limits<uint32_t>::max());

        params.out_strides[i] = static_cast<uint32_t>(out_stride);
        params.self_strides[i] = static_cast<uint32_t>(self_stride);
    }

    wgpu::BufferDescriptor uniform_descriptor{};
    uniform_descriptor.label = "Params";
    uniform_descriptor.usage = wgpu::BufferUsage::Uniform | wgpu::BufferUsage::CopyDst;
    uniform_descriptor.size = sizeof(Params);
    uniform_descriptor.mappedAtCreation = false;
    WebGPUContext &ctx = getWebGPUContext();
    wgpu::Buffer params_buffer = ctx.getDevice().CreateBuffer(&uniform_descriptor);
    ctx.getQueue().WriteBuffer(params_buffer, 0, &params, sizeof(Params));

    wgpu::BindGroupEntry bind_group_entries[3]{};
    bind_group_entries[0].binding = 0;
    bind_group_entries[0].buffer = self_buffer;
    bind_group_entries[0].offset = 0;
    bind_group_entries[0].size = self_buffer.GetSize();

    bind_group_entries[1].binding = 1;
    bind_group_entries[1].buffer = out_buffer;
    bind_group_entries[1].offset = 0;
    bind_group_entries[1].size = out_buffer.GetSize();

    bind_group_entries[2].binding = 2;
    bind_group_entries[2].buffer = params_buffer;
    bind_group_entries[2].offset = 0;
    bind_group_entries[2].size = sizeof(Params);

    wgpu::BindGroupDescriptor bind_group_descriptor{};
    bind_group_descriptor.layout = kernel.bind_group_layout;
    bind_group_descriptor.entryCount = 3;
    bind_group_descriptor.entries = bind_group_entries;

    wgpu::BindGroup bind_group = ctx.getDevice().CreateBindGroup(&bind_group_descriptor);

    wgpu::CommandEncoder encoder = ctx.getDevice().CreateCommandEncoder();
    wgpu::ComputePassDescriptor pass_descriptor;
    wgpu::ComputePassEncoder pass_encoder = encoder.BeginComputePass(&pass_descriptor);
    pass_encoder.SetPipeline(kernel.pipeline);
    pass_encoder.SetBindGroup(0, bind_group);

    const uint32_t workgroup_size = 64;
    uint32_t num_workgroups = (length + workgroup_size - 1) / workgroup_size;

    pass_encoder.DispatchWorkgroups(num_workgroups);
    pass_encoder.End();

    wgpu::CommandBuffer command_buffer = encoder.Finish();
    ctx.getQueue().Submit(1, &command_buffer);
}

at::Tensor relu(at::Tensor const &self)
{
    at::Tensor out = at::empty_like(self, self.options().device(at::DeviceType::PrivateUse1));
    at::TensorIteratorConfig config;
    config.add_output(out);
    config.add_input(self);
    config.check_all_same_dtype(true);
    auto iter = config.build();

    relu_kernel_webgpu(iter);

    return out;
}

at::Tensor &relu_out(
    at::Tensor const &self,
    at::Tensor &out)
{
    at::TensorIteratorConfig config;
    config.add_output(out);
    config.add_input(self);
    config.check_all_same_dtype(true);
    auto iter = config.build();

    relu_kernel_webgpu(iter);

    return out;
}

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m)
{
    m.impl("empty.memory_format", TORCH_FN(empty_memory_format));
    m.impl("empty_strided", TORCH_FN(empty_strided));
    m.impl("copy_", TORCH_FN(copy_));
    m.impl("_copy_from", TORCH_FN(_copy_from));
    m.impl("view", TORCH_FN(view));
    m.impl("resize_", TORCH_FN(resize_));
    m.impl("_copy_from_and_resize", TORCH_FN(_copy_from_and_resize));
    m.impl("add.out", TORCH_FN(add_out_webgpu));
    m.impl("relu", TORCH_FN(relu));
    m.impl("relu.out", TORCH_FN(relu_out));

    m.impl("abs", torch::CppFunction::makeFromBoxedFunction<&webgpu_cpu_fallback_boxed>());
    m.impl("ne.Scalar_out", torch::CppFunction::makeFromBoxedFunction<&webgpu_cpu_fallback_boxed>());
    m.impl("ne.Scalar", torch::CppFunction::makeFromBoxedFunction<&webgpu_cpu_fallback_boxed>());
    m.impl("eq.Tensor_out", torch::CppFunction::makeFromBoxedFunction<&webgpu_cpu_fallback_boxed>());
    m.impl("eq.Tensor", torch::CppFunction::makeFromBoxedFunction<&webgpu_cpu_fallback_boxed>());
    m.impl("mul.out", torch::CppFunction::makeFromBoxedFunction<&webgpu_cpu_fallback_boxed>());
    m.impl("bitwise_and.Tensor_out", torch::CppFunction::makeFromBoxedFunction<&webgpu_cpu_fallback_boxed>());
    m.impl("bitwise_and.Tensor", torch::CppFunction::makeFromBoxedFunction<&webgpu_cpu_fallback_boxed>());
    m.impl("masked_select", torch::CppFunction::makeFromBoxedFunction<&webgpu_cpu_fallback_boxed>());
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