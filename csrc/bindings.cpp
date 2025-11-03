#include <ATen/ATen.h>
#include <torch/library.h>
#include <ATen/EmptyTensor.h>
#include <ATen/native/CPUFallback.h>
#include <ATen/native/TensorFactories.h>
#include <c10/core/CPUAllocator.h>
#include <Python.h>
#include <vector>
#include <cstdlib>

struct WebGPUAllocator
{
    void allocate(void **ptr, size_t size)
    {
        *ptr = std::malloc(size);
    }
};

static WebGPUAllocator &getWebGPUAllocator()
{
    static WebGPUAllocator webgpu_allocator;
    return webgpu_allocator;
}

static void WebGPUCachingHostDeleter(void *ptr)
{
    free(ptr);
}

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

at::Tensor &copy_(
    at::Tensor &self, at::Tensor const &src, bool non_blocking = false)
{
    auto size = self.numel() * at::elementSize(self.dtype().toScalarType());
    std::memcpy(self.data_ptr(), src.data_ptr(), size);
    return self;
}

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m)
{
    m.impl("empty.memory_format", TORCH_FN(empty_memory_format));
    m.impl("copy_", TORCH_FN(copy_));
}

// void custom_cpu_fallback(const c10::OperatorHandle &op, torch::jit::Stack *stack)
// {
//     at::native::cpu_fallback(op, stack);
// }

// TORCH_LIBRARY_IMPL(_, PrivateUse1, m)
// {
//     m.fallback(torch::CppFunction::makeFromBoxedFunction<&custom_cpu_fallback>());
// }

TORCH_LIBRARY_IMPL(_, PrivateUse1, m)
{
    m.fallback(torch::CppFunction::makeFallthrough());
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