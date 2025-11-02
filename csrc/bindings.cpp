#include <ATen/ATen.h>
#include <torch/library.h>
#include <ATen/EmptyTensor.h>
#include <ATen/native/TensorFactories.h>
#include <c10/core/CPUAllocator.h>
#include <Python.h>
#include <vector>

static void WebGPUCachingHostDeleter(void *ptr);

struct WebGPUCachingAllocator final : public at::Allocator
{
    at::DataPtr allocate(size_t size) override
    {
        void *ptr = nullptr;
        return {ptr, ptr, &WebGPUCachingHostDeleter, at::DeviceType::PrivateUse1};
    }

    at::DeleterFnPtr raw_deleter() const override
    {
        return &WebGPUCachingHostDeleter;
    }
}

// static WebGPUCachingAllocator webgpu_caching_allocator;
// at::Allocator *getWebGPUCachingAllocator()
// {
//     return &webgpu_caching_allocator;
// }

at::Tensor empty_memory_format(
    c10::IntArrayRef size,
    c10::optional<at::ScalarType> dtype_opt,
    c10::optional<at::Layout> layout_opt,
    c10::optional<at::Device> device_opt,
    c10::optional<bool> pin_memory_opt,
    c10::optional<c10::MemoryFormat> memory_format_opt)
{
    auto allocator = WebGPUCachingAllocator();
    constexpr c10::DispatchKeySet cpu_ks(c10::DispatchKey::PrivateUse1);
    return at::detail::empty_generic(size, allocator, cpu_ks, dtype_or_default(dtype_opt), memory_format_opt);
}

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m)
{
    m.impl("empty.memory_format", TORCH_FN(empty_memory_format));
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