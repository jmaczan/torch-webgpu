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
#include "core/webgpu_context.h"
#include "core/webgpu_allocator.h"
#include "core/webgpu_device_guard.h"

namespace torch_webgpu
{

    C10_REGISTER_GUARD_IMPL(PrivateUse1, core::WebGPUGuardImpl);

    static void webgpu_cpu_fallback_boxed(const c10::OperatorHandle &op, torch::jit::Stack *stack)
    {
        at::native::cpu_fallback(op, stack);
    }

    TORCH_LIBRARY_IMPL(aten, PrivateUse1, m)
    {
        // cpu fallbacks for ops not natively implemented
        m.impl("masked_select", torch::CppFunction::makeFromBoxedFunction<&webgpu_cpu_fallback_boxed>());
        // Required for tensor printing/formatting
        m.impl("ceil", torch::CppFunction::makeFromBoxedFunction<&webgpu_cpu_fallback_boxed>());
        m.impl("ceil.out", torch::CppFunction::makeFromBoxedFunction<&webgpu_cpu_fallback_boxed>());
        m.impl("floor", torch::CppFunction::makeFromBoxedFunction<&webgpu_cpu_fallback_boxed>());
        m.impl("floor.out", torch::CppFunction::makeFromBoxedFunction<&webgpu_cpu_fallback_boxed>());
        m.impl("isfinite", torch::CppFunction::makeFromBoxedFunction<&webgpu_cpu_fallback_boxed>());
        m.impl("isinf", torch::CppFunction::makeFromBoxedFunction<&webgpu_cpu_fallback_boxed>());
        m.impl("isnan", torch::CppFunction::makeFromBoxedFunction<&webgpu_cpu_fallback_boxed>());
    }
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
    PyObject* module = PyModule_Create2(&module_def, python_api_version);
    return module;
}
