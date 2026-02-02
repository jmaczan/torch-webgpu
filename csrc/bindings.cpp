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
#include "core/command_batcher.h"

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

// Python bindings for command batcher control
static PyObject* flush_commands(PyObject* self, PyObject* args)
{
    torch_webgpu::core::getCommandBatcher().flush();
    Py_RETURN_NONE;
}

static PyObject* disable_batching(PyObject* self, PyObject* args)
{
    torch_webgpu::core::getCommandBatcher().setEnabled(false);
    Py_RETURN_NONE;
}

static PyObject* enable_batching(PyObject* self, PyObject* args)
{
    torch_webgpu::core::getCommandBatcher().setEnabled(true);
    Py_RETURN_NONE;
}

PyMODINIT_FUNC PyInit__C(void)
{
    static std::vector<PyMethodDef> methods = {
        {"flush_commands", flush_commands, METH_NOARGS, "Flush pending WebGPU commands"},
        {"disable_batching", disable_batching, METH_NOARGS, "Disable command batching"},
        {"enable_batching", enable_batching, METH_NOARGS, "Enable command batching"},
        {nullptr, nullptr, 0, nullptr}  // Sentinel
    };
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
