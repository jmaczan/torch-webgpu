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
#include "core/dispatch_profiler.h"
#include "core/bind_group_cache.h"

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

// Python bindings for dispatch profiler
static PyObject* enable_profiling(PyObject* self, PyObject* args)
{
    torch_webgpu::core::getDispatchProfiler().setEnabled(true);
    Py_RETURN_NONE;
}

static PyObject* disable_profiling(PyObject* self, PyObject* args)
{
    torch_webgpu::core::getDispatchProfiler().setEnabled(false);
    Py_RETURN_NONE;
}

static PyObject* reset_profiler(PyObject* self, PyObject* args)
{
    torch_webgpu::core::getDispatchProfiler().reset();
    Py_RETURN_NONE;
}

static PyObject* get_profile_stats(PyObject* self, PyObject* args)
{
    auto& profiler = torch_webgpu::core::getDispatchProfiler();
    auto& bind_cache = torch_webgpu::core::getBindGroupCache();

    PyObject* dict = PyDict_New();
    if (!dict) return nullptr;

    // Dispatch counts
    PyDict_SetItemString(dict, "dispatch_count",
        PyLong_FromSize_t(profiler.getDispatchCount()));
    PyDict_SetItemString(dict, "submission_count",
        PyLong_FromSize_t(profiler.getSubmissionCount()));

    // Timings in microseconds
    PyDict_SetItemString(dict, "total_encoder_time_us",
        PyFloat_FromDouble(profiler.getTotalEncoderTimeUs()));
    PyDict_SetItemString(dict, "total_bind_group_time_us",
        PyFloat_FromDouble(profiler.getTotalBindGroupTimeUs()));
    PyDict_SetItemString(dict, "total_submission_time_us",
        PyFloat_FromDouble(profiler.getTotalSubmissionTimeUs()));
    PyDict_SetItemString(dict, "total_gpu_sync_time_us",
        PyFloat_FromDouble(profiler.getTotalGpuSyncTimeUs()));

    // Averages
    PyDict_SetItemString(dict, "avg_encoder_time_us",
        PyFloat_FromDouble(profiler.getAvgEncoderTimeUs()));
    PyDict_SetItemString(dict, "avg_bind_group_time_us",
        PyFloat_FromDouble(profiler.getAvgBindGroupTimeUs()));
    PyDict_SetItemString(dict, "avg_submission_time_us",
        PyFloat_FromDouble(profiler.getAvgSubmissionTimeUs()));
    PyDict_SetItemString(dict, "avg_dispatch_overhead_us",
        PyFloat_FromDouble(profiler.getAvgDispatchOverheadUs()));

    // Bind group cache stats
    PyDict_SetItemString(dict, "bind_group_cache_hits",
        PyLong_FromSize_t(bind_cache.getHits()));
    PyDict_SetItemString(dict, "bind_group_cache_misses",
        PyLong_FromSize_t(bind_cache.getMisses()));
    PyDict_SetItemString(dict, "bind_group_cache_hit_rate",
        PyFloat_FromDouble(bind_cache.getHitRate()));
    PyDict_SetItemString(dict, "bind_group_cache_size",
        PyLong_FromSize_t(bind_cache.getSize()));

    return dict;
}

static PyObject* get_profile_summary(PyObject* self, PyObject* args)
{
    std::string summary = torch_webgpu::core::getDispatchProfiler().getSummary();
    return PyUnicode_FromString(summary.c_str());
}

PyMODINIT_FUNC PyInit__C(void)
{
    static std::vector<PyMethodDef> methods = {
        {"flush_commands", flush_commands, METH_NOARGS, "Flush pending WebGPU commands"},
        {"disable_batching", disable_batching, METH_NOARGS, "Disable command batching"},
        {"enable_batching", enable_batching, METH_NOARGS, "Enable command batching"},
        {"enable_profiling", enable_profiling, METH_NOARGS, "Enable dispatch profiling"},
        {"disable_profiling", disable_profiling, METH_NOARGS, "Disable dispatch profiling"},
        {"reset_profiler", reset_profiler, METH_NOARGS, "Reset profiler counters"},
        {"get_profile_stats", get_profile_stats, METH_NOARGS, "Get profiling statistics as dict"},
        {"get_profile_summary", get_profile_summary, METH_NOARGS, "Get profiling summary as string"},
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
