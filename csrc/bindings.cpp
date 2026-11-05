// Python bindings (torch_webgpu._C).

#include <torch/extension.h>

#include "core/allocator.h"
#include "core/context.h"
#include "core/generator.h"
#include "core/kernel.h"
#include "core/stream.h"

namespace torch_webgpu::ops
{
    std::vector<std::string> fallback_ops_seen();
}

namespace py = pybind11;
using namespace torch_webgpu;

PYBIND11_MODULE(_C, m)
{
    m.doc() = "torch_webgpu native extension";

    m.def("_is_available", &core::is_available);
    m.def("_device_count", []
          { return core::is_available() ? 1 : 0; });
    m.def("_synchronize", []
          { core::Stream::get().synchronize(); });
    m.def("_flush", []
          { core::Stream::get().flush(); });

    m.def("_device_info", []
          {
              auto &ctx = core::Context::get();
              const auto &i = ctx.info();
              const auto &l = ctx.limits();
              py::dict d;
              d["name"] = i.description.empty() ? i.device : i.description;
              d["vendor"] = i.vendor;
              d["architecture"] = i.architecture;
              d["device"] = i.device;
              d["description"] = i.description;
              d["backend"] = i.backend;
              d["adapter_type"] = i.adapter_type;
              d["shader_f16"] = ctx.has_f16();
              d["subgroups"] = ctx.has_subgroups();
              d["timestamp_query"] = ctx.has_timestamp_query();
              d["max_buffer_size"] = l.maxBufferSize;
              d["max_storage_buffer_binding_size"] = l.maxStorageBufferBindingSize;
              d["max_storage_buffers_per_shader_stage"] = l.maxStorageBuffersPerShaderStage;
              d["max_compute_workgroup_storage_size"] = l.maxComputeWorkgroupStorageSize;
              d["max_compute_invocations_per_workgroup"] = l.maxComputeInvocationsPerWorkgroup;
              d["max_compute_workgroup_size_x"] = l.maxComputeWorkgroupSizeX;
              d["max_compute_workgroups_per_dimension"] = l.maxComputeWorkgroupsPerDimension;
              return d; });

    m.def("_memory_stats", []
          {
              auto s = core::memory_stats();
              py::dict d;
              d["allocated_bytes"] = s.allocated_bytes;
              d["reserved_bytes"] = s.reserved_bytes;
              d["peak_allocated_bytes"] = s.peak_allocated_bytes;
              d["num_allocs"] = s.num_allocs;
              d["num_cache_hits"] = s.num_cache_hits;
              d["num_ooms"] = s.num_ooms;
              return d; });
    m.def("_reset_peak_memory_stats", &core::reset_peak_memory_stats);
    m.def("_empty_cache", &core::empty_cache);

    m.def("_runtime_stats", []
          {
              auto &s = core::Stream::get();
              py::dict d;
              d["dispatches"] = s.total_dispatches();
              d["submits"] = s.total_submits();
              d["pipelines"] = core::pipeline_cache_size();
              return d; });
    m.def("_fallback_ops", &ops::fallback_ops_seen);

    m.def("_default_generator", []
          { return core::default_generator(); });
}
