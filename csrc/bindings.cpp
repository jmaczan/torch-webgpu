// Python bindings (torch_webgpu._C).

#include <torch/extension.h>

#include "core/allocator.h"
#include "core/context.h"
#include "core/generator.h"
#include "core/kernel.h"
#include "core/stream.h"
#include "core/binding_set.h"
#include "kernels/math.h"
#include "kernels/special.h"
#include "kernels/wgsl.h"

namespace torch_webgpu::ops
{
    std::vector<std::string> fallback_ops_seen();
}

namespace
{
    // Pipelines compiled through _wgsl_compile, indexed by id.
    const std::vector<const torch_webgpu::core::Pipeline *> *&compiled_pipelines()
    {
        static const std::vector<const torch_webgpu::core::Pipeline *> *p = nullptr;
        return p;
    }
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

    // Runs a WGSL compute shader. Tensors are bound as storage buffers at
    // @binding(0..n-1) (whole buffers; use storage_offset in the shader),
    // optional uniform words at @binding(n). Entry point must be `main`.
    m.def(
        "_run_wgsl",
        [](const std::string &source, const std::vector<at::Tensor> &tensors, std::vector<uint32_t> uniforms,
           std::tuple<uint32_t, uint32_t, uint32_t> grid)
        {
            const auto &p = core::get_pipeline(source, "user_wgsl");
            std::vector<core::BufferBinding> bufs;
            for (auto &t : tensors)
            {
                TORCH_CHECK(t.device().type() == c10::DeviceType::PrivateUse1, "_run_wgsl: expected webgpu tensors");
                bufs.push_back({core::buffer_of(t)});
            }
            core::Grid g;
            g.x = std::get<0>(grid);
            g.y = std::get<1>(grid);
            g.z = std::get<2>(grid);
            core::launch(p, bufs, uniforms.empty() ? nullptr : uniforms.data(), uniforms.size() * 4, g);
        },
        py::arg("source"), py::arg("tensors"), py::arg("uniforms") = std::vector<uint32_t>{},
        py::arg("grid") = std::make_tuple(1u, 1u, 1u));

    // ---- Compiled-kernel API (used by the Inductor WGSL backend) ----------
    // _wgsl_compile(source, label) -> id. For a launch with n tensors (bound
    // as storage buffers 0..k-1, see _wgsl_binding_map), an optional error
    // flag (binding k) and p parameters, kernels declare the uniform block
    // as the last binding:
    //   var<uniform> P: struct { offs: array<vec4<u32>, ceil(n/4)>,
    //                            v: array<vec4<u32>, ceil(p/4)> }
    // offs[i] is tensor i's element offset inside its binding's window.
    m.def(
        "_wgsl_compile",
        [](const std::string &source, const std::string &label) -> int64_t
        {
            static std::vector<const core::Pipeline *> pipelines;
            static std::unordered_map<std::string, int64_t> ids;
            auto it = ids.find(source);
            if (it != ids.end())
                return it->second;
            const auto &p = core::get_pipeline(source, label.c_str());
            pipelines.push_back(&p);
            const int64_t id = static_cast<int64_t>(pipelines.size() - 1);
            ids.emplace(source, id);
            compiled_pipelines() = &pipelines;
            return id;
        },
        py::arg("source"), py::arg("label") = "inductor_wgsl");
    // Binding of each tensor: tensors in the same device buffer share one
    // binding (WebGPU forbids binding a buffer twice when one use writes);
    // empty tensors get dummy bindings after the real ones.
    struct WgslBindings
    {
        core::BindingSet bs;
        std::vector<int> regs;
        std::vector<int> map;
        std::vector<core::BufferBinding> buffers;
    };
    auto wgsl_bindings = [](const std::vector<at::Tensor> &tensors, WgslBindings &b)
    {
        const size_t n = tensors.size();
        b.regs.assign(n, -1);
        b.map.assign(n, -1);
        for (size_t i = 0; i < n; ++i)
        {
            const auto &t = tensors[i];
            TORCH_CHECK(t.device().type() == c10::DeviceType::PrivateUse1, "_wgsl_launch: expected webgpu tensors");
            if (t.numel() > 0)
                b.regs[i] = b.bs.add(t);
        }
        b.bs.finalize();
        b.buffers = b.bs.buffers();
        for (size_t i = 0; i < n; ++i)
        {
            if (b.regs[i] >= 0)
                b.map[i] = b.bs.binding(b.regs[i]);
            else
            {
                b.map[i] = static_cast<int>(b.buffers.size());
                b.buffers.push_back({core::dummy_buffer()});
            }
        }
    };
    m.def(
        "_wgsl_binding_map",
        [wgsl_bindings](const std::vector<at::Tensor> &tensors)
        {
            WgslBindings b;
            wgsl_bindings(tensors, b);
            return b.map;
        },
        py::arg("tensors"));
    // Launches a compiled kernel. Unless `merged`, the kernel was generated
    // for one binding per tensor; if tensors share buffers the launch is
    // skipped and false returned (the caller then compiles a variant for
    // _wgsl_binding_map(tensors) and launches it with merged=True).
    m.def(
        "_wgsl_launch",
        [wgsl_bindings](int64_t id, const std::vector<at::Tensor> &tensors, const std::vector<int64_t> &params,
                        std::tuple<uint32_t, uint32_t, uint32_t> grid, bool error_flag, bool merged)
        {
            auto *pl = compiled_pipelines();
            TORCH_CHECK(pl && id >= 0 && id < static_cast<int64_t>(pl->size()), "_wgsl_launch: unknown kernel id ", id);
            const size_t n = tensors.size();
            WgslBindings b;
            wgsl_bindings(tensors, b);
            if (!merged)
                for (size_t i = 0; i < n; ++i)
                    if (b.map[i] != static_cast<int>(i))
                        return false;
            std::vector<core::BufferBinding> bufs = b.buffers;
            std::vector<uint32_t> words((n + 3) / 4 * 4 + (params.size() + 3) / 4 * 4, 0u);
            for (size_t i = 0; i < n; ++i)
                words[i] = b.regs[i] >= 0 ? b.bs.offset(b.regs[i]) : 0u;
            // Optional device error flag (array<atomic<u32>>) after the tensor
            // bindings; the uniform block comes last. Bit 1 reports an
            // out-of-bounds index (raised as IndexError at the next sync).
            if (error_flag)
                bufs.push_back({core::Stream::get().index_error_flag()});
            const size_t pbase = (n + 3) / 4 * 4;
            for (size_t j = 0; j < params.size(); ++j)
            {
                TORCH_CHECK(params[j] >= INT32_MIN && params[j] <= UINT32_MAX, "_wgsl_launch: parameter out of 32-bit range");
                words[pbase + j] = static_cast<uint32_t>(params[j]);
            }
            core::Grid g;
            g.x = std::get<0>(grid);
            g.y = std::get<1>(grid);
            g.z = std::get<2>(grid);
            if (words.empty())
                words.push_back(0u);
            core::launch(*(*pl)[id], bufs, words.data(), words.size() * 4, g);
            return true;
        },
        py::arg("id"), py::arg("tensors"), py::arg("params") = std::vector<int64_t>{},
        py::arg("grid") = std::make_tuple(1u, 1u, 1u), py::arg("error_flag") = false, py::arg("merged") = false);
    // WGSL building blocks for code generators.
    m.def("_wgsl_prelude", [] { return wgsl::prelude(); });
    m.def("_wgsl_math", [] { return kernels::math_wgsl(); });
    m.def("_wgsl_special", [](const std::vector<std::string> &names) { return kernels::special_wgsl(names); });
    m.def("_wgsl_load_fn", [](const std::string &name, const std::string &var, at::ScalarType t, bool atomic)
          { return wgsl::load_fn(name, var, t, atomic); });
    m.def("_wgsl_store_fn", [](const std::string &name, const std::string &var, at::ScalarType t)
          { return wgsl::store_fn(name, var, t); });
    m.def("_wgsl_is_subword", [](at::ScalarType t) { return wgsl::is_subword(t); });

    m.def("_default_generator", []
          { return core::default_generator(); });
}
