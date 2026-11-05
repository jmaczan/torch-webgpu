#include "core/kernel.h"

#include "core/context.h"
#include "core/stream.h"

#include <c10/util/Exception.h>

#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <mutex>
#include <unordered_map>

namespace torch_webgpu::core
{
    namespace
    {
        std::unordered_map<std::string, Pipeline> &cache()
        {
            static auto *c = new std::unordered_map<std::string, Pipeline>();
            return *c;
        }

        // TORCH_WEBGPU_DUMP_SHADERS=<dir> writes every compiled shader there.
        void maybe_dump(const std::string &source, const char *label)
        {
            static const char *dir = std::getenv("TORCH_WEBGPU_DUMP_SHADERS");
            if (!dir || !*dir)
                return;
            static int counter = 0;
            std::string path = std::string(dir) + "/" + std::to_string(counter++) + "_" + label + ".wgsl";
            if (FILE *f = std::fopen(path.c_str(), "w"))
            {
                std::fwrite(source.data(), 1, source.size(), f);
                std::fclose(f);
            }
        }

        Pipeline build(const std::string &source, const char *label)
        {
            maybe_dump(source, label);
            auto &ctx = Context::get();
            const auto &device = ctx.device();

            wgpu::ShaderSourceWGSL wgsl{};
            wgsl.code = wgpu::StringView{source.data(), source.size()};
            wgpu::ShaderModuleDescriptor sm_desc{};
            sm_desc.nextInChain = &wgsl;
            sm_desc.label = label;

            device.PushErrorScope(wgpu::ErrorFilter::Validation);
            device.PushErrorScope(wgpu::ErrorFilter::Internal);
            wgpu::ShaderModule module = device.CreateShaderModule(&sm_desc);

            wgpu::ComputePipelineDescriptor p_desc{};
            p_desc.label = label;
            p_desc.compute.module = module;
            p_desc.compute.entryPoint = "main";
            wgpu::ComputePipeline pipeline = device.CreateComputePipeline(&p_desc);

            std::string error;
            auto collect = [&](wgpu::PopErrorScopeStatus status, wgpu::ErrorType type, wgpu::StringView msg)
            {
                if (status == wgpu::PopErrorScopeStatus::Success && type != wgpu::ErrorType::NoError && msg.data)
                    error += std::string(msg.data, msg.length == WGPU_STRLEN ? std::strlen(msg.data) : msg.length) + "\n";
            };
            ctx.wait(device.PopErrorScope(wgpu::CallbackMode::WaitAnyOnly, collect));
            ctx.wait(device.PopErrorScope(wgpu::CallbackMode::WaitAnyOnly, collect));

            if (!error.empty())
            {
                if (debug_enabled())
                    std::cerr << "[torch_webgpu] failed shader:\n"
                              << source << "\n";
                TORCH_CHECK(false, "torch_webgpu: failed to compile kernel '", label, "':\n", error,
                            debug_enabled() ? std::string("\n--- source ---\n") + source : std::string("(set TORCH_WEBGPU_DEBUG=1 to print the shader source)"));
            }
            return Pipeline{pipeline, pipeline.GetBindGroupLayout(0)};
        }
    }

    const Pipeline &get_pipeline(const std::string &key, const std::function<std::string()> &make_source,
                                 const char *label)
    {
        auto &ctx = Context::get();
        std::lock_guard<std::recursive_mutex> g(ctx.mutex());
        auto &c = cache();
        auto it = c.find(key);
        if (it != c.end())
            return it->second;
        Pipeline p = build(make_source(), label);
        return c.emplace(key, std::move(p)).first->second;
    }

    const Pipeline &get_pipeline(const std::string &source, const char *label)
    {
        return get_pipeline(source, [&]
                            { return source; }, label);
    }

    size_t pipeline_cache_size()
    {
        std::lock_guard<std::recursive_mutex> g(Context::get().mutex());
        return cache().size();
    }

    Grid grid_1d(uint64_t n, uint32_t workgroup_size)
    {
        Grid g;
        if (n == 0)
        {
            g.x = 0;
            return g;
        }
        const uint64_t max_dim = Context::get().limits().maxComputeWorkgroupsPerDimension;
        uint64_t groups = (n + workgroup_size - 1) / workgroup_size;
        if (groups <= max_dim)
        {
            g.x = static_cast<uint32_t>(groups);
            g.y = 1;
        }
        else
        {
            g.x = static_cast<uint32_t>(max_dim);
            g.y = static_cast<uint32_t>((groups + max_dim - 1) / max_dim);
            TORCH_CHECK(g.y <= max_dim, "torch_webgpu: launch of ", n, " invocations is too large");
        }
        g.row_stride = g.x * workgroup_size;
        return g;
    }

    void launch(const Pipeline &p, const std::vector<BufferBinding> &buffers, const void *params,
                size_t params_size, const Grid &grid)
    {
        if (grid.x == 0 || grid.y == 0 || grid.z == 0)
            return;
        auto &ctx = Context::get();
        std::lock_guard<std::recursive_mutex> g(ctx.mutex());
        auto &stream = Stream::get();

        std::vector<wgpu::BindGroupEntry> entries(buffers.size() + (params ? 1 : 0));
        for (size_t i = 0; i < buffers.size(); ++i)
        {
            entries[i].binding = static_cast<uint32_t>(i);
            entries[i].buffer = buffers[i].buffer;
            entries[i].offset = buffers[i].offset;
            entries[i].size = buffers[i].size;
        }
        if (params)
        {
            UniformSlice u = stream.alloc_uniform(params, params_size);
            auto &e = entries.back();
            e.binding = static_cast<uint32_t>(buffers.size());
            e.buffer = u.buffer;
            e.offset = u.offset;
            e.size = u.size;
        }
        wgpu::BindGroupDescriptor desc{};
        desc.layout = p.layout;
        desc.entryCount = entries.size();
        desc.entries = entries.data();
        wgpu::BindGroup bg = ctx.device().CreateBindGroup(&desc);
        stream.dispatch(p.pipeline, bg, grid.x, grid.y, grid.z);
    }
}
