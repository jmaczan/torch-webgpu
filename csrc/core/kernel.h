#pragma once
// Pipeline cache and kernel launch helpers.
//
// Shaders use the convention:
//   @group(0) @binding(0..N-1)  storage buffers (in the order passed to launch)
//   @group(0) @binding(N)       uniform parameters (optional)
// and the bind group layout is derived from the shader ("auto" layout), so
// every declared binding must be statically used by the entry point.

#include <webgpu/webgpu_cpp.h>

#include <cstdint>
#include <cstring>
#include <functional>
#include <string>
#include <vector>

namespace torch_webgpu::core
{
    struct Pipeline
    {
        wgpu::ComputePipeline pipeline;
        wgpu::BindGroupLayout layout;
    };

    // Returns the pipeline for `key`, building it from `make_source()` on a
    // miss. Compilation errors are raised as c10::Error with the WGSL source
    // attached when TORCH_WEBGPU_DEBUG=1.
    const Pipeline &get_pipeline(const std::string &key, const std::function<std::string()> &make_source,
                                 const char *label = "torch_webgpu kernel");

    // Same, with the source as key.
    const Pipeline &get_pipeline(const std::string &source, const char *label = "torch_webgpu kernel");

    size_t pipeline_cache_size();

    struct BufferBinding
    {
        wgpu::Buffer buffer;
        uint64_t offset = 0;
        uint64_t size = WGPU_WHOLE_SIZE;
    };

    struct Grid
    {
        uint32_t x = 1, y = 1, z = 1;
        // Number of invocations along the flattened x axis per y row, i.e.
        // linear_id = gid.x + gid.y * row_stride. Pass this to the shader.
        uint32_t row_stride = 0;
    };

    // 1-D launch over `n` invocations, spilling into the y dimension when the
    // workgroup count exceeds maxComputeWorkgroupsPerDimension.
    Grid grid_1d(uint64_t n, uint32_t workgroup_size);

    void launch(const Pipeline &p, const std::vector<BufferBinding> &buffers, const void *params,
                size_t params_size, const Grid &grid);

    // Builds a uniform-compatible parameter block of 32-bit words.
    class Params
    {
    public:
        Params &u32(uint32_t v)
        {
            words_.push_back(v);
            return *this;
        }
        Params &i32(int32_t v) { return u32(static_cast<uint32_t>(v)); }
        Params &f32(float v)
        {
            uint32_t w;
            static_assert(sizeof(w) == sizeof(v));
            std::memcpy(&w, &v, sizeof(w));
            return u32(w);
        }
        // Pads to a multiple of 4 words (one vec4).
        Params &align16()
        {
            while (words_.size() % 4)
                words_.push_back(0);
            return *this;
        }
        const void *data() const { return words_.data(); }
        size_t size() const { return words_.size() * 4; }
        size_t words() const { return words_.size(); }

    private:
        std::vector<uint32_t> words_;
    };
}
