#pragma once
#include <webgpu/webgpu_cpp.h>
#include <unordered_map>
#include <vector>
#include <cstdint>
#include <functional>

namespace torch_webgpu
{
    namespace core
    {
        /**
         * BindGroupCache caches WebGPU bind groups to avoid recreation overhead.
         *
         * Bind groups are keyed by:
         * - Pipeline pointer (identifies the shader/layout)
         * - Buffer pointers (the actual GPU buffers bound)
         * - Buffer sizes (for proper binding)
         *
         * Cache hit: ~0ms (just return cached bind group)
         * Cache miss: ~0.15-0.2ms (create new bind group)
         *
         * Usage:
         *   BindGroupKey key = makeBindGroupKey(pipeline, {buf1, buf2, buf3}, {size1, size2, size3});
         *   auto cached = getBindGroupCache().get(key);
         *   if (cached.has_value()) {
         *       bind_group = cached.value();
         *   } else {
         *       bind_group = device.CreateBindGroup(...);
         *       getBindGroupCache().put(key, bind_group);
         *   }
         */

        // Key for bind group cache
        struct BindGroupKey
        {
            // Pipeline pointer (identifies the shader)
            uint64_t pipeline_id;

            // Hash of buffer pointers and sizes
            size_t buffers_hash;

            bool operator==(const BindGroupKey &other) const
            {
                return pipeline_id == other.pipeline_id && buffers_hash == other.buffers_hash;
            }
        };

        // Hash function for BindGroupKey
        struct BindGroupKeyHash
        {
            size_t operator()(const BindGroupKey &key) const
            {
                return std::hash<uint64_t>{}(key.pipeline_id) ^ (std::hash<size_t>{}(key.buffers_hash) << 1);
            }
        };

        // Create a bind group key from pipeline and buffers
        inline BindGroupKey makeBindGroupKey(
            wgpu::ComputePipeline pipeline,
            const std::vector<wgpu::Buffer> &buffers,
            const std::vector<uint64_t> &sizes)
        {
            BindGroupKey key;

            // Use pipeline's internal pointer as ID
            key.pipeline_id = reinterpret_cast<uint64_t>(pipeline.Get());

            // Hash buffers and sizes together
            size_t h = 0;
            for (size_t i = 0; i < buffers.size(); ++i)
            {
                h ^= std::hash<uint64_t>{}(reinterpret_cast<uint64_t>(buffers[i].Get())) << (i * 8);
                if (i < sizes.size())
                {
                    h ^= std::hash<uint64_t>{}(sizes[i]) << ((i + buffers.size()) * 4);
                }
            }
            key.buffers_hash = h;

            return key;
        }

        class BindGroupCache
        {
        public:
            static constexpr size_t MAX_CACHE_SIZE = 1024;

            BindGroupCache() = default;
            ~BindGroupCache() = default;

            // Get cached bind group, returns nullopt if not found
            std::optional<wgpu::BindGroup> get(const BindGroupKey &key)
            {
                auto it = cache_.find(key);
                if (it != cache_.end())
                {
                    hits_++;
                    return it->second;
                }
                misses_++;
                return std::nullopt;
            }

            // Store bind group in cache
            void put(const BindGroupKey &key, wgpu::BindGroup bind_group)
            {
                // Evict oldest entries if cache is full
                if (cache_.size() >= MAX_CACHE_SIZE)
                {
                    // Simple strategy: clear half the cache
                    // A proper LRU would be better but more complex
                    size_t to_evict = MAX_CACHE_SIZE / 2;
                    auto it = cache_.begin();
                    while (to_evict > 0 && it != cache_.end())
                    {
                        it = cache_.erase(it);
                        to_evict--;
                        evictions_++;
                    }
                }
                cache_[key] = bind_group;
            }

            // Clear the cache
            void clear()
            {
                cache_.clear();
            }

            // Get statistics
            size_t getHits() const { return hits_; }
            size_t getMisses() const { return misses_; }
            size_t getEvictions() const { return evictions_; }
            size_t getSize() const { return cache_.size(); }
            double getHitRate() const
            {
                size_t total = hits_ + misses_;
                return total > 0 ? static_cast<double>(hits_) / total : 0.0;
            }
            void resetStats()
            {
                hits_ = 0;
                misses_ = 0;
                evictions_ = 0;
            }

        private:
            std::unordered_map<BindGroupKey, wgpu::BindGroup, BindGroupKeyHash> cache_;
            size_t hits_ = 0;
            size_t misses_ = 0;
            size_t evictions_ = 0;
        };

        // Global bind group cache
        BindGroupCache &getBindGroupCache();

    } // namespace core
} // namespace torch_webgpu
