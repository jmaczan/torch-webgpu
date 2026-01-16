#pragma once
#include <webgpu/webgpu_cpp.h>
#include <vector>
#include <functional>

namespace torch_webgpu
{
    namespace core
    {
        /**
         * CommandBatcher reduces WebGPU command submission overhead by batching
         * multiple compute dispatches into a single command buffer submission.
         *
         * Instead of:
         *   encoder1 -> dispatch -> submit
         *   encoder2 -> dispatch -> submit
         *   ... (200 times)
         *
         * We do:
         *   encoder -> dispatch1 -> dispatch2 -> ... -> dispatchN -> submit
         *   (repeated for batches of N)
         *
         * This can reduce submission overhead by 4-8x.
         */
        class CommandBatcher
        {
        public:
            // Default batch size - flush every N dispatches
            static constexpr size_t DEFAULT_BATCH_SIZE = 16;

            CommandBatcher(wgpu::Device device, wgpu::Queue queue, size_t batch_size = DEFAULT_BATCH_SIZE);
            ~CommandBatcher();

            // Get the current compute pass encoder to add dispatches
            // Returns nullptr if batching is disabled
            wgpu::ComputePassEncoder getComputePassEncoder();

            // Record a dispatch (increments counter, auto-flushes if batch full)
            void recordDispatch();

            // Explicitly flush all pending commands
            // Call this before reading GPU results or at end of forward pass
            void flush();

            // Enable/disable batching (disabled = immediate submission like before)
            void setEnabled(bool enabled);
            bool isEnabled() const { return enabled_; }

            // Set batch size (number of dispatches before auto-flush)
            void setBatchSize(size_t size) { batch_size_ = size; }
            size_t getBatchSize() const { return batch_size_; }

            // Get statistics
            size_t getTotalDispatches() const { return total_dispatches_; }
            size_t getTotalSubmissions() const { return total_submissions_; }
            void resetStats() { total_dispatches_ = 0; total_submissions_ = 0; }

        private:
            void beginNewBatch();
            void endCurrentBatch();

            wgpu::Device device_;
            wgpu::Queue queue_;

            wgpu::CommandEncoder current_encoder_;
            wgpu::ComputePassEncoder current_pass_;

            size_t batch_size_;
            size_t current_batch_count_;
            bool enabled_;
            bool has_active_pass_;

            // Stats
            size_t total_dispatches_;
            size_t total_submissions_;
        };

        // Global command batcher (lazily initialized)
        CommandBatcher& getCommandBatcher();

        // RAII helper to ensure flush at scope end
        class BatchScope
        {
        public:
            BatchScope() = default;
            ~BatchScope() { getCommandBatcher().flush(); }
        };

        /**
         * Helper function to dispatch a compute shader with batching support.
         *
         * Usage:
         *   dispatchCompute(pipeline, bind_group, workgroups_x, workgroups_y, workgroups_z);
         *
         * This automatically uses the CommandBatcher if enabled, or falls back
         * to immediate submission if disabled.
         */
        void dispatchCompute(
            wgpu::ComputePipeline pipeline,
            wgpu::BindGroup bind_group,
            uint32_t workgroups_x,
            uint32_t workgroups_y = 1,
            uint32_t workgroups_z = 1
        );

    } // namespace core
} // namespace torch_webgpu
