#include "command_batcher.h"
#include "webgpu_context.h"
#include "buffer_pool.h"
#include "dispatch_profiler.h"
#include <iostream>

namespace torch_webgpu
{
    namespace core
    {
        CommandBatcher::CommandBatcher(wgpu::Device device, wgpu::Queue queue, size_t batch_size)
            : device_(device)
            , queue_(queue)
            , batch_size_(batch_size)
            , current_batch_count_(0)
            , enabled_(false)  // Disable batching by default - causes RAW dependency issues
            , has_active_pass_(false)
            , total_dispatches_(0)
            , total_submissions_(0)
        {
        }

        CommandBatcher::~CommandBatcher()
        {
            if (has_active_pass_)
            {
                flush();
            }
        }

        void CommandBatcher::beginNewBatch()
        {
            if (has_active_pass_)
            {
                return;  // Already have an active batch
            }

            current_encoder_ = device_.CreateCommandEncoder();
            wgpu::ComputePassDescriptor pass_descriptor{};
            current_pass_ = current_encoder_.BeginComputePass(&pass_descriptor);
            has_active_pass_ = true;
            current_batch_count_ = 0;
        }

        void CommandBatcher::endCurrentBatch()
        {
            if (!has_active_pass_)
            {
                return;  // No active batch to end
            }

            current_pass_.End();
            wgpu::CommandBuffer command_buffer = current_encoder_.Finish();
            queue_.Submit(1, &command_buffer);

            has_active_pass_ = false;
            current_batch_count_ = 0;
            total_submissions_++;
        }

        wgpu::ComputePassEncoder CommandBatcher::getComputePassEncoder()
        {
            if (!enabled_)
            {
                return nullptr;  // Caller should create their own encoder
            }

            if (!has_active_pass_)
            {
                beginNewBatch();
            }

            return current_pass_;
        }

        void CommandBatcher::recordDispatch()
        {
            if (!enabled_)
            {
                return;
            }

            current_batch_count_++;
            total_dispatches_++;

            // Auto-flush if batch is full
            if (current_batch_count_ >= batch_size_)
            {
                flush();
            }
        }

        void CommandBatcher::flush()
        {
            if (has_active_pass_)
            {
                endCurrentBatch();
            }
            // Release pooled buffers back to the pool
            getBufferPool().releaseAll();
        }

        void CommandBatcher::setEnabled(bool enabled)
        {
            if (enabled_ && !enabled)
            {
                // Disabling - flush any pending work
                flush();
            }
            enabled_ = enabled;
        }

        CommandBatcher& getCommandBatcher()
        {
            static CommandBatcher batcher(
                getWebGPUContext().getDevice(),
                getWebGPUContext().getQueue()
            );
            return batcher;
        }

        void dispatchCompute(
            wgpu::ComputePipeline pipeline,
            wgpu::BindGroup bind_group,
            uint32_t workgroups_x,
            uint32_t workgroups_y,
            uint32_t workgroups_z
        )
        {
            CommandBatcher& batcher = getCommandBatcher();
            DispatchProfiler& profiler = getDispatchProfiler();

            // Record dispatch count
            profiler.recordDispatch();

            if (batcher.isEnabled())
            {
                // Batched mode - add to current batch
                wgpu::ComputePassEncoder pass_encoder = batcher.getComputePassEncoder();
                pass_encoder.SetPipeline(pipeline);
                pass_encoder.SetBindGroup(0, bind_group);
                pass_encoder.DispatchWorkgroups(workgroups_x, workgroups_y, workgroups_z);
                batcher.recordDispatch();
            }
            else
            {
                // Immediate mode - create encoder, dispatch, submit
                WebGPUContext& ctx = getWebGPUContext();

                // Profile encoder creation
                ScopedTimer encoder_timer;
                wgpu::CommandEncoder encoder = ctx.getDevice().CreateCommandEncoder();
                wgpu::ComputePassDescriptor pass_descriptor{};
                wgpu::ComputePassEncoder pass_encoder = encoder.BeginComputePass(&pass_descriptor);
                profiler.recordEncoderCreation(encoder_timer.elapsedUs());

                pass_encoder.SetPipeline(pipeline);
                pass_encoder.SetBindGroup(0, bind_group);
                pass_encoder.DispatchWorkgroups(workgroups_x, workgroups_y, workgroups_z);
                pass_encoder.End();
                wgpu::CommandBuffer command_buffer = encoder.Finish();

                // Profile submission
                ScopedTimer submit_timer;
                ctx.getQueue().Submit(1, &command_buffer);
                profiler.recordSubmission(submit_timer.elapsedUs());
            }
        }

        wgpu::Buffer acquireUniformBuffer(const void* data, size_t size)
        {
            // Acquire buffer from pool
            wgpu::Buffer buffer = getBufferPool().acquire(size);

            // Write data to buffer
            getWebGPUContext().getQueue().WriteBuffer(buffer, 0, data, size);

            return buffer;
        }

    } // namespace core
} // namespace torch_webgpu
