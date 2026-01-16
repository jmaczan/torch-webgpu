#include "buffer_pool.h"
#include "webgpu_context.h"
#include <algorithm>

namespace torch_webgpu
{
    namespace core
    {
        BufferPool::BufferPool(wgpu::Device device)
            : device_(device)
        {
            // Initialize available buffer lists for each size class
            available_.resize(NUM_SIZE_CLASSES);
        }

        size_t BufferPool::getSizeClass(size_t size) const
        {
            for (size_t i = 0; i < NUM_SIZE_CLASSES; ++i)
            {
                if (size <= SIZE_CLASSES[i])
                {
                    return i;
                }
            }
            // For sizes larger than max class, return max class
            // (will create a custom-sized buffer)
            return NUM_SIZE_CLASSES - 1;
        }

        wgpu::Buffer BufferPool::createBuffer(size_t size)
        {
            wgpu::BufferDescriptor descriptor{};
            descriptor.label = "Pooled Uniform Buffer";
            descriptor.usage = wgpu::BufferUsage::Uniform | wgpu::BufferUsage::CopyDst;
            descriptor.size = size;
            descriptor.mappedAtCreation = false;

            total_created_++;
            return device_.CreateBuffer(&descriptor);
        }

        wgpu::Buffer BufferPool::acquire(size_t size)
        {
            size_t sizeClass = getSizeClass(size);
            size_t actualSize = SIZE_CLASSES[sizeClass];

            // If size is larger than max class, use exact size
            if (size > SIZE_CLASSES[NUM_SIZE_CLASSES - 1])
            {
                actualSize = size;
            }

            wgpu::Buffer buffer;

            // Try to get from available pool
            if (!available_[sizeClass].empty())
            {
                buffer = available_[sizeClass].back();
                available_[sizeClass].pop_back();
            }
            else
            {
                // Create new buffer
                buffer = createBuffer(actualSize);
            }

            // Track as in-use
            in_use_.push_back(buffer);
            total_acquired_++;

            return buffer;
        }

        void BufferPool::releaseAll()
        {
            // Move all in-use buffers back to available pools
            for (auto& buffer : in_use_)
            {
                size_t size = buffer.GetSize();
                size_t sizeClass = getSizeClass(size);
                available_[sizeClass].push_back(buffer);
                total_released_++;
            }
            in_use_.clear();
        }

        size_t BufferPool::getTotalBuffers() const
        {
            return total_created_;
        }

        size_t BufferPool::getInUseBuffers() const
        {
            return in_use_.size();
        }

        size_t BufferPool::getAvailableBuffers() const
        {
            size_t count = 0;
            for (const auto& list : available_)
            {
                count += list.size();
            }
            return count;
        }

        void BufferPool::preallocate(size_t buffersPerClass)
        {
            for (size_t i = 0; i < NUM_SIZE_CLASSES; ++i)
            {
                for (size_t j = 0; j < buffersPerClass; ++j)
                {
                    wgpu::Buffer buffer = createBuffer(SIZE_CLASSES[i]);
                    available_[i].push_back(buffer);
                }
            }
        }

        BufferPool& getBufferPool()
        {
            static BufferPool pool(getWebGPUContext().getDevice());
            return pool;
        }

    } // namespace core
} // namespace torch_webgpu
