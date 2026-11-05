#pragma once
// Random number generator state for the WebGPU device.
//
// Like the CUDA generator, the state is (seed, offset) driving a
// counter-based Philox4x32-10 generator evaluated inside the kernels: each
// random kernel reserves a range of counters by advancing `offset`.

#include <ATen/core/Generator.h>
#include <c10/core/GeneratorImpl.h>

#include <mutex>

namespace torch_webgpu::core
{
    struct GeneratorImpl : public c10::GeneratorImpl
    {
        explicit GeneratorImpl(c10::DeviceIndex index = 0);

        void set_current_seed(uint64_t seed) override;
        void set_offset(uint64_t offset) override;
        uint64_t get_offset() const override;
        uint64_t current_seed() const override;
        uint64_t seed() override;
        void set_state(const c10::TensorImpl &new_state) override;
        c10::intrusive_ptr<c10::TensorImpl> get_state() const override;

        static c10::DeviceType device_type() { return c10::DeviceType::PrivateUse1; }

        // Returns the (seed, offset) to use for a kernel consuming
        // `increment` 128-bit Philox blocks per thread-slot, and advances.
        std::pair<uint64_t, uint64_t> philox_engine_inputs(uint64_t increment);

        std::mutex mutex_;

    private:
        GeneratorImpl *clone_impl() const override;
        uint64_t seed_;
        uint64_t offset_ = 0;
    };

    const at::Generator &default_generator();
    // Resolves an optional user generator to ours (falls back to the default).
    GeneratorImpl *get_generator(const std::optional<at::Generator> &gen);
}
