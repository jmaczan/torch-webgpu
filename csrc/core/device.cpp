// Device guard, PrivateUse1 hooks and generator registration.

#include <ATen/ATen.h>
#include <ATen/core/GeneratorForPrivateuseone.h>
#include <ATen/detail/PrivateUse1HooksInterface.h>
#include <c10/core/impl/DeviceGuardImplInterface.h>
#include <c10/core/CPUAllocator.h>

#include "core/allocator.h"
#include "core/context.h"
#include "core/generator.h"
#include "core/stream.h"

#include <random>

namespace torch_webgpu::core
{
    // ---------------------------------------------------------------- guard
    namespace
    {
        thread_local c10::DeviceIndex current_device = 0;

        struct GuardImpl final : public c10::impl::DeviceGuardImplInterface
        {
            static constexpr c10::DeviceType static_type = c10::DeviceType::PrivateUse1;

            c10::DeviceType type() const override { return static_type; }
            c10::Device exchangeDevice(c10::Device d) const override
            {
                auto old = getDevice();
                setDevice(d);
                return old;
            }
            c10::Device getDevice() const override { return {static_type, current_device}; }
            void setDevice(c10::Device d) const override
            {
                TORCH_CHECK(d.type() == static_type, "expected a webgpu device, got ", d);
                TORCH_CHECK(d.index() <= 0, "torch_webgpu: invalid device index ", int(d.index()),
                            " (only device 0 is available)");
                current_device = 0;
            }
            void uncheckedSetDevice(c10::Device) const noexcept override { current_device = 0; }
            c10::Stream getStream(c10::Device d) const override
            {
                return c10::Stream(c10::Stream::DEFAULT, c10::Device(static_type, 0));
            }
            c10::Stream getDefaultStream(c10::Device d) const override { return getStream(d); }
            c10::Stream getNewStream(c10::Device d, int) const override { return getStream(d); }
            c10::Stream getStreamFromGlobalPool(c10::Device d, bool) const override { return getStream(d); }
            c10::Stream exchangeStream(c10::Stream s) const override { return getStream(s.device()); }
            c10::DeviceIndex deviceCount() const noexcept override { return is_available() ? 1 : 0; }

            // Single in-order queue: events are ordering no-ops; waiting on one
            // means waiting for all submitted work.
            void record(void **event, const c10::Stream &, const c10::DeviceIndex, const c10::EventFlag) const override
            {
                if (*event == nullptr)
                    *event = new int(0);
                Stream::get().flush();
            }
            void block(void *, const c10::Stream &) const override {}
            bool queryEvent(void *) const override
            {
                Stream::get().synchronize();
                return true;
            }
            void destroyEvent(void *event, const c10::DeviceIndex) const noexcept override
            {
                delete static_cast<int *>(event);
            }
            void synchronizeEvent(void *) const override { Stream::get().synchronize(); }
            bool queryStream(const c10::Stream &) const override
            {
                Stream::get().synchronize();
                return true;
            }
            void synchronizeStream(const c10::Stream &) const override { Stream::get().synchronize(); }
            void synchronizeDevice(const c10::DeviceIndex) const override { Stream::get().synchronize(); }
            void recordDataPtrOnStream(const c10::DataPtr &, const c10::Stream &) const override {}
        };

        C10_REGISTER_GUARD_IMPL(PrivateUse1, GuardImpl);
    }

    // ------------------------------------------------------------ generator
    GeneratorImpl::GeneratorImpl(c10::DeviceIndex index)
        : c10::GeneratorImpl(c10::Device(c10::DeviceType::PrivateUse1, index),
                             c10::DispatchKeySet(c10::DispatchKey::PrivateUse1)),
          seed_(c10::detail::getNonDeterministicRandom()) {}

    void GeneratorImpl::set_current_seed(uint64_t seed)
    {
        seed_ = seed;
        offset_ = 0;
    }
    void GeneratorImpl::set_offset(uint64_t offset) { offset_ = offset; }
    uint64_t GeneratorImpl::get_offset() const { return offset_; }
    uint64_t GeneratorImpl::current_seed() const { return seed_; }
    uint64_t GeneratorImpl::seed()
    {
        auto s = c10::detail::getNonDeterministicRandom();
        set_current_seed(s);
        return s;
    }

    // State layout (CPU uint8 tensor of 16 bytes): seed (u64) | offset (u64).
    void GeneratorImpl::set_state(const c10::TensorImpl &new_state)
    {
        TORCH_CHECK(new_state.dtype() == caffe2::TypeMeta::Make<uint8_t>(), "RNG state must be a torch.ByteTensor");
        TORCH_CHECK(new_state.numel() == 16, "torch_webgpu: RNG state must have 16 bytes, got ", new_state.numel());
        const auto *data = static_cast<const uint8_t *>(new_state.data());
        std::memcpy(&seed_, data, 8);
        std::memcpy(&offset_, data + 8, 8);
    }
    c10::intrusive_ptr<c10::TensorImpl> GeneratorImpl::get_state() const
    {
        auto state = at::empty({16}, at::TensorOptions().dtype(at::kByte).device(at::kCPU));
        auto *data = state.data_ptr<uint8_t>();
        std::memcpy(data, &seed_, 8);
        std::memcpy(data + 8, &offset_, 8);
        return state.getIntrusivePtr();
    }
    std::pair<uint64_t, uint64_t> GeneratorImpl::philox_engine_inputs(uint64_t increment)
    {
        uint64_t off = offset_;
        offset_ += (increment + 3) / 4 * 4;
        return {seed_, off};
    }
    GeneratorImpl *GeneratorImpl::clone_impl() const
    {
        auto *g = new GeneratorImpl(device().index());
        g->seed_ = seed_;
        g->offset_ = offset_;
        return g;
    }

    const at::Generator &default_generator()
    {
        static at::Generator *gen = new at::Generator(at::make_generator<GeneratorImpl>(0));
        return *gen;
    }

    GeneratorImpl *get_generator(const std::optional<at::Generator> &gen)
    {
        const at::Generator &g = (gen.has_value() && gen->defined()) ? *gen : default_generator();
        TORCH_CHECK(g.device().type() == c10::DeviceType::PrivateUse1,
                    "torch_webgpu: expected a webgpu generator, got one for ", g.device());
        return at::check_generator<GeneratorImpl>(g);
    }

    // ---------------------------------------------------------------- hooks
    void resize_storage_bytes(c10::StorageImpl *storage, size_t new_size);

    namespace
    {
        struct Hooks final : public at::PrivateUse1HooksInterface
        {
            bool isBuilt() const override { return true; }
            bool isAvailable() const override { return is_available(); }
            c10::DeviceIndex deviceCount() const override { return is_available() ? 1 : 0; }
            void setCurrentDevice(c10::DeviceIndex) const override {}
            c10::DeviceIndex getCurrentDevice() const override { return 0; }
            c10::DeviceIndex exchangeDevice(c10::DeviceIndex) const override { return 0; }
            c10::DeviceIndex maybeExchangeDevice(c10::DeviceIndex) const override { return 0; }
            const at::Generator &getDefaultGenerator(c10::DeviceIndex) const override { return default_generator(); }
            at::Generator getNewGenerator(c10::DeviceIndex index) const override
            {
                return at::make_generator<GeneratorImpl>(index < 0 ? 0 : index);
            }
            at::Device getDeviceFromPtr(void *) const override { return {c10::DeviceType::PrivateUse1, 0}; }
            bool isPinnedPtr(const void *) const override { return false; }
            c10::Allocator *getPinnedMemoryAllocator() const override { return c10::GetCPUAllocator(); }
            bool hasPrimaryContext(c10::DeviceIndex) const override { return true; }
            void init() const override { Context::get(); }
            void resizePrivateUse1Bytes(const c10::Storage &storage, size_t newsize) const override
            {
                resize_storage_bytes(storage.unsafeGetStorageImpl(), newsize);
            }
        };

        struct RegisterHooks
        {
            RegisterHooks() { at::RegisterPrivateUse1HooksInterface(new Hooks()); }
        };
        static RegisterHooks register_hooks;
    }
}
