// Runtime internals reachable only from C++: dispatch grids, the virtual
// address allocator, buffer offsets, stream ordering, scalar encoding and
// the asynchronous index-error flag.

#include <gtest/gtest.h>
#include <torch/torch.h>

#include "core/allocator.h"
#include "core/binding_set.h"
#include "core/context.h"
#include "core/kernel.h"
#include "core/stream.h"
#include "kernels/wgsl.h"

#include <cstring>

using namespace torch_webgpu;

namespace
{
    torch::Device dev() { return torch::Device(torch::DeviceType::PrivateUse1, 0); }
}

TEST(Grid, CoversAllElementsWithinLimits)
{
    const uint64_t max_dim = core::Context::get().limits().maxComputeWorkgroupsPerDimension;
    for (uint64_t n : {1ull, 255ull, 256ull, 257ull, 1000000ull, 256ull * max_dim + 1, 3ull * 256 * max_dim})
    {
        auto g = core::grid_1d(n, 256);
        EXPECT_LE(g.x, max_dim);
        EXPECT_LE(g.y, max_dim);
        EXPECT_GE(static_cast<uint64_t>(g.x) * g.y * g.z * 256, n) << "n=" << n;
        // linear id = gid.x + gid.y * row_stride must address every element
        EXPECT_GE(static_cast<uint64_t>(g.row_stride) * (g.y - 1) + static_cast<uint64_t>(g.x) * 256, n);
    }
}

TEST(Allocator, VirtualAddressesResolveInteriorPointers)
{
    auto t = torch::empty({1000}, torch::TensorOptions().device(dev()));
    auto *a = core::allocation_of(t);
    ASSERT_NE(a, nullptr);
    const auto base = reinterpret_cast<uintptr_t>(t.data_ptr());
    EXPECT_EQ(base, a->va);
    EXPECT_EQ(base % 256, 0u);
    EXPECT_EQ(core::find_allocation(reinterpret_cast<void *>(base + 1234)), a);
    EXPECT_EQ(core::find_allocation(reinterpret_cast<void *>(base + a->capacity - 1)), a);
    EXPECT_NE(core::find_allocation(reinterpret_cast<void *>(base + a->capacity)), a);
    EXPECT_EQ(core::find_allocation(reinterpret_cast<void *>(16)), nullptr);
}

TEST(Allocator, DistinctLiveAllocationsDoNotOverlap)
{
    auto x = torch::empty({100}, torch::TensorOptions().device(dev()));
    auto y = torch::empty({100}, torch::TensorOptions().device(dev()));
    auto *ax = core::allocation_of(x), *ay = core::allocation_of(y);
    EXPECT_TRUE(ax->va + ax->capacity <= ay->va || ay->va + ay->capacity <= ax->va);
}

TEST(Allocator, StatsTrackLiveBytes)
{
    core::Stream::get().synchronize();
    const auto before = core::memory_stats().allocated_bytes;
    {
        auto t = torch::empty({1 << 20}, torch::TensorOptions().device(dev()));
        EXPECT_GE(core::memory_stats().allocated_bytes, before + (4 << 20));
    }
    EXPECT_EQ(core::memory_stats().allocated_bytes, before);
}

TEST(Offsets, ViewsAndForeignWrappers)
{
    auto t = torch::arange(64, torch::TensorOptions().dtype(torch::kFloat).device(dev()));
    auto v = t.slice(0, 10, 20);
    EXPECT_EQ(core::byte_offset_of(v), 40u);
    EXPECT_EQ(core::element_offset_of(v), 10u);
    // A from_blob wrapper of an interior device pointer reads the same data.
    auto w = at::from_blob(v.data_ptr(), {10}, {1}, nullptr, t.options(), {dev()});
    EXPECT_EQ(core::allocation_of(w), core::allocation_of(t));
    EXPECT_EQ(core::byte_offset_of(w), 40u);
    EXPECT_TRUE(torch::equal(w.cpu(), torch::arange(10, 20, torch::kFloat)));
    EXPECT_TRUE(torch::equal((w * 2).cpu(), torch::arange(10, 20, torch::kFloat) * 2));
}

TEST(Stream, UploadsAreOrderedWithKernels)
{
    // Overwrite a buffer from the host between two kernels: each kernel must
    // see the upload issued before it.
    auto t = torch::zeros({256}, torch::TensorOptions().device(dev()));
    std::vector<float> ones(256, 1.0f), twos(256, 2.0f);
    auto &s = core::Stream::get();
    s.write_buffer(core::buffer_of(t), 0, ones.data(), 1024);
    auto a = t * 10;
    s.write_buffer(core::buffer_of(t), 0, twos.data(), 1024);
    auto b = t * 10;
    EXPECT_TRUE(torch::equal(a.cpu(), torch::full({256}, 10.0f)));
    EXPECT_TRUE(torch::equal(b.cpu(), torch::full({256}, 20.0f)));
    std::vector<float> back(256);
    s.read_buffer(core::buffer_of(t), 0, back.data(), 1024);
    EXPECT_EQ(back[17], 2.0f);
}

TEST(Wgsl, ScalarEncoding)
{
    using wgsl::CT;
    uint32_t w[2];
    wgsl::encode_scalar(1.5, CT::F32, w);
    float f;
    std::memcpy(&f, &w[0], 4);
    EXPECT_EQ(f, 1.5f);
    wgsl::encode_scalar(int64_t(-3), CT::I32, w);
    EXPECT_EQ(static_cast<int32_t>(w[0]), -3);
    wgsl::encode_scalar(int64_t(-(int64_t(1) << 40) - 5), CT::I64, w);
    const int64_t v = static_cast<int64_t>((static_cast<uint64_t>(w[1]) << 32) | w[0]);
    EXPECT_EQ(v, -(int64_t(1) << 40) - 5);
    EXPECT_EQ(wgsl::compute_type(torch::kHalf), CT::F32);
    EXPECT_EQ(wgsl::compute_type(torch::kLong), CT::I64);
    EXPECT_TRUE(wgsl::is_storage_dtype(torch::kDouble));
    EXPECT_FALSE(wgsl::is_compute_dtype(torch::kDouble));
}

TEST(Errors, IndexErrorRaisedAtSyncAndCleared)
{
    auto x = torch::arange(5, torch::TensorOptions().dtype(torch::kFloat).device(dev()));
    auto bad = torch::tensor({7}, torch::kLong).to(dev());
    EXPECT_THROW({ x.index_select(0, bad).cpu(); }, c10::Error);
    // The flag is cleared: the device keeps working.
    EXPECT_TRUE(torch::equal(x.cpu(), torch::arange(5, torch::kFloat)));
}

TEST(BindingSet, TensorsSharingABufferShareABinding)
{
    auto base = torch::zeros({1024}, torch::TensorOptions().device(dev()));
    auto other = torch::zeros({16}, torch::TensorOptions().device(dev()));
    auto a = base.narrow(0, 0, 100), b = base.narrow(0, 512, 100);
    core::BindingSet bs;
    int ra = bs.add(a), rb = bs.add(b, /*written=*/true), ro = bs.add(other);
    bs.finalize();
    EXPECT_EQ(bs.binding(ra), bs.binding(rb));
    EXPECT_NE(bs.binding(ra), bs.binding(ro));
    ASSERT_EQ(bs.bindings().size(), 2u);
    const auto &shared = bs.bindings()[bs.binding(ra)];
    EXPECT_TRUE(shared.written);
    // Element offsets inside the shared window keep the views apart.
    EXPECT_EQ(static_cast<int64_t>(bs.offset(rb)) - static_cast<int64_t>(bs.offset(ra)), 512);
}

TEST(BindingSet, WindowCoversAccessedBytesAligned)
{
    auto base = torch::zeros({1 << 16}, torch::TensorOptions().device(dev()));
    auto v = base.narrow(0, 40000, 1000);
    core::BindingSet bs;
    int r = bs.add(v);
    bs.finalize();
    const auto &b = bs.bindings()[bs.binding(r)];
    EXPECT_EQ(b.offset % core::binding_offset_alignment(), 0u);
    // The window starts at or before the view and covers all of it.
    const uint64_t first = b.offset + uint64_t(bs.offset(r)) * 4;
    auto [lo, hi] = core::tensor_byte_range(v);
    EXPECT_EQ(first, lo);
    EXPECT_GE(b.offset + b.size, hi);
    EXPECT_LE(b.size, core::max_binding_size());
}

TEST(BindingSet, StridedViewRangeSpansAllElements)
{
    auto base = torch::zeros({64, 64}, torch::TensorOptions().device(dev()));
    auto col = base.select(1, 5);  // stride 64
    auto [lo, hi] = core::tensor_byte_range(col);
    EXPECT_EQ(hi - lo, uint64_t((63 * 64 + 1) * 4));
}
