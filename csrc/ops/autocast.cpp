// Autocast (torch.autocast(device_type="webgpu")): the same op policies as
// CUDA, from PyTorch's shared op lists. Ops not listed fall through.

#include <ATen/autocast_mode.h>
#include <torch/library.h>

namespace torch_webgpu::ops
{
    TORCH_LIBRARY_IMPL(_, AutocastPrivateUse1, m)
    {
        m.fallback(torch::CppFunction::makeFallthrough());
    }

    TORCH_LIBRARY_IMPL(aten, AutocastPrivateUse1, m)
    {
        using namespace at;
        using namespace at::autocast;
#define WG_LOWER(...) KERNEL_PRIVATEUSEONE(__VA_ARGS__, lower_precision_fp)
        AT_FORALL_LOWER_PRECISION_FP(WG_LOWER)
#undef WG_LOWER
#define WG_FP32(...) KERNEL_PRIVATEUSEONE(__VA_ARGS__, fp32)
        AT_FORALL_FP32(WG_FP32)
#undef WG_FP32
#define WG_FP32_SET_OPT_DTYPE(...) KERNEL_PRIVATEUSEONE(__VA_ARGS__, fp32_set_opt_dtype)
        AT_FORALL_FP32_SET_OPT_DTYPE(WG_FP32_SET_OPT_DTYPE)
#undef WG_FP32_SET_OPT_DTYPE
        AT_FORALL_DIFFERENT_REDISPATCH_SIGNATURE(KERNEL_DIFFERENT_REDISPATCH_SIGNATURE_PRIVATEUSEONE)
#define WG_PROMOTE(...) KERNEL_PRIVATEUSEONE(__VA_ARGS__, promote)
        AT_FORALL_PROMOTE(WG_PROMOTE)
#undef WG_PROMOTE
    }
}
