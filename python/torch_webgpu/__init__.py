import torch


def dummy():
    print("Dummy")


torch.utils.rename_privateuse1_backend("webgpu")
torch._register_device_module("webgpu", dummy)
torch.utils.generate_methods_for_privateuse1_backend(
    for_tensor=True,
    for_module=True,
    for_storage=True,
)
