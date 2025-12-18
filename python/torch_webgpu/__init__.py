import torch
from . import _C
from . import webgpu


torch.utils.rename_privateuse1_backend("webgpu")
torch._register_device_module("webgpu", webgpu)
torch.utils.generate_methods_for_privateuse1_backend(
    for_tensor=True,
    for_module=True,
    for_storage=True,
    for_packed_sequence=True,
)

from .compiler.webgpu_compiler import webgpu_backend  # noqa: F401 E402

__all__ = ["webgpu_backend", "webgpu_ops"]
