import os
import sys
import torch

# On Windows, add the libs directory to DLL search path before importing _C
if sys.platform == "win32":
    libs_dir = os.path.join(os.path.dirname(__file__), "libs")
    if os.path.isdir(libs_dir):
        if hasattr(os, "add_dll_directory"):
            os.add_dll_directory(libs_dir)
        # Also prepend to PATH for older Python/Windows compatibility
        os.environ["PATH"] = libs_dir + os.pathsep + os.environ.get("PATH", "")

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
