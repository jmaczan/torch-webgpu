"""torch_webgpu: a WebGPU backend for PyTorch.

Importing this package registers the ``webgpu`` device::

    import torch, torch_webgpu
    x = torch.randn(3, device="webgpu")
"""

import os
import sys

import torch

# On Windows, add the bundled Dawn directory to the DLL search path.
if sys.platform == "win32":
    _libs_dir = os.path.join(os.path.dirname(__file__), "libs")
    if os.path.isdir(_libs_dir):
        if hasattr(os, "add_dll_directory"):
            os.add_dll_directory(_libs_dir)
        os.environ["PATH"] = _libs_dir + os.pathsep + os.environ.get("PATH", "")

from . import _C  # noqa: E402
from . import webgpu  # noqa: E402

if torch._C._get_privateuse1_backend_name() != "webgpu":
    torch.utils.rename_privateuse1_backend("webgpu")
    torch._register_device_module("webgpu", webgpu)
    torch.utils.generate_methods_for_privateuse1_backend(
        for_tensor=True,
        for_module=True,
        for_storage=True,
        for_packed_sequence=True,
    )

from .compiler.webgpu_compiler import webgpu_backend  # noqa: F401 E402

__all__ = ["webgpu_backend", "webgpu"]
