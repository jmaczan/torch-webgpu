from setuptools import setup
from torch.utils.cpp_extension import (
    BuildExtension,
    CppExtension,
)

setup(
    name="torch-webgpu",
    description="WebGPU for PyTorch",
    ext_modules=[
        CppExtension("torch_webgpu._C", sources=["csrc/bindings.cpp"]),
    ],
    cmdclass={"build_ext": BuildExtension},
    package_dir={"": "python"},
    packages=["torch_webgpu"],
)
