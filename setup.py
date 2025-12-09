import glob
import os
from setuptools import setup
from torch.utils.cpp_extension import CppExtension, BuildExtension
from dotenv import load_dotenv

load_dotenv()

DAWN_PREFIX = os.environ.get("DAWN_PREFIX")
ROOT = os.path.dirname(os.path.abspath(__file__))

setup(
    name="torch-webgpu",
    description="WebGPU backend for PyTorch",
    ext_modules=[
        CppExtension(
            name="torch_webgpu._C",
            sources=glob.glob("csrc/**/*.cpp", recursive=True),
            include_dirs=[
                os.path.join(ROOT, "csrc"),
                os.path.join(DAWN_PREFIX, "include"),
            ],
            library_dirs=[
                os.path.join(DAWN_PREFIX, "lib"),
            ],
            libraries=[
                "webgpu_dawn",
            ],
            runtime_library_dirs=[
                os.path.join(DAWN_PREFIX, "lib"),
            ],
            extra_compile_args=["-g", "-O0"],
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
    package_dir={"": "python"},
    packages=["torch_webgpu"],
)
