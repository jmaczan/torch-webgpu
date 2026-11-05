import glob
import os
import platform
import shutil
from pathlib import Path
from setuptools import setup, find_packages
from torch.utils.cpp_extension import CppExtension, BuildExtension

# Try to load .env for local development
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

ROOT = Path(__file__).parent.absolute()

# Dawn prefix - required for building
DAWN_PREFIX = os.environ.get("DAWN_PREFIX")
if not DAWN_PREFIX:
    # Check common locations
    possible_paths = [
        ROOT / "dawn-install",
        ROOT / "dawn" / "install" / "Release",
        Path.home() / "dawn" / "install" / "Release",
    ]
    for p in possible_paths:
        if p.exists():
            DAWN_PREFIX = str(p)
            break

if not DAWN_PREFIX:
    raise RuntimeError(
        "DAWN_PREFIX environment variable not set and Dawn not found in common locations.\n"
        "Please set DAWN_PREFIX to the Dawn installation directory, or run:\n"
        "  ./scripts/build-dawn.sh"
    )

DAWN_PREFIX = Path(DAWN_PREFIX)


def get_dawn_library_name():
    """Get the Dawn library filename for the current platform."""
    system = platform.system()
    if system == "Linux":
        return "libwebgpu_dawn.so"
    elif system == "Darwin":
        return "libwebgpu_dawn.dylib"
    elif system == "Windows":
        return "webgpu_dawn.dll"
    else:
        raise RuntimeError(f"Unsupported platform: {system}")


def get_dawn_lib_path():
    """Get the path to the Dawn shared library, or None if statically linked."""
    lib_name = get_dawn_library_name()
    lib_path = DAWN_PREFIX / "lib" / lib_name
    if lib_path.exists():
        return lib_path
    # Check if static library exists (static linking)
    static_name = lib_name.replace(".so", ".a").replace(".dylib", ".a").replace(".dll", ".lib")
    static_path = DAWN_PREFIX / "lib" / static_name
    if static_path.exists():
        print(f"Dawn static library found at {static_path}, using static linking")
        return None
    raise RuntimeError(f"Dawn library not found at {lib_path} or {static_path}")


class BuildExtWithDawn(BuildExtension):
    """Custom build extension that copies Dawn library into the package."""

    def run(self):
        # Run the normal build
        super().run()

        # Copy Dawn library into the built package (only for dynamic linking)
        lib_path = get_dawn_lib_path()

        if lib_path is None:
            # Static linking - no need to copy library
            print("Using static Dawn library, skipping library copy")
            return

        # Copy next to every built extension. get_ext_fullpath honours
        # --inplace / editable installs, unlike get_outputs().
        output_dirs = {
            Path(self.get_ext_fullpath(ext.name)).parent for ext in self.extensions
        }
        for output_dir in output_dirs:
            libs_dir = output_dir / "libs"
            libs_dir.mkdir(parents=True, exist_ok=True)
            dst = libs_dir / lib_path.name
            print(f"Copying {lib_path} -> {dst}")
            shutil.copy2(lib_path, dst)

        # RPATH ($ORIGIN/libs, @loader_path/libs) is set at link time via
        # extra_link_args, so no post-processing of the binary is needed.


# Determine extra compile/link args based on platform
extra_compile_args = []
extra_link_args = []
system = platform.system()

if system == "Linux":
    extra_compile_args = ["-std=c++20", "-O2"]
    extra_link_args = ["-Wl,-rpath,$ORIGIN/libs"]
elif system == "Darwin":
    extra_compile_args = ["-std=c++20", "-O2"]
    extra_link_args = ["-Wl,-rpath,@loader_path/libs"]
elif system == "Windows":
    extra_compile_args = ["/std:c++20", "/O2"]

setup(
    name="torch-webgpu",
    version="0.0.1",
    description="WebGPU backend for PyTorch",
    long_description=Path("README.md").read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    author="Jedrzej Maczan",
    author_email="jedrzejpawel@maczan.pl",
    url="https://github.com/jmaczan/torch-webgpu",
    ext_modules=[
        CppExtension(
            name="torch_webgpu._C",
            sources=glob.glob("csrc/**/*.cpp", recursive=True),
            include_dirs=[
                str(ROOT / "csrc"),
                str(DAWN_PREFIX / "include"),
            ],
            library_dirs=[
                str(DAWN_PREFIX / "lib"),
            ],
            libraries=[
                "webgpu_dawn",
            ],
            extra_compile_args=extra_compile_args,
            extra_link_args=extra_link_args,
        ),
    ],
    cmdclass={"build_ext": BuildExtWithDawn},
    package_dir={"": "python"},
    packages=find_packages(where="python"),
    package_data={
        "torch_webgpu": ["libs/*"],
    },
    include_package_data=True,
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.0.0",
        "numpy",
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
