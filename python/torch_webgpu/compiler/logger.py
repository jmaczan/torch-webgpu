"""
Debug logger for torch-webgpu compiler.

Only prints when DEBUG=1 environment variable is set.
This avoids polluting user's console and slowing down inference.

Usage:
    from .logger import debug, debug_enabled

    debug("Some debug message")
    debug(f"Tensor shape: {tensor.shape}")

    # For expensive operations, check first:
    if debug_enabled():
        debug(format_complex_object(obj))
"""

import os
from functools import lru_cache


@lru_cache(maxsize=1)
def debug_enabled() -> bool:
    """Check if debug mode is enabled via DEBUG=1 environment variable."""
    return os.environ.get("DEBUG", "0") == "1"


def debug(msg: str) -> None:
    """Print debug message only if DEBUG=1 is set."""
    if debug_enabled():
        print(f"[torch-webgpu] {msg}")


def debug_ir(title: str, content: str) -> None:
    """Print IR graph debug info only if DEBUG=1 is set."""
    if debug_enabled():
        print(f"\n[torch-webgpu] {title}")
        print(content)
