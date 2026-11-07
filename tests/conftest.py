"""Shared pytest configuration for the torch-webgpu suites.

* Every test ends with a device synchronization, so asynchronous device
  errors (WebGPU validation errors, out-of-bounds index flags, device loss)
  fail the test that caused them instead of a later, unrelated one.
* `no_fallback` / `@pytest.mark.no_fallback` fail a test whose ops ran
  through the CPU fallback.
* Markers: `slow` (deselect with -m "not slow"), `models` (needs Hugging
  Face models in the local cache; skipped when absent).
"""

import os

import pytest
import torch

os.environ.setdefault("TORCH_WEBGPU_FALLBACK_WARN", "0")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import torch_webgpu  # noqa: E402


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: long-running test (deselect with -m 'not slow')")
    config.addinivalue_line("markers", "models: needs Hugging Face models in the local cache")
    config.addinivalue_line("markers", "no_fallback: fail if any op ran through the CPU fallback")
    config.addinivalue_line("markers", "allow_fallback: CPU fallbacks are expected (suites that forbid them)")


def pytest_report_header(config):
    try:
        info = torch.webgpu.get_device_properties() if hasattr(torch.webgpu, "get_device_properties") else None
        name = torch.webgpu.get_device_name() if hasattr(torch.webgpu, "get_device_name") else "?"
    except Exception as e:  # noqa: BLE001
        return f"webgpu: unavailable ({e})"
    lines = [f"webgpu: {name}"]
    if info is not None:
        lines.append(f"webgpu properties: {info}")
    for var in ("TORCH_WEBGPU_BACKEND", "TORCH_WEBGPU_LIMITS", "TORCH_WEBGPU_DAWN_TOGGLES"):
        if os.environ.get(var):
            lines.append(f"{var}={os.environ[var]}")
    return lines


def model_cached(model_id: str) -> bool:
    try:
        from huggingface_hub import try_to_load_from_cache

        return isinstance(try_to_load_from_cache(model_id, "config.json"), str)
    except Exception:  # noqa: BLE001
        return False


@pytest.fixture(autouse=True)
def _device_sync_after_test():
    yield
    # Raises pending device errors here, inside the test that caused them.
    torch.webgpu.synchronize()


@pytest.fixture
def no_fallback():
    before = set(torch_webgpu._C._fallback_ops())
    yield
    new = set(torch_webgpu._C._fallback_ops()) - before
    assert not new, f"ops ran through the CPU fallback: {sorted(new)}"


def pytest_collection_modifyitems(config, items):
    for item in items:
        if item.get_closest_marker("no_fallback") and "no_fallback" not in item.fixturenames:
            item.fixturenames.append("no_fallback")
