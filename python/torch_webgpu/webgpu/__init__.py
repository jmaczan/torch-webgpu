"""``torch.webgpu`` device module.

Registered with :func:`torch._register_device_module`, so everything here is
reachable as ``torch.webgpu.<name>``. The API mirrors ``torch.cuda`` /
``torch.mps`` where it makes sense for a single-queue WebGPU device.
"""

from __future__ import annotations

from typing import Any, Union

import time

import torch

from .. import _C

__all__ = [
    "is_available",
    "is_initialized",
    "init",
    "device_count",
    "current_device",
    "set_device",
    "synchronize",
    "get_device_name",
    "get_device_properties",
    "get_device_capability",
    "manual_seed",
    "manual_seed_all",
    "seed",
    "seed_all",
    "initial_seed",
    "get_rng_state",
    "set_rng_state",
    "get_rng_state_all",
    "set_rng_state_all",
    "memory_allocated",
    "max_memory_allocated",
    "memory_reserved",
    "memory_stats",
    "reset_peak_memory_stats",
    "empty_cache",
    "get_amp_supported_dtype",
    "is_bf16_supported",
    "current_stream",
    "default_stream",
    "Stream",
    "Event",
    "device",
]

_DeviceLike = Union[torch.device, str, int, None]


def is_available() -> bool:
    """True if a WebGPU adapter and device could be created."""
    return bool(_C._is_available())


def is_initialized() -> bool:
    return is_available()


def init() -> None:
    _C._is_available()


def _lazy_init() -> None:
    init()


def _is_in_bad_fork() -> bool:
    return False


def device_count() -> int:
    return int(_C._device_count())


def current_device() -> int:
    return 0


def _check_index(d: _DeviceLike) -> None:
    if d is None:
        return
    if isinstance(d, str):
        d = torch.device(d)
    if isinstance(d, torch.device):
        if d.type != "webgpu":
            raise ValueError(f"Expected a webgpu device, got {d}")
        d = d.index if d.index is not None else 0
    if d not in (0, -1):
        raise ValueError(f"Invalid webgpu device index {d}: only device 0 exists")


def set_device(device: _DeviceLike) -> None:
    _check_index(device)


class device:
    """Context manager that selects a webgpu device (only index 0 exists)."""

    def __init__(self, device: _DeviceLike):
        _check_index(device)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


def synchronize(device: _DeviceLike = None) -> None:
    """Wait for all queued WebGPU work to finish."""
    _check_index(device)
    _C._synchronize()


def get_device_properties(device: _DeviceLike = None) -> dict[str, Any]:
    """Adapter information and limits as a dict."""
    _check_index(device)
    return dict(_C._device_info())


def get_device_name(device: _DeviceLike = None) -> str:
    return get_device_properties(device)["name"]


def get_device_capability(device: _DeviceLike = None) -> dict[str, Any]:
    props = get_device_properties(device)
    return {
        "shader_f16": props["shader_f16"],
        "subgroups": props["subgroups"],
        "supported_dtypes": [
            torch.float32,
            torch.float16,
            torch.bfloat16,
            torch.int64,
            torch.int32,
            torch.int16,
            torch.int8,
            torch.uint8,
            torch.bool,
        ],
    }


# ----------------------------------------------------------------- RNG
# Generator state lives outside any graph: Dynamo must not trace into these
# (it can't fake the CPU state tensors), so calls are graph breaks.
def _generator() -> torch.Generator:
    return _C._default_generator()


@torch.compiler.disable
def manual_seed(seed: int) -> None:
    _generator().manual_seed(int(seed))


def manual_seed_all(seed: int) -> None:
    manual_seed(seed)


@torch.compiler.disable
def seed() -> None:
    _generator().seed()


def seed_all() -> None:
    seed()


def initial_seed() -> int:
    return _generator().initial_seed()


@torch.compiler.disable
def get_rng_state(device: _DeviceLike = None) -> torch.Tensor:
    _check_index(device)
    return _generator().get_state()


@torch.compiler.disable
def set_rng_state(new_state: torch.Tensor, device: _DeviceLike = None) -> None:
    _check_index(device)
    _generator().set_state(new_state)


def get_rng_state_all() -> list[torch.Tensor]:
    return [get_rng_state()]


def set_rng_state_all(new_states) -> None:
    for state in new_states:
        set_rng_state(state)


# -------------------------------------------------------------- memory
def memory_stats(device: _DeviceLike = None) -> dict[str, int]:
    _check_index(device)
    return dict(_C._memory_stats())


def memory_allocated(device: _DeviceLike = None) -> int:
    return memory_stats(device)["allocated_bytes"]


def max_memory_allocated(device: _DeviceLike = None) -> int:
    return memory_stats(device)["peak_allocated_bytes"]


def memory_reserved(device: _DeviceLike = None) -> int:
    return memory_stats(device)["reserved_bytes"]


def reset_peak_memory_stats(device: _DeviceLike = None) -> None:
    _check_index(device)
    _C._reset_peak_memory_stats()


def empty_cache() -> None:
    """Release cached, unused device buffers."""
    _C._empty_cache()


# ----------------------------------------------------------------- AMP
def get_amp_supported_dtype() -> list[torch.dtype]:
    return [torch.float16, torch.bfloat16]


def is_bf16_supported(including_emulation: bool = True) -> bool:
    return True


# ------------------------------------------------------ streams/events
class Stream:
    """The single in-order WebGPU queue."""

    def __init__(self, device: _DeviceLike = None, priority: int = 0):
        _check_index(device)
        self.device = torch.device("webgpu", 0)

    def synchronize(self) -> None:
        synchronize()

    def query(self) -> bool:
        synchronize()
        return True

    def wait_stream(self, stream: "Stream") -> None:
        pass

    def wait_event(self, event: "Event") -> None:
        pass

    def record_event(self, event: "Event | None" = None) -> "Event":
        event = event or Event()
        event.record(self)
        return event

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def __eq__(self, other):
        return isinstance(other, Stream)

    def __hash__(self):
        return hash("webgpu-stream-0")


_DEFAULT_STREAM: Stream | None = None


def current_stream(device: _DeviceLike = None) -> Stream:
    global _DEFAULT_STREAM
    _check_index(device)
    if _DEFAULT_STREAM is None:
        _DEFAULT_STREAM = Stream()
    return _DEFAULT_STREAM


def default_stream(device: _DeviceLike = None) -> Stream:
    return current_stream(device)


def stream(stream: Stream | None = None) -> Stream:
    return stream or current_stream()


class Event:
    """Events on a single in-order queue: waiting means synchronizing.

    With ``enable_timing=True``, ``record()`` waits for all previously
    submitted work and takes a host timestamp, so ``elapsed_time`` measures
    the device time between two records (plus submission latency).
    """

    def __init__(self, enable_timing: bool = False, blocking: bool = False, interprocess: bool = False):
        self.enable_timing = enable_timing
        self._time_ns: int | None = None

    def record(self, stream: Stream | None = None) -> None:
        if self.enable_timing:
            synchronize()
            self._time_ns = time.perf_counter_ns()
        else:
            _C._flush()

    def wait(self, stream: Stream | None = None) -> None:
        pass

    def query(self) -> bool:
        synchronize()
        return True

    def synchronize(self) -> None:
        synchronize()

    def elapsed_time(self, end_event: "Event") -> float:
        """Milliseconds between this event and ``end_event``."""
        if not (self.enable_timing and end_event.enable_timing):
            raise RuntimeError("Both events must be created with enable_timing=True")
        if self._time_ns is None or end_event._time_ns is None:
            raise RuntimeError("Both events must be recorded before calling elapsed_time")
        return (end_event._time_ns - self._time_ns) / 1e6

from . import profiler  # noqa: E402,F401
