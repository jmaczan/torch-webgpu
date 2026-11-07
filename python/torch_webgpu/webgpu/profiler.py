"""GPU kernel profiler for webgpu (timestamp queries).

    with torch.webgpu.profiler.profile() as prof:
        model(x)
    print(prof.table())

Every dispatch issued inside the block is timed on the GPU. Records carry
the kernel's name (eager kernels: the op family, e.g. "add" or "gemv";
compiled graphs: the Inductor kernel name), its workgroup grid and its GPU
time. Timing gives each dispatch its own compute pass, so the total is
slightly above an unprofiled run; host overhead is not included.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from .. import _C

__all__ = ["profile", "KernelRecord", "is_supported"]


def is_supported() -> bool:
    """Whether the adapter has the timestamp-query feature."""
    return bool(_C._profiler_supported())


@dataclass
class KernelRecord:
    name: str
    grid: tuple[int, int, int]
    gpu_us: float


class profile:
    def __init__(self) -> None:
        self.records: list[KernelRecord] = []

    def __enter__(self) -> "profile":
        _C._profiler_start()
        return self

    def __exit__(self, *exc) -> None:
        self.records = [KernelRecord(n, tuple(g), t) for n, g, t in _C._profiler_stop()]

    @property
    def total_us(self) -> float:
        return sum(r.gpu_us for r in self.records)

    def summary(self) -> list[tuple[str, int, float]]:
        """(name, calls, total GPU microseconds), slowest first."""
        acc: dict[str, list[float]] = defaultdict(lambda: [0, 0.0])
        for r in self.records:
            a = acc[r.name]
            a[0] += 1
            a[1] += r.gpu_us
        return sorted(((n, int(c), t) for n, (c, t) in acc.items()), key=lambda x: -x[2])

    def table(self, top: int = 25) -> str:
        total = self.total_us or 1.0
        lines = [f"{'kernel':40s} {'calls':>7s} {'total us':>11s} {'mean us':>9s} {'%':>6s}"]
        for name, calls, t in self.summary()[:top]:
            lines.append(f"{name[:40]:40s} {calls:7d} {t:11.1f} {t / calls:9.2f} {100 * t / total:6.1f}")
        lines.append(f"{'total':40s} {len(self.records):7d} {self.total_us:11.1f}")
        return "\n".join(lines)
