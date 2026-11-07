"""Runtime support for Inductor-generated WGSL kernels."""

from __future__ import annotations

import torch

from .. import _C

_MAX_GRID = 65535


def _word_atomic_fns(name: str, var: str, dtype: torch.dtype, off: str, add: bool) -> str:
    """st_<name> (and atomic_add_<name>) for a 32-bit dtype in an atomic binding."""
    ty = "f32" if dtype == torch.float32 else "i32"
    out = f"fn st_{name}(i: u32, v: {ty}) {{ atomicStore(&{var}[{off} + i], bitcast<u32>(v)); }}\n"
    if add:
        if ty == "f32":
            out += (
                f"fn atomic_add_{name}(i: u32, v: f32) {{\n"
                f"    let p = &{var}[{off} + i];\n"
                f"    var old = atomicLoad(p);\n"
                f"    loop {{\n"
                f"        let r = atomicCompareExchangeWeak(p, old, bitcast<u32>(bitcast<f32>(old) + v));\n"
                f"        if (r.exchanged) {{ break; }}\n"
                f"        old = r.old_value;\n"
                f"    }}\n"
                f"}}\n"
            )
        else:
            out += f"fn atomic_add_{name}(i: u32, v: i32) {{ atomicAdd(&{var}[{off} + i], bitcast<u32>(v)); }}\n"
    return out


_CT = {torch.float32: "f32", torch.float16: "f32", torch.bfloat16: "f32", torch.bool: "bool", torch.int64: "i64"}


def buffer_declarations(buffers, mapping, error_flag: bool, nparams: int) -> str:
    """WGSL bindings, uniform block and ld_/st_ helpers for tensors
    `buffers` = [(name, dtype, written, atomic_add)] bound per `mapping`
    (tensor index -> binding index)."""
    nb = max(mapping, default=-1) + 1
    written = [False] * nb
    atomic = [False] * nb
    for (name, dtype, w, add), b in zip(buffers, mapping):
        if w:
            written[b] = True
            if dtype.itemsize < 4 or add:
                atomic[b] = True
    s = []
    n = max(1, len(buffers))
    s.append("struct Params {")
    s.append(f"    offs: array<vec4<u32>, {(n + 3) // 4}>,")
    if nparams:
        s.append(f"    v: array<vec4<u32>, {(nparams + 3) // 4}>,")
    s.append("};")
    for b in range(nb):
        mode = "read_write" if written[b] else "read"
        elem = "atomic<u32>" if atomic[b] else "u32"
        s.append(f"@group(0) @binding({b}) var<storage, {mode}> bind{b}: array<{elem}>;")
    nbind = nb
    if error_flag:
        s.append(f"@group(0) @binding({nbind}) var<storage, read_write> err_flag: array<atomic<u32>>;")
        nbind += 1
    s.append(f"@group(0) @binding({nbind}) var<uniform> P: Params;")
    for i, ((name, dtype, w, add), b) in enumerate(zip(buffers, mapping)):
        var = f"bind{b}"
        off = f"P.offs[{i // 4}][{i % 4}]"
        ty = _CT.get(dtype, "i32")
        s.append(_C._wgsl_load_fn(f"raw_ld_{name}", var, dtype, atomic[b]))
        s.append(f"fn ld_{name}(i: u32) -> {ty} {{ return raw_ld_{name}({off} + i); }}")
        if w:
            if atomic[b] and dtype.itemsize == 4:
                s.append(_word_atomic_fns(name, var, dtype, off, add))
            else:
                s.append(_C._wgsl_store_fn(f"raw_st_{name}", var, dtype))
                s.append(f"fn st_{name}(i: u32, v: {ty}) {{ raw_st_{name}({off} + i, v); }}")
    return "\n".join(s) + "\n"


class WGSLKernelRunner:
    """A compiled WGSL kernel. Called with (tensors, params, xnumel): tensors
    are bound in order, params are 32-bit integer uniforms, and xnumel
    elements of the x range are covered `block` per workgroup.

    `source` contains the marker //@BUFFERS@ where buffer declarations go;
    they depend on which tensors share a device buffer, so a variant is
    compiled per aliasing pattern (the common one: none)."""

    __slots__ = ("name", "source", "buffers", "nparams", "block", "error_flag", "kid", "variants")

    def __init__(self, name: str, source: str, buffers, nparams: int, block: int, error_flag: bool = False) -> None:
        self.name = name
        self.source = source
        self.buffers = [(n, getattr(torch, d), w, a) for n, d, w, a in buffers]
        self.nparams = nparams
        self.block = block
        self.error_flag = error_flag
        self.variants: dict[tuple[int, ...], int] = {}
        self.kid = self._compile(tuple(range(len(self.buffers))))

    def _compile(self, mapping: tuple[int, ...]) -> int:
        kid = self.variants.get(mapping)
        if kid is None:
            decls = buffer_declarations(self.buffers, mapping, self.error_flag, self.nparams)
            kid = _C._wgsl_compile(self.source.replace("//@BUFFERS@", decls), self.name)
            self.variants[mapping] = kid
        return kid

    def __call__(self, tensors, params, xnumel: int) -> None:
        nwg = -(-int(xnumel) // self.block)
        if nwg <= 0:
            return
        gx = min(nwg, _MAX_GRID)
        grid = (gx, -(-nwg // gx), 1)
        params = [int(p) for p in params]
        # Inductor feeds zero-dim CPU tensors (e.g. optimizer step counts)
        # to device kernels. Stage them on the device; written ones are
        # copied back after the launch.
        staged = None
        for i, t in enumerate(tensors):
            if t.device.type != "webgpu":
                if staged is None:
                    tensors = list(tensors)
                    staged = []
                tensors[i] = t.to("webgpu")
                if self.buffers[i][2]:
                    staged.append((t, tensors[i]))
        if not _C._wgsl_launch(self.kid, tensors, params, grid, self.error_flag, False):
            kid = self._compile(tuple(_C._wgsl_binding_map(tensors)))
            _C._wgsl_launch(kid, tensors, params, grid, self.error_flag, True)
        if staged:
            for cpu, dev in staged:
                cpu.copy_(dev)
