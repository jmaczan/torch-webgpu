"""Inductor code generation for WebGPU: fused WGSL kernels.

Modeled on Inductor's Metal backend (torch/_inductor/codegen/mps.py), built
on SIMDKernel / SIMDScheduling. Kernel shapes:

* pointwise: one invocation per element of the (flattened) x range,
  256 invocations per workgroup.
* reduction: a workgroup of 256 invocations handles XB = 256 / RB outputs;
  RB invocations per output stride over the reduction range (a loop per
  reduction pass), then combine through workgroup memory.

Storage follows the eager runtime (csrc/kernels/wgsl.h): every tensor is
bound as array<u32> and accessed through generated ld_* / st_* helpers by
element index plus a per-launch element offset (P.offs). Values are
computed as f32 (all float dtypes), i32 (<= 32-bit integers), bool, or
emulated two's-complement i64 (vec2<u32>), so int64 stays exact.

Indexing is 32-bit (i32); kernels over >= 2**31 elements are rejected at
codegen time.
"""

from __future__ import annotations

import itertools
import math
import struct
from typing import Any

import sympy
import torch
from torch._inductor.codegen.common import (
    BackendFeature,
    CSEVariable,
    DeferredLine,
    IndentedBuffer,
    OpOverrides,
)
from torch._inductor.codegen.simd import IterationRangesEntry, SIMDKernel, SIMDScheduling
from torch._inductor.ops_handler import StoreMode
from torch._inductor.utils import get_bounds_index_expr, get_kernel_metadata, sympy_index_symbol
from torch._inductor.virtualized import V, ops
from torch.utils._ordered_set import OrderedSet
from torch.utils._sympy.printers import ExprPrinter
from sympy.printing.precedence import PRECEDENCE

from .. import _C

WG = 256  # invocations per workgroup
MAX_INDEX = 2**31 - 1

# ---------------------------------------------------------------------------
# dtypes
# ---------------------------------------------------------------------------

_CT = {
    torch.float32: "f32",
    torch.float16: "f32",
    torch.bfloat16: "f32",
    torch.int32: "i32",
    torch.int16: "i32",
    torch.int8: "i32",
    torch.uint8: "i32",
    torch.bool: "bool",
    torch.int64: "i64",
}


def ct(dtype: torch.dtype) -> str:
    """WGSL compute type for a torch dtype."""
    try:
        return _CT[dtype]
    except KeyError:
        raise NotImplementedError(f"WebGPU Inductor backend: dtype {dtype} is not supported in kernels") from None


def _dtype(x: Any) -> torch.dtype | None:
    return getattr(x, "dtype", None)


def _rank(dtype: torch.dtype) -> int:
    if dtype == torch.bool:
        return 0
    if dtype.is_floating_point:
        return 3
    if dtype == torch.int64:
        return 2
    return 1


def _promote(*dtypes: torch.dtype | None) -> torch.dtype | None:
    ds = [d for d in dtypes if d is not None]
    if not ds:
        return None
    out = ds[0]
    for d in ds[1:]:
        if _rank(d) > _rank(out) or (_rank(d) == _rank(out) and d.itemsize > out.itemsize):
            out = d
    return out


def convert(x: Any, src: torch.dtype | None, dst: torch.dtype) -> str:
    """WGSL expression converting x (of dtype src) to dst's compute type and
    value range (narrow integer and half-float dtypes are rounded / wrapped)."""
    if src is None:
        return str(x)
    s, d = ct(src), ct(dst)
    e = str(x)
    if s != d:
        table = {
            ("f32", "i32"): f"i32({e})",
            ("f32", "bool"): f"ne_f({e}, 0.0f)",
            ("f32", "i64"): f"i64_from_f32({e})",
            ("i32", "f32"): f"f32({e})",
            ("i32", "bool"): f"({e} != 0i)",
            ("i32", "i64"): f"i64_from_i32({e})",
            ("bool", "f32"): f"select(0.0f, 1.0f, {e})",
            ("bool", "i32"): f"select(0i, 1i, {e})",
            ("bool", "i64"): f"i64(select(0u, 1u, {e}), 0u)",
            ("i64", "f32"): f"i64_to_f32({e})",
            ("i64", "i32"): f"i64_to_i32({e})",
            ("i64", "bool"): f"(!i64_is_zero({e}))",
        }
        e = table[(s, d)]
    # Round / wrap into the narrower destination dtype.
    if dst == src:
        return e
    if dst == torch.float16:
        return f"unpack2x16float(pack2x16float(vec2<f32>({e}, 0.0f))).x"
    if dst == torch.bfloat16:
        return f"bitcast<f32>(f32_to_bf16_bits({e}) << 16u)"
    if dst == torch.uint8:
        return f"({e} & 255i)"
    if dst == torch.int8:
        return f"(({e} << 24u) >> 24u)"
    if dst == torch.int16:
        return f"(({e} << 16u) >> 16u)"
    return e


def _f32_literal(v: float) -> str:
    if math.isnan(v):
        return "nan_f()"
    if math.isinf(v):
        return "inf_f()" if v > 0 else "(-inf_f())"
    r = struct.unpack("f", struct.pack("f", v))[0] if abs(v) < 3.4028234663852886e38 else v
    if math.isinf(r) or abs(v) >= 3.4028235677973366e38:
        return "inf_f()" if v > 0 else "(-inf_f())"
    if r == 0.0:
        return "-0.0f" if math.copysign(1.0, r) < 0 else "0.0f"
    if r != v:
        # Distinct Python values that round to the same f32 must print
        # differently: Inductor's CSE would merge them and intersect their
        # (disjoint) value ranges.
        return f"{r!r}f /*{v!r}*/"
    return f"{r!r}f"


def literal(v: Any, dtype: torch.dtype) -> str:
    t = ct(dtype)
    if t == "f32":
        return _f32_literal(float(v))
    if t == "bool":
        return "true" if v else "false"
    if t == "i32":
        v = int(v)
        if v == -(2**31):
            return "(-2147483647i - 1i)"
        return f"{v}i"
    v = int(v) & (2**64 - 1)
    return f"i64({v & 0xFFFFFFFF}u, {v >> 32}u)"


# ---------------------------------------------------------------------------
# index expressions (i32)
# ---------------------------------------------------------------------------


_PRIVATE_SPECIALS = """var<private> NAN_BITS: u32 = 0x7fc00000u;
var<private> INF_BITS: u32 = 0x7f800000u;
fn nan_f() -> f32 { return bitcast<f32>(NAN_BITS); }
fn inf_f() -> f32 { return bitcast<f32>(INF_BITS); }"""

# Fused kernels compute with special values in registers. Constants the
# driver can see (even through never-written private variables) are folded
# under its fast-math assumptions (x * inf -> garbage), so NaN / Inf and the
# opq() barrier come from the uniform block, which the runtime fills with
# KERNEL_CONSTANTS.
_UNIFORM_SPECIALS = """fn nan_f() -> f32 { return bitcast<f32>(P.k.x); }
fn inf_f() -> f32 { return bitcast<f32>(P.k.y); }
// Identity the compiler can't see through (P.k.z is 0).
fn opq(x: f32) -> f32 { return bitcast<f32>(bitcast<u32>(x) | P.k.z); }"""


def opaque_math() -> str:
    math = _C._wgsl_math()
    assert _PRIVATE_SPECIALS in math, "math prelude changed: update _PRIVATE_SPECIALS"
    return math.replace(_PRIVATE_SPECIALS, _UNIFORM_SPECIALS)


class WGSLExprPrinter(ExprPrinter):
    def _print_Integer(self, expr: sympy.Expr) -> str:
        v = int(expr)
        if not -(2**31) <= v <= MAX_INDEX:
            raise NotImplementedError(f"WebGPU Inductor backend: index constant {v} exceeds 32 bits")
        return f"{v}" if v != -(2**31) else "(-2147483647 - 1)"

    def _print_Symbol(self, expr: sympy.Expr) -> str:
        return str(expr)

    def _print_FloorDiv(self, expr: sympy.Expr) -> str:
        x, div = expr.args
        if expr.args[0].is_nonnegative and div.is_positive:
            return f"(({self.doprint(x)}) / ({self.doprint(div)}))"
        return f"div_floor_i({self.doprint(x)}, {self.doprint(div)})"

    _print_CleanDiv = _print_FloorDiv

    def _print_ModularIndexing(self, expr: sympy.Expr) -> str:
        x, div, mod = expr.args
        xs = self.doprint(x)
        if div != 1:
            xs = f"(({xs}) / ({self.doprint(div)}))"
        return f"(({xs}) % ({self.doprint(mod)}))"

    def _print_PythonMod(self, expr: sympy.Expr) -> str:
        a, b = expr.args
        if a.is_nonnegative and b.is_positive:
            return f"(({self.doprint(a)}) % ({self.doprint(b)}))"
        return f"remainder_i({self.doprint(a)}, {self.doprint(b)})"

    def _print_Min(self, expr: sympy.Expr) -> str:
        out = self.doprint(expr.args[0])
        for a in expr.args[1:]:
            out = f"min({out}, {self.doprint(a)})"
        return out

    def _print_Max(self, expr: sympy.Expr) -> str:
        out = self.doprint(expr.args[0])
        for a in expr.args[1:]:
            out = f"max({out}, {self.doprint(a)})"
        return out

    def _print_Abs(self, expr: sympy.Expr) -> str:
        return f"abs({self.doprint(expr.args[0])})"

    def _print_Not(self, expr: sympy.Expr) -> str:
        return f"!({self.doprint(expr.args[0])})"

    def _print_And(self, expr: sympy.Expr) -> str:
        return " && ".join(f"({self.doprint(a)})" for a in expr.args)

    def _print_Or(self, expr: sympy.Expr) -> str:
        return " || ".join(f"({self.doprint(a)})" for a in expr.args)

    def _print_Where(self, expr: sympy.Expr) -> str:
        c, p, q = (self.doprint(a) for a in expr.args)
        return f"select({q}, {p}, {c})"

    def _print_PowByNatural(self, expr: sympy.Expr) -> str:
        base, e = expr.args
        if e.is_Integer and 0 <= int(e) <= 8:
            return "(" + ("*".join([f"({self.doprint(base)})"] * int(e)) or "1") + ")"
        return f"ipow({self.doprint(base)}, {self.doprint(e)})"

    def _print_ToFloat(self, expr: sympy.Expr) -> str:
        return f"f32({self.doprint(expr.args[0])})"

    def _print_IntTrueDiv(self, expr: sympy.Expr) -> str:
        a, b = expr.args
        return f"(f32({self.doprint(a)}) / f32({self.doprint(b)}))"

    def _print_FloatTrueDiv(self, expr: sympy.Expr) -> str:
        a, b = expr.args
        return f"(({self.doprint(a)}) / ({self.doprint(b)}))"

    def _print_FloorToInt(self, expr: sympy.Expr) -> str:
        return f"i32(floor({self.doprint(expr.args[0])}))"

    _print_floor = _print_FloorToInt

    def _print_CeilToInt(self, expr: sympy.Expr) -> str:
        return f"i32(ceil({self.doprint(expr.args[0])}))"

    _print_ceiling = _print_CeilToInt

    def _print_TruncToInt(self, expr: sympy.Expr) -> str:
        return f"i32(trunc({self.doprint(expr.args[0])}))"

    def _print_RoundToInt(self, expr: sympy.Expr) -> str:
        return f"i32(round_half_even({self.doprint(expr.args[0])}))"

    def _print_Float(self, expr: sympy.Expr) -> str:
        if expr.is_integer:
            return str(int(expr))
        return _f32_literal(float(expr))

    def _print_Infinity(self, expr: sympy.Expr) -> str:
        return "inf_f()"

    def _print_NegativeInfinity(self, expr: sympy.Expr) -> str:
        return "(-inf_f())"

    def _print_BooleanTrue(self, expr: sympy.Expr) -> str:
        return "true"

    def _print_BooleanFalse(self, expr: sympy.Expr) -> str:
        return "false"


wexpr = WGSLExprPrinter().doprint


# ---------------------------------------------------------------------------
# op overrides
# ---------------------------------------------------------------------------


def _arg(a: Any, dt: torch.dtype) -> str:
    """a as a WGSL expression of dtype dt's compute type. Python scalars
    become literals; CSE variables are converted from their dtype."""
    if isinstance(a, (bool, int, float)):
        return literal(a, dt)
    if _dtype(a) is not None:
        return convert(a, _dtype(a), dt)
    return str(a)


def _scalar_dtype(a: Any) -> torch.dtype | None:
    if isinstance(a, bool):
        return torch.bool
    if isinstance(a, int):
        return torch.int32
    if isinstance(a, float):
        return torch.float32
    return None


def _unify(*args: Any) -> tuple[list[str], torch.dtype | None]:
    dt = _promote(*(_dtype(a) for a in args))
    if dt is None:
        dt = _promote(*(_scalar_dtype(a) for a in args))
    elif dt in (torch.bool,) and any(isinstance(a, (int, float)) and not isinstance(a, bool) for a in args):
        dt = _promote(dt, *(_scalar_dtype(a) for a in args))
    if dt is None:
        return [str(a) for a in args], None
    return [_arg(a, dt) for a in args], dt


def _float_args(*args: Any) -> list[str]:
    return [_arg(a, torch.float32) for a in args]


_ZEROS = ("0.0f", "-0.0f")


def _binary(op: str, i64fn: str, a: Any, b: Any, boolop: str | None = None) -> str:
    (x, y), dt = _unify(a, b)
    if dt is None:
        return f"({x} {op} {y})"
    t = ct(dt)
    if t == "i64":
        return f"{i64fn}({x}, {y})"
    if t == "bool":
        if boolop is None:
            # Arithmetic on bools: compute on i32 (result converted back by the
            # caller's dtype).
            xi, yi = convert(a, torch.bool, torch.int32), convert(b, torch.bool, torch.int32)
            return f"(({xi} {op} {yi}) != 0i)"
        return f"({x} {boolop} {y})"
    if t == "f32" and x == y:
        # x - x (or x + x, x * x) on one value: fast-math folds x - x to 0,
        # which is wrong for Inf / NaN.
        y = f"opq({y})"
    elif t == "f32" and op == "*" and (x in _ZEROS or y in _ZEROS):
        # x * 0 folds to 0 (NaN for x = Inf / NaN).
        x, y = (f"opq({x})", y) if x in _ZEROS else (x, f"opq({y})")
    return f"({x} {op} {y})"


def _compare(op: str, i64fn: str, a: Any, b: Any) -> str:
    (x, y), dt = _unify(a, b)
    if dt is not None and ct(dt) == "i64":
        return f"{i64fn}({x}, {y})"
    if dt is not None and ct(dt) == "f32" and op == "!=":
        return f"ne_f({x}, {y})"
    if dt is not None and ct(dt) == "bool" and op in ("<", "<=", ">", ">="):
        # WGSL orders no bools; compare as 0 / 1.
        return f"(select(0i, 1i, {x}) {op} select(0i, 1i, {y}))"
    return f"({x} {op} {y})"


def _float_unary(fn: str):
    def impl(x: Any) -> str:
        (e,) = _float_args(x)
        return fn.format(e)

    return staticmethod(impl)


def _special(fn: str, *deps: str):
    """A special function from the eager kernels' library (kernels/special.h)."""

    def impl(*args: Any) -> str:
        V.kernel.helpers.update(f"special:{d}" for d in (fn, *deps))
        return f"{fn}({', '.join(_float_args(*args))})"

    return staticmethod(impl)


# polygamma(n, x) for a runtime order n: (-1)^(n+1) n! zeta(n+1, x).
POLYGAMMA_WGSL = """
fn polygamma_n(n: f32, x: f32) -> f32 {
    if (n == 0.0f) { return digamma_f(x); }
    if (n == 1.0f) { return trigamma_f(x); }
    let fac = select(-1.0f, 1.0f, (i32(n) & 1) == 1) * exp(lgamma_f(n + 1.0f));
    return polygamma_f(n + 1.0f, fac, x);
}
"""


class WGSLOverrides(OpOverrides):
    """Expression generation. Every argument is a CSE variable whose dtype
    (from Inductor's dtype propagation) says which compute type it holds."""

    @staticmethod
    def to_dtype(x: Any, dtype: torch.dtype, src_dtype: torch.dtype | None = None, use_compute_types: bool = True) -> str:
        # The variable's own dtype decides its WGSL type; Inductor's src_dtype
        # hint can disagree after index-dtype narrowing (int64 vs int32).
        return convert(x, _dtype(x) or src_dtype, dtype)

    @staticmethod
    def to_dtype_bitcast(x: Any, dtype: torch.dtype, src_dtype: torch.dtype) -> str:
        s, d = ct(src_dtype), ct(dtype)
        if src_dtype.itemsize == 4 and dtype.itemsize == 4 and s in ("f32", "i32") and d in ("f32", "i32"):
            return f"bitcast<{d}>({x})"
        if src_dtype.itemsize == 2 and dtype.itemsize == 2:
            # 16-bit floats are held as f32; integers as sign-extended i32.
            if src_dtype == torch.float16:
                return convert(f"i32(pack2x16float(vec2<f32>({x}, 0.0f)) & 0xffffu)", torch.int32, dtype)
            if src_dtype == torch.bfloat16:
                return convert(f"i32(f32_to_bf16_bits({x}))", torch.int32, dtype)
            if dtype == torch.float16:
                return f"unpack2x16float(u32({x}) & 0xffffu).x"
            if dtype == torch.bfloat16:
                return f"bitcast<f32>((u32({x}) & 0xffffu) << 16u)"
        if src_dtype.itemsize == 1 and dtype.itemsize == 1:
            return convert(x, torch.int32, dtype)
        raise NotImplementedError(f"WebGPU Inductor backend: bitcast {src_dtype} -> {dtype}")

    @staticmethod
    def constant(val: Any, dtype: torch.dtype) -> str:
        lit = literal(val, dtype)
        # Float zero constants are opaque: fast-math folds x * 0 to 0.
        return f"opq({lit})" if lit in _ZEROS else lit

    @staticmethod
    def index_expr(expr: sympy.Expr, dtype: torch.dtype) -> Any:
        # Index math is i32; Inductor's index-dtype strength reduction may
        # record the result as int32 or int64, so make the variable's dtype
        # explicit and convert.
        s = V.kernel.index_to_str(V.kernel.prepare_indexing(expr))
        var = V.kernel.cse.generate(
            V.kernel.compute, f"i32({s})", bounds=get_bounds_index_expr(expr), dtype=torch.int32
        )
        return ops.to_dtype(var, dtype)

    @staticmethod
    def value_expr(expr: sympy.Expr, dtype: torch.dtype) -> str:
        return WGSLOverrides.index_expr(expr, dtype)

    @staticmethod
    def masked(mask: CSEVariable, body: Any, other: Any) -> str:
        kernel = V.kernel
        scoped = IndentedBuffer()
        with kernel.swap_buffers(scoped), scoped.indent():
            kernel.cse.iter_buffer_ids = itertools.count()
            kernel.cse.name_prefix += "_scoped"
            rc = body()
        dt = _dtype(rc) or torch.float32
        if isinstance(other, float) or isinstance(other, int) or isinstance(other, bool):
            other_s = literal(other, dt)
        else:
            other_s = convert(other, _dtype(other), dt)
        key = f"{mask}:{scoped.getvalue()}:{rc}:{other_s}"
        var = kernel.cse.try_get(key)
        if not var:
            var = kernel.cse.newvar(dtype=dt)
            kernel.cse.put(key, var)
            kernel.compute.writeline(f"var {var}: {ct(dt)} = {other_s};")
            kernel.compute.writeline(f"if ({mask}) {{")
            with kernel.compute.indent():
                kernel.compute.splice(scoped)
                kernel.compute.writeline(f"{var} = {convert(rc, _dtype(rc), dt)};")
            kernel.compute.writeline("}")
        return var

    @staticmethod
    def where(c: Any, a: Any, b: Any) -> str:
        (x, y), _ = _unify(a, b)
        cond = convert(c, _dtype(c), torch.bool) if _dtype(c) is not None else str(c)
        return f"select({y}, {x}, {cond})"

    @staticmethod
    def indirect_indexing(var: Any, size: Any, check: bool = True, wrap_neg: bool = True) -> sympy.Symbol:
        if _dtype(var) is not None and ct(_dtype(var)) != "i32":
            var = V.kernel.cse.generate(
                V.kernel.compute, convert(var, _dtype(var), torch.int32), bounds=var.bounds, dtype=torch.int32
            )
        return sympy_index_symbol(str(var))

    # -- arithmetic ---------------------------------------------------------
    @staticmethod
    def add(a: Any, b: Any) -> str:
        return _binary("+", "i64_add", a, b, boolop="||")

    @staticmethod
    def sub(a: Any, b: Any) -> str:
        return _binary("-", "i64_sub", a, b)

    @staticmethod
    def mul(a: Any, b: Any) -> str:
        return _binary("*", "i64_mul", a, b, boolop="&&")

    @staticmethod
    def truediv(a: Any, b: Any) -> str:
        # Both operands go through opq(): drivers compile with fast-math and
        # would fold x / x to 1, 0 / x to 0 and x / 0.0 to garbage.
        x, y = _float_args(a, b)
        return f"(opq({x}) / opq({y}))"

    int_truediv = truediv

    @staticmethod
    def div_rn(a: Any, b: Any) -> str:
        x, y = _float_args(a, b)
        return f"(opq({x}) / opq({y}))"

    @staticmethod
    def floordiv(a: Any, b: Any) -> str:
        (x, y), dt = _unify(a, b)
        t = ct(dt) if dt is not None else "i32"
        if t == "i64":
            return f"i64_div_floor({x}, {y})"
        if t == "f32":
            return f"div_floor_f({x}, {y})"
        return f"div_floor_i({x}, {y})"

    @staticmethod
    def truncdiv(a: Any, b: Any) -> str:
        (x, y), dt = _unify(a, b)
        t = ct(dt) if dt is not None else "i32"
        if t == "i64":
            return f"i64_div_trunc({x}, {y})"
        if t == "f32":
            return f"trunc(opq({x}) / opq({y}))"
        return f"({x} / {y})"

    @staticmethod
    def mod(a: Any, b: Any) -> str:
        (x, y), dt = _unify(a, b)
        t = ct(dt) if dt is not None else "i32"
        if t == "i64":
            return f"i64_rem_trunc({x}, {y})"
        if t == "f32":
            return f"fmod_f({x}, {y})"
        return f"({x} % {y})"

    @staticmethod
    def fmod(a: Any, b: Any) -> str:
        return WGSLOverrides.mod(a, b)

    @staticmethod
    def remainder(a: Any, b: Any) -> str:
        (x, y), dt = _unify(a, b)
        t = ct(dt) if dt is not None else "i32"
        if t == "i64":
            return f"i64_remainder({x}, {y})"
        if t == "f32":
            return f"remainder_f({x}, {y})"
        return f"remainder_i({x}, {y})"

    @staticmethod
    def neg(x: Any) -> str:
        t = ct(_dtype(x)) if _dtype(x) is not None else "f32"
        if t == "i64":
            return f"i64_neg({x})"
        return f"(-{x})"

    @staticmethod
    def abs(x: Any) -> str:
        t = ct(_dtype(x)) if _dtype(x) is not None else "f32"
        if t == "i64":
            return f"i64_abs({x})"
        if t == "bool":
            return str(x)
        return f"abs({x})"

    @staticmethod
    def sign(x: Any) -> str:
        t = ct(_dtype(x)) if _dtype(x) is not None else "f32"
        if t == "i64":
            return f"i64_sign({x})"
        if t == "bool":
            return str(x)
        if t == "f32":
            return f"select(sign({x}), {x}, isnan_f({x}))"
        return f"sign({x})"

    @staticmethod
    def signbit(x: Any) -> str:
        t = ct(_dtype(x)) if _dtype(x) is not None else "f32"
        if t == "f32":
            return f"signbit_f({x})"
        if t == "i64":
            return f"i64_is_neg({x})"
        if t == "bool":
            return "false"
        return f"({x} < 0i)"

    @staticmethod
    def minimum(a: Any, b: Any) -> str:
        (x, y), dt = _unify(a, b)
        t = ct(dt) if dt is not None else "f32"
        return {"i64": f"i64_min({x}, {y})", "f32": f"min_nan({x}, {y})", "bool": f"({x} && {y})"}.get(
            t, f"min({x}, {y})"
        )

    @staticmethod
    def maximum(a: Any, b: Any) -> str:
        (x, y), dt = _unify(a, b)
        t = ct(dt) if dt is not None else "f32"
        return {"i64": f"i64_max({x}, {y})", "f32": f"max_nan({x}, {y})", "bool": f"({x} || {y})"}.get(
            t, f"max({x}, {y})"
        )

    @staticmethod
    def pow(a: Any, b: Any) -> str:
        (x, y), dt = _unify(a, b)
        t = ct(dt) if dt is not None else "f32"
        if t == "f32":
            return f"pow_f({x}, {y})"
        if t == "i64":
            return f"i64_pow({x}, {y})"
        if t == "i32":
            return f"ipow({x}, {y})"
        raise NotImplementedError("pow on bool")

    @staticmethod
    def fma(a: Any, b: Any, c: Any) -> str:
        x, y, z = _float_args(a, b, c)
        return f"fma({x}, {y}, {z})"

    # -- comparisons ---------------------------------------------------------
    @staticmethod
    def eq(a: Any, b: Any) -> str:
        return _compare("==", "i64_eq", a, b)

    @staticmethod
    def ne(a: Any, b: Any) -> str:
        (x, y), dt = _unify(a, b)
        if dt is not None and ct(dt) == "i64":
            return f"(!i64_eq({x}, {y}))"
        return _compare("!=", "", a, b)

    @staticmethod
    def lt(a: Any, b: Any) -> str:
        return _compare("<", "i64_lt", a, b)

    @staticmethod
    def le(a: Any, b: Any) -> str:
        return _compare("<=", "i64_le", a, b)

    @staticmethod
    def gt(a: Any, b: Any) -> str:
        return _compare(">", "i64_gt", a, b)

    @staticmethod
    def ge(a: Any, b: Any) -> str:
        return _compare(">=", "i64_ge", a, b)

    # -- logical / bitwise ---------------------------------------------------
    @staticmethod
    def logical_not(a: Any) -> str:
        return f"(!{convert(a, _dtype(a), torch.bool)})"

    @staticmethod
    def logical_and(a: Any, b: Any) -> str:
        return f"({convert(a, _dtype(a), torch.bool)} && {convert(b, _dtype(b), torch.bool)})"

    @staticmethod
    def logical_or(a: Any, b: Any) -> str:
        return f"({convert(a, _dtype(a), torch.bool)} || {convert(b, _dtype(b), torch.bool)})"

    @staticmethod
    def logical_xor(a: Any, b: Any) -> str:
        return f"({convert(a, _dtype(a), torch.bool)} != {convert(b, _dtype(b), torch.bool)})"

    @staticmethod
    def bitwise_not(a: Any) -> str:
        t = ct(_dtype(a)) if _dtype(a) is not None else "i32"
        if t == "bool":
            return f"(!{a})"
        if t == "i64":
            return f"(~{a})"
        return f"(~{a})"

    @staticmethod
    def _bitwise(op: str, a: Any, b: Any) -> str:
        (x, y), dt = _unify(a, b)
        t = ct(dt) if dt is not None else "i32"
        if t == "bool":
            return f"({x} {op}{op if op != '^' else ''} {y})" if op != "^" else f"({x} != {y})"
        return f"({x} {op} {y})"

    @staticmethod
    def bitwise_and(a: Any, b: Any) -> str:
        return WGSLOverrides._bitwise("&", a, b)

    @staticmethod
    def bitwise_or(a: Any, b: Any) -> str:
        return WGSLOverrides._bitwise("|", a, b)

    @staticmethod
    def bitwise_xor(a: Any, b: Any) -> str:
        return WGSLOverrides._bitwise("^", a, b)

    @staticmethod
    def and_(a: Any, b: Any) -> str:
        return WGSLOverrides._bitwise("&", a, b)

    @staticmethod
    def or_(a: Any, b: Any) -> str:
        return WGSLOverrides._bitwise("|", a, b)

    @staticmethod
    def xor(a: Any, b: Any) -> str:
        return WGSLOverrides._bitwise("^", a, b)

    @staticmethod
    def lshift(a: Any, b: Any) -> str:
        return WGSLOverrides.bitwise_left_shift(a, b)

    @staticmethod
    def rshift(a: Any, b: Any) -> str:
        return WGSLOverrides.bitwise_right_shift(a, b)

    @staticmethod
    def bitwise_left_shift(a: Any, b: Any) -> str:
        t = ct(_dtype(a)) if _dtype(a) is not None else "i32"
        sh = convert(b, _dtype(b), torch.int32)
        if t == "i64":
            return f"i64_shl({a}, u32({sh}))"
        # Shifts by >= the bit width give 0 in PyTorch.
        return f"select(0i, ({a} << u32({sh})), u32({sh}) < 32u)"

    @staticmethod
    def bitwise_right_shift(a: Any, b: Any) -> str:
        t = ct(_dtype(a)) if _dtype(a) is not None else "i32"
        sh = convert(b, _dtype(b), torch.int32)
        if t == "i64":
            return f"i64_shr({a}, u32({sh}))"
        return f"({a} >> min(u32({sh}), 31u))"

    # -- floating point --------------------------------------------------------
    @staticmethod
    def isnan(x: Any) -> str:
        if ct(_dtype(x) or torch.float32) != "f32":
            return "false"
        return f"isnan_f({x})"

    @staticmethod
    def isinf(x: Any) -> str:
        if ct(_dtype(x) or torch.float32) != "f32":
            return "false"
        return f"isinf_f({x})"

    @staticmethod
    def isfinite(x: Any) -> str:
        if ct(_dtype(x) or torch.float32) != "f32":
            return "true"
        return f"isfinite_f({x})"

    exp = _float_unary("exp({})")
    exp2 = _float_unary("exp2({})")
    expm1 = _float_unary("expm1_f({})")
    log = _float_unary("log({})")
    log2 = _float_unary("log2({})")
    log10 = _float_unary("(log({}) * 0.4342944819032518f)")
    log1p = _float_unary("log1p_f({})")
    sqrt = _float_unary("sqrt({})")
    rsqrt = _float_unary("inverseSqrt({})")
    sin = _float_unary("sin_f({})")
    cos = _float_unary("cos_f({})")
    tan = _float_unary("tan_f({})")
    asin = _float_unary("asin({})")
    acos = _float_unary("acos({})")
    atan = _float_unary("atan({})")
    sinh = _float_unary("sinh({})")
    cosh = _float_unary("cosh({})")
    tanh = _float_unary("tanh_f({})")
    asinh = _float_unary("asinh({})")
    acosh = _float_unary("acosh({})")
    atanh = _float_unary("atanh({})")
    erf = _float_unary("erf_f({})")
    erfc = _float_unary("erfc_f({})")
    erfinv = _float_unary("erfinv_f({})")
    sigmoid = _float_unary("sigmoid_f({})")
    reciprocal = _float_unary("(1.0f / opq({}))")

    # Special functions (shared with the eager kernels).
    lgamma = _special("lgamma_f")
    digamma = _special("digamma_f")
    i0 = _special("i0_f")
    i0e = _special("i0e_f")
    i1 = _special("i1_f")
    i1e = _special("i1e_f")
    ndtri = _special("ndtri_f")
    log_ndtr = _special("log_ndtr_f")
    erfcx = _special("erfcx_f")
    zeta = _special("zeta_f")

    @staticmethod
    def ndtr(x: Any) -> str:
        (e,) = _float_args(x)
        return f"(0.5f * erfc_f({e} * -0.7071067811865476f))"

    @staticmethod
    def polygamma(n: Any, x: Any) -> str:
        V.kernel.helpers.update(["special:digamma_f", "special:trigamma_f", "special:lgamma_f",
                                 "special:polygamma_f", "polygamma_n"])
        return f"polygamma_n({', '.join(_float_args(n, x))})"

    @staticmethod
    def frexp(x: Any) -> tuple[Any, Any]:
        V.kernel.helpers.add("special:frexp_f")
        (e,) = _float_args(x)
        fr = V.kernel.cse.generate(V.kernel.compute, f"frexp_f({e})", dtype=torch.int32)
        mant = V.kernel.cse.generate(V.kernel.compute, f"bitcast<f32>({fr}.x)", dtype=torch.float32)
        expo = V.kernel.cse.generate(V.kernel.compute, f"bitcast<i32>({fr}.y)", dtype=torch.int32)
        return mant, expo

    @staticmethod
    def nextafter(a: Any, b: Any) -> str:
        # float32 bit stepping (other dtypes fall back to eager: see
        # _inductor/__init__.py).
        x, y = _float_args(a, b)
        return (f"select(select(select(bitcast<f32>(select(bitcast<u32>({x}) - 1u, bitcast<u32>({x}) + 1u, "
                f"({x} < {y}) == ({x} > 0.0f))), bitcast<f32>(select(1u, 0x80000001u, {y} < 0.0f)), {x} == 0.0f), "
                f"{y}, {x} == {y}), nan_f(), isnan_f({x}) || isnan_f({y}))")

    @staticmethod
    def _rounding(fn: str, x: Any) -> str:
        if ct(_dtype(x) or torch.float32) != "f32":
            return str(x)
        return f"{fn}({x})"

    @staticmethod
    def floor(x: Any) -> str:
        return WGSLOverrides._rounding("floor", x)

    @staticmethod
    def ceil(x: Any) -> str:
        return WGSLOverrides._rounding("ceil", x)

    @staticmethod
    def trunc(x: Any) -> str:
        return WGSLOverrides._rounding("trunc", x)

    @staticmethod
    def round(x: Any) -> str:
        return WGSLOverrides._rounding("round_half_even", x)

    @staticmethod
    def floor_to_int(x: Any, dtype: torch.dtype) -> str:
        return convert(f"floor({_float_args(x)[0]})", torch.float32, dtype)

    @staticmethod
    def ceil_to_int(x: Any, dtype: torch.dtype) -> str:
        return convert(f"ceil({_float_args(x)[0]})", torch.float32, dtype)

    @staticmethod
    def trunc_to_int(x: Any, dtype: torch.dtype) -> str:
        return convert(f"trunc({_float_args(x)[0]})", torch.float32, dtype)

    @staticmethod
    def round_to_int(x: Any, dtype: torch.dtype) -> str:
        return convert(f"round_half_even({_float_args(x)[0]})", torch.float32, dtype)

    @staticmethod
    def atan2(a: Any, b: Any) -> str:
        x, y = _float_args(a, b)
        return f"atan2({x}, {y})"

    @staticmethod
    def hypot(a: Any, b: Any) -> str:
        x, y = _float_args(a, b)
        return f"hypot_f({x}, {y})"

    @staticmethod
    def copysign(a: Any, b: Any) -> str:
        x, y = _float_args(a, b)
        return f"copysign_f({x}, {y})"

    @staticmethod
    def ldexp(a: Any, b: Any) -> str:
        x = _float_args(a)[0]
        return f"ldexp({x}, {convert(b, _dtype(b), torch.int32)})"

    @staticmethod
    def logaddexp(a: Any, b: Any) -> str:
        x, y = _float_args(a, b)
        return f"logaddexp_f({x}, {y})"

    # -- random ----------------------------------------------------------------
    @staticmethod
    def load_seed(name: str, offset: Any) -> str:
        var = V.kernel.args.input(name)
        V.kernel.used_buffers.add(var)
        return f"ld_{var}({offset}u)"

    @staticmethod
    def rand(seed: Any, offset: Any) -> str:
        return f"rand_f({seed}, u32({convert(offset, _dtype(offset), torch.int32)}))"

    @staticmethod
    def randn(seed: Any, offset: Any) -> str:
        return f"randn_f({seed}, u32({convert(offset, _dtype(offset), torch.int32)}))"

    @staticmethod
    def randint64(seed: Any, offset: Any, low: Any, high: Any) -> str:
        return (
            f"randint64_i({seed}, u32({convert(offset, _dtype(offset), torch.int32)}), "
            f"{convert(low, _dtype(low), torch.int64)}, {convert(high, _dtype(high), torch.int64)})"
        )


# Counter-based RNG (Philox4x32-10, as in curand / Triton) for compiled
# kernels.
RANDOM_WGSL = """
fn philox4x32(ctr: vec4<u32>, key: vec2<u32>) -> vec4<u32> {
    var c = ctr;
    var k = key;
    for (var i = 0; i < 10; i++) {
        let p0 = mulhilo(0xD2511F53u, c.x);
        let p1 = mulhilo(0xCD9E8D57u, c.z);
        c = vec4<u32>(p1.x ^ c.y ^ k.x, p1.y, p0.x ^ c.w ^ k.y, p0.y);
        k = k + vec2<u32>(0x9E3779B9u, 0xBB67AE85u);
    }
    return c;
}
fn rand_f(seed: i64, offset: u32) -> f32 {
    return f32(philox4x32(vec4<u32>(offset, 0u, 0u, 0u), seed).x >> 8u) * 5.9604644775390625e-8f;
}
fn randn_f(seed: i64, offset: u32) -> f32 {
    let r = philox4x32(vec4<u32>(offset, 0u, 0u, 0u), seed);
    let u1 = (f32(r.x >> 8u) + 1.0f) * 5.9604644775390625e-8f;
    let u2 = f32(r.y >> 8u) * 5.9604644775390625e-8f;
    return sqrt(-2.0f * log(u1)) * cos_f(6.283185307179586f * u2);
}
fn randint64_i(seed: i64, offset: u32, low: i64, high: i64) -> i64 {
    let r = philox4x32(vec4<u32>(offset, 0u, 0u, 0u), seed);
    let span = i64_sub(high, low);
    let v = i64(r.x, r.y & 0x7fffffffu);
    return i64_add(low, i64_rem_trunc(v, span));
}
"""


# ---------------------------------------------------------------------------
# reductions
# ---------------------------------------------------------------------------


def _reduction_identity(rtype: str, dtype: torch.dtype) -> str:
    t = ct(dtype)
    if rtype in ("sum", "xor_sum", "any"):
        return literal(0, dtype)
    if rtype == "prod":
        return literal(1, dtype)
    if rtype in ("max", "argmax"):
        if t == "f32":
            return "(-inf_f())"
        if t == "bool":
            return "false"
        return literal(-(2**63) if t == "i64" else -(2**31), dtype)
    if rtype in ("min", "argmin"):
        if t == "f32":
            return "inf_f()"
        if t == "bool":
            return "true"
        return literal(2**63 - 1 if t == "i64" else 2**31 - 1, dtype)
    raise NotImplementedError(f"WebGPU Inductor backend: reduction {rtype}")


def _combine(rtype: str, dtype: torch.dtype, a: str, b: str) -> str:
    t = ct(dtype)
    if rtype == "sum":
        return {"i64": f"i64_add({a}, {b})", "bool": f"({a} || {b})"}.get(t, f"({a} + {b})")
    if rtype == "prod":
        return {"i64": f"i64_mul({a}, {b})", "bool": f"({a} && {b})"}.get(t, f"({a} * {b})")
    if rtype == "max":
        return {"i64": f"i64_max({a}, {b})", "f32": f"max_nan({a}, {b})", "bool": f"({a} || {b})"}.get(
            t, f"max({a}, {b})"
        )
    if rtype == "min":
        return {"i64": f"i64_min({a}, {b})", "f32": f"min_nan({a}, {b})", "bool": f"({a} && {b})"}.get(
            t, f"min({a}, {b})"
        )
    if rtype == "any":
        return f"({a} || {b})"
    if rtype == "xor_sum":
        return {"bool": f"({a} != {b})"}.get(t, f"({a} ^ {b})")
    raise NotImplementedError(f"WebGPU Inductor backend: reduction {rtype}")


def _arg_better(rtype: str, dtype: torch.dtype, v: str, a: str) -> str:
    """True if value v strictly beats the current value a (NaN beats all)."""
    t = ct(dtype)
    op = ">" if rtype == "argmax" else "<"
    if t == "f32":
        # NaN first: some drivers evaluate ordered comparisons with a NaN
        # operand as true (a > b compiled as !(a <= b)).
        return f"(!isnan_f({a}) && (isnan_f({v}) || ({v} {op} {a})))"
    if t == "i64":
        return f"i64_{'gt' if op == '>' else 'lt'}({v}, {a})"
    if t == "bool":
        return f"({v} && !{a})" if op == ">" else f"(!{v} && {a})"
    return f"({v} {op} {a})"


def _arg_equal(dtype: torch.dtype, v: str, a: str) -> str:
    t = ct(dtype)
    if t == "f32":
        return f"select({v} == {a} && !isnan_f({v}) && !isnan_f({a}), isnan_f({v}), isnan_f({a}))"
    if t == "i64":
        return f"i64_eq({v}, {a})"
    return f"({v} == {a})"


# ---------------------------------------------------------------------------
# kernel
# ---------------------------------------------------------------------------


class WGSLKernel(SIMDKernel):
    overrides = WGSLOverrides  # type: ignore[assignment]
    suffix = ";"
    newvar_prefix = "let "
    max_threadgroup_size = WG
    kexpr = sexpr = staticmethod(wexpr)  # type: ignore[assignment]

    def __init__(self, tiling: dict[str, sympy.Expr], **kwargs: Any) -> None:
        super().__init__(tiling, **kwargs)
        if self._index_dtype != torch.int32:
            raise NotImplementedError("WebGPU Inductor backend: kernels over >= 2**31 elements are not supported")
        self.used_buffers: OrderedSet[str] = OrderedSet()
        self.atomic_buffers: OrderedSet[str] = OrderedSet()
        self.workgroup_vars: list[str] = []
        self.helpers: OrderedSet[str] = OrderedSet()
        self.uses_error_flag = False
        self.loop_open = False
        self.red_ids = itertools.count()
        trees = [t for t in self.range_trees if not t.is_reduction]
        rtrees = [t for t in self.range_trees if t.is_reduction]
        if len(trees) > 1 or len(rtrees) > 1:
            raise NotImplementedError("WebGPU Inductor backend: expected 1-d tiling")
        rnumel = self.features.reduction_numel
        if self.features.is_reduction():
            if isinstance(rnumel, sympy.Integer):
                self.rblock = min(WG, 1 << max(0, (int(rnumel) - 1).bit_length()))
            else:
                self.rblock = WG
        else:
            self.rblock = 1
        self.xblock = WG // self.rblock

    # -- configuration ---------------------------------------------------------
    def should_use_persistent_reduction(self) -> bool:
        return False

    def should_use_cooperative_reduction(self) -> bool:
        return False

    def dtype_to_str(self, dtype: torch.dtype) -> str:
        return ct(dtype)

    @property
    def index_dtype(self) -> str:
        return "i32"

    # -- indexing --------------------------------------------------------------
    def _rtree(self):
        return next(t for t in self.range_trees if t.is_reduction)

    def _open_loop(self) -> None:
        if self.loop_open:
            return
        tree = self._rtree()
        self.body.writeline(
            f"for (var {tree.name}: i32 = lr; {tree.name} < {tree.prefix}numel; {tree.name} += {self.rblock}i) {{"
        )
        self.loop_open = True

    def codegen_iteration_ranges_entry(self, entry: IterationRangesEntry) -> None:
        index_str = wexpr(self.rename_indexing(entry.expr))
        if not entry.is_reduction:
            self.indexing_code.writeline(f"let {entry.name}: i32 = {index_str};")
            return
        self._open_loop()
        with self.body.indent():
            self.body.writeline(f"let {entry.name}: i32 = {index_str};")

    def codegen_body(self) -> None:
        if self.loop_open:
            with self.body.indent():
                self.body.splice(self.loads)
                self.body.splice(self.compute)
            self.body.writeline("}")
            self.loop_open = False
            self.cse.invalidate(
                OrderedSet(
                    v
                    for item in self.cse.reduction_cache.values()
                    for v in (item if isinstance(item, tuple) else (item,))
                )
            )
            for node in self.range_tree_nodes.values():
                if node.is_reduction:
                    node.cache_clear()
        else:
            self.body.splice(self.loads)
            self.body.splice(self.compute)
        self.body.splice(self.stores)
        self.loads.clear()
        self.compute.clear()
        self.stores.clear()

    def check_bounds(self, expr: sympy.Expr, size: sympy.Expr, lower: bool, upper: bool) -> None:
        if not (lower or upper):
            return
        self.uses_error_flag = True
        e = wexpr(self.rename_indexing(expr))
        conds = []
        if lower:
            conds.append(f"({e}) < 0")
        if upper:
            conds.append(f"({e}) >= ({wexpr(self.rename_indexing(size))})")
        self.compute.writeline(f"if ({' || '.join(conds)}) {{ atomicOr(&err_flag[0], 1u); }}")

    # -- memory ----------------------------------------------------------------
    def load(self, name: str, index: sympy.Expr) -> CSEVariable:
        var = self.args.input(name)
        self.used_buffers.add(var)
        index = self.prepare_indexing(index)
        dtype = V.graph.get_dtype(name)
        line = f"ld_{var}(u32({self.index_to_str(index)}))"
        return self.cse.generate(self.loads, line, dtype=dtype)

    def store(self, name: str, index: sympy.Expr, value: CSEVariable, mode: StoreMode = None) -> None:
        var = self.args.output(name)
        self.used_buffers.add(var)
        index = self.prepare_indexing(index)
        dtype = V.graph.get_dtype(name)
        v = convert(value, _dtype(value), dtype) if _dtype(value) is not None else str(value)
        idx = self.index_to_str(index)
        if mode is None:
            line = f"st_{var}(u32({idx}), {v});"
        elif mode == "atomic_add":
            if dtype not in (torch.float32, torch.int32, torch.float16, torch.bfloat16):
                raise NotImplementedError(f"WebGPU Inductor backend: atomic_add on {dtype}")
            self.atomic_buffers.add(var)
            line = f"atomic_add_{var}(u32({idx}), {v});"
        else:
            raise NotImplementedError(f"WebGPU Inductor backend: store mode {mode}")
        guard = self._store_guard(in_loop=self.inside_reduction)
        if guard:
            line = f"if ({guard}) {{ {line} }}"
        if self.inside_reduction:
            self.compute.writeline(DeferredLine(name, line))
        else:
            self.stores.writeline(DeferredLine(name, line))

    def _store_guard(self, in_loop: bool) -> str:
        if not self.features.is_reduction():
            return ""
        return "xmask" if in_loop else "xmask && lr == 0i"

    def store_reduction(self, name: str, index: sympy.Expr, value: CSEVariable) -> None:
        var = self.args.output(name)
        self.used_buffers.add(var)
        index = self.prepare_indexing(index)
        dtype = V.graph.get_dtype(name)
        v = convert(value, _dtype(value), dtype) if _dtype(value) is not None else str(value)
        line = f"if (xmask && lr == 0i) {{ st_{var}(u32({self.index_to_str(index)}), {v}); }}"
        self.stores.writeline(DeferredLine(name, line))

    # -- reductions ------------------------------------------------------------
    def _workgroup_array(self, ty: str) -> str:
        name = f"red{len(self.workgroup_vars)}"
        self.workgroup_vars.append(f"var<workgroup> {name}: array<{ty}, {WG}>;")
        return name

    def _tree_reduce(self, buf: str, combine_stmt: str) -> None:
        """Workgroup tree reduction over each RB-wide segment of `buf`.
        combine_stmt updates slot `i` from slot `j`."""
        rb = self.rblock
        self.stores.writeline("workgroupBarrier();")
        if rb > 1:
            self.stores.writeline(f"for (var s: u32 = {rb // 2}u; s > 0u; s >>= 1u) {{")
            with self.stores.indent():
                self.stores.writeline("if (u32(lr) < s) {")
                with self.stores.indent():
                    self.stores.writeline("let i = lidx; let j = lidx + s;")
                    self.stores.splice(combine_stmt)
                self.stores.writeline("}")
                self.stores.writeline("workgroupBarrier();")
            self.stores.writeline("}")

    def reduction(self, dtype: torch.dtype, src_dtype: torch.dtype, reduction_type: str, value: Any) -> Any:
        cache_key = (src_dtype, reduction_type, value)
        if cache_key in self.cse.reduction_cache:
            return self.cse.reduction_cache[cache_key]
        result = self._reduction(dtype, src_dtype, reduction_type, value)
        self.cse.reduction_cache[cache_key] = result  # type: ignore[assignment]
        return result

    def _reduction(self, dtype: torch.dtype, src_dtype: torch.dtype, rtype: str, value: Any) -> Any:
        assert self.inside_reduction
        self._open_loop()
        k = next(self.red_ids)
        seg = f"(u32(ly) * {self.rblock}u)"
        if rtype in ("sum", "prod", "max", "min", "any", "xor_sum"):
            acc_dt = src_dtype
            if rtype == "any":
                acc_dt = torch.bool
            elif rtype in ("sum", "prod") and src_dtype in (torch.float16, torch.bfloat16):
                acc_dt = torch.float32
            t = ct(acc_dt)
            acc = f"racc{k}"
            self.indexing_code.writeline(f"var {acc}: {t} = {_reduction_identity(rtype, acc_dt)};")
            v = convert(value, _dtype(value), acc_dt) if _dtype(value) is not None else str(value)
            self.compute.writeline(f"{acc} = {_combine(rtype, acc_dt, acc, v)};")
            buf = self._workgroup_array(t)
            self.stores.writeline(f"{buf}[lidx] = {acc};")
            self._tree_reduce(buf, f"{buf}[i] = {_combine(rtype, acc_dt, f'{buf}[i]', f'{buf}[j]')};")
            out = self.cse.newvar(dtype=acc_dt)
            self.stores.writeline(f"let {out} = {buf}[{seg}];")
            return out
        if rtype in ("argmax", "argmin"):
            t = ct(src_dtype)
            av, ai = f"racc{k}_v", f"racc{k}_i"
            self.indexing_code.writeline(f"var {av}: {t} = {_reduction_identity(rtype, src_dtype)};")
            self.indexing_code.writeline(f"var {ai}: i32 = -1i;")
            v = convert(value, _dtype(value), src_dtype) if _dtype(value) is not None else str(value)
            idx = self._rtree().name
            self.compute.writeline(
                f"if ({ai} < 0i || {_arg_better(rtype, src_dtype, v, av)}) {{ {av} = {v}; {ai} = {idx}; }}"
            )
            bi = self._workgroup_array("i32")
            if t == "f32":
                # Floats cross lanes as bit patterns: some drivers lose NaN
                # tests on floats read back from workgroup memory.
                bv = self._workgroup_array("u32")
                self.stores.writeline(f"{bv}[lidx] = bitcast<u32>({av}); {bi}[lidx] = {ai};")
                op = ">" if rtype == "argmax" else "<"
                vj, vi = f"bitcast<f32>({bv}[j])", f"bitcast<f32>({bv}[i])"
                nj, ni = f"(({bv}[j] & 0x7fffffffu) > 0x7f800000u)", f"(({bv}[i] & 0x7fffffffu) > 0x7f800000u)"
                better = f"(!{ni} && ({nj} || ({vj} {op} {vi})))"
                equal = f"select({vj} == {vi} && !{nj} && !{ni}, {nj}, {ni})"
            else:
                bv = self._workgroup_array(t)
                self.stores.writeline(f"{bv}[lidx] = {av}; {bi}[lidx] = {ai};")
                better = _arg_better(rtype, src_dtype, f"{bv}[j]", f"{bv}[i]")
                equal = _arg_equal(src_dtype, f"{bv}[j]", f"{bv}[i]")
            take = f"{bi}[i] < 0i || ({bi}[j] >= 0i && ({better} || ({equal} && {bi}[j] < {bi}[i])))"
            self._tree_reduce(bv, f"if ({take}) {{ {bv}[i] = {bv}[j]; {bi}[i] = {bi}[j]; }}")
            out = self.cse.newvar(dtype=dtype)
            self.stores.writeline(f"let {out} = {convert(f'max({bi}[{seg}], 0i)', torch.int32, dtype)};")
            return out
        if rtype in ("welford_reduce", "welford_combine"):
            acc = f"racc{k}"
            self.indexing_code.writeline(f"var {acc}: vec3<f32> = vec3<f32>(0.0f);")
            self.helpers.add("welford")
            if rtype == "welford_reduce":
                (v,) = _float_args(value)
                self.compute.writeline(f"{acc} = welford_add({acc}, {v});")
            else:
                m, m2, w = _float_args(*value)
                self.compute.writeline(f"{acc} = welford_merge({acc}, vec3<f32>({m}, {m2}, {w}));")
            buf = self._workgroup_array("vec3<f32>")
            self.stores.writeline(f"{buf}[lidx] = {acc};")
            self._tree_reduce(buf, f"{buf}[i] = welford_merge({buf}[i], {buf}[j]);")
            out = self.cse.newvar(dtype=torch.float32)
            self.stores.writeline(f"let {out} = {buf}[{seg}];")
            # The mean of nothing is NaN (as in eager), not the initial 0.
            mean = f"select({out}.x, nan_f(), {out}.z == 0.0f)"
            return tuple(self.cse.generate(self.stores, e, dtype=torch.float32) for e in (mean, f"{out}.y", f"{out}.z"))
        raise NotImplementedError(f"WebGPU Inductor backend: reduction {rtype}")

    # -- final source ----------------------------------------------------------
    def _buffers(self) -> list[tuple[str, str, torch.dtype, bool]]:
        """(inner name, outer name, dtype, written) for used tensor arguments."""
        out = []
        arg_defs, call_args, _, arg_types = self.args.python_argdefs()
        for d, c, t in zip(arg_defs, call_args, arg_types):
            inner = str(getattr(d, "name", d))
            if not isinstance(t, torch.dtype):
                continue
            if inner not in self.used_buffers:
                continue
            written = inner.startswith("out_ptr") or inner.startswith("in_out_ptr")
            out.append((inner, str(c), t, written))
        return out

    def _sizevars(self) -> list[tuple[str, str]]:
        return [(str(inner), str(outer)) for outer, inner in self.args.sizevars.items()]

    def codegen_kernel(self, name: str | None = None) -> str:
        self.codegen_body()
        bufs = self._buffers()
        sizevars = self._sizevars()
        reduction = self.features.is_reduction()
        xtree = next(t for t in self.range_trees if not t.is_reduction)
        numel_params: list[tuple[str, Any]] = []
        for tree in self.range_trees:
            if not isinstance(tree.numel, (int, sympy.Integer)):
                numel_params.append((f"{tree.prefix}numel", tree.numel))
        nparams = len(sizevars) + len(numel_params)

        code = IndentedBuffer()
        code.splice(_C._wgsl_prelude())
        code.splice(opaque_math())
        code.splice(RANDOM_WGSL)
        if "welford" in self.helpers:
            code.splice(WELFORD_WGSL)
        special = sorted(h.removeprefix("special:") for h in self.helpers if h.startswith("special:"))
        if special:
            code.splice(_C._wgsl_special(special))
        if "polygamma_n" in self.helpers:
            code.splice(POLYGAMMA_WGSL)
        # Buffer declarations depend on runtime aliasing (runtime.py); the spec
        # comment keeps sources that differ only in buffer dtypes distinct.
        self.buffer_specs = [
            (inner, str(dtype).removeprefix("torch."), written, inner in self.atomic_buffers)
            for inner, _, dtype, written in bufs
        ]
        self.nparams = nparams
        code.writeline(f"// buffers: {self.buffer_specs}")
        code.writeline("//@BUFFERS@")
        for w in self.workgroup_vars:
            code.writeline(w)

        code.writeline(f"@compute @workgroup_size({WG})")
        code.writeline(
            "fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>, "
            "@builtin(local_invocation_index) lidx: u32) {"
        )
        with code.indent():
            code.writeline("_ = P.offs[0];")
            for j, (inner, _) in enumerate(sizevars):
                code.writeline(f"let {inner}: i32 = bitcast<i32>(P.v[{j // 4}][{j % 4}]);")
            for j, (pname, _) in enumerate(numel_params):
                jj = len(sizevars) + j
                code.writeline(f"let {pname}: i32 = bitcast<i32>(P.v[{jj // 4}][{jj % 4}]);")
            for tree in self.range_trees:
                if isinstance(tree.numel, (int, sympy.Integer)):
                    code.writeline(f"let {tree.prefix}numel: i32 = {int(tree.numel)};")
            code.writeline("let wg_lin = wid.y * nwg.x + wid.x;")
            if reduction:
                code.writeline(f"let lr = i32(lidx % {self.rblock}u);")
                code.writeline(f"let ly = i32(lidx / {self.rblock}u);")
                code.writeline(f"let xraw = i32(wg_lin) * {self.xblock}i + ly;")
                code.writeline(f"let xmask = xraw < {xtree.prefix}numel;")
                code.writeline(f"let {xtree.name} = min(xraw, {xtree.prefix}numel - 1i);")
            else:
                code.writeline(f"let {xtree.name} = i32(wg_lin * {WG}u + lidx);")
                code.writeline(f"if ({xtree.name} >= {xtree.prefix}numel) {{ return; }}")
            code.splice(self.indexing_code)
            code.splice(self.body)
        code.writeline("}")
        return code.getvalue()

    def call_kernel(self, name: str, node: Any = None, deallocate_ws: bool = True) -> None:
        wrapper = V.graph.wrapper_code
        for v in self.args.sizevars:
            wrapper.ensure_size_computed(v)
        bufs = self._buffers()
        params = [outer for _, outer in self._sizevars()]
        xtree = next(t for t in self.range_trees if not t.is_reduction)
        for tree in self.range_trees:
            if not isinstance(tree.numel, (int, sympy.Integer)):
                params.append(wrapper.generate_numel_expr(name, tree).inner)
        xnumel = (
            str(int(xtree.numel))
            if isinstance(xtree.numel, (int, sympy.Integer))
            else wrapper.generate_numel_expr(name, xtree).inner
        )
        args = [
            "[" + ", ".join(outer for _, outer, _, _ in bufs) + "]",
            "[" + ", ".join(str(p) for p in params) + "]",
            str(xnumel),
        ]
        wrapper.generate_kernel_call(name, args, device=torch.device("webgpu"), triton=False)


WELFORD_WGSL = """
// Welford state: (mean, m2, weight).
fn welford_add(a: vec3<f32>, v: f32) -> vec3<f32> {
    let w = a.z + 1.0f;
    let d = v - a.x;
    let mean = a.x + d / w;
    return vec3<f32>(mean, a.y + d * (v - mean), w);
}
fn welford_merge(a: vec3<f32>, b: vec3<f32>) -> vec3<f32> {
    if (b.z == 0.0f) { return a; }
    if (a.z == 0.0f) { return b; }
    let w = a.z + b.z;
    let nb = b.z / w;
    let d = b.x - a.x;
    return vec3<f32>(a.x + d * nb, a.y + b.y + d * d * a.z * nb, w);
}
"""


class WGSLScheduling(SIMDScheduling):
    kernel_type = WGSLKernel  # type: ignore[assignment]

    def get_backend_features(self, device: torch.device) -> OrderedSet[BackendFeature]:
        return OrderedSet([BackendFeature.INPLACE_BUFFERS])

    @classmethod
    def get_tiling_and_scores(cls, node_schedule, numel, reduction_numel=sympy.S.One, coalesce_analysis=None):
        return cls.create_tiling([numel], [reduction_numel]), None

    def define_kernel(self, src_code: str, node_schedule: list[Any], kernel: WGSLKernel) -> str:
        wrapper = V.graph.wrapper_code
        if src_code in wrapper.src_to_kernel:
            return wrapper.src_to_kernel[src_code]
        name = f"wgsl_kernel_{wrapper.next_kernel_suffix()}"
        wrapper.src_to_kernel[src_code] = name
        wrapper.add_import_once("from torch_webgpu._inductor.runtime import WGSLKernelRunner")
        origins, detailed = get_kernel_metadata(node_schedule, wrapper)
        block = kernel.xblock if kernel.features.is_reduction() else WG
        wrapper.header.writeline(f"# {origins}")
        wrapper.header.writeline(f"{name} = WGSLKernelRunner({name!r}, r'''")
        wrapper.header.splice(src_code)
        wrapper.header.writeline(
            f"''', {kernel.buffer_specs!r}, {kernel.nparams}, {block}, {kernel.uses_error_flag})"
        )
        return name
