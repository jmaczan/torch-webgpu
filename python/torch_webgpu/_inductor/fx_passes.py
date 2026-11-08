"""Post-grad FX passes for webgpu graphs.

strip_repeat_kv: attention inputs repeated per query head (Hugging Face's
repeat_kv: k[:, :, None].expand(...).reshape(...), a full copy of the KV
cache per layer and token) feed the fused SDPA kernel, which reads shared
key / value heads directly (grouped-query attention), unrepeated.

fuse_parallel_linears: decode-size linear layers that read the same input
(attention q / k / v projections, MLP gate / up projections) become one
torch.ops.webgpu.multi_linear call, i.e. one GEMV dispatch instead of
several, with the individual results taken as column slices.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import torch
import torch.fx

aten = torch.ops.aten

# Inputs with more rows than this are GEMMs, where separate launches are
# already efficient.
MAX_ROWS = 16
# Weights per fused call. WebGPU guarantees 8 storage buffers per shader
# stage: input, output and up to 3 weights + 3 biases (or 4 without biases).
MAX_GROUP = 4
MAX_BINDINGS = 8

_VIEWS = (aten.view.default, aten.reshape.default, aten._unsafe_view.default)


def _val(n: Any) -> torch.Tensor | None:
    v = getattr(n, "meta", {}).get("val") if isinstance(n, torch.fx.Node) else None
    return v if isinstance(v, torch.Tensor) else None


_GEMV_DTYPES = (torch.float32, torch.float16, torch.bfloat16)


def _is_webgpu_float(v: torch.Tensor | None) -> bool:
    return v is not None and v.device.type == "webgpu" and v.dtype in _GEMV_DTYPES


def _linear_parts(node: torch.fx.Node):
    """(x, weight, bias | None) for mm(x, w.t()) / addmm(b, x, w.t())."""
    if node.op != "call_function":
        return None
    if node.target is aten.mm.default:
        x, wt = node.args
        b = None
    elif node.target is aten.addmm.default:
        if node.kwargs.get("beta", 1) != 1 or node.kwargs.get("alpha", 1) != 1 or len(node.args) != 3:
            return None
        b, x, wt = node.args
        bv = _val(b)
        if bv is None or bv.dim() != 1 or not _is_webgpu_float(bv):
            return None
    else:
        return None
    if not isinstance(wt, torch.fx.Node) or wt.op != "call_function":
        return None
    if wt.target is aten.permute.default and list(wt.args[1]) == [1, 0] or wt.target is aten.t.default:
        w = wt.args[0]
    else:
        return None
    xv, wv = _val(x), _val(w)
    if not (_is_webgpu_float(xv) and _is_webgpu_float(wv)) or xv.dim() != 2 or wv.dim() != 2:
        return None
    if not isinstance(xv.shape[0], int) or xv.shape[0] > MAX_ROWS or not isinstance(wv.shape[0], int):
        return None
    if xv.shape[1] != wv.shape[1] or not wv.is_contiguous() or wv.dtype != xv.dtype:
        return None
    if b is not None and _val(b).dtype != xv.dtype:
        return None
    return x, w, b


def _input_key(x: torch.fx.Node):
    # Each linear may view the shared activation through its own view node.
    if x.op == "call_function" and x.target in _VIEWS:
        return (x.args[0], tuple(_val(x).shape))
    return (x, None)


def _chunks(items):
    group: list = []
    bindings = 2
    for it in items:
        need = 1 + (it[3] is not None)
        if group and (len(group) == MAX_GROUP or bindings + need > MAX_BINDINGS):
            yield group
            group, bindings = [], 2
        group.append(it)
        bindings += need
    if group:
        yield group


def fuse_parallel_linears(gm: torch.fx.GraphModule) -> int:
    """Returns the number of linear nodes fused."""
    graph = gm.graph
    order = {n: i for i, n in enumerate(graph.nodes)}
    groups: dict[Any, list] = defaultdict(list)
    for n in graph.nodes:
        parts = _linear_parts(n)
        if parts is not None:
            groups[_input_key(parts[0])].append((n, *parts))
    fused = 0
    for items in groups.values():
        for group in _chunks(items):
            if len(group) < 2:
                continue
            first = min(group, key=lambda it: order[it[0]])[0]
            x = group[0][1]
            # Every operand must exist before the first linear of the group.
            operands = [x] + [it[2] for it in group] + [it[3] for it in group if it[3] is not None]
            if any(isinstance(o, torch.fx.Node) and order[o] >= order[first] for o in operands):
                continue
            xv = _val(x)
            fake_mode = getattr(xv, "fake_mode", None)
            if fake_mode is None:
                continue
            with graph.inserting_before(first), fake_mode:
                call = graph.call_function(
                    torch.ops.webgpu.multi_linear.default,
                    (x, [it[2] for it in group], [it[3] for it in group]),
                )
                total = sum(_val(it[2]).shape[0] for it in group)
                call.meta["val"] = xv.new_empty((xv.shape[0], total))
                start = 0
                slices = []
                for node, _, w, _ in group:
                    n_out = _val(w).shape[0]
                    sl = graph.call_function(aten.slice.Tensor, (call, 1, start, start + n_out))
                    sl.meta["val"] = call.meta["val"][:, start:start + n_out]
                    slices.append((node, sl))
                    start += n_out
            # (Erase only after inserting: `first` is the insertion point.)
            for node, sl in slices:
                node.replace_all_uses_with(sl)
                graph.erase_node(node)
            fused += len(group)
    if fused:
        graph.eliminate_dead_code()
        gm.recompile()
    return fused


_SDPA = aten._scaled_dot_product_fused_attention_overrideable.default


def _full_slice(n: torch.fx.Node) -> bool:
    if n.op != "call_function" or n.target is not aten.slice.Tensor:
        return False
    dim = n.args[1] if len(n.args) > 1 else 0
    start = n.args[2] if len(n.args) > 2 else None
    end = n.args[3] if len(n.args) > 3 else None
    step = n.args[4] if len(n.args) > 4 else 1
    size = _val(n.args[0]).shape[dim]
    return (start in (None, 0)) and (end is None or (isinstance(end, int) and end >= size)) and step == 1


def _unrepeated(x: Any):
    """(base, n_rep) if x == base[:, :, None].expand(B, Hk, n_rep, L, D)
    .reshape(B, Hk * n_rep, L, D) (materialized by a clone), else None."""
    n = x
    if not isinstance(n, torch.fx.Node) or n.op != "call_function" or n.target not in _VIEWS:
        return None
    n = n.args[0]
    if isinstance(n, torch.fx.Node) and n.op == "call_function" and n.target is aten.clone.default:
        n = n.args[0]
    if not isinstance(n, torch.fx.Node) or n.op != "call_function" or n.target is not aten.expand.default:
        return None
    u = n.args[0]
    if not isinstance(u, torch.fx.Node) or u.op != "call_function" or u.target is not aten.unsqueeze.default \
            or u.args[1] != 2:
        return None
    base = u.args[0]
    while isinstance(base, torch.fx.Node) and _full_slice(base):
        base = base.args[0]
    bv, ev, xv = _val(base), _val(n), _val(x)
    if bv is None or ev is None or xv is None or bv.dim() != 4 or ev.dim() != 5:
        return None
    B, Hk, L, Dh = bv.shape
    if tuple(ev.shape) != (B, Hk, ev.shape[2], L, Dh) or tuple(xv.shape) != (B, Hk * ev.shape[2], L, Dh):
        return None
    return base, ev.shape[2]


def strip_repeat_kv(gm: torch.fx.GraphModule) -> int:
    """Returns the number of attention calls rewritten."""
    count = 0
    for node in gm.graph.nodes:
        if node.op != "call_function" or node.target is not _SDPA:
            continue
        qv = _val(node.args[0])
        if qv is None or qv.device.type != "webgpu":
            continue
        k, v = _unrepeated(node.args[1]), _unrepeated(node.args[2])
        if k is None or v is None or k[1] != v[1] or k[1] < 2:
            continue
        if _val(k[0]).shape[1] * k[1] != qv.shape[1]:
            continue
        node.args = (node.args[0], k[0], v[0], *node.args[3:])
        count += 1
    if count:
        gm.graph.eliminate_dead_code()
        gm.recompile()
    return count
