"""Post-grad FX passes for webgpu graphs.

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


def _is_webgpu_f32(v: torch.Tensor | None) -> bool:
    return v is not None and v.device.type == "webgpu" and v.dtype == torch.float32


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
        if bv is None or bv.dim() != 1 or not _is_webgpu_f32(bv):
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
    if not (_is_webgpu_f32(xv) and _is_webgpu_f32(wv)) or xv.dim() != 2 or wv.dim() != 2:
        return None
    if not isinstance(xv.shape[0], int) or xv.shape[0] > MAX_ROWS or not isinstance(wv.shape[0], int):
        return None
    if xv.shape[1] != wv.shape[1] or not wv.is_contiguous():
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
