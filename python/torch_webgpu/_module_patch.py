"""Keep nn.Module parameters (and weight tying) intact across webgpu moves.

``Module._apply`` (used by ``.to()``, ``.webgpu()``, ``.cpu()``, ``.half()``,
...) updates a Parameter in place via ``param.data = new`` only when
``TensorImpl::has_compatible_shallow_copy_type`` holds, whose list of dense
backends includes CUDA and MPS but not PrivateUse1. Otherwise it replaces the
Parameter with a new object, which silently unties shared weights (e.g. an
LM head tied to the embedding gets its own copy and training diverges) and
detaches optimizers created before the move.

For conversions involving webgpu tensors we use PyTorch's swap-tensors path
(``torch.__future__.set_swap_module_params_on_conversion``), which converts
Parameters in place, preserving object identity like CUDA does. If a swap is
impossible (e.g. a parameter with weakrefs), we fall back to the default path
and re-tie parameters that were shared before the move.
"""

from __future__ import annotations

import itertools
import threading

import torch

_state = threading.local()
_orig_apply = torch.nn.Module._apply


def _involves_webgpu(module: torch.nn.Module, fn) -> bool:
    if any(t.device.type == "webgpu" for t in itertools.chain(module.parameters(), module.buffers())):
        return True
    return _maps_to_webgpu(fn)


def _maps_to_webgpu(fn) -> bool:
    try:
        with torch.no_grad():
            out = fn(torch.empty(0))
        return isinstance(out, torch.Tensor) and out.device.type == "webgpu"
    except Exception:  # noqa: BLE001 - arbitrary user fns
        return False


def _shared_params(module: torch.nn.Module):
    """Groups of (module, name) slots holding the same Parameter object."""
    slots: dict[int, list[tuple[torch.nn.Module, str]]] = {}
    for m in module.modules():
        for name, p in m._parameters.items():
            if p is not None:
                slots.setdefault(id(p), []).append((m, name))
    return [v for v in slots.values() if len(v) > 1]


def _retie(groups) -> None:
    for group in groups:
        m0, n0 = group[0]
        p = m0._parameters[n0]
        for m, n in group[1:]:
            m._parameters[n] = p


def _apply(self, fn, recurse=True):
    if getattr(_state, "active", False) or torch.__future__.get_swap_module_params_on_conversion():
        return _orig_apply(self, fn, recurse)
    if not _involves_webgpu(self, fn):
        return _orig_apply(self, fn, recurse)
    groups = _shared_params(self)
    _state.active = True
    torch.__future__.set_swap_module_params_on_conversion(True)
    try:
        try:
            return _orig_apply(self, fn, recurse)
        except RuntimeError as e:
            if "Couldn't swap" not in str(e):
                raise
        # Swapping impossible: convert the usual way, then restore sharing.
        torch.__future__.set_swap_module_params_on_conversion(False)
        result = _orig_apply(self, fn, recurse)
        _retie(groups)
        return result
    finally:
        torch.__future__.set_swap_module_params_on_conversion(False)
        _state.active = False


def install() -> None:
    if torch.nn.Module._apply is not _apply:
        torch.nn.Module._apply = _apply
