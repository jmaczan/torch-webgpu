from typing import Callable, List
from functools import partial
from .low_ir import (
    LowIRCreateBuffer,
    LowIRMoveTo,
    LowIRNode,
    LowIROp,
    LowIROutput,
    LowIRRunShader,
    LowIRWriteBuffer,
)
from .logger import debug, debug_enabled
import torch

Runtime = dict


def create_buffer(node: LowIRCreateBuffer, runtime: Runtime) -> torch.Tensor:
    # TODO: take into account wher the buffer should be allocated (device)
    buf = torch.ops.webgpu.create_buffer(
        node.size,
        node.stride,
        node.dtype,
    )
    runtime[node.value_id] = buf
    return buf


def write_buffer(node: LowIRWriteBuffer, runtime: Runtime) -> torch.Tensor:
    # TODO: take into account wher the buffer should be allocated (device)
    dst = runtime[node.value_id]
    src = torch.tensor(node.constant_data)  # TODO: handle also other kinds of data
    return torch.ops.webgpu.write_buffer(dst, src)


import torch.nn.functional as F


# --- Fused op fallback functions ---
# These provide CPU fallbacks for fused ops that are only registered for WebGPU (PrivateUse1)

def _is_webgpu_device(x):
    """Check if tensor is on WebGPU device."""
    return x.device.type in ('webgpu', 'privateuseone', 'privateuse1')


def _rms_norm_with_fallback(x, weight, eps=1e-6):
    """RMSNorm with CPU fallback."""
    if _is_webgpu_device(x):
        return torch.ops.webgpu.rms_norm(x, weight, eps)
    # CPU fallback: y = x * rsqrt(mean(x^2) + eps) * weight
    variance = x.pow(2).mean(dim=-1, keepdim=True)
    x_norm = x * torch.rsqrt(variance + eps)
    return x_norm * weight


def _fused_qkv_proj_with_fallback(x, q_w, k_w, v_w):
    """Fused Q, K, V projection with CPU fallback."""
    if _is_webgpu_device(x):
        # Only use fused kernel if all weights have same output dimension (not GQA)
        if q_w.size(0) == k_w.size(0) == v_w.size(0):
            try:
                return torch.ops.webgpu.fused_qkv_proj(x, q_w, k_w, v_w)
            except Exception:
                pass  # Fall through to separate projections
        # GQA or error - use separate projections (still on WebGPU)
        q = F.linear(x, q_w)
        k = F.linear(x, k_w)
        v = F.linear(x, v_w)
        return (q, k, v)
    # CPU fallback: three separate linear projections
    q = F.linear(x, q_w)
    k = F.linear(x, k_w)
    v = F.linear(x, v_w)
    return (q, k, v)


def _fused_gate_up_silu_with_fallback(x, gate_w, up_w):
    """Fused gate+up+silu MLP with CPU fallback."""
    if _is_webgpu_device(x):
        return torch.ops.webgpu.fused_gate_up_silu(x, gate_w, up_w)
    # CPU fallback: gate_proj -> silu -> * up_proj
    gate = F.linear(x, gate_w)
    up = F.linear(x, up_w)
    return F.silu(gate) * up


# Map shader names to actual torch functions
SHADER_TO_FUNC = {
    "linear": F.linear,
    "relu": torch.relu,
    "silu": F.silu,
    "gelu": F.gelu,
    "softmax": F.softmax,
    "layer_norm": F.layer_norm,
    "embedding": F.embedding,
    "add": torch.add,
    "sub": torch.sub,
    "mul": torch.mul,
    "div": torch.div,
    "neg": torch.neg,
    "mm": torch.mm,
    "matmul": torch.matmul,
    "cos": torch.cos,
    "sin": torch.sin,
    "exp": torch.exp,
    "sqrt": torch.sqrt,
    "rsqrt": torch.rsqrt,
    "pow": torch.pow,
    "sum": torch.sum,
    "mean": torch.mean,
    "max": torch.max,
    "min": torch.min,
    "argmax": torch.argmax,
    "tanh": torch.tanh,
    "cat": torch.cat,
    "view": lambda x, *args: x.view(*args),
    "reshape": lambda x, *shape: x.reshape(*shape) if len(shape) > 1 else x.reshape(shape[0]),
    "transpose": torch.transpose,
    "permute": lambda x, *args: x.permute(*args),
    "unsqueeze": torch.unsqueeze,
    "squeeze": torch.squeeze,
    "contiguous": lambda x: x.contiguous(),
    "clone": torch.clone,
    "expand": lambda x, *args: x.expand(*args),
    "eq": torch.eq,
    "ne": torch.ne,
    "lt": torch.lt,
    "le": torch.le,
    "gt": torch.gt,
    "ge": torch.ge,
    "where": torch.where,
    "getitem": lambda x, idx: x[idx],  # indexing/slicing
    "fused_add_relu": lambda a, b: torch.relu(torch.add(a, b)),
    "fused_mul_silu": lambda a, b: a * F.silu(b),  # GLU pattern: gate * silu(up)
    "fused_add_silu": lambda a, b: F.silu(torch.add(a, b)),
    "fused_add_gelu": lambda a, b: F.gelu(torch.add(a, b)),
    # RMSNorm: Use fused C++ kernel if on WebGPU, otherwise CPU fallback
    "rms_norm": _rms_norm_with_fallback,
    # Fused WebGPU ops from fusion pass - with CPU fallbacks
    "fused_rms_norm": _rms_norm_with_fallback,
    "fused_qkv_proj": _fused_qkv_proj_with_fallback,
    "fused_gate_up_silu": _fused_gate_up_silu_with_fallback,
    "cast": lambda x, dtype: x.to(dtype) if dtype else x,  # Perform dtype cast
    "scaled_dot_product_attention": F.scaled_dot_product_attention,
    "repeat_interleave": torch.repeat_interleave,
    # Dtype casting methods (no dtype argument, implicit type)
    "float": lambda x: x.float(),  # cast to float32
    "half": lambda x: x.half(),    # cast to float16
    "int": lambda x: x.int(),      # cast to int32
    "long": lambda x: x.long(),    # cast to int64
    "bool": lambda x: x.bool(),    # cast to bool
    # Scalar extraction
    "item": lambda x: x.item(),    # extract scalar from single-element tensor
    # vmap batch dimension operations (use actual PyTorch implementations)
    "add_batch_dim": lambda *args, **kwargs: torch._functorch.predispatch._add_batch_dim(*args, **kwargs),
    "remove_batch_dim": lambda *args, **kwargs: torch._functorch.predispatch._remove_batch_dim(*args, **kwargs),
    "vmap_increment_nesting": lambda *args, **kwargs: torch._functorch.predispatch._vmap_increment_nesting(*args, **kwargs),
    "vmap_decrement_nesting": lambda *args, **kwargs: torch._functorch.predispatch._vmap_decrement_nesting(*args, **kwargs),
    # Other internal ops
    "lazy_load_decompositions": lambda *args, **kwargs: None,  # Just load decompositions, return nothing
    "enter_autocast": lambda *args, **kwargs: None,  # Autocast context enter
    "exit_autocast": lambda *args, **kwargs: None,  # Autocast context exit
    "log_api_usage": lambda *args, **kwargs: None,  # API logging, no-op
    # MoE ops
    "topk": torch.topk,
    "scatter": lambda x, dim, idx, src: x.scatter(dim, idx, src),
    "scatter_add": lambda x, dim, idx, src: x.scatter_add(dim, idx, src),
    "gather": torch.gather,
    "any": torch.any,
}




def _resolve_arg(arg, runtime: Runtime):
    """Recursively resolve FX Node arguments to actual tensor values."""
    if hasattr(arg, 'name'):
        # This is an FX Node, get its value from runtime
        if arg.name in runtime:
            return runtime[arg.name]
        # If not in runtime, might be a placeholder that wasn't used
        return arg
    elif isinstance(arg, (int, float, bool, type(None), torch.dtype)):
        return arg
    elif isinstance(arg, tuple):
        return tuple(_resolve_arg(a, runtime) for a in arg)
    elif isinstance(arg, list):
        return [_resolve_arg(a, runtime) for a in arg]
    elif isinstance(arg, slice):
        # Resolve any nodes inside the slice
        start = _resolve_arg(arg.start, runtime) if arg.start is not None else None
        stop = _resolve_arg(arg.stop, runtime) if arg.stop is not None else None
        step = _resolve_arg(arg.step, runtime) if arg.step is not None else None
        return slice(start, stop, step)
    else:
        return arg


def run_shader(node: LowIRRunShader, runtime: Runtime) -> torch.Tensor:
    # Build args in the same order as the original FX node args
    # by replacing tensor placeholders with actual tensors from runtime
    fx_args = node.high_ir_node.fx_node.args

    all_args = [_resolve_arg(arg, runtime) for arg in fx_args]

    if debug_enabled():
        debug(f"run_shader: {node.shader_name}")
        for i, a in enumerate(all_args):
            if hasattr(a, 'shape'):
                debug(f"  Arg {i}: tensor shape={a.shape}")
            else:
                debug(f"  Arg {i}: {type(a).__name__} = {a}")

    # Get kwargs and resolve any Node objects
    fx_kwargs = node.high_ir_node.fx_node.kwargs
    resolved_kwargs = {k: _resolve_arg(v, runtime) for k, v in fx_kwargs.items()}

    # shader_name can be an enum (with .value) or a plain string
    shader_name = node.shader_name.value if hasattr(node.shader_name, 'value') else node.shader_name

    # Look up the function
    if shader_name in SHADER_TO_FUNC:
        func = SHADER_TO_FUNC[shader_name]
        out = func(*all_args, **resolved_kwargs)
    elif hasattr(torch.ops.webgpu, shader_name):
        out = getattr(torch.ops.webgpu, shader_name)(*all_args, **resolved_kwargs)
    elif hasattr(torch, shader_name):
        out = getattr(torch, shader_name)(*all_args, **resolved_kwargs)
    else:
        raise Exception(
            f"I don't know where to put a relevant op for this shader: {node.shader_name}. Node: {node}"
        )
    if debug_enabled():
        debug(f"  Output: shape={out.shape if hasattr(out, 'shape') else 'N/A'}")
    runtime[node.value_id] = out
    return out


def move_to(node: LowIRMoveTo, runtime: Runtime):
    assert len(node.inputs) == 1
    assert node.to_device
    to_be_moved = runtime[node.inputs[0].name]
    result = to_be_moved.to(device=node.to_device)
    runtime[node.value_id] = result
    return result


def output(node: LowIROutput, runtime: Runtime):
    # torch.compile expects a tuple of outputs (matching FX GraphModule format)
    # The output format is (tensor1, tensor2, ...) wrapped in a tuple
    results = tuple(runtime[inp.name] for inp in node.inputs)
    if debug_enabled():
        debug(f"output: returning tuple with {len(results)} tensor(s)")
        for i, t in enumerate(results):
            debug(f"  [{i}] shape={t.shape if hasattr(t, 'shape') else 'N/A'}")
    return results


low_ir_to_webgpu_ops: dict[LowIROp, Callable] = {
    LowIROp.CREATE_BUFFER: create_buffer,
    LowIROp.WRITE_BUFFER: write_buffer,
    LowIROp.RUN_SHADER: run_shader,
    LowIROp.MOVE_TO: move_to,
    LowIROp.OUTPUT: output,
}


def lowering(nodes: List[LowIRNode], placeholder_names: List[str] = None) -> Callable:
    # Build list of ops to execute
    calls: List[Callable] = []
    for node in nodes:
        webgpu_op = low_ir_to_webgpu_ops.get(node.ir_op)
        if webgpu_op is not None:
            calls.append(partial(webgpu_op, node))
        else:
            debug(f"WebGPU op is none for LowIROp: {node.ir_op}")

    placeholder_names = placeholder_names or []

    def program(*args):
        # Initialize runtime with placeholder values
        runtime: Runtime = {}
        for name, value in zip(placeholder_names, args):
            runtime[name] = value
            if debug_enabled():
                debug(f"Placeholder '{name}': shape={value.shape if hasattr(value, 'shape') else 'N/A'}")

        # Execute all ops
        output = None
        for call in calls:
            output = call(runtime)
        if debug_enabled():
            debug(f"program returns: type={type(output)}, shape={output.shape if hasattr(output, 'shape') else 'N/A'}")
        return output

    return program
