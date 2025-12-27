from typing import Any, Callable, List
from functools import partial
from .low_ir import LowIRCreateBuffer, LowIRNode, LowIROp, LowIRWriteBuffer
import torch_webgpu
import torch

Runtime = dict


def create_buffer(
    node: LowIRCreateBuffer, runtime: Runtime
) -> Callable[[], torch.Tensor]:
    def _call():
        buf = torch.ops.webgpu.create_buffer(
            node.size,
            node.stride,
            node.dtype,
        )
        runtime[node.value_id] = buf
        return buf

    return _call


def write_buffer(
    node: LowIRWriteBuffer, runtime: Runtime
) -> Callable[[], torch.Tensor]:
    def _call():
        dst = runtime[node.value_id]
        src = torch.tensor(node.constant_data)  # TODO: handle also other kinds of data
        return torch.ops.webgpu.write_buffer(dst, src)

    return _call


low_ir_to_webgpu_ops: dict[LowIROp, Callable] = {
    LowIROp.CREATE_BUFFER: create_buffer,
    LowIROp.WRITE_BUFFER: write_buffer,
}


def lowering(nodes: List[LowIRNode]) -> Callable:
    runtime: Runtime = {}
    # hardcode to just see if the whole pipeline works
    calls: List[Callable] = []
    # TODO: use nodes instead of ops
    for node in nodes:
        webgpu_op = low_ir_to_webgpu_ops.get(node.ir_op)
        if webgpu_op is not None:
            calls.append(partial(webgpu_op, node))
        else:
            print(f"WebGPU op is none for LowIROp: {node.ir_op}")

    def program():
        # ultra naive and non-flexible, just to start with something
        output = None
        for call in calls:
            output = call(runtime)()
        return output

    return program
