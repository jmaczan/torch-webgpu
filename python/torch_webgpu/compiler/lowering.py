from typing import Any, Callable, List
from functools import partial
from .low_ir import (
    LowIRCreateBuffer,
    LowIRNode,
    LowIROp,
    LowIRRunShader,
    LowIRWriteBuffer,
)
import torch_webgpu
import torch

Runtime = dict


def create_buffer(
    node: LowIRCreateBuffer, runtime: Runtime
) -> Callable[[], torch.Tensor]:
    # TODO: take into account wher the buffer should be allocated (device)
    buf = torch.ops.webgpu.create_buffer(
        node.size,
        node.stride,
        node.dtype,
    )
    runtime[node.value_id] = buf
    return buf


def write_buffer(
    node: LowIRWriteBuffer, runtime: Runtime
) -> Callable[[], torch.Tensor]:
    # TODO: take into account wher the buffer should be allocated (device)
    dst = runtime[node.value_id]
    src = torch.tensor(node.constant_data)  # TODO: handle also other kinds of data
    return torch.ops.webgpu.write_buffer(dst, src)


def run_shader(node: LowIRRunShader, runtime: Runtime):
    inputs = {}
    for node_input in node.inputs:
        inputs[node_input.name] = runtime[node_input.name]
    assert len(inputs) == len(node.inputs)
    # TODO: this is a mock implementation, because I need to rethink
    # if I want to implement a generic machinery for running
    # an arbitraty shader or just want to invoke a
    # particular shader with known parameters etc
    out = inputs["a"].data + inputs["b"].data
    return out  # TODO


low_ir_to_webgpu_ops: dict[LowIROp, Callable] = {
    LowIROp.CREATE_BUFFER: create_buffer,
    LowIROp.WRITE_BUFFER: write_buffer,
    LowIROp.RUN_SHADER: run_shader,
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
            output = call(runtime)
        return output

    return program
