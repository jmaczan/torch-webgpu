from typing import Any, Callable, List
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
import torch_webgpu
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


def run_shader(node: LowIRRunShader, runtime: Runtime) -> torch.Tensor:
    inputs = {}
    for node_input in node.inputs:
        inputs[node_input.name] = runtime[node_input.name]
    assert len(inputs) == len(node.inputs)
    # TODO: this is a mock implementation, because I need to rethink
    # if I want to implement a generic machinery for running
    # an arbitraty shader or just want to invoke a
    # particular shader with known parameters etc
    out = torch.ops.webgpu.fused_add_relu(inputs["a"], inputs["b"])
    # out = torch.relu(inputs["a"].data + inputs["b"].data)
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
    assert len(node.inputs) == 1
    return runtime[node.inputs[0].name]


low_ir_to_webgpu_ops: dict[LowIROp, Callable] = {
    LowIROp.CREATE_BUFFER: create_buffer,
    LowIROp.WRITE_BUFFER: write_buffer,
    LowIROp.RUN_SHADER: run_shader,
    LowIROp.MOVE_TO: move_to,
    LowIROp.OUTPUT: output,
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
        # ultra naive and non-flexible "scheduler", just to start with something
        output = None
        for call in calls:
            output = call(runtime)
        return output

    return program
