from typing import Any, Callable, List

from .low_ir import LowIRNode, LowIROp
import torch_webgpu
import torch

low_ir_to_webgpu_ops: dict[LowIROp, Callable] = {
    LowIROp.CREATE_BUFFER: torch.ops.webgpu.create_buffer,
    LowIROp.WRITE_BUFFER: torch.ops.webgpu.write_buffer,
}


def lowering(nodes: List[LowIRNode]) -> Callable:
    # hardcode to just see if the whole pipeline works
    calls: List[Callable] = []
    # TODO: use nodes instead of ops
    for node in nodes:
        webgpu_op = low_ir_to_webgpu_ops.get(node.ir_op)
        if webgpu_op is not None:
            calls.append(webgpu_op)
        else:
            print(f"WebGPU op is none for LowIROp: {node.ir_op}")

    def program():
        output = None
        for call in calls:
            output = call(output)
        return output

    return program
