from typing import Any
import torch
from enum import StrEnum, auto
from .ir import IRNode
from .high_ir import HighIROp


class LowIROp(StrEnum):
    CREATE_BUFFER = auto()
    WRITE_BUFFER = auto()
    RUN_SHADER = auto()


class LowIRCreateBuffer(IRNode):
    ir_op = LowIROp.CREATE_BUFFER

    def __init__(self, operator=None, *args, **kwargs):
        self.operator = operator
        self.args = args
        self.kwargs = kwargs


class LowIRWriteBuffer(IRNode):
    ir_op = LowIROp.CREATE_BUFFER

    def __init__(self, operator=None, *args, **kwargs):
        self.operator = operator
        self.args = args
        self.kwargs = kwargs


class LowIRRunShader(IRNode):
    ir_op = LowIROp.CREATE_BUFFER

    def __init__(self, operator=None, *args, **kwargs):
        self.operator = operator
        self.args = args
        self.kwargs = kwargs

        # TODO: somehow store which shader it should run


high_ir_op_to_low_ir_op: dict[HighIROp, list[LowIROp]] = {
    HighIROp.CREATE_TENSOR: [LowIROp.CREATE_BUFFER, LowIROp.WRITE_BUFFER],
    HighIROp.FUSED_ADD_RELU: [LowIROp.RUN_SHADER],
}

low_ir_op_to_low_ir_node: dict[LowIROp, type[IRNode]] = {
    LowIROp.CREATE_BUFFER: LowIRCreateBuffer,
    LowIROp.WRITE_BUFFER: LowIRWriteBuffer,
    LowIROp.RUN_SHADER: LowIRRunShader,
}

low_ir_compiler_passes = []  # TODO


def get_low_ir(high_ir_op):
    low_ir_ops = high_ir_op_to_low_ir_op.get(high_ir_op)
    if not low_ir_ops or len(low_ir_ops) == 0:
        return None
    low_ir_nodes = []
    for op in low_ir_ops:
        ir_node_type = low_ir_op_to_low_ir_node.get(op)
        if ir_node_type:
            ir_node = ir_node_type()
            ir_node.operator = op
            low_ir_nodes.append(ir_node)
    return low_ir_nodes


def high_ir_to_low_ir(high_ir_graph):
    ir_graph: list[IRNode] = []
    for i, node in enumerate(high_ir_graph):
        ir_nodes = get_low_ir(node.ir_op)
        if ir_nodes:
            for node in ir_nodes:
                node(fx_node=node)
                ir_graph.append(node)
        else:
            print(
                f"Unsupported FX op: {node.ir_op} (node: {node}). ir_graph: {ir_graph}"
            )
            raise Exception(f"Unsupported FX op: {node.ir_op} for node: {node}")
    return ir_graph
