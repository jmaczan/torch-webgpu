from typing import Any
import torch
from enum import StrEnum, auto
from .ir import IRNode
from .high_ir import HighIROp


class LowIROp(StrEnum):
    CREATE_BUFFER = auto()
    WRITE_BUFFER = auto()


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


high_ir_op_to_low_ir_op: dict[HighIROp, list[LowIROp]] = {
    HighIROp.CREATE_TENSOR: [LowIROp.CREATE_BUFFER, LowIROp.WRITE_BUFFER]
}

low_ir_op_to_low_ir_node: dict[LowIROp, type[IRNode]] = {
    LowIROp.CREATE_BUFFER: LowIRCreateBuffer,
    LowIROp.WRITE_BUFFER: LowIRWriteBuffer,
}

low_ir_compiler_passes = []


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
        ir_nodes = get_low_ir(node.target)
        if ir_nodes:
            for node in ir_nodes:
                node(fx_node=node)
        else:
            source_fn_stack = node.meta.get("source_fn_stack")
            if source_fn_stack and len(source_fn_stack) > 0:
                source_fn_stack = source_fn_stack[0]
                if source_fn_stack and len(source_fn_stack) > 0:
                    node_key = source_fn_stack[0]
                    if node_key:
                        ir_nodes = get_low_ir(node_key)
        if ir_nodes:
            for node in ir_nodes:
                ir_graph.append(node)
        else:
            print(f"Unsupported FX op: {node.target}. ir_graph: {ir_graph}")
            raise Exception(f"Unsupported FX op: {node.target}")
    return ir_graph
