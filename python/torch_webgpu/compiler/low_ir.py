import torch
from enum import StrEnum, auto

from .compiler_pass import CompilerPass
from .ir import IRNode
from .high_ir import HighIRNode, HighIROp


class LowIROp(StrEnum):
    CREATE_BUFFER = auto()
    WRITE_BUFFER = auto()
    RUN_SHADER = auto()


class LowIRNode(IRNode):
    def __init__(self, high_ir_node: HighIRNode):
        super().__init__()
        self.high_ir_node = high_ir_node


class LowIRCreateBuffer(LowIRNode):
    ir_op = LowIROp.CREATE_BUFFER


class LowIRWriteBuffer(LowIRNode):
    ir_op = LowIROp.WRITE_BUFFER


class LowIRRunShader(LowIRNode):
    ir_op = LowIROp.RUN_SHADER

    def __init__(self, high_ir_node: HighIRNode):
        super().__init__(high_ir_node)
        self.shader_name = high_ir_node.ir_op


high_ir_op_to_low_ir_op: dict[HighIROp, list[LowIROp]] = {
    HighIROp.CREATE_TENSOR: [LowIROp.CREATE_BUFFER, LowIROp.WRITE_BUFFER],
    HighIROp.FUSED_ADD_RELU: [LowIROp.RUN_SHADER],
}

low_ir_op_to_low_ir_node: dict[LowIROp, type[LowIRNode]] = {
    LowIROp.CREATE_BUFFER: LowIRCreateBuffer,
    LowIROp.WRITE_BUFFER: LowIRWriteBuffer,
    LowIROp.RUN_SHADER: LowIRRunShader,
}

low_ir_compiler_passes: list[CompilerPass[LowIRNode]] = []  # TODO


def get_low_ir(high_ir_op, high_ir_node):
    low_ir_ops = high_ir_op_to_low_ir_op.get(high_ir_op)
    if not low_ir_ops or len(low_ir_ops) == 0:
        return None
    low_ir_nodes = []
    for op in low_ir_ops:
        ir_node_type = low_ir_op_to_low_ir_node.get(op)
        if ir_node_type:
            ir_node = ir_node_type(high_ir_node)
            low_ir_nodes.append(ir_node)
    return low_ir_nodes


def high_ir_to_low_ir(high_ir_graph):
    ir_graph: list[IRNode] = []
    for i, node in enumerate(high_ir_graph):
        ir_nodes = get_low_ir(high_ir_op=node.ir_op, high_ir_node=node)
        if ir_nodes:
            for node in ir_nodes:
                ir_graph.append(node)
        else:
            print(
                f"Unsupported FX op: {node.ir_op} (node: {node}). ir_graph: {ir_graph}"
            )
            raise Exception(f"Unsupported FX op: {node.ir_op} for node: {node}")
    return ir_graph
