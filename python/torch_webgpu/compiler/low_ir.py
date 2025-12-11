from typing import Any
import torch
from enum import StrEnum, auto
from .ir import IRNode
from .high_ir import HighIROp


class LowIROp(StrEnum):
    CREATE_TENSOR = auto()
    ADD = auto()
    RELU = auto()
    FUSED_ADD_RELU = auto()
    MOVE_TO = auto()
    OUTPUT = auto()


class LowIRCreateTensor(IRNode):
    ir_op = LowIROp.CREATE_TENSOR

    def __init__(self, operator=None, *args, **kwargs):
        self.operator = operator
        self.args = args
        self.kwargs = kwargs

    # def __call__(self):
    #     # torch_webgpu.create_buffer or smth like this
    #     return None


class LowIRAdd(IRNode):
    ir_op = LowIROp.ADD

    def __init__(self, operator=None, *args, **kwargs):
        self.operator = operator
        self.args = args
        self.kwargs = kwargs


class LowIRRelu(IRNode):
    ir_op = LowIROp.RELU

    def __init__(self, operator=None, *args, **kwargs):
        self.operator = operator
        self.args = args
        self.kwargs = kwargs


class LowIRFusedAddRelu(IRNode):
    ir_op = LowIROp.FUSED_ADD_RELU

    def __init__(self, operator=None, *args, **kwargs):
        self.operator = operator
        self.args = args
        self.kwargs = kwargs


class LowIRMoveTo(IRNode):
    ir_op = LowIROp.MOVE_TO

    def __init__(self, operator=None, *args, **kwargs):
        self.operator = operator
        self.args = args
        self.kwargs = kwargs


class LowIROutput(IRNode):
    ir_op = LowIROp.OUTPUT

    def __init__(self, operator=None, *args, **kwargs):
        self.operator = operator
        self.args = args
        self.kwargs = kwargs


fx_op_to_low_ir_op: dict[Any, LowIROp] = {
    torch.tensor: LowIROp.CREATE_TENSOR,
    "add": LowIROp.ADD,
    torch.relu: LowIROp.RELU,
    "to": LowIROp.MOVE_TO,
    "output": LowIROp.OUTPUT,
}

low_ir_op_to_low_ir_node: dict[LowIROp, type[IRNode]] = {
    LowIROp.CREATE_TENSOR: LowIRCreateTensor,
    LowIROp.ADD: LowIRAdd,
    LowIROp.RELU: LowIRRelu,
    LowIROp.MOVE_TO: LowIRMoveTo,
    LowIROp.OUTPUT: LowIROutput,
    LowIROp.FUSED_ADD_RELU: LowIRFusedAddRelu,
}


def get_low_ir(fx_operator):
    ir_op = fx_op_to_low_ir_op.get(fx_operator)
    if not ir_op:
        return None
    ir_node_type = low_ir_op_to_low_ir_node.get(ir_op)
    if ir_node_type:
        ir_node = ir_node_type()
        ir_node.operator = ir_op
    return ir_node


def high_ir_to_low_ir(high_ir_graph):
    ir_graph: list[IRNode] = []
    for i, node in enumerate(gm.graph.nodes):
        ir_node = get_low_ir(node.target)
        if ir_node:
            ir_node(fx_node=node)
        else:
            source_fn_stack = node.meta.get("source_fn_stack")
            if source_fn_stack and len(source_fn_stack) > 0:
                source_fn_stack = source_fn_stack[0]
                if source_fn_stack and len(source_fn_stack) > 0:
                    node_key = source_fn_stack[0]
                    if node_key:
                        ir_node = get_low_ir(node_key)
        if ir_node:
            ir_graph.append(ir_node)
        else:
            print(f"Unsupported FX op: {node.target}. ir_graph: {ir_graph}")
            raise Exception(f"Unsupported FX op: {node.target}")
    return ir_graph
