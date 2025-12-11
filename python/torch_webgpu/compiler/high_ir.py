from typing import Any, Optional
import torch
from enum import StrEnum, auto
from .ir import IRNode


class HighIROp(StrEnum):
    CREATE_TENSOR = auto()
    ADD = auto()
    RELU = auto()
    FUSED_ADD_RELU = auto()
    MOVE_TO = auto()
    OUTPUT = auto()


class HighIRCreateTensor(IRNode):
    ir_op = HighIROp.CREATE_TENSOR

    def __init__(self, operator=None, *args, **kwargs):
        self.operator = operator
        self.args = args
        self.kwargs = kwargs

    # def __call__(self):
    #     # torch_webgpu.create_buffer or smth like this
    #     return None


class HighIRAdd(IRNode):
    ir_op = HighIROp.ADD

    def __init__(self, operator=None, *args, **kwargs):
        self.operator = operator
        self.args = args
        self.kwargs = kwargs


class HighIRRelu(IRNode):
    ir_op = HighIROp.RELU

    def __init__(self, operator=None, *args, **kwargs):
        self.operator = operator
        self.args = args
        self.kwargs = kwargs


class HighIRFusedAddRelu(IRNode):
    ir_op = HighIROp.FUSED_ADD_RELU

    def __init__(self, operator=None, *args, **kwargs):
        self.operator = operator
        self.args = args
        self.kwargs = kwargs


class HighIRMoveTo(IRNode):
    ir_op = HighIROp.MOVE_TO

    def __init__(self, operator=None, *args, **kwargs):
        self.operator = operator
        self.args = args
        self.kwargs = kwargs


class HighIROutput(IRNode):
    ir_op = HighIROp.OUTPUT

    def __init__(self, operator=None, *args, **kwargs):
        self.operator = operator
        self.args = args
        self.kwargs = kwargs


fx_op_to_high_ir_op: dict[Any, HighIROp] = {
    torch.tensor: HighIROp.CREATE_TENSOR,
    "add": HighIROp.ADD,
    torch.relu: HighIROp.RELU,
    "to": HighIROp.MOVE_TO,
    "output": HighIROp.OUTPUT,
}

high_ir_op_to_high_ir_node: dict[HighIROp, type[IRNode]] = {
    HighIROp.CREATE_TENSOR: HighIRCreateTensor,
    HighIROp.ADD: HighIRAdd,
    HighIROp.RELU: HighIRRelu,
    HighIROp.MOVE_TO: HighIRMoveTo,
    HighIROp.OUTPUT: HighIROutput,
    HighIROp.FUSED_ADD_RELU: HighIRFusedAddRelu,
}


def get_high_ir(fx_operator) -> Optional[IRNode]:
    ir_op = fx_op_to_high_ir_op.get(fx_operator)
    if not ir_op:
        return None
    ir_node_type = high_ir_op_to_high_ir_node.get(ir_op)
    if ir_node_type:
        ir_node = ir_node_type()
        ir_node.operator = ir_op
    return ir_node


def fx_to_high_ir(gm: torch.fx.GraphModule) -> list[IRNode]:
    ir_graph: list[IRNode] = []
    for i, node in enumerate(gm.graph.nodes):
        ir_node = get_high_ir(node.target)
        if ir_node:
            ir_node(fx_node=node)
        else:
            source_fn_stack = node.meta.get("source_fn_stack")
            if source_fn_stack and len(source_fn_stack) > 0:
                source_fn_stack = source_fn_stack[0]
                if source_fn_stack and len(source_fn_stack) > 0:
                    node_key = source_fn_stack[0]
                    if node_key:
                        ir_node = get_high_ir(node_key)
        if ir_node:
            ir_graph.append(ir_node)
        else:
            print(f"Unsupported FX op: {node.target}. ir_graph: {ir_graph}")
            raise Exception(f"Unsupported FX op: {node.target}")
    return ir_graph
