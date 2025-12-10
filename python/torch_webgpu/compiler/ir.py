from typing import Any
import torch
from enum import StrEnum, auto


class IRNode:
    def __init__(self, operator=None, fx_node=None):
        super().__init__()
        print(f"IR Node init operator = {operator}")
        self.operator = operator
        self.fx_node = fx_node

    def __call__(self, operator=None, fx_node=None):
        print(f"IR Node call operator = {operator}")
        if operator:
            self.operator = operator
        if fx_node:
            self.fx_node = fx_node


class IRTensor(IRNode):
    def __init__(self, operator=None, *args, **kwargs):
        self.operator = operator
        self.args = args
        self.kwargs = kwargs


class IRPointwise(IRNode):
    def __init__(self, operator=None, *args, **kwargs):
        self.operator = operator
        self.args = args
        self.kwargs = kwargs


class IROperation(IRNode):
    def __init__(self, operator=None, *args, **kwargs):
        self.operator = operator
        self.args = args
        self.kwargs = kwargs


class IRReturn(IRNode):
    def __init__(self, operator=None, *args, **kwargs):
        self.operator = operator
        self.args = args
        self.kwargs = kwargs


class IROp(StrEnum):
    TENSOR = auto()
    ADD = auto()
    RELU = auto()
    TO = auto()
    OUTPUT = auto()


fx_op_to_ir_op: dict[Any, IROp] = {
    torch.tensor: IROp.TENSOR,
    "add": IROp.ADD,
    torch.relu: IROp.RELU,
    "to": IROp.TO,
    "output": IROp.OUTPUT,
}

ir_op_to_ir_node: dict[IROp, type[IRNode]] = {
    IROp.TENSOR: IRTensor,
    IROp.ADD: IRPointwise,
    IROp.RELU: IROperation,
    IROp.TO: IROperation,
    IROp.OUTPUT: IRReturn,
}


def get_ir(fx_operator):
    ir_op = fx_op_to_ir_op.get(fx_operator)
    if not ir_op:
        return None
    ir_node_type = ir_op_to_ir_node.get(ir_op)
    if ir_node_type:
        ir_node = ir_node_type()
        ir_node.operator = ir_op
    return ir_node


def fx_to_ir(gm):
    ir_graph: list[IRNode] = []
    for i, node in enumerate(gm.graph.nodes):
        ir_node = get_ir(node.target)
        if ir_node:
            ir_node(fx_node=node)
        else:
            source_fn_stack = node.meta.get("source_fn_stack")
            if source_fn_stack and len(source_fn_stack) > 0:
                source_fn_stack = source_fn_stack[0]
                if source_fn_stack and len(source_fn_stack) > 0:
                    node_key = source_fn_stack[0]
                    if node_key:
                        ir_node = get_ir(node_key)
        if ir_node:
            ir_graph.append(ir_node)
        else:
            print(f"Unsupported FX op: {node.target}. ir_graph: {ir_graph}")
            raise Exception(f"Unsupported FX op: {node.target}")
        print(f"#{i}: node:{node} node.target:{node.target}")
    return ir_graph
