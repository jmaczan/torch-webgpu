from typing import Any, Optional
import torch
from enum import StrEnum, auto

from .compiler_pass import CompilerPass, Transform, Pattern
from .ir import IRNode


class HighIROp(StrEnum):
    CREATE_TENSOR = auto()
    ADD = auto()
    RELU = auto()
    FUSED_ADD_RELU = auto()
    MOVE_TO = auto()
    OUTPUT = auto()


class HighIRNode(IRNode):
    def __init__(self, fx_node: torch.fx.Node):
        self.fx_node = fx_node


class HighIRCreateTensor(HighIRNode):
    ir_op = HighIROp.CREATE_TENSOR


class HighIRAdd(HighIRNode):
    ir_op = HighIROp.ADD


class HighIRRelu(HighIRNode):
    ir_op = HighIROp.RELU


class HighIRFusedAddRelu(HighIRNode):
    ir_op = HighIROp.FUSED_ADD_RELU


class HighIRMoveTo(HighIRNode):
    ir_op = HighIROp.MOVE_TO


class HighIROutput(HighIRNode):
    ir_op = HighIROp.OUTPUT


fx_op_to_high_ir_op: dict[Any, HighIROp] = {
    torch.tensor: HighIROp.CREATE_TENSOR,
    "add": HighIROp.ADD,
    torch.relu: HighIROp.RELU,
    "to": HighIROp.MOVE_TO,
    "output": HighIROp.OUTPUT,
}

high_ir_op_to_high_ir_node: dict[HighIROp, type[HighIRNode]] = {
    HighIROp.CREATE_TENSOR: HighIRCreateTensor,
    HighIROp.ADD: HighIRAdd,
    HighIROp.RELU: HighIRRelu,
    HighIROp.MOVE_TO: HighIRMoveTo,
    HighIROp.OUTPUT: HighIROutput,
    HighIROp.FUSED_ADD_RELU: HighIRFusedAddRelu,
}

high_ir_compiler_passes: list[CompilerPass[HighIRNode]] = [
    CompilerPass(
        transforms=[
            Transform(
                pattern=[
                    Pattern("ir_op", HighIROp.ADD),
                    Pattern("ir_op", HighIROp.RELU),
                ],
                output=HighIROp.FUSED_ADD_RELU,
            )
        ]
    ),
]


def get_high_ir(fx_op, fx_node: torch.fx.Node) -> Optional[HighIRNode]:
    ir_op = fx_op_to_high_ir_op.get(fx_op)
    if not ir_op:
        return None
    ir_node_type = high_ir_op_to_high_ir_node.get(ir_op)
    if ir_node_type:
        ir_node = ir_node_type(fx_node)
    return ir_node


def fx_to_high_ir(gm: torch.fx.GraphModule) -> list[HighIRNode]:
    ir_graph: list[HighIRNode] = []
    for i, node in enumerate(gm.graph.nodes):
        ir_node = get_high_ir(fx_op=node.target, fx_node=node)
        if not ir_node:
            source_fn_stack = node.meta.get("source_fn_stack")
            if source_fn_stack and len(source_fn_stack) > 0:
                source_fn_stack = source_fn_stack[0]
                if source_fn_stack and len(source_fn_stack) > 0:
                    node_key = source_fn_stack[0]
                    if node_key:
                        ir_node = get_high_ir(fx_op=node_key, fx_node=node)
        if ir_node:
            ir_graph.append(ir_node)
        else:
            print(f"Unsupported FX op: {node.target}. ir_graph: {ir_graph}")
            raise Exception(f"Unsupported FX op: {node.target}")
    return ir_graph
