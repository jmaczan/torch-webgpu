from typing import Any, List, Optional
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
    ir_op: HighIROp
    value_id: Any = None
    inputs: List[Any] = []

    def __init__(
        self, fx_node: torch.fx.Node, value_id: Any = None, inputs: List[Any] = []
    ):
        self.fx_node = fx_node
        if value_id:
            self.value_id = value_id
        if inputs:
            self.inputs = inputs

    def __str__(self):
        return self.ir_op


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


def get_high_ir_node(fx_op, fx_node: torch.fx.Node) -> Optional[HighIRNode]:
    ir_op = fx_op_to_high_ir_op.get(fx_op)
    if not ir_op:
        return None
    ir_node_type = high_ir_op_to_high_ir_node.get(ir_op)
    if ir_node_type:
        ir_node = ir_node_type(
            fx_node=fx_node, value_id=fx_node.name, inputs=fx_node.all_input_nodes
        )
    return ir_node


def fx_to_high_ir(gm: torch.fx.GraphModule) -> list[HighIRNode]:
    ir_graph: list[HighIRNode] = []
    for i, node in enumerate(gm.graph.nodes):
        ir_node = get_high_ir_node(fx_op=node.target, fx_node=node)
        if not ir_node:
            source_fn_stack = node.meta.get("source_fn_stack")
            if source_fn_stack and len(source_fn_stack) > 0:
                source_fn_stack = source_fn_stack[0]
                if source_fn_stack and len(source_fn_stack) > 0:
                    node_key = source_fn_stack[0]
                    if node_key:
                        ir_node = get_high_ir_node(fx_op=node_key, fx_node=node)
        if ir_node:
            ir_graph.append(ir_node)
        else:
            print(f"Unsupported FX op: {node.target}. ir_graph: {ir_graph}")
            raise Exception(f"Unsupported FX op: {node.target}")
    return ir_graph


def high_ir_print_tabular(nodes: List[HighIRNode]) -> None:
    if nodes is None or len(nodes) == 0:
        print("IR Nodes list is empty")
        return None

    # took most of the code from PyTorch torch/fx/graph.py
    try:
        from tabulate import tabulate
    except ImportError:
        print(
            "`print_tabular` relies on the library `tabulate`, "
            "which could not be found on this machine. Run `pip "
            "install tabulate` to install the library."
        )
        raise

    node_specs = [
        [
            n.ir_op,
            n.value_id,
            n.inputs,
            n.fx_node.args,
            n.fx_node.kwargs,
        ]
        for n in nodes
    ]
    print(
        tabulate(
            node_specs,
            headers=[
                "opcode",
                "value_id",
                "inputs",
                "args",
                "kwargs",
            ],
        )
    )
