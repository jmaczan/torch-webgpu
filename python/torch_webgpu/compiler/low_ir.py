from typing import Any, List
import torch
from enum import StrEnum, auto

from .compiler_pass import CompilerPass
from .ir import IRNode
from .high_ir import HighIRCreateTensor, HighIRNode, HighIROp


class LowIROp(StrEnum):
    CREATE_BUFFER = auto()
    WRITE_BUFFER = auto()
    RUN_SHADER = auto()
    MOVE_TO = auto()
    OUTPUT = auto()


class LowIRNode(IRNode):
    ir_op: LowIROp

    def __init__(
        self,
        high_ir_node: HighIRNode,
        value_id: Any = None,
        inputs: List[Any] = [],
    ):
        super().__init__(value_id, inputs)
        self.high_ir_node = high_ir_node

    def __str__(self):
        return self.ir_op


class LowIRCreateBuffer(LowIRNode):
    ir_op = LowIROp.CREATE_BUFFER
    shape = None
    stride = None
    dtype = None
    device = None
    numel = None
    size = None
    # TODO adjust the list, for now I copied it from High IR

    def __init__(
        self,
        high_ir_node: HighIRCreateTensor,
        value_id: Any = None,
        inputs: List[Any] = [],
    ):
        super().__init__(high_ir_node, value_id, inputs)
        self.shape = high_ir_node.shape
        self.dtype = high_ir_node.dtype
        self.device = high_ir_node.device
        self.numel = high_ir_node.numel
        self.stride = high_ir_node.stride
        self.size = high_ir_node.size
        # I don't put data here, because it should go to LowIRWriteBuffer


class LowIRWriteBuffer(LowIRNode):
    ir_op = LowIROp.WRITE_BUFFER
    shape = None
    stride = None
    dtype = None
    device = None
    numel = None
    size = None
    data = None
    constant_data = None

    def __init__(
        self,
        high_ir_node: HighIRCreateTensor,
        value_id: Any = None,
        inputs: List[Any] = [],
    ):
        super().__init__(high_ir_node, value_id, inputs)
        self.shape = high_ir_node.shape
        self.dtype = high_ir_node.dtype
        self.device = high_ir_node.device
        self.numel = high_ir_node.numel
        self.stride = high_ir_node.stride
        self.size = high_ir_node.size
        self.constant_data = high_ir_node.constant_data


class LowIRRunShader(LowIRNode):
    ir_op = LowIROp.RUN_SHADER

    def __init__(
        self, high_ir_node: HighIRNode, value_id: Any = None, inputs: List[Any] = []
    ):
        super().__init__(high_ir_node, value_id=value_id, inputs=inputs)
        self.shader_name = high_ir_node.ir_op


class LowIRMoveTo(LowIRNode):
    ir_op = LowIROp.MOVE_TO


class LowIROutput(LowIRNode):
    ir_op = LowIROp.OUTPUT


high_ir_op_to_low_ir_op: dict[HighIROp, list[LowIROp]] = {
    HighIROp.CREATE_TENSOR: [LowIROp.CREATE_BUFFER, LowIROp.WRITE_BUFFER],
    HighIROp.FUSED_ADD_RELU: [LowIROp.RUN_SHADER],
    HighIROp.MOVE_TO: [LowIROp.MOVE_TO],
    HighIROp.OUTPUT: [LowIROp.OUTPUT],
}

low_ir_op_to_low_ir_node: dict[LowIROp, type[LowIRNode]] = {
    LowIROp.CREATE_BUFFER: LowIRCreateBuffer,
    LowIROp.WRITE_BUFFER: LowIRWriteBuffer,
    LowIROp.RUN_SHADER: LowIRRunShader,
    LowIROp.MOVE_TO: LowIRMoveTo,
    LowIROp.OUTPUT: LowIROutput,
}

low_ir_compiler_passes: list[CompilerPass[LowIRNode]] = []  # TODO


def get_low_ir_node(high_ir_op: HighIROp, high_ir_node: HighIRNode):
    low_ir_ops = high_ir_op_to_low_ir_op.get(high_ir_op)
    if not low_ir_ops or len(low_ir_ops) == 0:
        print(f"Didn't find a Low IR Op for High IR Op: {high_ir_op}")
        return None
    low_ir_nodes = []
    for op in low_ir_ops:
        ir_node_type = low_ir_op_to_low_ir_node.get(op)
        if ir_node_type:
            ir_node = ir_node_type(
                high_ir_node, value_id=high_ir_node.value_id, inputs=high_ir_node.inputs
            )
            low_ir_nodes.append(ir_node)
        else:
            print(f"Didn't find a Low IR Node for Low IR Op: {op}")
    return low_ir_nodes


def high_ir_to_low_ir(high_ir_graph: List[HighIRNode]) -> List[LowIRNode]:
    ir_graph: list[LowIRNode] = []
    for i, node in enumerate(high_ir_graph):
        ir_nodes = get_low_ir_node(high_ir_op=node.ir_op, high_ir_node=node)
        if ir_nodes:
            for node in ir_nodes:
                ir_graph.append(node)
        else:
            print(
                f"Unsupported FX op: {node.ir_op} (node: {node}). ir_graph: {ir_graph}"
            )
            raise Exception(f"Unsupported FX op: {node.ir_op} for node: {node}")
    return ir_graph


def low_ir_print_tabular(nodes: List[LowIRNode]) -> None:
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

    node_specs = [[n.ir_op, n.high_ir_node, n.value_id, n.inputs] for n in nodes]
    print(
        tabulate(
            node_specs,
            headers=[
                "opcode",
                "high ir node",
                "value_id",
                "inputs",
            ],
        )
        + "\n"
    )
