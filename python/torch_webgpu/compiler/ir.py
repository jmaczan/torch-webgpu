import torch


class IRNode:
    def __init__(self, fx_node=None):
        super().__init__()
        self.fx_node = fx_node
        self.operator = None

    def __call__(self, operator=None, fx_node=None):
        print(f"IR Node call operator = {operator}")
        if operator:
            self.operator = operator
        if fx_node:
            self.fx_node = fx_node


class IRTensor(IRNode):
    def __init__(self, name: str = None, *args, **kwargs):
        self.name = name
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


fx_op_to_ir_op = {
    torch.tensor: "tensor",
    "add": "add",
    torch.relu: "relu",
    "to": "to",
    "output": "output",
}

ir_op_to_ir_node = {
    "tensor": IRTensor,
    "add": IRPointwise,
    "relu": IROperation,
    "to": IROperation,
    "output": IRReturn,
}


def get_ir(fx_operator):
    ir_op = fx_op_to_ir_op.get(fx_operator)
    ir_node = ir_op_to_ir_node.get(ir_op)
    if ir_node:
        ir_node(operator=ir_op)
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
            ir_graph.append(ir_node(node.meta))
        else:
            print(f"Unsupported FX op: {node.target}. ir_graph: {ir_graph}")
            raise Exception(f"Unsupported FX op: {node.target}")
        print(f"#{i}: node:{node} node.target:{node.target}")
    return ir_graph
