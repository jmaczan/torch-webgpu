import torch
from torch._dynamo import register_backend
from typing import Callable, List
from .ir import IRPointwise, IRTensor, IROperation, IRReturn

fx_to_ir_dict = {
    torch.tensor: IRTensor,
    "add": IRPointwise,
    torch.relu: IROperation,
    "to": IROperation,
    "output": IRReturn,
}


def get_ir(fx_operator):
    ir_node = fx_to_ir_dict.get(fx_operator)
    if ir_node:
        ir_node(operator=fx_operator)
    return ir_node


@register_backend
def webgpu_backend(
    gm: torch.fx.GraphModule, example_inputs: List[torch.Tensor]
) -> Callable:
    gm.graph.print_tabular()
    print(gm.graph)

    ir_graph = []  #: List[IRNode]'
    for i, node in enumerate(gm.graph.nodes):
        ir_node = get_ir(node.target)
        if ir_node:
            ir_node(node)
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
            print(f"Unsupported FX op: {node.target}. Current ir_graph: {ir_graph}")
            raise Exception(
                f"Unsupported FX op: {node.target}. Current ir_graph: {ir_graph}"
            )
        print(
            f"#{i}: node:{node} node.target:{node.target}"  # Noqa E501
        )  # Noqa E501

    # TODO: see if it's still relevant
    # https://docs.pytorch.org/docs/stable/generated/torch.jit.optimize_for_inference.html
    return gm.forward
