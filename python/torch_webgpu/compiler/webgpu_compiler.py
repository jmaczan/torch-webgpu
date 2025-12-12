import torch
from torch._dynamo import register_backend
from typing import Callable, List

from .ir import IRNode
from .high_ir import HighIROp, fx_to_high_ir, high_ir_op_to_high_ir_node
from .low_ir import high_ir_to_low_ir
from .compiler_pass import CompilerPass, Transform, Pattern


@register_backend
def webgpu_backend(
    gm: torch.fx.GraphModule, example_inputs: List[torch.Tensor]
) -> Callable:
    gm.graph.print_tabular()

    high_ir = fx_to_high_ir(gm)
    high_ir = run_compiler_passes(high_ir)
    # low_ir = high_ir_to_low_ir(high_ir)

    # BUILD A COMPILED FN (closure? lambda?) AND RETURN

    # Noqa 501 TODO: see if it's still relevant https://docs.pytorch.org/docs/stable/generated/torch.jit.optimize_for_inference.html
    return gm.forward


def run_compiler_passes(input_ir_graph: list[IRNode]):
    passes = [
        CompilerPass(
            transforms=[
                Transform(
                    pattern=[
                        Pattern("operator", HighIROp.ADD),
                        Pattern("operator", HighIROp.RELU),
                    ],
                    output=HighIROp.FUSED_ADD_RELU,
                )
            ]
        ),
    ]
    output_ir_graph = []
    for p, compiler_pass in enumerate(passes):
        output_ir_graph = compiler_pass.run(
            ir_graph=input_ir_graph, ir_op_to_ir_node=high_ir_op_to_high_ir_node
        )
        input_ir_graph = output_ir_graph
    return output_ir_graph
