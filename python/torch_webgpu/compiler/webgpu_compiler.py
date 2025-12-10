import torch
from torch._dynamo import register_backend
from typing import Callable, List
from .ir import IROp, fx_to_ir
from .compiler_pass import CompilerPass, Transform, Pattern


@register_backend
def webgpu_backend(
    gm: torch.fx.GraphModule, example_inputs: List[torch.Tensor]
) -> Callable:
    gm.graph.print_tabular()
    print(gm.graph)

    # OPTIMIZATION PASSES

    ir_graph = fx_to_ir(gm)
    ir_graph = optimize(ir_graph)

    # LOWERING

    # output_ir_graph

    # BUILD A COMPILED FN (closure? lambda?) AND RETURN

    # TODO: see if it's still relevant
    # https://docs.pytorch.org/docs/stable/generated/torch.jit.optimize_for_inference.html
    return gm.forward


def optimize(input_ir_graph):
    passes = [
        CompilerPass(
            transforms=[
                Transform(
                    pattern=[
                        Pattern("operator", IROp.ADD),
                        Pattern("operator", IROp.RELU),
                    ],
                    output=IROp.FUSED_ADD_RELU,
                )
            ]
        ),
    ]
    output_ir_graph = []
    for p, compiler_pass in enumerate(passes):
        output_ir_graph = compiler_pass.run(input_ir_graph)
        input_ir_graph = output_ir_graph
    return output_ir_graph
