import torch
from torch._dynamo import register_backend
from typing import Callable, List

from .high_ir import (
    fx_to_high_ir,
    high_ir_op_to_high_ir_node,
    high_ir_compiler_passes,
)
from .low_ir import high_ir_to_low_ir, low_ir_op_to_low_ir_node, low_ir_compiler_passes
from .compiler_pass import run_compiler_passes


@register_backend
def webgpu_backend(
    gm: torch.fx.GraphModule,
    example_inputs: List[torch.Tensor],  # TODO: don't ignore example_inputs
) -> Callable:
    gm.graph.print_tabular()

    high_ir = fx_to_high_ir(gm)
    high_ir = run_compiler_passes(
        input_ir_graph=high_ir,
        ir_op_to_ir_node=high_ir_op_to_high_ir_node,
        passes=high_ir_compiler_passes,
    )
    low_ir = high_ir_to_low_ir(high_ir)
    low_ir = run_compiler_passes(
        input_ir_graph=low_ir,
        ir_op_to_ir_node=low_ir_op_to_low_ir_node,
        passes=low_ir_compiler_passes,
    )

    # BUILD A COMPILED FN (closure? lambda?) AND RETURN

    # Noqa 501 TODO: see if it's still relevant https://docs.pytorch.org/docs/stable/generated/torch.jit.optimize_for_inference.html
    return gm.forward
