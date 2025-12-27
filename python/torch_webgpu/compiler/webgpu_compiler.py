import torch
from torch._dynamo import register_backend
from typing import Callable, List

from .lowering import lowering
from .high_ir import (
    fx_to_high_ir,
    high_ir_op_to_high_ir_node,
    high_ir_compiler_passes,
    high_ir_print_tabular,
)
from .low_ir import (
    LowIRNode,
    high_ir_to_low_ir,
    low_ir_op_to_low_ir_node,
    low_ir_compiler_passes,
    low_ir_print_tabular,
)
from .compiler_pass import run_compiler_passes


@register_backend
def webgpu_backend(
    gm: torch.fx.GraphModule,
    example_inputs: List[torch.Tensor],  # TODO: don't ignore example_inputs
) -> Callable:
    print("\nFX graph (input):")
    gm.graph.print_tabular()

    high_ir = fx_to_high_ir(gm)
    print("\nHigh IR graph:")
    high_ir_print_tabular(high_ir)

    high_ir = run_compiler_passes(
        input_ir_graph=high_ir,
        ir_op_to_ir_node=high_ir_op_to_high_ir_node,
        passes=high_ir_compiler_passes,
    )
    print("\nHigh IR graph (after compilation):")
    high_ir_print_tabular(high_ir)

    low_ir: List[LowIRNode] = high_ir_to_low_ir(high_ir)
    print("\nLow IR graph:")
    low_ir_print_tabular(low_ir)

    low_ir: List[LowIRNode] = run_compiler_passes(
        input_ir_graph=low_ir,
        ir_op_to_ir_node=low_ir_op_to_low_ir_node,
        passes=low_ir_compiler_passes,
    )
    # print("\nLow IR graph (after compilation):")
    # low_ir_print_tabular(low_ir)

    program = lowering(low_ir)
    print(program())
    # BUILD A COMPILED FN (closure? lambda?) AND RETURN

    # Noqa 501 TODO: see if it's still relevant https://docs.pytorch.org/docs/stable/generated/torch.jit.optimize_for_inference.html
    return gm.forward
