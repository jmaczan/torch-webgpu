import torch
from torch._dynamo import register_backend
from typing import Callable, List

from .logger import debug, debug_enabled
from .lowering import lowering
from .high_ir import (
    fx_to_high_ir,
    high_ir_op_to_high_ir_node,
    high_ir_compiler_passes,
    high_ir_print_tabular,
    HighIROp,
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
    example_inputs: List[torch.Tensor],
) -> Callable:
    if debug_enabled():
        debug("FX graph (input):")
        gm.graph.print_tabular()

    high_ir = fx_to_high_ir(gm)
    if debug_enabled():
        debug("High IR graph:")
        high_ir_print_tabular(high_ir)

    # Extract placeholder names for later binding
    placeholder_names = [
        node.value_id for node in high_ir if node.ir_op == HighIROp.PLACEHOLDER
    ]

    high_ir = run_compiler_passes(
        input_ir_graph=high_ir,
        ir_op_to_ir_node=high_ir_op_to_high_ir_node,
        passes=high_ir_compiler_passes,
    )
    if debug_enabled():
        debug("High IR graph (after compilation):")
        high_ir_print_tabular(high_ir)

    low_ir: List[LowIRNode] = high_ir_to_low_ir(high_ir)
    if debug_enabled():
        debug("Low IR graph:")
        low_ir_print_tabular(low_ir)

    low_ir: List[LowIRNode] = run_compiler_passes(
        input_ir_graph=low_ir,
        ir_op_to_ir_node=low_ir_op_to_low_ir_node,
        passes=low_ir_compiler_passes,
    )

    # Create the compiled program with placeholder binding
    program = lowering(low_ir, placeholder_names)

    def compiled_fn(*args):
        return program(*args)

    return compiled_fn
