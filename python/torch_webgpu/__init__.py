from typing import Callable, List
import torch
from . import _C
from torch._dynamo import register_backend


def todo():
    print("TODO")


torch.utils.rename_privateuse1_backend("webgpu")
torch._register_device_module("webgpu", todo)
torch.utils.generate_methods_for_privateuse1_backend(
    for_tensor=True,
    for_module=True,
    for_storage=True,
)


@register_backend
def webgpu_backend(
    gm: torch.fx.GraphModule, example_inputs: List[torch.Tensor]
) -> Callable:
    gm.graph.print_tabular()

    # compile
    def compiled_fn(x):
        return 1

    # TODO: see if it's still relevant
    # https://docs.pytorch.org/docs/stable/generated/torch.jit.optimize_for_inference.html
    return compiled_fn
