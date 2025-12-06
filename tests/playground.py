from typing import List
import torch


def my_compiler(gm: torch.fx.GraphModule, example_inputs: List[torch.Tensor]):
    print("my_compiler() called with FX graph:")
    gm.graph.print_tabular()
    return gm.forward  # return a python callable


@torch.compile(backend=my_compiler)
def fn(x, y):
    a = torch.cos(x)
    b = torch.sin(y)
    return a + b


fn(torch.randn(10), torch.randn(10))

if __name__ == "__main__":
    fn(torch.randn(10), torch.randn(10))
