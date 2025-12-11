# This is a rabbit-hole but I need to just start somewhere,
# so lot's of assumptions and limitations here


from typing import Callable, Optional

from .ir import IRNode, IROp, ir_op_to_ir_node


class Pattern:
    def __init__(self, trait: str, value: IROp):
        if not trait or not value:
            raise Exception("Trait and value are required to create a pattern")

        self.trait = trait
        self.value = value


class Transform:
    def __init__(
        self,
        pattern: Optional[list[Pattern]] = None,
        input=None,
        output: Optional[IROp] = None,
    ):
        self.input = input
        self.output = output
        if not pattern:
            raise Exception(
                "Transformations without a pattern ",
                "aren't supported yet",
            )
        self.pattern = pattern


class CompilerPass:
    def __init__(self, transforms: list[Transform] = []):
        self.transforms = transforms

    def run(self, ir_graph: list[IRNode] = [], ir_op_to_ir_node: Callable[IROp, type[IRNode]]):
        output_graph = []
        input_graph = ir_graph
        for t, transform in enumerate(self.transforms):
            output_graph = []
            skips_left = 0
            for i, input_node in enumerate(input_graph):
                if skips_left > 0:
                    skips_left -= 1
                    continue
                if (
                    transform.pattern is not None
                    and len(transform.pattern) > 0
                    and len(transform.pattern) + i - 1 < len(input_graph)
                ):
                    # the mechanism of matching needs to be more flexible;
                    # a TODO for the future
                    is_pattern_match = True
                    for p, pattern in enumerate(transform.pattern):
                        if p == 0:
                            current_node_pattern_check = input_node
                        else:
                            current_node_pattern_check = input_graph[i + p]
                        trait_value = getattr(
                            current_node_pattern_check, pattern.trait, None
                        )
                        if trait_value is None or trait_value != pattern.value:
                            is_pattern_match = False
                            break

                    if is_pattern_match:
                        if transform.output:
                            output_node = ir_op_to_ir_node.get(transform.output)  # Noqa E501
                            if not output_node:
                                raise Exception("Trying to add empty node")
                            output_graph.append(output_node())
                        skips_left = len(transform.pattern) - 1
                    else:
                        output_graph.append(input_node)
                else:
                    output_graph.append(input_node)
            input_graph = output_graph

        return output_graph
