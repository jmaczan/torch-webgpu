# This is a rabbit-hole but I need to just start somewhere,
# so lot's of assumptions and limitations here


from enum import StrEnum
from typing import Any, Callable, Optional

from .ir import IRNode


class Pattern:
    def __init__(self, trait: str, value: str):
        if not trait or not value:
            raise Exception("Trait and value are required to create a pattern")

        self.trait = trait
        self.value = value


class Transform:
    def __init__(
        self,
        pattern: Optional[list[Pattern]] = None,
        input=None,
        output: Optional[str] = None,
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

    def run(
        self,
        ir_graph: list[IRNode],
        ir_op_to_ir_node: dict[Any, type[IRNode]],
    ):
        if not ir_graph:
            raise Exception("Compiler pass needs an IR graph to run")
        if not ir_op_to_ir_node:
            raise Exception(
                "Compiler pass needs a dictionary that maps IR tops to IR nodes"
            )
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


def run_compiler_passes(
    input_ir_graph: list[IRNode],
    ir_op_to_ir_node: dict[Any, type[IRNode]],
    passes: list[CompilerPass],
):
    if not passes:
        print("Provide compiler passes to actually run the compiler")
        return input_ir_graph
    output_ir_graph = []
    for p, compiler_pass in enumerate(passes):
        output_ir_graph = compiler_pass.run(
            ir_graph=input_ir_graph, ir_op_to_ir_node=ir_op_to_ir_node
        )
        input_ir_graph = output_ir_graph
    return output_ir_graph
