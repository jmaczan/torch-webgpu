# This is a rabbit-hole but I need to just start somewhere,
# so lot's of assumptions and limitations here


from typing import Any, Generic, List, Mapping, TypeVar, Type, Optional

from .ir import IRNode

T_IRNode = TypeVar("T_IRNode", bound="IRNode")


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


class CompilerPass(Generic[T_IRNode]):
    def __init__(self, transforms: list[Transform] = []):
        self.transforms = transforms

    def run(
        self,
        ir_graph: list[T_IRNode],
        ir_op_to_ir_node: Mapping[Any, Type[T_IRNode]],
    ):
        from .high_ir import HighIRNode
        from .low_ir import LowIRNode

        if not ir_graph:
            raise Exception("Compiler pass needs an IR graph to run")
        if not ir_op_to_ir_node:
            raise Exception(
                "Compiler pass needs a dictionary that maps IR tops to IR nodes"
            )
        output_graph: list[T_IRNode] = []
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
                    # The assumption here is that each element of pattern corresponds to one graph node
                    # And they are in the same order as elements in graph
                    # the mechanism of matching needs to be more flexible;
                    # to perhaps support more than a single pattern to check for a single node
                    # and maybe be more flexible about the pattern ordering or something like that
                    # In general, this is written mostly for fusion and needs to be generalized
                    # For instance there could be value_id/inputs manipulation rules defined per transform
                    # instead of one rule for all transforms
                    # a big TODO for the future
                    is_pattern_match = True
                    reuse_value_id_from_this_node: Optional[T_IRNode] = None
                    inputs = []
                    stale_value_ids = []
                    num_patterns = len(transform.pattern)
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
                        if p == num_patterns - 1:  # last pattern
                            reuse_value_id_from_this_node = current_node_pattern_check
                        else:
                            inputs += current_node_pattern_check.inputs
                            stale_value_ids += current_node_pattern_check.value_id

                    if is_pattern_match:
                        inputs = list(dict.fromkeys(inputs))  # prevent duplicates
                        stale_value_ids = list(
                            dict.fromkeys(stale_value_ids)
                        )  # value_ids that will be removed because of fusion

                        inputs = [
                            x for x in inputs if x not in set(stale_value_ids)
                        ]  # remove value_ids that will not exist from inputs for fused node
                        assert reuse_value_id_from_this_node is not None, (
                            "Error scenario: since is_pattern_match is True, there had to be pattern matching and there has to be last node that was a part of a pattern"
                        )
                        if transform.output:  # TODO: maybe explicitly disallow empty outputs since they are not fully supported yet anyway
                            output_node = ir_op_to_ir_node.get(transform.output)  # Noqa E501
                            if not output_node:
                                raise Exception("Trying to add empty node")
                            # the dirty part, to be refactored in the near never
                            if isinstance(input_node, HighIRNode):
                                output_graph.append(
                                    output_node(
                                        fx_node=input_node.fx_node,  # pyright: ignore[reportCallIssue] TODO remove the ignore
                                        value_id=reuse_value_id_from_this_node.value_id,
                                        inputs=inputs,
                                    )
                                )
                            elif isinstance(input_node, LowIRNode):
                                output_graph.append(
                                    output_node(
                                        high_ir_node=input_node.high_ir_node,  # pyright: ignore[reportCallIssue] TODO remove the ignore
                                        value_id=reuse_value_id_from_this_node.value_id,
                                        inputs=inputs,
                                    )
                                )
                            else:
                                raise Exception(f"Unrecognized IR: {input_node}")
                        skips_left = len(transform.pattern) - 1
                    else:
                        output_graph.append(input_node)
                else:
                    output_graph.append(input_node)
            input_graph = output_graph

        return output_graph


def run_compiler_passes(
    input_ir_graph: list[T_IRNode],
    ir_op_to_ir_node: Mapping[Any, Type[T_IRNode]],
    passes: list[CompilerPass],
) -> List[T_IRNode]:
    if not passes:
        print(
            "No compiler passes provided to run_compiler_passes. Skipping this optimization step"
        )
        return input_ir_graph
    output_ir_graph = []
    for p, compiler_pass in enumerate(passes):
        output_ir_graph = compiler_pass.run(
            ir_graph=input_ir_graph, ir_op_to_ir_node=ir_op_to_ir_node
        )
        input_ir_graph = output_ir_graph
    return output_ir_graph
