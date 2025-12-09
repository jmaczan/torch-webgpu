# This is a rabbit-hole but I need to just start somewhere,
# so lot's of assumptions and limitations here


class Pattern:
    def __init__(self, trait, value):
        if not trait or not value:
            raise ("Trait and value are required to create a pattern")

        self.trait = trait
        self.value = value


class Transform:
    def __init__(self, pattern: list[Pattern] = None, input=None, output=None):
        self.input = input
        self.output = output
        self.pattern = pattern or input


class CompilerPass:
    def __init__(self, transforms: list[Transform] = []):
        self.transforms = transforms

    def run(self, ir_graph=[]):
        output_graph = []
        input_graph = ir_graph
        for t, transform in enumerate(self.transforms):
            output_graph = []
            for i, input_node in enumerate(input_graph):
                if len(transform.pattern) > 0 and len(transform.pattern) + i - 1 < len(  # Noqa 501
                    input_graph
                ):
                    # the mechanism of matching needs to be more flexible;
                    # a TODO for the future
                    is_pattern_match = True
                    for p, pattern in enumerate(transform.pattern):
                        trait_value = getattr(input_node, pattern.trait, None)
                        if trait_value is None or trait_value != pattern.value:
                            is_pattern_match = False
                            break

                    if is_pattern_match:
                        output_graph.append(transform.output)
            input_graph = output_graph

        return output_graph
