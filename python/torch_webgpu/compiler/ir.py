class IRNode:
    def __init__(self, fx_node=None):
        super().__init__()
        self.fx_node = fx_node
        self.operator = None

    def __call__(self, operator=None, fx_node=None):
        if operator:
            self.operator = operator
        if fx_node:
            self.fx_node = fx_node


class IRTensor(IRNode):
    def __init__(self, name: str = None, *args, **kwargs):
        self.name = name
        self.args = args
        self.kwargs = kwargs


class IRPointwise(IRNode):
    def __init__(self, operator=None, *args, **kwargs):
        self.operator = operator
        self.args = args
        self.kwargs = kwargs


class IROperation(IRNode):
    def __init__(self, operator=None, *args, **kwargs):
        self.operator = operator
        self.args = args
        self.kwargs = kwargs


class IRReturn(IRNode):
    def __init__(self, operator=None, *args, **kwargs):
        self.operator = operator
        self.args = args
        self.kwargs = kwargs
