class IRNode:
    def __init__(self, meta):
        super().__init__()
        self.meta = meta


class IRTensor(IRNode):
    def __init__(self, meta, name: str = None, *args, **kwargs):
        self.meta = meta
        self.name = name
        self.args = args
        self.kwargs = kwargs


class IRPointwise(IRNode):
    def __init__(self, meta, operator=None, *args, **kwargs):
        self.meta = meta
        self.operator = operator
        self.args = args
        self.kwargs = kwargs


class IROperation(IRNode):
    def __init__(self, meta, operator=None, *args, **kwargs):
        self.meta = meta
        self.operator = operator
        self.args = args
        self.kwargs = kwargs
