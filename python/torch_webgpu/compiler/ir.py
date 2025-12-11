class IRNode:
    def __init__(self, operator=None, fx_node=None):
        super().__init__()
        self.operator = operator
        self.fx_node = fx_node

    def __call__(self, operator=None, fx_node=None):
        if operator:
            self.operator = operator
        if fx_node:
            self.fx_node = fx_node
