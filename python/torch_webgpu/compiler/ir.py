from abc import ABC
from typing import Any, List


class IROp(ABC):
    def __init__(self):
        super().__init__()


class IRNode(ABC):
    value_id: Any = None  # I assume that it's always defined for any concrete IRNode
    inputs: List[Any] = []  # I assume that it can't be a None, but can be an empty list

    def __init__(self, value_id: Any = None, inputs: List[Any] = []):
        super().__init__()

        if value_id:
            self.value_id = value_id
        if inputs:
            self.inputs = inputs
