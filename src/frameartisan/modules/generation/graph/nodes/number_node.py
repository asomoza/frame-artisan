from __future__ import annotations

from typing import Union

from frameartisan.modules.generation.graph.nodes.node import Node


class NumberNode(Node):
    PRIORITY = 2
    OUTPUTS = ["value"]
    INPUTS = []

    def __init__(self, number: Union[int, float] = None):
        super().__init__()
        self.number = number

    def update_value(self, number: Union[int, float]):
        self.number = number
        self.set_updated()

    def __call__(self):
        self.values["value"] = self.number
        return self.values
