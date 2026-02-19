from __future__ import annotations

from frameartisan.modules.generation.graph.nodes.node import Node


class BooleanNode(Node):
    PRIORITY = 2
    OUTPUTS = ["value"]

    def __init__(self, value: bool = False):
        super().__init__()
        self.value = bool(value)

    def update_value(self, value: object):
        self.value = bool(value)
        self.set_updated()

    def __call__(self):
        self.values["value"] = bool(self.value)
        return self.values
