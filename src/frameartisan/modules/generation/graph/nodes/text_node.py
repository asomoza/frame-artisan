from __future__ import annotations

from frameartisan.modules.generation.graph.nodes.node import Node


class TextNode(Node):
    PRIORITY = 2
    OUTPUTS = ["value"]
    INPUTS = []

    def __init__(self, text: str = None):
        super().__init__()
        self.text = text

    def update_value(self, text: str):
        self.text = text
        self.set_updated()

    def __call__(self):
        self.values["value"] = self.text
        return self.values
