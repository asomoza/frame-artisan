from __future__ import annotations

import gc
import json
import time
from collections import defaultdict, deque
from typing import TYPE_CHECKING

import torch

from frameartisan.modules.generation.graph.node_error import NodeError


if TYPE_CHECKING:
    from frameartisan.modules.generation.graph.nodes.node import Node


class FrameArtisanNodeGraph:
    def __init__(self):
        self.node_counter = 0
        self.nodes = []
        self.updated = False
        self.abort_function = lambda: None
        self.executing_node = None

        self.device = None
        self.dtype = None

        self.total_elapsed_time: float | None = None
        self.additional_generation_data = {}

    def add_node(self, node: Node, name: str = None):
        if name is not None:
            for existing_node in self.nodes:
                if existing_node.name == name:
                    raise NodeError(
                        f"A node with the name {name} already exists in the graph.", self.__class__.__name__
                    )
        node.name = name
        node.id = self.node_counter
        node.abort_callable = self.abort_function
        node.updated = True
        self.nodes.append(node)
        self.node_counter += 1

    def get_node(self, node_id):
        for node in self.nodes:
            if node.id == node_id:
                return node
        return None

    def get_node_by_name(self, name: str):
        for node in self.nodes:
            if node.name == name:
                return node
        return None

    def get_all_nodes_class(self, node_class):
        return [node for node in self.nodes if isinstance(node, node_class)]

    def delete_node_by_id(self, node_id):
        node = self.get_node(node_id)
        if node is not None:
            self.delete_node(node)

    def delete_node_by_name(self, name: str):
        node = self.get_node_by_name(name)
        if node is not None:
            self.delete_node(node)

    def delete_node(self, node: Node):
        node.before_delete()

        # Make copies to avoid mutation during iteration
        related = list(set(node.dependencies + node.dependents))
        for other_node in related:
            other_node.disconnect_from_node(node)
            node.disconnect_from_node(other_node)

        node.delete()
        self.nodes.remove(node)
        del node

        gc.collect()
        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
            except Exception:
                pass

    @torch.no_grad()
    def __call__(self):
        from frameartisan.app.model_manager import get_model_manager

        self.updated = False
        sorted_nodes = deque()
        visited = set()
        visiting = set()

        def dfs(node):
            visiting.add(node)

            for dependency in sorted(node.dependencies, key=lambda x: x.PRIORITY, reverse=True):
                if dependency in visiting:
                    raise ValueError("Graph contains a cycle")
                if dependency not in visited:
                    dfs(dependency)

            visiting.remove(node)
            visited.add(node)
            sorted_nodes.append(node)

        for node in sorted(self.nodes, key=lambda x: x.PRIORITY, reverse=True):
            if node not in visited:
                dfs(node)

        mm = get_model_manager()
        self.total_elapsed_time = None
        graph_start = time.time()

        with mm.device_scope(device=self.device, dtype=self.dtype):
            for node in sorted_nodes:
                if node.updated and not node.enabled:
                    # Clear stale output values on disabled nodes so downstream
                    # get_input_value never sees leftover data from a previous run.
                    if hasattr(node, "values"):
                        node.values.clear()
                    node.updated = False
                    continue

                if node.updated and node.enabled:
                    if hasattr(node, "values"):
                        node.values.clear()
                    node.device = self.device
                    node.dtype = self.dtype
                    start_time = time.time()

                    try:
                        self.executing_node = node
                        node()
                    except NodeError:
                        raise
                    except Exception as e:
                        raise NodeError(str(e), node.name or node.__class__.__name__) from e

                    end_time = time.time()
                    node.elapsed_time = end_time - start_time
                    self.updated = True
                    self.executing_node = None

                    if node.abort:
                        node.abort = False
                        self.abort_function()
                        break

                    node.updated = False

            # Ensure offload strategy is applied even when the model node was
            # skipped (e.g. only the strategy changed, not the model_path).
            mm.apply_offload_strategy(self.device)

        gc.collect()
        self.total_elapsed_time = time.time() - graph_start

    def to_json(self, additional_generation_data: dict | None = None):
        if additional_generation_data is None:
            additional_generation_data = self.additional_generation_data or {}

        graph_dict = {
            "format_version": 1,
            "nodes": [node.to_dict() for node in self.nodes],
            "connections": [],
            "additional_generation_data": additional_generation_data,
        }

        for node in self.nodes:
            for input_name, connections in node.connections.items():
                for connected_node, output_name in connections:
                    graph_dict["connections"].append(
                        {
                            "from_node_id": connected_node.id,
                            "from_output_name": output_name,
                            "to_node_id": node.id,
                            "to_input_name": input_name,
                        }
                    )

        return json.dumps(graph_dict)

    def from_json(self, json_str, node_classes, callbacks=None):
        graph_dict = json.loads(json_str)

        self.nodes.clear()
        self.node_counter = 0

        self.additional_generation_data = graph_dict.get("additional_generation_data", {}) or {}

        id_to_node = {}
        max_id = 0

        for node_dict in graph_dict["nodes"]:
            NodeClass = node_classes[node_dict["class"]]
            node = NodeClass.from_dict(node_dict, callbacks)
            id_to_node[node.id] = node
            self.nodes.append(node)
            max_id = max(max_id, node.id)

        for connection_dict in graph_dict["connections"]:
            from_node = id_to_node[connection_dict["from_node_id"]]
            to_node = id_to_node[connection_dict["to_node_id"]]
            to_node.connect(
                connection_dict["to_input_name"],
                from_node,
                connection_dict["from_output_name"],
            )

        self.node_counter = max_id + 1

    def save_to_json(self, filename, additional_generation_data: dict | None = None):
        json_str = self.to_json(additional_generation_data=additional_generation_data)
        with open(filename, "w", encoding="utf-8") as f:
            f.write(json_str)

    def load_from_json(self, filename, node_classes, callbacks=None):
        with open(filename, "r", encoding="utf-8") as f:
            json_str = f.read()

        self.from_json(json_str, node_classes, callbacks)

    def update_from_json(self, json_str, node_classes, callbacks=None):
        graph_dict = json.loads(json_str)
        current_id_to_node = {node.id: node for node in self.nodes}
        updated_nodes = set()

        self.additional_generation_data = (
            graph_dict.get("additional_generation_data", self.additional_generation_data) or {}
        )

        new_id_to_node = {}
        max_id = 0

        for node_dict in graph_dict["nodes"]:
            node_class = node_classes[node_dict["class"]]

            if node_dict["id"] in current_id_to_node and isinstance(current_id_to_node[node_dict["id"]], node_class):
                node = current_id_to_node[node_dict["id"]]
                new_node = node_class.from_dict(node_dict, callbacks)

                if node.to_dict() != new_node.to_dict():
                    node.update_inputs(node_dict, callbacks=callbacks)
                    node.set_updated(updated_nodes)
            else:
                node = node_class.from_dict(node_dict, callbacks)
                node.set_updated(updated_nodes)
                if node_dict["id"] in current_id_to_node:
                    self.delete_node_by_id(node_dict["id"])
                self.nodes.append(node)

            new_id_to_node[node.id] = node
            max_id = max(max_id, node.id)

        for node_id in list(current_id_to_node.keys()):
            if node_id not in new_id_to_node:
                self.delete_node_by_id(node_id)

        for node in self.nodes:
            if node.updated and node.id not in updated_nodes:
                node.set_updated(updated_nodes)

        new_connections = defaultdict(list)
        for connection_dict in graph_dict["connections"]:
            to_id = connection_dict["to_node_id"]
            new_connections[to_id].append(
                (
                    connection_dict["to_input_name"],
                    connection_dict["from_node_id"],
                    connection_dict["from_output_name"],
                )
            )

        for node in self.nodes:
            desired = new_connections[node.id]
            if node.connections_changed(desired):
                node.set_updated(updated_nodes)
                node.clear_all_connections()

                for to_input_name, from_node_id, output_name in desired:
                    from_node = new_id_to_node[from_node_id]
                    node.connect(to_input_name, from_node, output_name)

        for node in self.nodes:
            if node.id not in updated_nodes:
                node.updated = False

        self.node_counter = max_id + 1

    def update_from_json_file(self, filename, node_classes, callbacks=None):
        with open(filename, "r", encoding="utf-8") as f:
            json_str = f.read()
        self.update_from_json(json_str, node_classes, callbacks)

    def abort_graph(self):
        if self.executing_node is not None:
            self.executing_node.abort = True

    def set_abort_function(self, abort_callable: callable):
        self.abort_function = abort_callable

    def clean_up(self):
        for i in range(len(self.nodes) - 1, -1, -1):
            self.delete_node(self.nodes[i])

        self.node_counter = 0
