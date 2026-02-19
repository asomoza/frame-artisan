from __future__ import annotations

import inspect
from collections import defaultdict
from typing import Any, Callable, ClassVar, Optional

from frameartisan.modules.generation.graph.node_error import NodeError


class Node:
    PRIORITY = 0
    REQUIRED_INPUTS = []
    OPTIONAL_INPUTS = []
    OUTPUTS = []

    SERIALIZE_INCLUDE: ClassVar[Optional[set[str]]] = None
    SERIALIZE_EXCLUDE: ClassVar[set[str]] = set()

    SERIALIZE_CONVERTERS: ClassVar[dict[str, tuple[Callable[[Any], Any], Callable[[Any], Any]]]] = {}

    STRICT_SERIALIZATION: ClassVar[bool] = True

    _RUNTIME_EXCLUDE: ClassVar[set[str]] = {
        "dependencies",
        "dependents",
        "connections",
        "values",
        "device",
        "dtype",
        "elapsed_time",
        "updated",
        "abort",
        "abort_callable",
    }

    _DICT_RESERVED_KEYS: ClassVar[set[str]] = {"class", "id", "name", "enabled", "state"}

    def __init__(self):
        self.id = None
        self.enabled = True
        self.name = None
        self.elapsed_time = None
        self.updated = True
        self.abort = False
        self.dependencies = []
        self.dependents = []
        self.values = {}
        self.connections = defaultdict(list)

        self.device = None
        self.dtype = None

    def connect(self, input_name: str, node, output_name: str):
        if not isinstance(node, Node):
            raise TypeError("node must be an instance of Node or its subclass")

        if input_name not in self.REQUIRED_INPUTS + self.OPTIONAL_INPUTS:
            raise ValueError(f'The input "{input_name}" is not present in "{self.__class__.__name__}"')
        if output_name not in node.OUTPUTS:
            raise ValueError(f'The output "{output_name}" is not present in "{node.__class__.__name__}"')

        # Avoid duplicate dependencies when multiple inputs come from same node
        if node not in self.dependencies:
            self.dependencies.append(node)

        self.connections[input_name].append((node, output_name))
        if self not in node.dependents:
            node.dependents.append(self)

        self.updated = True
        for dependent_node in self.dependents:
            dependent_node.set_updated()

    def disconnect(self, input_name: str, node, output_name: str):
        if input_name in self.connections:
            self.connections[input_name] = [
                (n, out_name)
                for n, out_name in self.connections[input_name]
                if not (n == node and out_name == output_name)
            ]
            if not self.connections[input_name]:
                del self.connections[input_name]

        still_connected = any(n == node for conns in self.connections.values() for n, _output_name in conns)
        if not still_connected:
            if node in self.dependencies:
                self.dependencies.remove(node)
            if self in node.dependents:
                node.dependents.remove(self)

        self.updated = True
        for dependent_node in self.dependents:
            dependent_node.set_updated()

    def disconnect_from_node(self, node):
        changed = False

        if node in self.dependencies:
            changed = True
        self.dependencies = [dep for dep in self.dependencies if dep != node]
        for input_name, conns in list(self.connections.items()):
            new_conns = [(n, output_name) for n, output_name in conns if n != node]
            if len(new_conns) != len(conns):
                changed = True
            self.connections[input_name] = new_conns
            if not new_conns:
                del self.connections[input_name]
        if self in node.dependents:
            node.dependents.remove(self)

        if changed:
            self.set_updated()

    def clear_all_connections(self):
        # Fully remove wiring both ways
        for dep in list(self.dependencies):
            if self in dep.dependents:
                dep.dependents.remove(self)
        self.dependencies.clear()
        self.connections = defaultdict(list)

        # Changing wiring should invalidate this node (and its dependents)
        self.set_updated()

    def connections_changed(self, new_connections):
        """
        new_connections: list of tuples (to_input_name, from_node_id, from_output_name)
        """
        current_connections = [
            (input_name, dep.id, output_name)
            for input_name, deps in self.connections.items()
            for dep, output_name in deps
        ]
        return set(current_connections) != set(new_connections)

    def set_updated(self, updated_nodes=None, update_dependents=True):
        if updated_nodes is None:
            updated_nodes = set()
        if self.id in updated_nodes:
            return
        self.updated = True
        updated_nodes.add(self.id)
        if update_dependents:
            for dependent in self.dependents:
                dependent.set_updated(updated_nodes)

    def __getattr__(self, name):
        if name in self.REQUIRED_INPUTS + self.OPTIONAL_INPUTS:
            return self.get_input_value(name)
        raise NodeError(f"There is no attribute '{name}'", self.__class__.__name__)

    def get_input_value(self, input_name):
        if input_name in self.connections:
            values = []
            for node, output_name in self.connections[input_name]:
                if not node.enabled:
                    if input_name in self.OPTIONAL_INPUTS:
                        return None
                    raise NodeError(
                        f"The required input '{input_name}' is connected to disabled node '{node.name}'.",
                        self.__class__.__name__,
                    )
                if output_name not in node.values:
                    if input_name in self.OPTIONAL_INPUTS:
                        return None
                    raise NodeError(
                        f"The required output '{output_name}' is not in node.values.",
                        self.__class__.__name__,
                    )
                values.append(node.values[output_name])
            return values if len(values) > 1 else values[0]
        elif input_name in self.OPTIONAL_INPUTS:
            return None
        else:
            raise NodeError(
                f"The required input '{input_name}' is not connected.",
                self.__class__.__name__,
            )

    def __call__(self):
        pass

    def _is_json_primitive(self, v: Any) -> bool:
        return v is None or isinstance(v, (str, int, float, bool))

    def _to_jsonable(self, v: Any):
        if self._is_json_primitive(v):
            return v

        if isinstance(v, (list, tuple)):
            return [self._to_jsonable(x) for x in v]

        if isinstance(v, dict):
            out = {}
            for k, item in v.items():
                if not isinstance(k, str):
                    raise TypeError(f"Only string dict keys are supported (got {type(k)})")
                out[k] = self._to_jsonable(item)
            return out

        raise TypeError(f"Value of type {type(v)} is not JSON-serializable by default")

    def _serialization_error(self, field: str, value: Any, reason: str) -> NodeError:
        return NodeError(
            f"Field '{field}' on {type(self).__name__} is not JSON-serializable ({reason}). "
            f"Either add it to SERIALIZE_EXCLUDE or define SERIALIZE_CONVERTERS['{field}']. "
            f"(value type: {type(value)})",
            self.__class__.__name__,
        )

    def get_state(self) -> dict:
        if self.SERIALIZE_INCLUDE is None:
            keys = set(self.__dict__.keys())
        else:
            keys = set(self.SERIALIZE_INCLUDE)

        keys -= self._RUNTIME_EXCLUDE
        keys -= set(self.SERIALIZE_EXCLUDE)
        keys = {k for k in keys if not k.startswith("_")}

        state: dict[str, Any] = {}
        for k in sorted(keys):
            v = getattr(self, k, None)

            # Converter takes precedence (so callables can be serialized via converter)
            if k in self.SERIALIZE_CONVERTERS:
                to_json, _from_json = self.SERIALIZE_CONVERTERS[k]
                try:
                    converted = to_json(v)
                    state[k] = self._to_jsonable(converted)
                except Exception as e:
                    if self.STRICT_SERIALIZATION:
                        raise self._serialization_error(k, v, f"converter failed: {e!r}") from e
                    continue
                continue

            # Callables are never JSON-serializable unless converted/excluded
            if callable(v):
                if self.STRICT_SERIALIZATION:
                    raise self._serialization_error(k, v, "callable values are not supported")
                continue

            try:
                state[k] = self._to_jsonable(v)
            except TypeError as e:
                if self.STRICT_SERIALIZATION:
                    raise self._serialization_error(k, v, str(e)) from e
                continue

        return state

    def apply_state(self, state: dict, callbacks: dict | None = None):
        state = state or {}
        for k, v in state.items():
            if k in self.SERIALIZE_CONVERTERS:
                _to_json, from_json = self.SERIALIZE_CONVERTERS[k]
                try:
                    # Allow from_json(value) or from_json(value, callbacks)
                    if len(inspect.signature(from_json).parameters) >= 2:
                        setattr(self, k, from_json(v, callbacks))
                    else:
                        setattr(self, k, from_json(v))
                except Exception as e:
                    if self.STRICT_SERIALIZATION:
                        raise NodeError(
                            f"Failed to restore field '{k}' on {type(self).__name__} from JSON ({e!r}).",
                            self.__class__.__name__,
                        ) from e
                    setattr(self, k, v)
            else:
                setattr(self, k, v)

    def to_dict(self) -> dict:
        """
        Graph JSON representation.
        """
        out = {
            "class": type(self).__name__,
            "id": self.id,
            "name": self.name,
            "enabled": self.enabled,
        }

        state = self.get_state()
        if state:
            out["state"] = state

        return out

    @classmethod
    def from_dict(cls, node_dict, callbacks=None):
        node = cls()
        node.id = node_dict["id"]
        node.name = node_dict.get("name", None)
        node.enabled = node_dict.get("enabled", True)

        legacy_state = {k: v for k, v in node_dict.items() if k not in cls._DICT_RESERVED_KEYS}
        state = {}
        state.update(legacy_state)
        state.update(node_dict.get("state", {}) or {})
        node.apply_state(state, callbacks=callbacks)

        return node

    def update_inputs(self, node_dict, callbacks=None):
        if "name" in node_dict:
            self.name = node_dict.get("name", None)
        if "enabled" in node_dict:
            self.enabled = node_dict.get("enabled", True)

        legacy_state = {k: v for k, v in node_dict.items() if k not in self._DICT_RESERVED_KEYS}
        state = {}
        state.update(legacy_state)
        state.update(node_dict.get("state", {}) or {})
        self.apply_state(state, callbacks=callbacks)

    def before_delete(self):
        pass

    def delete(self):
        # Clean up the node's data
        self.dependencies.clear()
        self.dependents.clear()
        self.values.clear()
        self.connections.clear()

        # Reset other attributes
        self.id = None
        self.name = None
        self.updated = False
        self.device = None
        self.dtype = None

    def abort_call(self):
        self.abort = True
