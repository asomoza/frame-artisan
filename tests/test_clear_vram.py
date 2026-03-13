"""Tests for the Clear VRAM & RAM action.

Verifies that clearing VRAM/RAM releases all model references, restores
memory to pre-load levels, and leaves the graph functional for the next run.
"""

from __future__ import annotations

import gc
import weakref

import psutil
import pytest
import torch
import torch.nn as nn

from frameartisan.app.model_manager import ModelManager


def _rss_mb() -> float:
    """Current process RSS in MiB."""
    return psutil.Process().memory_info().rss / (1024**2)


def _make_large_model(size_mb: int = 100) -> nn.Module:
    """Create a dummy model that consumes roughly *size_mb* MiB of RAM."""
    # Each float32 param = 4 bytes → need size_mb * 1024^2 / 4 params
    num_params = size_mb * 1024 * 1024 // 4
    model = nn.Linear(num_params, 1, bias=False)
    return model


class TestClearVRAMReleasesMemory:
    """Simulate the real clear VRAM & RAM flow with RSS measurement."""

    def test_clear_releases_ram(self):
        """End-to-end: models in ModelManager + node.values, freed via
        clear_node_values() + mm.clear(). Setup is done in a helper function
        so local refs go out of scope naturally (matching real app behavior
        where LTX2ModelNode.__call__ locals don't persist).
        RSS should drop back near baseline.
        """
        from unittest.mock import MagicMock

        from frameartisan.modules.generation.graph.frameartisan_node_graph import FrameArtisanNodeGraph
        from frameartisan.modules.generation.graph.nodes.node import Node
        from frameartisan.modules.generation.threads.generation_thread import NodeGraphThread

        mm = ModelManager()

        def _setup() -> NodeGraphThread:
            """Simulate a generation run. Local refs (transformer, vae, etc.)
            go out of scope when this function returns — just like in the real
            LTX2ModelNode.__call__."""
            transformer = _make_large_model(200)
            vae = _make_large_model(100)
            text_encoder = _make_large_model(50)

            mm.register_component("transformer", transformer)
            mm.register_component("vae", vae)
            mm.register_component("text_encoder", text_encoder)
            mm.set_cached("some_hash", torch.randn(1000, 1000))

            graph = FrameArtisanNodeGraph()

            model_node = Node()
            model_node.name = "model"
            model_node.id = 0
            # Model node stores component refs in values + _prev_values
            model_node.values = {
                "transformer": transformer,
                "vae": vae,
                "text_encoder": text_encoder,
            }
            model_node._prev_values = {
                "transformer": transformer,
                "vae": vae,
                "text_encoder": text_encoder,
            }
            model_node._prev_component_paths = {"transformer": "/t", "vae": "/v", "text_encoder": "/te"}

            denoise_node = Node()
            denoise_node.name = "denoise"
            denoise_node.id = 1
            denoise_node.values = {"latents": torch.randn(1, 4096, 128)}

            graph.nodes = [model_node, denoise_node]

            dirs = MagicMock()
            staged_graph = FrameArtisanNodeGraph()
            thread = NodeGraphThread(
                dirs, staged_graph, torch.bfloat16, torch.device("cpu"),
                graph_factory=FrameArtisanNodeGraph,
            )
            thread._persistent_run_graph = graph
            return thread

        gc.collect()
        gc.collect()
        baseline_rss = _rss_mb()

        thread = _setup()
        # Local refs from _setup are now out of scope

        gc.collect()
        loaded_rss = _rss_mb()
        assert loaded_rss - baseline_rss > 100, (
            f"Models should consume significant RAM: baseline={baseline_rss:.0f} loaded={loaded_rss:.0f}"
        )

        # --- Clear VRAM & RAM (exactly what on_module_event does) ---
        thread.clear_node_values()
        mm.clear()

        post_clear_rss = _rss_mb()

        freed = loaded_rss - post_clear_rss
        expected_freed = loaded_rss - baseline_rss
        assert freed > expected_freed * 0.8, (
            f"Expected to free ~{expected_freed:.0f} MiB but only freed {freed:.0f} MiB "
            f"(baseline={baseline_rss:.0f} loaded={loaded_rss:.0f} post={post_clear_rss:.0f})"
        )

    def test_clear_drops_all_references(self):
        """Weak refs to registered components should be dead after clear()."""
        mm = ModelManager()

        transformer = nn.Linear(4, 4)
        vae = nn.Linear(4, 4)
        cached_tensor = torch.randn(4, 4)

        mm.register_component("transformer", transformer)
        mm.register_component("vae", vae)
        mm.set_cached("tensor_key", cached_tensor)

        weak_transformer = weakref.ref(transformer)
        weak_vae = weakref.ref(vae)
        weak_cached = weakref.ref(cached_tensor)

        # Drop local refs — ModelManager is the only holder now
        del transformer, vae, cached_tensor

        # Before clear: refs should still be alive
        assert weak_transformer() is not None
        assert weak_vae() is not None
        assert weak_cached() is not None

        mm.clear()
        gc.collect()

        # After clear: refs should be dead
        assert weak_transformer() is None, "transformer still alive after clear()"
        assert weak_vae() is None, "vae still alive after clear()"
        assert weak_cached() is None, "cached tensor still alive after clear()"

    def test_clear_resets_all_state(self):
        """All ModelManager internal state should be reset after clear()."""
        mm = ModelManager()

        mm._model_id = "some/model"
        mm.register_component("transformer", nn.Linear(4, 4))
        mm.set_cached("k", "v")
        mm._component_hashes["transformer"] = "abc123"
        mm._lora_sources["lora1"] = "/path/to/lora"
        mm._compiled_components[("key",)] = nn.Linear(4, 4)
        mm._applied_strategy = "no_offload"
        mm._group_offload_use_stream = True
        mm._group_offload_low_cpu_mem = True

        mm.clear()

        assert mm._model_id is None
        assert mm._components == {}
        assert mm._component_cache == {}
        assert mm._component_hashes == {}
        assert mm._lora_sources == {}
        assert mm._compiled_components == {}
        assert mm._managed_components == {}
        assert mm._applied_strategy is None
        assert mm._group_offload_use_stream is False
        assert mm._group_offload_low_cpu_mem is False


class TestClearVRAMWithPersistentGraph:
    """Verify that clear_node_values on the thread frees node output tensors."""

    def test_clear_node_values_frees_tensors(self):
        """Simulates the full clear_vram flow: mm.clear() + thread.clear_node_values()
        should free both model components and node output tensors.
        """
        from unittest.mock import MagicMock

        from frameartisan.modules.generation.graph.frameartisan_node_graph import FrameArtisanNodeGraph
        from frameartisan.modules.generation.graph.nodes.node import Node
        from frameartisan.modules.generation.threads.generation_thread import NodeGraphThread

        mm = ModelManager()

        # Register a large model
        large_model = _make_large_model(100)
        mm.register_component("transformer", large_model)

        # Build a persistent run graph with nodes holding tensor values
        graph = FrameArtisanNodeGraph()
        node = Node()
        node.name = "denoise"
        node.id = 0
        node.values = {"latents": torch.randn(1, 128, 4096)}
        graph.nodes = [node]

        # Set up the thread with the persistent graph
        dirs = MagicMock()
        staged_graph = FrameArtisanNodeGraph()
        thread = NodeGraphThread(
            dirs, staged_graph, torch.bfloat16, torch.device("cpu"),
            graph_factory=FrameArtisanNodeGraph,
        )
        thread._persistent_run_graph = graph

        weak_model = weakref.ref(large_model)
        weak_latents = weakref.ref(node.values["latents"])

        del large_model, node

        # --- Simulate the full clear_vram flow ---
        thread.clear_node_values()
        mm.clear()
        gc.collect()

        assert weak_model() is None, "Model should be freed after clear()"
        assert weak_latents() is None, "Node tensors should be freed after clear_node_values()"

    def test_clear_node_values_noop_without_persistent_graph(self):
        """clear_node_values should not raise when there is no persistent graph."""
        from unittest.mock import MagicMock

        from frameartisan.modules.generation.graph.frameartisan_node_graph import FrameArtisanNodeGraph
        from frameartisan.modules.generation.threads.generation_thread import NodeGraphThread

        dirs = MagicMock()
        staged_graph = FrameArtisanNodeGraph()
        thread = NodeGraphThread(
            dirs, staged_graph, torch.bfloat16, torch.device("cpu"),
            graph_factory=FrameArtisanNodeGraph,
        )
        assert thread._persistent_run_graph is None

        # Should not raise
        thread.clear_node_values()


class TestGraphFunctionalAfterClear:
    """After clear_node_values + mm.clear(), the graph must still work on next run."""

    def test_graph_executes_after_clear(self):
        """Build a mini 2-node graph, run it, clear, run again — no errors."""
        from unittest.mock import MagicMock

        from frameartisan.modules.generation.graph.frameartisan_node_graph import FrameArtisanNodeGraph
        from frameartisan.modules.generation.graph.nodes.node import Node
        from frameartisan.modules.generation.threads.generation_thread import NodeGraphThread

        # --- Define two simple node types ---
        class ProducerNode(Node):
            OUTPUTS = ["value"]

            def __init__(self):
                super().__init__()
                self._prev_component_paths = {}
                self._prev_values = {}

            def __call__(self):
                self.values = {"value": torch.randn(4, 4)}

        class ConsumerNode(Node):
            REQUIRED_INPUTS = ["value"]
            OUTPUTS = ["result"]

            def __call__(self):
                val = self.get_input_value("value")
                self.values = {"result": val.sum()}

        # --- Build and wire graph ---
        graph = FrameArtisanNodeGraph()
        producer = ProducerNode()
        consumer = ConsumerNode()
        graph.add_node(producer, name="producer")
        graph.add_node(consumer, name="consumer")
        consumer.connect("value", producer, "value")

        graph.device = torch.device("cpu")
        graph.dtype = torch.float32

        mm = ModelManager()
        mm.register_component("transformer", nn.Linear(4, 4))

        # --- First run: should succeed ---
        with mm.device_scope(device="cpu"):
            graph()
        assert "result" in consumer.values

        # --- Set up thread with persistent graph ---
        dirs = MagicMock()
        staged_graph = FrameArtisanNodeGraph()
        thread = NodeGraphThread(
            dirs, staged_graph, torch.float32, torch.device("cpu"),
            graph_factory=FrameArtisanNodeGraph,
        )
        thread._persistent_run_graph = graph

        # --- Clear VRAM & RAM ---
        thread.clear_node_values()
        mm.clear()
        gc.collect()

        # Verify values are cleared
        assert producer.values == {}
        assert consumer.values == {}

        # Verify nodes are marked for re-execution
        assert producer.updated is True
        assert consumer.updated is True

        # Verify model node caches are cleared
        assert producer._prev_component_paths == {}
        assert producer._prev_values == {}

        # --- Second run: should succeed (nodes are marked updated) ---
        with mm.device_scope(device="cpu"):
            graph()
        assert "result" in consumer.values
        assert consumer.values["result"].shape == ()

    def test_clear_without_marking_updated_would_fail(self):
        """Demonstrate that clearing values without marking updated causes errors.

        The real scenario: clear_node_values empties values, then update_from_json
        marks only downstream nodes as updated (because no settings changed on the
        producer). The consumer runs but producer was skipped → empty values → crash.
        """
        from frameartisan.modules.generation.graph.frameartisan_node_graph import FrameArtisanNodeGraph
        from frameartisan.modules.generation.graph.nodes.node import Node
        from frameartisan.modules.generation.graph.node_error import NodeError

        class ProducerNode(Node):
            OUTPUTS = ["value"]

            def __call__(self):
                self.values = {"value": torch.tensor(42)}

        class ConsumerNode(Node):
            REQUIRED_INPUTS = ["value"]
            OUTPUTS = ["result"]

            def __call__(self):
                val = self.get_input_value("value")
                self.values = {"result": val}

        graph = FrameArtisanNodeGraph()
        producer = ProducerNode()
        consumer = ConsumerNode()
        graph.add_node(producer, name="producer")
        graph.add_node(consumer, name="consumer")
        consumer.connect("value", producer, "value")
        graph.device = torch.device("cpu")
        graph.dtype = torch.float32

        mm = ModelManager()

        # First run succeeds
        with mm.device_scope(device="cpu"):
            graph()
        assert "result" in consumer.values

        # Simulate the old bug: clear values but only mark consumer as updated
        # (producer is not updated → skipped → its values stay empty)
        for node in graph.nodes:
            node.values.clear()
        consumer.updated = True  # e.g. user changed a setting on consumer
        # producer.updated remains False

        # Consumer runs, tries to read producer.values["value"] → crash
        with pytest.raises(NodeError, match="not in node.values"):
            with mm.device_scope(device="cpu"):
                graph()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestClearVRAMGPU:
    """Test that clear() actually frees GPU memory."""

    def test_clear_frees_gpu_memory(self):
        mm = ModelManager()

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        baseline_vram = torch.cuda.memory_allocated()

        # Put a model on GPU and register it
        model = nn.Linear(1024, 1024, bias=False).cuda()
        mm.register_component("transformer", model)
        mm.set_cached("gpu_tensor", torch.randn(1024, 1024, device="cuda"))

        loaded_vram = torch.cuda.memory_allocated()
        assert loaded_vram > baseline_vram, "GPU memory should increase after loading"

        # Drop local refs
        del model

        mm.clear()
        gc.collect()

        post_clear_vram = torch.cuda.memory_allocated()
        assert post_clear_vram <= baseline_vram + 1024, (
            f"GPU memory not freed: baseline={baseline_vram} post_clear={post_clear_vram}"
        )
