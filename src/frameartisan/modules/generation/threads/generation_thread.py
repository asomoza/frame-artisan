from __future__ import annotations

import logging
from collections.abc import Callable
from typing import TYPE_CHECKING

import torch
from PyQt6.QtCore import QThread, pyqtSignal

from frameartisan.app.model_manager import get_model_manager
from frameartisan.modules.generation.graph.node_error import NodeError
from frameartisan.modules.generation.graph.nodes.node_registry import NODE_CLASSES


if TYPE_CHECKING:
    from frameartisan.app.directories import DirectoriesObject
    from frameartisan.modules.generation.graph.frameartisan_node_graph import FrameArtisanNodeGraph


logger = logging.getLogger(__name__)


class NodeGraphThread(QThread):
    status_changed = pyqtSignal(str)
    progress_update = pyqtSignal(int, torch.Tensor)
    stage_started = pyqtSignal(int)  # total_steps for the new stage
    generation_finished = pyqtSignal(object, object)
    generation_error = pyqtSignal(str, bool)
    generation_aborted = pyqtSignal()

    def __init__(
        self,
        directories: DirectoriesObject,
        node_graph: FrameArtisanNodeGraph,
        dtype: torch.dtype,
        device: torch.device,
        *,
        graph_factory: Callable[[], FrameArtisanNodeGraph] | None = None,
        node_classes: dict | None = None,
    ):
        super().__init__()

        self.node_graph = node_graph
        self.dtype = dtype
        self.device = device
        self.directories = directories

        self.node_graph.set_abort_function(self.on_aborted)
        self.save_video_metadata: bool = False
        self.save_source_images: bool = False
        self.save_source_audio: bool = False
        self.save_source_video: bool = False
        self.auto_save_videos: bool = True
        self._job_json_graph: str | None = None
        self._active_graph: FrameArtisanNodeGraph | None = None
        self._persistent_run_graph: FrameArtisanNodeGraph | None = None
        self._completion_emitted: bool = False

        self._graph_factory = graph_factory or FrameArtisanNodeGraph
        self._node_classes = node_classes or NODE_CLASSES

    def get_staged_json_graph(self) -> str:
        return self.node_graph.to_json()

    def start_generation(self, json_graph: str) -> None:
        self._job_json_graph = json_graph
        self.start()

    def wire_callbacks(self, graph: FrameArtisanNodeGraph) -> None:
        model_node = graph.get_node_by_name("model")
        if model_node is not None:
            model_node.status_callback = lambda msg: self.status_changed.emit(msg)

        node = graph.get_node_by_name("denoise")
        if node is not None:
            node.callback = self.step_progress_update
            node.status_callback = lambda msg: self.status_changed.emit(msg)

        # Wire 2nd pass denoise callback (per-stage progress: resets to 0)
        sp_denoise = graph.get_node_by_name("second_pass_denoise")
        if sp_denoise is not None:
            sp_denoise.on_start_callback = lambda total_steps: self.stage_started.emit(total_steps)
            sp_denoise.callback = self.step_progress_update

        image_send = graph.get_node_by_name("image_send")
        if image_send is not None:
            image_send.image_callback = self.preview_image

        video_send = graph.get_node_by_name("video_send")
        if video_send is not None:
            video_send.video_callback = self.preview_video
            video_send.json_graph_metadata = self._job_json_graph if self.save_video_metadata else None
            if self.auto_save_videos:
                video_send.output_dir = str(self.directories.outputs_videos)
                if self.save_source_images:
                    video_send.source_image_output_dir = str(self.directories.outputs_source_images)
                else:
                    video_send.source_image_output_dir = None
                if self.save_source_audio:
                    video_send.source_audio_output_dir = str(self.directories.outputs_source_audio)
                else:
                    video_send.source_audio_output_dir = None
                if self.save_source_video:
                    video_send.source_video_output_dir = str(self.directories.outputs_source_videos)
                else:
                    video_send.source_video_output_dir = None
            else:
                video_send.output_dir = str(self.directories.temp_path)
                # Save source image/audio/video to temp so manual save button can find them
                video_send.source_image_output_dir = str(self.directories.temp_path)
                video_send.source_audio_output_dir = str(self.directories.temp_path)
                video_send.source_video_output_dir = str(self.directories.temp_path)
            if self.node_graph is not None:
                staged_vs = self.node_graph.get_node_by_name("video_send")
                if staged_vs is not None:
                    for _attr in ("video_codec", "video_crf", "video_preset", "audio_codec", "audio_bitrate_kbps"):
                        setattr(video_send, _attr, getattr(staged_vs, _attr))

    def create_run_graph_from_json(self, json_graph: str) -> FrameArtisanNodeGraph:
        callbacks = {"preview_image": self.preview_image}

        if self._persistent_run_graph is not None:
            self._persistent_run_graph.update_from_json(
                json_graph, node_classes=self._node_classes, callbacks=callbacks
            )
        else:
            self._persistent_run_graph = self._graph_factory()
            self._persistent_run_graph.set_abort_function(self.on_aborted)
            self._persistent_run_graph.from_json(json_graph, node_classes=self._node_classes, callbacks=callbacks)

        self.wire_callbacks(self._persistent_run_graph)
        return self._persistent_run_graph

    def run(self):
        self.status_changed.emit("Generating video...")
        self._completion_emitted = False
        self._pending_video_path = None
        self._pending_image = None

        json_graph = self._job_json_graph or self.get_staged_json_graph()

        run_graph = self.create_run_graph_from_json(json_graph)
        run_graph.dtype = self.dtype
        run_graph.device = self.device
        self._active_graph = run_graph

        try:
            run_graph()
        except ValueError as e:
            logger.debug(f"Configuration error: {e}")
            self._completion_emitted = True
            self.generation_error.emit(str(e), False)
        except NodeError as e:
            logger.debug(f"Error in node: '{e.node_name}': {e}")
            self._completion_emitted = True
            self.generation_error.emit(f"Error in node '{e.node_name}': {e}", False)
        except Exception as e:
            logger.error(f"Unexpected generation error: {e}")
            self._completion_emitted = True
            self.generation_error.emit(str(e), False)
        else:
            duration = run_graph.total_elapsed_time
            result = self._pending_video_path or self._pending_image
            if result is not None:
                self._completion_emitted = True
                self.generation_finished.emit(result, duration)
        finally:
            self._active_graph = None

        if not self._completion_emitted:
            self.generation_aborted.emit()

    def clean_up(self):
        if self._active_graph is not None:
            model_node = self._active_graph.get_node_by_name("model")
            if model_node is not None:
                model_node.clear_models()

        if self.node_graph is not None:
            model_node = self.node_graph.get_node_by_name("model")
            if model_node is not None:
                model_node.clear_models()

        get_model_manager().clear()

        self._persistent_run_graph = None
        self.node_graph = None
        self.dtype = None
        self.device = None

    def load_json_graph(self, json_graph: str, callbacks: dict | None = None):
        self._persistent_run_graph = None
        self.node_graph.update_from_json(json_graph, node_classes=self._node_classes, callbacks=callbacks)

    def step_progress_update(self, step, _timestep, latents):
        self.progress_update.emit(step, latents)

    def preview_image(self, image):
        self._pending_image = image

    def preview_video(self, video_path: str):
        self._pending_video_path = video_path

    def abort_graph(self):
        if self._active_graph is not None:
            self._active_graph.abort_graph()
        elif self.node_graph is not None:
            self.node_graph.abort_graph()

    def on_aborted(self):
        self._completion_emitted = True
        self.generation_aborted.emit()
