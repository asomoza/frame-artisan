from __future__ import annotations

import logging
import os
import tempfile
from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np
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
    preview_video_ready = pyqtSignal(str)  # temp video path
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
        self.preview_decode: bool = False
        self.preview_time_upscale: bool = False
        self.preview_space_upscale: bool = True
        self.tiny_vae_path: str | None = None
        self._job_json_graph: str | None = None
        self._active_graph: FrameArtisanNodeGraph | None = None
        self._persistent_run_graph: FrameArtisanNodeGraph | None = None
        self._completion_emitted: bool = False
        self._tiny_vae = None
        self._preview_temp_paths: list[str] = []
        self._preview_temp_index: int = 0

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
            sp_denoise.on_start_callback = self._on_stage_started
            sp_denoise.callback = self.step_progress_update

        # RAM-saving mode: evict transformers between stages
        self._wire_ram_saving_callbacks(graph, node, sp_denoise)

        image_send = graph.get_node_by_name("image_send")
        if image_send is not None:
            image_send.image_callback = self.preview_image

        video_send = graph.get_node_by_name("video_send")
        if video_send is not None:
            video_send.video_callback = self.preview_video
            video_send.json_graph_metadata = self._job_json_graph if self.save_video_metadata else None
            if self.auto_save_videos:
                video_send.output_dir = str(self.directories.outputs_videos)
            else:
                video_send.output_dir = str(self.directories.temp_path)
            # Source files are always persisted to permanent dirs (paths are
            # embedded in the video metadata and must survive temp cleanup).
            video_send.source_image_output_dir = (
                str(self.directories.outputs_source_images) if self.save_source_images else None
            )
            video_send.source_audio_output_dir = (
                str(self.directories.outputs_source_audio) if self.save_source_audio else None
            )
            video_send.source_video_output_dir = (
                str(self.directories.outputs_source_videos) if self.save_source_video else None
            )
            video_send.lora_mask_output_dir = (
                str(self.directories.outputs_lora_masks) if self.save_source_images else None
            )
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

    def _load_tiny_vae(self) -> None:
        """Load the tiny VAE for live preview if enabled and available."""
        if not self.preview_decode or not self.tiny_vae_path:
            self._tiny_vae = None
            return
        if not os.path.isfile(self.tiny_vae_path):
            logger.warning("Tiny VAE not found at %s, disabling preview", self.tiny_vae_path)
            self._tiny_vae = None
            return

        from frameartisan.modules.generation.graph.nodes.taehv import TAEHV

        decoder_time_upscale = None if self.preview_time_upscale else (False, False)
        decoder_space_upscale = (True, True, True) if self.preview_space_upscale else (False, False, False)
        self._tiny_vae = TAEHV(
            checkpoint_path=self.tiny_vae_path,
            decoder_time_upscale=decoder_time_upscale,
            decoder_space_upscale=decoder_space_upscale,
        ).to(
            device=self.device, dtype=torch.float16
        )
        self._tiny_vae.eval()
        logger.info("Tiny VAE loaded for live preview")

    def _get_preview_temp_path(self) -> str:
        """Return the next temp file path, alternating between two files."""
        if not self._preview_temp_paths:
            for i in range(2):
                fd, path = tempfile.mkstemp(suffix=".mp4", prefix=f"fa_preview_{i}_")
                os.close(fd)
                self._preview_temp_paths.append(path)
        path = self._preview_temp_paths[self._preview_temp_index % 2]
        self._preview_temp_index += 1
        return path

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

        self._load_tiny_vae()

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
            self._tiny_vae = None  # free VRAM

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
        self._tiny_vae = None
        self._cleanup_preview_temp()

    def _cleanup_preview_temp(self):
        for path in self._preview_temp_paths:
            try:
                os.unlink(path)
            except OSError:
                pass
        self._preview_temp_paths = []
        self._preview_temp_index = 0

    def load_json_graph(self, json_graph: str, callbacks: dict | None = None):
        self._persistent_run_graph = None
        self.node_graph.update_from_json(json_graph, node_classes=self._node_classes, callbacks=callbacks)

    def _wire_ram_saving_callbacks(self, graph, denoise_node, sp_denoise_node) -> None:
        """Wire model eviction callbacks for RAM-saving (sequential_group_offload) mode."""
        from frameartisan.app.model_manager import get_model_manager

        mm = get_model_manager()
        if not mm.is_ram_saving_mode:
            return
        if denoise_node is None or sp_denoise_node is None:
            return
        if not sp_denoise_node.enabled:
            return

        model_node = graph.get_node_by_name("model")
        sp_model_node = graph.get_node_by_name("second_pass_model")
        if model_node is None or sp_model_node is None:
            return

        def _evict_stage1_transformer(_node):
            mm = get_model_manager()
            s1_hash = getattr(model_node, "_transformer_cache_hash", None)
            s2_hash = getattr(sp_model_node, "_transformer_cache_hash", None)
            if s1_hash and s2_hash and s1_hash != s2_hash:
                mm.evict_component_and_cache("transformer", s1_hash)
                logger.info("RAM-saving: evicted stage 1 transformer after denoise")

        def _evict_stage2_transformer(_node):
            mm = get_model_manager()
            s2_hash = getattr(sp_model_node, "_transformer_cache_hash", None)
            mm.evict_component_and_cache("sp_transformer", s2_hash)
            logger.info("RAM-saving: evicted stage 2 transformer after second_pass_denoise")

        denoise_node.on_complete_callback = _evict_stage1_transformer
        sp_denoise_node.on_complete_callback = _evict_stage2_transformer

    def _on_stage_started(self, total_steps: int) -> None:
        """Called when the 2nd pass starts. Free the tiny VAE to reclaim VRAM."""
        if self._tiny_vae is not None:
            del self._tiny_vae
            self._tiny_vae = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info("Tiny VAE unloaded before 2nd pass to free VRAM")
        self.stage_started.emit(total_steps)

    def step_progress_update(self, step, _timestep, latents):
        self.progress_update.emit(step, latents)
        self._emit_preview_video(latents)

    def _emit_preview_video(self, latents: torch.Tensor) -> None:
        """Decode packed latents with the tiny VAE, encode to temp video, and emit path."""
        if self._tiny_vae is None:
            return

        graph = self._active_graph
        if graph is None:
            return

        # Only use first-pass denoise — preview is disabled for 2nd pass
        denoise = graph.get_node_by_name("denoise")
        if denoise is None or denoise._latent_height == 0:
            return

        h = denoise._latent_height
        w = denoise._latent_width
        f = denoise._latent_num_frames
        frame_rate = int(getattr(denoise, "frame_rate", 24) or 24)

        try:
            video_path = self._get_preview_temp_path()
            _decode_and_encode_preview(latents, h, w, f, self._tiny_vae, video_path, frame_rate)
            self.preview_video_ready.emit(video_path)
        except Exception:
            logger.debug("Preview decode failed", exc_info=True)

    def preview_image(self, image):
        self._pending_image = image

    def preview_video(self, video_path: str):
        self._pending_video_path = video_path

    def clear_node_values(self):
        """Clear all node output values and mark nodes for re-execution."""
        if self._persistent_run_graph is not None:
            for node in self._persistent_run_graph.nodes:
                node.values.clear()
                node.updated = True
                # Clear model node caches so components are reloaded from disk
                if "_prev_component_paths" in node.__dict__:
                    node._prev_component_paths.clear()
                if "_prev_values" in node.__dict__:
                    node._prev_values.clear()

    def abort_graph(self):
        if self._active_graph is not None:
            self._active_graph.abort_graph()
        elif self.node_graph is not None:
            self.node_graph.abort_graph()

    def on_aborted(self):
        self._completion_emitted = True
        self.generation_aborted.emit()


@torch.no_grad()
def _decode_and_encode_preview(
    latents: torch.Tensor,
    h: int,
    w: int,
    f: int,
    tiny_vae,
    output_path: str,
    frame_rate: int,
) -> None:
    """Unpack latents, decode with the tiny VAE, and encode to an MP4 preview."""
    import av

    from frameartisan.modules.generation.graph.nodes.ltx2_utils import unpack_latents

    # Unpack from [B, seq_len, C] to [B, C, F, H, W]
    unpacked = unpack_latents(latents, f, h, w, patch_size=1, patch_size_t=1)

    # TAEHV expects NTCHW — transpose from NCFHW to NFCHW
    ntchw = unpacked.transpose(1, 2)  # [N, F, C, H, W]

    # Decode with tiny VAE (float16, same device)
    ntchw = ntchw.to(device=tiny_vae.decoder[1].weight.device, dtype=torch.float16)
    decoded = tiny_vae.decode_video(ntchw, parallel=True)  # [N, T', 3, H', W'] in [0, 1]

    # Convert to uint8 numpy: [T', H', W', 3]
    frames = (decoded[0] * 255).clamp(0, 255).to(torch.uint8).cpu().numpy()
    # frames shape: [T', 3, H', W'] — need [T', H', W', 3]
    frames = np.transpose(frames, (0, 2, 3, 1))
    frames = np.ascontiguousarray(frames)

    # Encode to MP4 with PyAV
    with av.open(output_path, "w") as container:
        stream = container.add_stream("libx264", rate=frame_rate)
        stream.height = frames.shape[1]
        stream.width = frames.shape[2]
        stream.pix_fmt = "yuv420p"
        stream.options = {"crf": "23", "preset": "ultrafast"}

        for frame_arr in frames:
            avf = av.VideoFrame.from_ndarray(frame_arr, format="rgb24")
            for pkt in stream.encode(avf):
                container.mux(pkt)
        for pkt in stream.encode(None):
            container.mux(pkt)
