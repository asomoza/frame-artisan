from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import ClassVar

import numpy as np

from frameartisan.modules.generation.graph.node_error import NodeError
from frameartisan.modules.generation.graph.nodes.node import Node


logger = logging.getLogger(__name__)

_AV1_CPU_USED = {
    "ultrafast": 8,
    "superfast": 7,
    "veryfast": 6,
    "faster": 5,
    "medium": 4,
    "slow": 2,
    "slower": 1,
    "veryslow": 0,
}


def _av1_cpu_used(preset: str) -> int:
    return _AV1_CPU_USED.get(preset, 4)


def _encode_video_pyav(
    video,
    fps: int,
    output_path: str,
    audio=None,
    audio_sample_rate: int = 24000,
    video_codec: str = "libx264",
    video_crf: int = 23,
    video_preset: str = "medium",
    audio_codec: str = "aac",
    audio_bitrate_kbps: int = 192,
    metadata: str | None = None,
) -> None:
    import av

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    with av.open(output_path, "w") as container:
        if metadata:
            container.metadata["comment"] = metadata

        # --- video stream ---
        v_stream = container.add_stream(video_codec, rate=fps)
        v_stream.height = video.shape[1]
        v_stream.width = video.shape[2]
        v_stream.pix_fmt = "yuv420p"
        if video_codec == "libaom-av1":
            v_stream.options = {
                "crf": str(video_crf),
                "b:v": "0",
                "cpu-used": str(_av1_cpu_used(video_preset)),
            }
        else:
            v_stream.options = {"crf": str(video_crf), "preset": video_preset}

        # --- audio stream (optional) ---
        a_stream = None
        audio_frames = []
        if audio is not None:
            try:
                audio_np = audio.numpy() if hasattr(audio, "numpy") else np.asarray(audio)
                if audio_np.ndim == 1:
                    audio_np = audio_np[np.newaxis, :]  # (1, samples)
                a_stream = container.add_stream(audio_codec)
                a_stream.bit_rate = audio_bitrate_kbps * 1000
                a_stream.sample_rate = audio_sample_rate
                a_stream.layout = "stereo" if audio_np.shape[0] >= 2 else "mono"
                # Build audio frames in chunks
                chunk_size = 1024
                num_samples = audio_np.shape[1]
                for start in range(0, num_samples, chunk_size):
                    chunk = audio_np[:, start : start + chunk_size]
                    af = av.AudioFrame.from_ndarray(chunk.astype(np.float32), format="fltp", layout=a_stream.layout)
                    af.sample_rate = audio_sample_rate
                    audio_frames.append(af)
            except Exception as exc:
                logger.warning("Audio preparation failed (%s); saving video-only", exc)
                a_stream = None
                audio_frames = []

        # --- encode video frames ---
        for frame_arr in video:
            avf = av.VideoFrame.from_ndarray(np.asarray(frame_arr, dtype=np.uint8), format="rgb24")
            for pkt in v_stream.encode(avf):
                container.mux(pkt)
        for pkt in v_stream.encode(None):
            container.mux(pkt)

        # --- encode audio frames ---
        if a_stream is not None:
            for af in audio_frames:
                for pkt in a_stream.encode(af):
                    container.mux(pkt)
            for pkt in a_stream.encode(None):
                container.mux(pkt)


class LTX2VideoSendNode(Node):
    PRIORITY = -1
    REQUIRED_INPUTS: ClassVar[list[str]] = ["video", "audio", "frame_rate_out"]
    OUTPUTS: ClassVar[list[str]] = []

    SERIALIZE_INCLUDE: ClassVar[set[str]] = set()

    def __init__(self):
        super().__init__()
        self.video_callback = None
        self.output_dir = "."
        self.video_codec: str = "libx264"
        self.video_crf: int = 23
        self.video_preset: str = "medium"
        self.audio_codec: str = "aac"
        self.audio_bitrate_kbps: int = 192
        self.json_graph_metadata: str | None = None
        self.source_image_output_dir: str | None = None
        self.source_audio_output_dir: str | None = None
        self.source_video_output_dir: str | None = None

    def __call__(self):
        video = self.video
        audio = self.audio
        frame_rate_out = self.frame_rate_out

        if video is None:
            raise NodeError("video input is None", self.__class__.__name__)

        output_dir = Path(self.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = output_dir / f"ltx2_{timestamp}.mp4"

        fps = int(round(frame_rate_out)) if frame_rate_out else 24

        # Persist source files and rewrite paths in metadata before encoding
        metadata = self.json_graph_metadata
        if metadata:
            from frameartisan.utils.json_utils import persist_source_paths_in_graph

            metadata = persist_source_paths_in_graph(
                metadata,
                source_image_dir=self.source_image_output_dir,
                source_audio_dir=self.source_audio_output_dir,
                source_video_dir=self.source_video_output_dir,
            )

        try:
            _encode_video_pyav(
                video,
                fps=fps,
                output_path=str(output_path),
                audio=audio,
                audio_sample_rate=24000,
                video_codec=self.video_codec,
                video_crf=self.video_crf,
                video_preset=self.video_preset,
                audio_codec=self.audio_codec,
                audio_bitrate_kbps=self.audio_bitrate_kbps,
                metadata=metadata,
            )
            logger.debug("Video saved to %s", output_path)
        except Exception as e:
            raise NodeError(f"Failed to encode video: {e}", self.__class__.__name__) from e

        if self.video_callback is not None:
            self.video_callback(str(output_path))

        return self.values
