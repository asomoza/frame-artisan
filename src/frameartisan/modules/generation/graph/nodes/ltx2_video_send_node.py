from __future__ import annotations

import hashlib
import logging
import shutil
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
    OPTIONAL_INPUTS: ClassVar[list[str]] = ["source_images", "source_audio_path", "source_video_frames"]
    OUTPUTS: ClassVar[list[str]] = []

    SERIALIZE_INCLUDE: ClassVar[set[str]] = set()

    @staticmethod
    def _get_db():
        """Get a Database instance for the current thread, or None."""
        from frameartisan.app.app import get_app_database_path
        from frameartisan.utils.database import Database

        db_path = get_app_database_path()
        if db_path is None:
            return None
        return Database(db_path)

    @staticmethod
    def _find_existing_source(db, kind: str, content_hash: str) -> str | None:
        """Look up a source file by kind+hash. Returns filepath if it exists on disk."""
        row = db.fetch_one(
            "SELECT filepath FROM source_file WHERE kind = ? AND content_hash = ?",
            (kind, content_hash),
        )
        if row is None:
            return None
        filepath = row[0]
        if Path(filepath).exists():
            return filepath
        # File was deleted from disk — remove stale DB row
        db.execute(
            "DELETE FROM source_file WHERE kind = ? AND content_hash = ?",
            (kind, content_hash),
        )
        return None

    @staticmethod
    def _record_source(db, kind: str, content_hash: str, filepath: str) -> None:
        """Insert or update a source file record."""
        db.execute(
            "INSERT INTO source_file (kind, content_hash, filepath) VALUES (?, ?, ?) "
            "ON CONFLICT(kind, content_hash) DO UPDATE SET filepath = excluded.filepath",
            (kind, content_hash, filepath),
        )

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
                metadata=self.json_graph_metadata,
            )
            logger.debug("Video saved to %s", output_path)
        except Exception as e:
            raise NodeError(f"Failed to encode video: {e}", self.__class__.__name__) from e

        # Save all condition source images
        self._save_source_images(timestamp, output_dir)

        # Save source audio file
        self._save_source_audio(timestamp, output_dir)

        # Save source video file
        self._save_source_video(timestamp, output_dir, fps)

        if self.video_callback is not None:
            self.video_callback(str(output_path))

        return self.values

    def _save_source_images(self, timestamp: str, output_dir: Path) -> None:
        if self.source_image_output_dir is None:
            return
        raw = getattr(self, "source_images", None)
        if raw is None:
            return

        images = raw if isinstance(raw, list) else [raw]

        try:
            from PIL import Image

            src_dir = Path(self.source_image_output_dir)
            src_dir.mkdir(parents=True, exist_ok=True)
            db = self._get_db()

            for idx, img in enumerate(images):
                if not isinstance(img, np.ndarray):
                    continue
                img_hash = hashlib.md5(img.tobytes()).hexdigest()
                kind = f"image:{idx}"

                if db is not None and self._find_existing_source(db, kind, img_hash) is not None:
                    logger.debug("Source image %d unchanged, skipping save (hash=%s)", idx, img_hash[:8])
                    continue

                suffix = f"_{idx}" if len(images) > 1 else ""
                src_path = src_dir / f"ltx2_{timestamp}{suffix}.png"
                Image.fromarray(img.astype(np.uint8)).save(str(src_path))
                logger.debug("Source image %d saved to %s", idx, src_path)
                if db is not None:
                    self._record_source(db, kind, img_hash, str(src_path))
        except Exception as exc:
            logger.warning("Failed to save source image(s): %s", exc)

    def _save_source_audio(self, timestamp: str, output_dir: Path) -> None:
        if self.source_audio_output_dir is None:
            return
        audio_path = getattr(self, "source_audio_path", None)
        if not audio_path:
            return

        try:
            src = Path(audio_path)
            if not src.is_file():
                return

            audio_dir = Path(self.source_audio_output_dir)
            audio_dir.mkdir(parents=True, exist_ok=True)
            db = self._get_db()

            audio_hash = hashlib.md5(src.read_bytes()).hexdigest()
            if db is not None and self._find_existing_source(db, "audio", audio_hash) is not None:
                logger.debug("Source audio unchanged, skipping save (hash=%s)", audio_hash[:8])
                return

            dest = audio_dir / f"ltx2_{timestamp}{src.suffix}"
            shutil.copy2(str(src), str(dest))
            logger.debug("Source audio saved to %s", dest)
            if db is not None:
                self._record_source(db, "audio", audio_hash, str(dest))
        except Exception as exc:
            logger.warning("Failed to save source audio: %s", exc)

    def _save_source_video(self, timestamp: str, output_dir: Path, fps: int) -> None:
        if self.source_video_output_dir is None:
            return
        frames = getattr(self, "source_video_frames", None)
        if frames is None:
            return
        if not isinstance(frames, np.ndarray) or frames.ndim != 4:
            return

        try:
            video_dir = Path(self.source_video_output_dir)
            video_dir.mkdir(parents=True, exist_ok=True)
            db = self._get_db()

            video_hash = hashlib.md5(frames.tobytes()).hexdigest()
            if db is not None and self._find_existing_source(db, "video", video_hash) is not None:
                logger.debug("Source video unchanged, skipping save (hash=%s)", video_hash[:8])
                return

            dest = video_dir / f"ltx2_{timestamp}_source_video.mp4"
            _encode_video_pyav(frames, fps=fps, output_path=str(dest))
            logger.debug("Source video saved to %s", dest)
            if db is not None:
                self._record_source(db, "video", video_hash, str(dest))
        except Exception as exc:
            logger.warning("Failed to save source video: %s", exc)
