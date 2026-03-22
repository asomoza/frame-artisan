"""Background thread for video trimming and scaling via PyAV."""

from __future__ import annotations

import logging

from PyQt6.QtCore import QThread, pyqtSignal

logger = logging.getLogger(__name__)


class VideoProcessThread(QThread):
    """Trim and optionally scale a video file to a new temp file.

    Decodes only the requested frame range and resizes per-frame (never
    materialises the full-resolution video in RAM).
    """

    progress = pyqtSignal(int)  # 0-100
    finished_path = pyqtSignal(str)
    error = pyqtSignal(str)

    def __init__(
        self,
        source_path: str,
        output_path: str,
        trim_start_frame: int,
        trim_end_frame: int,
        output_width: int | None,
        output_height: int | None,
        fps: float,
        include_audio: bool = True,
    ):
        super().__init__()
        self.source_path = source_path
        self.output_path = output_path
        self.trim_start_frame = trim_start_frame  # 0-based inclusive
        self.trim_end_frame = trim_end_frame  # 0-based exclusive
        self.output_width = output_width
        self.output_height = output_height
        self.fps = fps
        self.include_audio = include_audio
        self._stop_requested = False

    def stop(self) -> None:
        self._stop_requested = True

    def run(self) -> None:
        try:
            self._process()
        except Exception as e:
            logger.exception("Video processing failed")
            self.error.emit(str(e))

    def _process(self) -> None:
        import av
        from PIL import Image

        total_frames = self.trim_end_frame - self.trim_start_frame
        if total_frames <= 0:
            self.error.emit("No frames to process")
            return

        needs_resize = self.output_width is not None and self.output_height is not None

        with av.open(self.source_path) as in_container:
            in_video = in_container.streams.video[0]

            # Seek to nearest keyframe before trim start
            if self.trim_start_frame > 0:
                time_base = in_video.time_base
                start_sec = self.trim_start_frame / self.fps
                pts = int(start_sec / time_base)
                in_container.seek(pts, stream=in_video)

            # Check for audio
            has_audio = self.include_audio and len(in_container.streams.audio) > 0
            audio_start_sec = self.trim_start_frame / self.fps
            audio_end_sec = self.trim_end_frame / self.fps

            # Set up output
            with av.open(self.output_path, "w") as out_container:
                out_width = self.output_width if needs_resize else in_video.width
                out_height = self.output_height if needs_resize else in_video.height

                from fractions import Fraction

                out_video = out_container.add_stream("libx264", rate=Fraction(self.fps).limit_denominator(10000))
                out_video.width = out_width
                out_video.height = out_height
                out_video.pix_fmt = "yuv420p"
                out_video.options = {"crf": "18", "preset": "fast"}

                # Decode and re-encode video frames
                frame_idx = 0
                written = 0
                for frame in in_container.decode(video=0):
                    if self._stop_requested:
                        return

                    if frame_idx < self.trim_start_frame:
                        frame_idx += 1
                        continue
                    if frame_idx >= self.trim_end_frame:
                        break

                    if needs_resize:
                        img = frame.to_image()
                        img = img.resize((out_width, out_height), Image.Resampling.LANCZOS)
                        out_frame = av.VideoFrame.from_image(img)
                    else:
                        out_frame = frame.reformat(format="yuv420p")

                    out_frame.pts = written
                    for packet in out_video.encode(out_frame):
                        out_container.mux(packet)

                    written += 1
                    pct = int(written / total_frames * 100)
                    self.progress.emit(min(pct, 100))
                    frame_idx += 1

                # Flush video
                for packet in out_video.encode(None):
                    out_container.mux(packet)

                # Copy audio (trim to matching time range)
                if has_audio:
                    try:
                        self._copy_audio(
                            self.source_path, out_container,
                            audio_start_sec, audio_end_sec,
                        )
                    except Exception as e:
                        logger.warning("Audio copy failed, producing video-only: %s", e)

        if self._stop_requested:
            return

        logger.info(
            "Video processed: %d frames, %dx%d → %s",
            written, out_width, out_height, self.output_path,
        )
        self.finished_path.emit(self.output_path)

    @staticmethod
    def _copy_audio(
        source_path: str,
        out_container,
        start_sec: float,
        end_sec: float,
    ) -> None:
        """Re-encode audio for the trimmed time range."""
        import av

        with av.open(source_path) as audio_in:
            in_audio = audio_in.streams.audio[0]
            out_audio = out_container.add_stream("aac", rate=in_audio.rate)
            out_audio.layout = in_audio.layout

            # Seek audio to start
            if start_sec > 0:
                time_base = in_audio.time_base
                pts = int(start_sec / time_base)
                audio_in.seek(pts, stream=in_audio)

            for frame in audio_in.decode(audio=0):
                if frame.time is not None:
                    if frame.time < start_sec:
                        continue
                    if frame.time >= end_sec:
                        break
                for packet in out_audio.encode(frame):
                    out_container.mux(packet)

            for packet in out_audio.encode(None):
                out_container.mux(packet)
