from __future__ import annotations

import logging
from fractions import Fraction


logger = logging.getLogger(__name__)

PREVIEW_DURATION_S = 2.0


def extract_preview_clip(video_path: str, start_ms: int, output_path: str) -> bool:
    """Extract a short MP4 clip starting at *start_ms* and save to *output_path*.

    The clip is re-encoded at a small resolution (max 345px on the long side)
    for use as a hover preview.  Much smaller than APNG for equivalent quality.

    Returns ``True`` on success, ``False`` on error.
    """
    try:
        import av
    except ImportError:
        logger.warning("PyAV not available — cannot extract preview clip")
        return False

    try:
        src = av.open(video_path)
    except Exception:
        logger.warning("Failed to open video: %s", video_path)
        return False

    dst = None
    try:
        src_stream = src.streams.video[0]
        src_rate = src_stream.average_rate or src_stream.guessed_rate or Fraction(15, 1)
        src_fps = float(src_rate)

        # Seek to start_ms
        time_base = src_stream.time_base
        if time_base is None:
            time_base = src_stream.codec_context.time_base
        start_sec = start_ms / 1000.0
        start_pts = int(start_sec / float(time_base))
        src.seek(start_pts, stream=src_stream)

        # Determine output size — fit within 345x345 keeping aspect ratio
        src_w = src_stream.codec_context.width
        src_h = src_stream.codec_context.height
        max_dim = 345
        if src_w >= src_h:
            out_w = max_dim
            out_h = int(src_h * max_dim / src_w)
        else:
            out_h = max_dim
            out_w = int(src_w * max_dim / src_h)
        # Ensure even dimensions (required by libx264)
        out_w = out_w + (out_w % 2)
        out_h = out_h + (out_h % 2)

        max_frames = int(PREVIEW_DURATION_S * src_fps)

        dst = av.open(output_path, mode="w")
        dst_stream = dst.add_stream("libx264", rate=src_rate)
        dst_stream.width = out_w
        dst_stream.height = out_h
        dst_stream.pix_fmt = "yuv420p"
        dst_stream.options = {"crf": "28", "preset": "fast"}

        frame_count = 0
        for frame in src.decode(video=0):
            # Skip frames before the actual seek target (seek lands on the
            # nearest keyframe which may be earlier than requested).
            if frame.time is not None and frame.time < start_sec:
                continue
            frame = frame.reformat(width=out_w, height=out_h, format="yuv420p")
            for packet in dst_stream.encode(frame):
                dst.mux(packet)
            frame_count += 1
            if frame_count >= max_frames:
                break

        # Flush encoder
        for packet in dst_stream.encode():
            dst.mux(packet)

        dst.close()
        src.close()

        return frame_count > 0
    except Exception:
        logger.exception("Failed to extract preview clip from %s", video_path)
        try:
            src.close()
        except Exception:
            pass
        if dst is not None:
            try:
                dst.close()
            except Exception:
                pass
        return False
