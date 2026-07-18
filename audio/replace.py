"""
audio/replace.py
-----------------
Muxes the new dubbed audio track into the original video, replacing the
original audio stream. The video stream is stream-copied (`-c:v copy`),
never re-encoded, per the spec — this keeps output quality identical to
the source and makes the remux step very fast (no GPU/CPU-heavy encoding).
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from config import CONFIG, OUTPUT_DIR
from utils.logger import get_logger

logger = get_logger()


class AudioReplaceError(RuntimeError):
    pass


class AudioReplacer:
    """Replaces a video's audio track with a new dubbed audio file."""

    def __init__(self, output_dir: Path = OUTPUT_DIR):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def replace(self, video_path: Path, dubbed_audio_path: Path, output_name: str) -> Path:
        video_path = Path(video_path)
        dubbed_audio_path = Path(dubbed_audio_path)

        if not video_path.exists():
            raise AudioReplaceError(f"Video not found: {video_path}")
        if not dubbed_audio_path.exists():
            raise AudioReplaceError(f"Dubbed audio not found: {dubbed_audio_path}")

        output_path = self.output_dir / f"{output_name}.{CONFIG.video.output_container}"

        cmd = [
            "ffmpeg", "-y",
            "-i", str(video_path),
            "-i", str(dubbed_audio_path),
            "-map", "0:v:0",              # take video from the source file
            "-map", "1:a:0",              # take audio from the dubbed file
        ]

        if CONFIG.video.copy_video_stream:
            cmd += ["-c:v", "copy"]       # never re-encode video
        cmd += [
            "-c:a", CONFIG.video.audio_codec,
            "-b:a", CONFIG.video.audio_bitrate,
            "-shortest",                  # avoid trailing silence/frozen frame mismatch
            "-movflags", "+faststart",
            str(output_path),
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise AudioReplaceError(f"ffmpeg remux failed: {result.stderr[-2000:]}")

        if not output_path.exists() or output_path.stat().st_size == 0:
            raise AudioReplaceError(f"Remux produced an empty output file: {output_path}")

        logger.debug("Final dubbed video written to %s", output_path)
        return output_path
