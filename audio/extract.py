"""
audio/extract.py
-----------------
Extracts a mono 16kHz WAV audio track from the downloaded video, suitable
for Whisper transcription. Uses ffmpeg directly via subprocess (through
ffmpeg-python) rather than moviepy for this step, since it is far faster
and uses far less RAM for long videos.
"""

from __future__ import annotations

from pathlib import Path

import ffmpeg

from config import TEMP_DIR
from utils.logger import get_logger

logger = get_logger()


class AudioExtractionError(RuntimeError):
    pass


class AudioExtractor:
    """Extracts audio from a video file into a Whisper-friendly WAV file."""

    def __init__(self, output_dir: Path = TEMP_DIR):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def extract(self, video_path: Path, sample_rate: int = 16000) -> Path:
        video_path = Path(video_path)
        if not video_path.exists():
            raise AudioExtractionError(f"Video file not found: {video_path}")

        audio_path = self.output_dir / f"{video_path.stem}_original_audio.wav"

        try:
            (
                ffmpeg
                .input(str(video_path))
                .output(
                    str(audio_path),
                    ac=1,                  # mono
                    ar=sample_rate,        # 16kHz for Whisper
                    format="wav",
                    acodec="pcm_s16le",
                    vn=None,               # no video stream
                )
                .overwrite_output()
                .run(quiet=True, capture_stdout=True, capture_stderr=True)
            )
        except ffmpeg.Error as exc:
            stderr = exc.stderr.decode("utf-8", errors="ignore") if exc.stderr else str(exc)
            raise AudioExtractionError(f"ffmpeg failed to extract audio: {stderr}") from exc

        if not audio_path.exists() or audio_path.stat().st_size == 0:
            raise AudioExtractionError(f"Audio extraction produced an empty file: {audio_path}")

        logger.debug("Extracted audio to %s", audio_path)
        return audio_path
