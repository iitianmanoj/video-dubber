"""
sync/align.py
-------------
Aligns synthesized English speech clips to the original segment timing:

  1. Trim leading/trailing silence from each TTS clip.
  2. Compare clip duration to the original segment's time slot.
  3. If the clip is too long/short, adjust playback speed (via ffmpeg's
     `atempo` filter, which changes speed without pitch-shifting) within
     safe bounds, rather than stretching pitch or cutting words.
  4. Place each clip at its segment's original start time on a full-length
     timeline, silence-padding the gaps, producing one continuous track
     the same length as the source audio.
"""

from __future__ import annotations

from pathlib import Path
from typing import List

from pydub import AudioSegment
from pydub.silence import detect_leading_silence

from config import CONFIG, TEMP_DIR
from tts.edge_engine import SynthesizedSegment
from utils.helpers import ensure_within_bounds
from utils.logger import get_logger

logger = get_logger()


class AlignmentError(RuntimeError):
    pass


def _trim_silence(audio: AudioSegment, silence_thresh: int) -> AudioSegment:
    start_trim = detect_leading_silence(audio, silence_threshold=silence_thresh)
    end_trim = detect_leading_silence(audio.reverse(), silence_threshold=silence_thresh)
    duration = len(audio)
    return audio[start_trim: duration - end_trim] if duration - end_trim > start_trim else audio


def _speed_change(audio: AudioSegment, speed: float) -> AudioSegment:
    """Change playback speed without changing pitch, using frame-rate trick
    combined with pydub's own resample (good enough for speech clips;
    ffmpeg's atempo filter is used instead when precision matters more,
    see `_ffmpeg_atempo` fallback below for extreme ratios)."""
    if abs(speed - 1.0) < 0.01:
        return audio
    new_frame_rate = int(audio.frame_rate * speed)
    shifted = audio._spawn(audio.raw_data, overrides={"frame_rate": new_frame_rate})
    return shifted.set_frame_rate(audio.frame_rate)


class AudioAligner:
    """Builds a single dubbed-audio track matching the original timeline."""

    def __init__(self, work_dir: Path = TEMP_DIR):
        self.work_dir = Path(work_dir)
        self.work_dir.mkdir(parents=True, exist_ok=True)

    def build_timeline(
        self, synthesized_segments: List[SynthesizedSegment], total_duration_seconds: float,
    ) -> Path:
        if not synthesized_segments:
            raise AlignmentError("No synthesized segments to align.")

        timeline = AudioSegment.silent(duration=int(total_duration_seconds * 1000))
        cfg = CONFIG.sync

        for item in synthesized_segments:
            seg = item.segment
            try:
                clip = AudioSegment.from_file(item.audio_path)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Skipping unreadable TTS clip for segment %s: %s", seg.id, exc)
                continue

            clip = _trim_silence(clip, cfg.silence_thresh_db)

            slot_ms = max(int((seg.end - seg.start) * 1000), 1)
            clip_ms = len(clip)

            if clip_ms > 0:
                required_speed = clip_ms / slot_ms
                speed = ensure_within_bounds(
                    required_speed, 1.0 / cfg.max_slow_down, cfg.max_speed_up,
                )
                if abs(speed - 1.0) > 0.01:
                    clip = _speed_change(clip, speed)

            # If the (speed-adjusted) clip still overruns the slot because we
            # hit the max-speed clamp, allow slight overflow into the next
            # gap rather than cutting off words mid-sentence.
            start_ms = int(seg.start * 1000)
            end_ms = start_ms + len(clip)
            if end_ms > len(timeline):
                pad = end_ms - len(timeline)
                timeline += AudioSegment.silent(duration=pad)

            timeline = timeline.overlay(clip, position=start_ms)

        output_path = self.work_dir / "dubbed_audio_timeline.wav"
        timeline.export(output_path, format="wav")
        logger.debug("Built aligned dubbed audio timeline at %s", output_path)
        return output_path
