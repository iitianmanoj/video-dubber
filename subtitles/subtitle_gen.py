"""
subtitles/subtitle_gen.py
--------------------------
Generates SRT and WebVTT subtitle files, plus a plain translated
transcript TXT file, from the translated segments.
"""

from __future__ import annotations

from pathlib import Path
from typing import List

from transcription.whisper_engine import Segment
from utils.logger import get_logger

logger = get_logger()


def _format_srt_timestamp(seconds: float) -> str:
    ms_total = int(round(seconds * 1000))
    hours, rem = divmod(ms_total, 3_600_000)
    minutes, rem = divmod(rem, 60_000)
    secs, ms = divmod(rem, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{ms:03d}"


def _format_vtt_timestamp(seconds: float) -> str:
    ms_total = int(round(seconds * 1000))
    hours, rem = divmod(ms_total, 3_600_000)
    minutes, rem = divmod(rem, 60_000)
    secs, ms = divmod(rem, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{ms:03d}"


def write_srt(segments: List[Segment], out_path: Path) -> Path:
    lines = []
    for i, seg in enumerate(segments, start=1):
        lines.append(str(i))
        lines.append(f"{_format_srt_timestamp(seg.start)} --> {_format_srt_timestamp(seg.end)}")
        lines.append(seg.text)
        lines.append("")
    out_path.write_text("\n".join(lines), encoding="utf-8")
    return out_path


def write_vtt(segments: List[Segment], out_path: Path) -> Path:
    lines = ["WEBVTT", ""]
    for seg in segments:
        lines.append(f"{_format_vtt_timestamp(seg.start)} --> {_format_vtt_timestamp(seg.end)}")
        lines.append(seg.text)
        lines.append("")
    out_path.write_text("\n".join(lines), encoding="utf-8")
    return out_path


def write_transcript_txt(segments: List[Segment], out_path: Path) -> Path:
    lines = [seg.text for seg in segments]
    out_path.write_text("\n".join(lines), encoding="utf-8")
    return out_path


def generate_all(segments: List[Segment], output_dir: Path, base_name: str) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "srt": write_srt(segments, output_dir / f"{base_name}.srt"),
        "vtt": write_vtt(segments, output_dir / f"{base_name}.vtt"),
        "txt": write_transcript_txt(segments, output_dir / f"{base_name}_transcript.txt"),
    }
    logger.debug("Generated subtitles/transcript: %s", paths)
    return paths
