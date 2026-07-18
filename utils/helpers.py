"""
utils/helpers.py
-----------------
Small, dependency-light helper functions shared across modules:
dependency checks, disk-space checks, filename sanitization, time
formatting, and GPU detection.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path
from typing import List, Tuple


class DependencyError(RuntimeError):
    """Raised when a required external binary or library is missing."""


def check_ffmpeg() -> None:
    """Verify ffmpeg/ffprobe are installed and on PATH.

    Raises DependencyError with an actionable message if not.
    """
    for binary in ("ffmpeg", "ffprobe"):
        if shutil.which(binary) is None:
            raise DependencyError(
                f"'{binary}' was not found on PATH. Install ffmpeg first:\n"
                "  - Ubuntu/Debian: sudo apt install ffmpeg\n"
                "  - macOS (Homebrew): brew install ffmpeg\n"
                "  - Windows: https://ffmpeg.org/download.html "
                "(and add it to your PATH)"
            )


def check_disk_space(path: Path, required_gb: float = 2.0) -> None:
    """Raise DependencyError if free space at `path` is below `required_gb`."""
    usage = shutil.disk_usage(path)
    free_gb = usage.free / (1024 ** 3)
    if free_gb < required_gb:
        raise DependencyError(
            f"Low disk space at {path}: {free_gb:.2f} GB free, "
            f"need at least {required_gb} GB."
        )


def check_internet(host: str = "8.8.8.8", port: int = 53, timeout: float = 3.0) -> bool:
    """Best-effort connectivity check (DNS root server on port 53)."""
    import socket

    try:
        socket.setdefaulttimeout(timeout)
        socket.socket(socket.AF_INET, socket.SOCK_STREAM).connect((host, port))
        return True
    except OSError:
        return False


def gpu_available() -> bool:
    """Return True if a CUDA-capable GPU is visible to torch."""
    try:
        import torch  # imported lazily; torch is a heavy dependency

        return torch.cuda.is_available()
    except ImportError:
        return False


def sanitize_filename(name: str, max_length: int = 120) -> str:
    """Strip characters that are unsafe for filenames across platforms."""
    name = re.sub(r'[\\/*?:"<>|]', "", name).strip()
    name = re.sub(r"\s+", "_", name)
    return name[:max_length] or "video"


def format_duration(seconds: float) -> str:
    """Format seconds as H:MM:SS (or M:SS for short clips)."""
    seconds = int(round(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes}:{secs:02d}"


def get_media_duration(path: Path) -> float:
    """Return the duration in seconds of an audio/video file via ffprobe."""
    result = subprocess.run(
        [
            "ffprobe", "-v", "error", "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1", str(path),
        ],
        capture_output=True, text=True, check=True,
    )
    return float(result.stdout.strip())


def split_into_chunks(total_seconds: float, chunk_minutes: int) -> List[Tuple[float, float]]:
    """Split a total duration into (start, end) chunk boundaries in seconds."""
    chunk_seconds = chunk_minutes * 60
    chunks: List[Tuple[float, float]] = []
    start = 0.0
    while start < total_seconds:
        end = min(start + chunk_seconds, total_seconds)
        chunks.append((start, end))
        start = end
    return chunks


def ensure_within_bounds(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))
