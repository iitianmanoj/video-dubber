"""
downloader/download.py
-----------------------
Wraps yt-dlp to download a YouTube video at the best available quality,
with retries and clear, specific errors for the failure modes called out
in the spec (invalid URL, private video, age restriction, no internet,
interrupted downloads).
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import yt_dlp

from config import CONFIG, TEMP_DIR
from utils.helpers import check_internet, sanitize_filename
from utils.logger import get_logger

logger = get_logger()

_YOUTUBE_URL_RE = re.compile(
    r"^(https?://)?(www\.)?(youtube\.com/(watch\?v=|shorts/|live/)|youtu\.be/)[\w\-]+",
    re.IGNORECASE,
)


class DownloadError(RuntimeError):
    """Raised for any unrecoverable download failure, with a clear reason."""


@dataclass
class VideoInfo:
    video_path: Path
    title: str
    duration_seconds: float
    uploader: str
    original_url: str


class VideoDownloader:
    """Downloads a single YouTube video to `temp/` using yt-dlp."""

    def __init__(self, output_dir: Path = TEMP_DIR):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def validate_url(url: str) -> str:
        url = url.strip()
        if not _YOUTUBE_URL_RE.match(url):
            raise DownloadError(
                f"'{url}' does not look like a valid YouTube URL. "
                "Expected formats: https://www.youtube.com/watch?v=... or "
                "https://youtu.be/..."
            )
        return url

    def download(self, url: str) -> VideoInfo:
        url = self.validate_url(url)

        if not check_internet():
            raise DownloadError(
                "No internet connection detected. Check your network and try again."
            )

        last_error: Optional[Exception] = None
        for attempt in range(1, CONFIG.max_download_retries + 1):
            try:
                return self._download_once(url)
            except yt_dlp.utils.DownloadError as exc:
                last_error = exc
                message = str(exc).lower()

                if "private video" in message:
                    raise DownloadError("This video is private and cannot be downloaded.") from exc
                if "sign in to confirm your age" in message or "age" in message and "restrict" in message:
                    raise DownloadError(
                        "This video is age-restricted. Provide cookies via "
                        "yt-dlp's --cookies option (see README) to bypass this."
                    ) from exc
                if "video unavailable" in message:
                    raise DownloadError("This video is unavailable (deleted, region-locked, or removed).") from exc
                if "unsupported url" in message:
                    raise DownloadError(f"'{url}' is not a supported/valid video URL.") from exc

                logger.warning(
                    "Download attempt %d/%d failed: %s",
                    attempt, CONFIG.max_download_retries, exc,
                )
                time.sleep(2 * attempt)  # simple backoff for interrupted downloads

        raise DownloadError(
            f"Failed to download video after {CONFIG.max_download_retries} attempts: {last_error}"
        )

    def _download_once(self, url: str) -> VideoInfo:
        with yt_dlp.YoutubeDL({"quiet": True, "skip_download": True}) as probe:
            info = probe.extract_info(url, download=False)

        title = sanitize_filename(info.get("title", "video"))
        out_template = str(self.output_dir / f"{title}.%(ext)s")

        ydl_opts = {
            "format": "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
            "outtmpl": out_template,
            "merge_output_format": "mp4",
            "quiet": True,
            "no_warnings": True,
            "noprogress": False,
            "continuedl": True,        # resume interrupted downloads
            "retries": 3,
            "fragment_retries": 3,
            "concurrent_fragment_downloads": 4,
            "progress_hooks": [self._progress_hook],
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            result = ydl.extract_info(url, download=True)
            video_path = Path(ydl.prepare_filename(result)).with_suffix(".mp4")

        if not video_path.exists():
            raise DownloadError(f"Download reported success but file not found at {video_path}")

        return VideoInfo(
            video_path=video_path,
            title=result.get("title", title),
            duration_seconds=float(result.get("duration") or 0.0),
            uploader=result.get("uploader", "unknown"),
            original_url=url,
        )

    @staticmethod
    def _progress_hook(status: dict) -> None:
        if status.get("status") == "downloading":
            pct = status.get("_percent_str", "").strip()
            speed = status.get("_speed_str", "").strip()
            logger.debug("Downloading: %s at %s", pct, speed)
        elif status.get("status") == "finished":
            logger.debug("Download finished, now post-processing...")
