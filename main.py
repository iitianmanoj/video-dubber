#!/usr/bin/env python3
"""
main.py
-------
Entry point for the Automated Video Dubbing System.

Usage:
    python main.py                       # interactive prompt
    python main.py "<youtube-url>"       # direct CLI argument
    python main.py "<url>" --voice male  # choose default voice gender
    python main.py "<url>" --no-diarization --no-subtitles

Pipeline:
    1. Download video (yt-dlp)
    2. Extract audio (ffmpeg)
    3. Transcribe + detect language (faster-whisper)
    4. (optional) Diarize speakers (pyannote.audio)
    5. Translate transcript to English (deep-translator / NLLB / IndicTrans2)
    6. Synthesize English speech (Edge TTS, or XTTS voice cloning if enabled)
    7. Align dubbed speech to original timing
    8. Replace audio in the video (ffmpeg, video stream copied, not re-encoded)
    9. (optional) Generate SRT/VTT/TXT subtitles
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from audio.extract import AudioExtractionError, AudioExtractor
from audio.replace import AudioReplaceError, AudioReplacer
from config import CONFIG, OUTPUT_DIR
from diarization.diarize import Diarizer
from downloader.download import DownloadError, VideoDownloader
from subtitles.subtitle_gen import generate_all as generate_subtitles
from sync.align import AlignmentError, AudioAligner
from transcription.whisper_engine import Transcriber, TranscriptionError
from translation.translator import Translator, TranslationError
from tts.edge_engine import TTSError, EdgeTTSEngine
from utils.helpers import (
    DependencyError,
    check_disk_space,
    check_ffmpeg,
    format_duration,
    get_media_duration,
)
from utils.logger import StageTimer, get_logger
from utils.progress import banner, console, spinner, summary_table

logger = get_logger()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Automated Video Dubbing System — YouTube video to English dub.",
    )
    parser.add_argument("url", nargs="?", default=None, help="YouTube video URL")
    parser.add_argument(
        "--voice", choices=["female", "male"], default="female",
        help="Default narrator voice gender when only one speaker is detected.",
    )
    parser.add_argument("--no-diarization", action="store_true", help="Disable speaker diarization.")
    parser.add_argument("--no-subtitles", action="store_true", help="Skip generating SRT/VTT/TXT files.")
    return parser.parse_args()


def run_pipeline(url: str, voice_gender: str, do_diarization: bool, do_subtitles: bool) -> Path:
    overall_start = time.perf_counter()

    # --- Pre-flight checks -------------------------------------------------
    check_ffmpeg()
    check_disk_space(OUTPUT_DIR, required_gb=2.0)

    # --- 1. Download ---------------------------------------------------
    with StageTimer("Downloading video"):
        with spinner("Downloading video from YouTube..."):
            downloader = VideoDownloader()
            video_info = downloader.download(url)
    console.print(
        f"[bold]{video_info.title}[/bold]  "
        f"({format_duration(video_info.duration_seconds)}, by {video_info.uploader})"
    )

    # --- 2. Extract audio ------------------------------------------------
    with StageTimer("Extracting audio"):
        extractor = AudioExtractor()
        original_audio_path = extractor.extract(video_info.video_path)

    # --- 3. Transcribe -----------------------------------------------------
    with StageTimer("Transcribing speech"):
        with spinner("Running Whisper transcription (this can take a while)..."):
            transcriber = Transcriber()
            transcription = transcriber.transcribe(original_audio_path)
    console.print(f"Detected Language: [bold yellow]{transcription.language_name}[/bold yellow]")
    console.print(f"Found {len(transcription.segments)} speech segments.")

    segments = transcription.segments

    # --- 4. Diarization (optional) ------------------------------------------
    if do_diarization:
        with StageTimer("Detecting speakers"):
            diarizer = Diarizer()
            segments = diarizer.diarize(original_audio_path, segments)

    # --- 5. Translate --------------------------------------------------
    with StageTimer("Translating to English"):
        with spinner(f"Translating {len(segments)} segments..."):
            translator = Translator()
            translated_segments = translator.translate_segments(segments, transcription.language)

    # --- 6. Synthesize English speech --------------------------------------
    default_voice = (
        CONFIG.tts.default_male_voice if voice_gender == "male" else CONFIG.tts.default_female_voice
    )
    with StageTimer("Generating dubbed speech"):
        with spinner("Synthesizing English speech with Edge TTS..."):
            tts_engine = EdgeTTSEngine()
            speaker_ids = {s.speaker for s in translated_segments}
            if len(speaker_ids) <= 1:
                voice_map = {next(iter(speaker_ids), "SPEAKER_00"): default_voice}
            else:
                voice_map = tts_engine.assign_voices([s.speaker for s in translated_segments])
            synthesized = tts_engine.synthesize_segments(translated_segments, voice_map)

    # --- 7. Align to original timing ----------------------------------------
    with StageTimer("Synchronizing timing"):
        aligner = AudioAligner()
        dubbed_audio_path = aligner.build_timeline(synthesized, video_info.duration_seconds)

    # --- 8. Replace audio in the video --------------------------------------
    with StageTimer("Replacing audio in video"):
        replacer = AudioReplacer()
        final_video_path = replacer.replace(
            video_info.video_path, dubbed_audio_path, output_name=f"{video_info.video_path.stem}_dubbed_en",
        )

    # --- 9. Subtitles (optional) ---------------------------------------------
    subtitle_paths = {}
    if do_subtitles:
        with StageTimer("Generating subtitles"):
            subtitle_paths = generate_subtitles(
                translated_segments, OUTPUT_DIR, base_name=f"{video_info.video_path.stem}_dubbed_en",
            )

    total_elapsed = time.perf_counter() - overall_start

    summary_table({
        "Title": video_info.title,
        "Source language": transcription.language_name,
        "Duration": format_duration(video_info.duration_seconds),
        "Speakers detected": len({s.speaker for s in translated_segments}),
        "Output video": final_video_path,
        "Subtitles": ", ".join(str(p) for p in subtitle_paths.values()) or "skipped",
        "Total processing time": format_duration(total_elapsed),
    })

    return final_video_path


def main() -> int:
    args = parse_args()
    banner(
        "🎬 Automated Video Dubbing System",
        "YouTube video → English-dubbed MP4, same voice energy, original quality",
    )

    url = args.url
    if not url:
        url = console.input("[bold cyan]Enter YouTube URL: [/bold cyan]").strip()

    if not url:
        console.print("[bold red]No URL provided. Exiting.[/bold red]")
        return 1

    try:
        run_pipeline(
            url=url,
            voice_gender=args.voice,
            do_diarization=not args.no_diarization,
            do_subtitles=not args.no_subtitles,
        )
    except (DependencyError, DownloadError, AudioExtractionError, TranscriptionError,
            TranslationError, TTSError, AlignmentError, AudioReplaceError) as exc:
        # All expected, "graceful" failure modes land here with a clear message.
        logger.error("[bold red]Pipeline stopped:[/bold red] %s", exc)
        console.print(f"\n[bold red]Error:[/bold red] {exc}")
        return 1
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted by user. Partial files are left in temp/.[/yellow]")
        return 130
    except Exception as exc:  # noqa: BLE001 - last resort, never crash silently
        logger.exception("Unexpected error")
        console.print(f"\n[bold red]Unexpected error:[/bold red] {exc}")
        return 1

    console.print("\n[bold green]Done.[/bold green]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
