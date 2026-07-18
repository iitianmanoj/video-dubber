"""
config.py
---------
Central configuration for the Automated Video Dubbing System.

All tunable parameters live here so the rest of the codebase never hardcodes
paths, model names, or voice IDs. Values can be overridden via a `.env` file
in the project root (see `.env.example`).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict

from dotenv import load_dotenv

load_dotenv()  # loads .env if present; safe no-op otherwise

# --------------------------------------------------------------------------
# Filesystem layout
# --------------------------------------------------------------------------
BASE_DIR: Path = Path(__file__).resolve().parent
OUTPUT_DIR: Path = BASE_DIR / "output"
TEMP_DIR: Path = BASE_DIR / "temp"
MODELS_DIR: Path = BASE_DIR / "models"
LOGS_DIR: Path = BASE_DIR / "logs"

for _dir in (OUTPUT_DIR, TEMP_DIR, MODELS_DIR, LOGS_DIR):
    _dir.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class WhisperConfig:
    """Settings for the faster-whisper transcription engine."""

    # "large-v3" is the most accurate but needs a decent GPU / a lot of RAM.
    # "small"/"medium" are good CPU-friendly fallbacks used automatically
    # when no GPU is detected (see transcription/whisper_engine.py).
    model_size_gpu: str = os.getenv("WHISPER_MODEL_GPU", "large-v3")
    model_size_cpu: str = os.getenv("WHISPER_MODEL_CPU", "medium")
    compute_type_gpu: str = "float16"
    compute_type_cpu: str = "int8"
    beam_size: int = 5
    vad_filter: bool = True  # trims silence before transcription


@dataclass(frozen=True)
class TranslationConfig:
    """Settings for the translation stage."""

    # Default engine is Google Translate via deep-translator: no API key,
    # works out of the box, good general quality. Swap `engine` to
    # "nllb" or "indictrans2" if you have those models available locally
    # (see translation/translator.py for the pluggable backend interface).
    engine: str = os.getenv("TRANSLATION_ENGINE", "google")
    target_language: str = "en"
    max_chars_per_request: int = 4500  # Google Translate's practical limit


@dataclass(frozen=True)
class TTSConfig:
    """Settings for Edge TTS speech synthesis."""

    default_female_voice: str = "en-US-AvaMultilingualNeural"
    default_male_voice: str = "en-US-AndrewMultilingualNeural"

    # Additional voices available for multi-speaker dubbing (diarization).
    # The pipeline cycles through this pool, alternating gender where
    # possible, once more than one speaker is detected.
    voice_pool: tuple = (
        "en-US-AvaMultilingualNeural",
        "en-US-AndrewMultilingualNeural",
        "en-US-EmmaMultilingualNeural",
        "en-US-BrianMultilingualNeural",
        "en-GB-SoniaNeural",
        "en-GB-RyanNeural",
    )

    rate: str = "+0%"     # e.g. "+10%" for faster speech
    pitch: str = "+0Hz"
    volume: str = "+0%"


@dataclass(frozen=True)
class SyncConfig:
    """Settings for aligning dubbed audio to the original timing."""

    max_speed_up: float = 1.35   # never speed a clip up more than this
    max_slow_down: float = 0.75  # never slow a clip down more than this
    silence_thresh_db: int = -40
    min_silence_len_ms: int = 200
    crossfade_ms: int = 15


@dataclass(frozen=True)
class DiarizationConfig:
    """Settings for optional speaker diarization (pyannote.audio)."""

    enabled: bool = os.getenv("ENABLE_DIARIZATION", "true").lower() == "true"
    model_name: str = "pyannote/speaker-diarization-3.1"
    hf_token_env: str = "HUGGINGFACE_TOKEN"


@dataclass(frozen=True)
class VoiceCloneConfig:
    """Settings for optional voice cloning via Coqui XTTS (GPU only)."""

    enabled: bool = os.getenv("ENABLE_VOICE_CLONE", "false").lower() == "true"
    model_name: str = "tts_models/multilingual/multi-dataset/xtts_v2"
    reference_clip_seconds: float = 8.0


@dataclass(frozen=True)
class VideoConfig:
    """Container / codec handling for the final remux step."""

    output_container: str = "mp4"
    audio_codec: str = "aac"
    audio_bitrate: str = "192k"
    copy_video_stream: bool = True  # never re-encode video, per spec


@dataclass(frozen=True)
class AppConfig:
    whisper: WhisperConfig = field(default_factory=WhisperConfig)
    translation: TranslationConfig = field(default_factory=TranslationConfig)
    tts: TTSConfig = field(default_factory=TTSConfig)
    sync: SyncConfig = field(default_factory=SyncConfig)
    diarization: DiarizationConfig = field(default_factory=DiarizationConfig)
    voice_clone: VoiceCloneConfig = field(default_factory=VoiceCloneConfig)
    video: VideoConfig = field(default_factory=VideoConfig)

    max_download_retries: int = 3
    chunk_processing_threshold_minutes: int = 20  # >20min videos are chunked
    chunk_length_minutes: int = 10


CONFIG = AppConfig()

# Language code -> friendly display name, used only for nicer CLI output.
LANGUAGE_NAMES: Dict[str, str] = {
    "en": "English", "hi": "Hindi", "de": "German", "fr": "French",
    "es": "Spanish", "it": "Italian", "pt": "Portuguese", "ru": "Russian",
    "ja": "Japanese", "ko": "Korean", "zh": "Chinese", "ar": "Arabic",
    "ta": "Tamil", "te": "Telugu", "bn": "Bengali", "mr": "Marathi",
    "gu": "Gujarati", "ur": "Urdu", "tr": "Turkish", "nl": "Dutch",
}
