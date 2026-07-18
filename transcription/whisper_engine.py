"""
transcription/whisper_engine.py
--------------------------------
Speech-to-text using faster-whisper (a CTranslate2 reimplementation of
OpenAI Whisper — much faster and lighter on RAM than the reference
implementation, which matters for 2-hour videos).

Automatically:
  - detects the spoken language
  - picks large-v3 on GPU, or a smaller CPU-friendly model on CPU
  - returns word-free but sentence-level segments with start/end timestamps
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

from config import CONFIG, LANGUAGE_NAMES, MODELS_DIR
from utils.helpers import gpu_available
from utils.logger import get_logger

logger = get_logger()


class TranscriptionError(RuntimeError):
    pass


@dataclass
class Segment:
    id: int
    start: float
    end: float
    text: str
    speaker: str = "SPEAKER_00"  # overwritten later if diarization runs


@dataclass
class TranscriptionResult:
    language: str
    language_name: str
    segments: List[Segment]


class Transcriber:
    """Transcribes an audio file to timestamped segments with faster-whisper."""

    def __init__(self):
        self._model = None  # lazy-loaded; heavy import/download

    def _load_model(self):
        if self._model is not None:
            return self._model

        from faster_whisper import WhisperModel  # heavy import, done lazily

        use_gpu = gpu_available()
        model_size = CONFIG.whisper.model_size_gpu if use_gpu else CONFIG.whisper.model_size_cpu
        device = "cuda" if use_gpu else "cpu"
        compute_type = (
            CONFIG.whisper.compute_type_gpu if use_gpu else CONFIG.whisper.compute_type_cpu
        )

        logger.info(
            "Loading Whisper model '%s' on %s (compute_type=%s)",
            model_size, device, compute_type,
        )
        try:
            self._model = WhisperModel(
                model_size,
                device=device,
                compute_type=compute_type,
                download_root=str(MODELS_DIR),
            )
        except Exception as exc:  # noqa: BLE001 - surface as our own error type
            if use_gpu:
                logger.warning(
                    "GPU model load failed (%s); falling back to CPU model '%s'.",
                    exc, CONFIG.whisper.model_size_cpu,
                )
                self._model = WhisperModel(
                    CONFIG.whisper.model_size_cpu,
                    device="cpu",
                    compute_type=CONFIG.whisper.compute_type_cpu,
                    download_root=str(MODELS_DIR),
                )
            else:
                raise TranscriptionError(f"Failed to load Whisper model: {exc}") from exc

        return self._model

    def transcribe(self, audio_path: Path) -> TranscriptionResult:
        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise TranscriptionError(f"Audio file not found: {audio_path}")

        model = self._load_model()

        try:
            segments_iter, info = model.transcribe(
                str(audio_path),
                beam_size=CONFIG.whisper.beam_size,
                vad_filter=CONFIG.whisper.vad_filter,
                word_timestamps=False,
            )
        except Exception as exc:  # noqa: BLE001
            raise TranscriptionError(f"Whisper transcription failed: {exc}") from exc

        segments: List[Segment] = []
        for idx, seg in enumerate(segments_iter):
            text = seg.text.strip()
            if not text:
                continue
            segments.append(Segment(id=idx, start=seg.start, end=seg.end, text=text))

        if not segments:
            raise TranscriptionError(
                "No speech was detected in this video. It may be silent, "
                "music-only, or the audio track may be corrupted."
            )

        language = info.language
        language_name = LANGUAGE_NAMES.get(language, language.upper())
        logger.info(
            "Detected language: [bold]%s[/bold] (confidence %.0f%%)",
            language_name, info.language_probability * 100,
        )

        return TranscriptionResult(
            language=language, language_name=language_name, segments=segments
        )
