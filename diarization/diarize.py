"""
diarization/diarize.py
-----------------------
Optional speaker diarization using pyannote.audio, so that different
speakers can be dubbed with different English voices.

This is genuinely optional: pyannote's pretrained pipelines require a
free Hugging Face account + accepting the model's license + an access
token (HUGGINGFACE_TOKEN in .env). If that token isn't configured, or the
pipeline fails to load (no internet, no GPU, etc.), we gracefully fall
back to treating the whole video as a single speaker — the pipeline
keeps working end-to-end either way.
"""

from __future__ import annotations

import os
from typing import List

from config import CONFIG
from transcription.whisper_engine import Segment
from utils.logger import get_logger

logger = get_logger()


class DiarizationUnavailable(RuntimeError):
    """Raised (and caught internally) when diarization can't run; callers
    should treat this as 'fall back to single speaker', not a hard error."""


class Diarizer:
    """Assigns a speaker label to each transcript segment."""

    def __init__(self):
        self._pipeline = None

    def _load_pipeline(self):
        if self._pipeline is not None:
            return self._pipeline

        token = os.getenv(CONFIG.diarization.hf_token_env)
        if not token:
            raise DiarizationUnavailable(
                f"{CONFIG.diarization.hf_token_env} is not set; skipping diarization."
            )

        try:
            from pyannote.audio import Pipeline

            self._pipeline = Pipeline.from_pretrained(
                CONFIG.diarization.model_name, use_auth_token=token,
            )
        except Exception as exc:  # noqa: BLE001
            raise DiarizationUnavailable(f"Failed to load diarization pipeline: {exc}") from exc

        return self._pipeline

    def diarize(self, audio_path, segments: List[Segment]) -> List[Segment]:
        """Return a copy of `segments` with `.speaker` populated.

        On any failure (missing token, no GPU, pyannote not installed,
        etc.) this returns the segments unmodified (all "SPEAKER_00"),
        which is a perfectly valid single-speaker dub.
        """
        if not CONFIG.diarization.enabled:
            logger.debug("Diarization disabled in config; using single-speaker mode.")
            return segments

        try:
            pipeline = self._load_pipeline()
            diarization = pipeline(str(audio_path))
        except DiarizationUnavailable as exc:
            logger.info("Diarization skipped (%s). Using a single voice for all speech.", exc)
            return segments
        except Exception as exc:  # noqa: BLE001
            logger.warning("Diarization failed unexpectedly (%s); using single-speaker mode.", exc)
            return segments

        # Build a list of (start, end, speaker_label) turns.
        turns = [
            (turn.start, turn.end, speaker)
            for turn, _, speaker in diarization.itertracks(yield_label=True)
        ]

        if not turns:
            return segments

        updated: List[Segment] = []
        for seg in segments:
            mid = (seg.start + seg.end) / 2
            speaker = next(
                (spk for (t_start, t_end, spk) in turns if t_start <= mid <= t_end),
                None,
            )
            updated.append(
                Segment(
                    id=seg.id, start=seg.start, end=seg.end,
                    text=seg.text, speaker=speaker or seg.speaker,
                )
            )

        num_speakers = len({s.speaker for s in updated})
        logger.info("Diarization detected %d speaker(s).", num_speakers)
        return updated
