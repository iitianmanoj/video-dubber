"""
tts/voice_clone.py
-------------------
Optional voice cloning using Coqui XTTS-v2, so the dubbed voice can match
the original speaker's timbre instead of a generic Edge TTS voice.

Requires a GPU for reasonable speed (XTTS is a large model). When no GPU
is available, or Coqui-TTS isn't installed, this module raises
`VoiceCloneUnavailable`, and the caller (main.py) falls back to
Edge TTS — exactly as the spec requires ("Fallback: Edge TTS").
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

from pydub import AudioSegment

from config import CONFIG, TEMP_DIR
from utils.helpers import gpu_available
from utils.logger import get_logger

logger = get_logger()


class VoiceCloneUnavailable(RuntimeError):
    pass


class VoiceCloner:
    """Clones each detected speaker's voice using a short reference clip."""

    def __init__(self, work_dir: Path = TEMP_DIR / "voice_clone"):
        self.work_dir = Path(work_dir)
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self._tts = None

    def _load_model(self):
        if self._tts is not None:
            return self._tts

        if not CONFIG.voice_clone.enabled:
            raise VoiceCloneUnavailable("Voice cloning disabled in config.")
        if not gpu_available():
            raise VoiceCloneUnavailable("Voice cloning requires a GPU; none detected.")

        try:
            from TTS.api import TTS as CoquiTTS
        except ImportError as exc:
            raise VoiceCloneUnavailable(
                "coqui-tts is not installed. Install it per the README to enable "
                "voice cloning, or leave ENABLE_VOICE_CLONE=false to use Edge TTS."
            ) from exc

        try:
            self._tts = CoquiTTS(CONFIG.voice_clone.model_name, gpu=True)
        except Exception as exc:  # noqa: BLE001
            raise VoiceCloneUnavailable(f"Failed to load XTTS model: {exc}") from exc

        return self._tts

    def extract_reference_clips(
        self, original_audio_path: Path, speaker_segments: Dict[str, List],
    ) -> Dict[str, Path]:
        """Cut a short reference clip per speaker from the original audio."""
        audio = AudioSegment.from_file(original_audio_path)
        clips: Dict[str, Path] = {}
        target_ms = int(CONFIG.voice_clone.reference_clip_seconds * 1000)

        for speaker, segments in speaker_segments.items():
            longest = max(segments, key=lambda s: s.end - s.start)
            start_ms = int(longest.start * 1000)
            end_ms = min(start_ms + target_ms, int(longest.end * 1000), len(audio))
            clip = audio[start_ms:end_ms]
            clip_path = self.work_dir / f"ref_{speaker}.wav"
            clip.export(clip_path, format="wav")
            clips[speaker] = clip_path

        return clips

    def synthesize(self, text: str, reference_clip: Path, out_path: Path, language: str = "en") -> Path:
        tts = self._load_model()
        try:
            tts.tts_to_file(
                text=text,
                speaker_wav=str(reference_clip),
                language=language,
                file_path=str(out_path),
            )
        except Exception as exc:  # noqa: BLE001
            raise VoiceCloneUnavailable(f"XTTS synthesis failed: {exc}") from exc
        return out_path
