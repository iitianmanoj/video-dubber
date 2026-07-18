"""
tts/edge_engine.py
-------------------
Generates natural, expressive English speech per transcript segment using
Microsoft Edge TTS (free, high-quality neural voices, no API key needed).

Supports configurable voices and per-speaker voice assignment for
multi-speaker dubbing (see diarization/diarize.py), with rate/pitch
control used later by sync/align.py to fit each clip into its original
time slot.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import edge_tts

from config import CONFIG, TEMP_DIR
from transcription.whisper_engine import Segment
from utils.logger import get_logger

logger = get_logger()


class TTSError(RuntimeError):
    pass


@dataclass
class SynthesizedSegment:
    segment: Segment
    audio_path: Path


class EdgeTTSEngine:
    """Synthesizes English speech for each transcript segment."""

    def __init__(self, output_dir: Path = TEMP_DIR / "tts_segments"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def assign_voices(self, speaker_ids: List[str]) -> Dict[str, str]:
        """Assign one Edge TTS voice per unique speaker, cycling the pool."""
        unique_speakers = sorted(set(speaker_ids))
        pool = CONFIG.tts.voice_pool
        if len(unique_speakers) <= 1:
            return {unique_speakers[0] if unique_speakers else "SPEAKER_00": CONFIG.tts.default_female_voice}
        return {
            speaker: pool[i % len(pool)]
            for i, speaker in enumerate(unique_speakers)
        }

    async def _synthesize_one(
        self, text: str, voice: str, out_path: Path,
        rate: str = None, pitch: str = None,
    ) -> None:
        rate = rate or CONFIG.tts.rate
        pitch = pitch or CONFIG.tts.pitch

        last_error: Exception | None = None
        for attempt in range(1, 4):
            try:
                communicate = edge_tts.Communicate(
                    text, voice, rate=rate, pitch=pitch, volume=CONFIG.tts.volume,
                )
                await communicate.save(str(out_path))
                if out_path.exists() and out_path.stat().st_size > 0:
                    return
                raise TTSError("Edge TTS produced an empty audio file.")
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                logger.warning("TTS attempt %d/3 failed for a segment: %s", attempt, exc)
                await asyncio.sleep(1.5 * attempt)

        raise TTSError(f"Edge TTS failed for segment after 3 attempts: {last_error}")

    def synthesize_segments(
        self, segments: List[Segment], voice_map: Dict[str, str] = None,
    ) -> List[SynthesizedSegment]:
        """Synthesize speech for every segment, returning paths in order."""
        if not segments:
            return []

        if voice_map is None:
            voice_map = self.assign_voices([s.speaker for s in segments])

        async def _run_all() -> List[SynthesizedSegment]:
            results: List[SynthesizedSegment] = []
            # Edge TTS handles concurrency fine; cap it to be a good network citizen.
            semaphore = asyncio.Semaphore(6)

            async def _work(seg: Segment) -> SynthesizedSegment:
                async with semaphore:
                    voice = voice_map.get(seg.speaker, CONFIG.tts.default_female_voice)
                    out_path = self.output_dir / f"seg_{seg.id:05d}.mp3"
                    await self._synthesize_one(seg.text, voice, out_path)
                    return SynthesizedSegment(segment=seg, audio_path=out_path)

            tasks = [_work(seg) for seg in segments]
            for coro in asyncio.as_completed(tasks):
                results.append(await coro)
            return results

        results = asyncio.run(_run_all())
        results.sort(key=lambda r: r.segment.id)
        return results
