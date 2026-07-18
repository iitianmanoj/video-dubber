# 🎬 Automated Video Dubbing System

Turn any YouTube video into an English-dubbed version: same video, same
energy, just with natural English audio. Give it a URL, get back an
`.mp4` with the original video quality untouched and a translated,
English-dubbed audio track in its place.

```
Enter YouTube URL: https://www.youtube.com/watch?v=example
→ Downloading video...
→ Extracting audio...
→ Transcribing speech...
Detected Language: Hindi
→ Detecting speakers...
→ Translating to English...
→ Generating dubbed speech...
→ Synchronizing timing...
→ Replacing audio in video...
→ Generating subtitles...
Done.
```

---

## How it works (architecture)

```
                     ┌─────────────┐
   YouTube URL  ───▶ │  yt-dlp      │  downloader/download.py
                     └──────┬──────┘
                            │ video.mp4
                     ┌──────▼──────┐
                     │  ffmpeg      │  audio/extract.py
                     └──────┬──────┘
                            │ audio.wav
                     ┌──────▼──────┐
                     │ faster-      │  transcription/whisper_engine.py
                     │ whisper      │  → segments with timestamps + language
                     └──────┬──────┘
                            │
                ┌───────────┴───────────┐
                │ pyannote.audio         │  diarization/diarize.py (optional)
                │ (speaker labels)       │  → falls back to 1 speaker if unavailable
                └───────────┬───────────┘
                            │
                     ┌──────▼──────┐
                     │ Google /     │  translation/translator.py
                     │ NLLB /       │  → natural English text per segment
                     │ IndicTrans2  │
                     └──────┬──────┘
                            │
                     ┌──────▼──────┐
                     │  Edge TTS    │  tts/edge_engine.py
                     │ (+ XTTS      │  tts/voice_clone.py (optional, GPU)
                     │  cloning)    │
                     └──────┬──────┘
                            │ per-segment English audio clips
                     ┌──────▼──────┐
                     │  pydub /     │  sync/align.py
                     │  ffmpeg      │  → trims silence, time-stretches clips,
                     │  atempo      │    places them on the original timeline
                     └──────┬──────┘
                            │ dubbed_audio_timeline.wav
                     ┌──────▼──────┐
                     │  ffmpeg      │  audio/replace.py
                     │ (-c:v copy)  │  → muxes new audio into original video,
                     └──────┬──────┘    video stream is never re-encoded
                            │
                     ┌──────▼──────┐
                     │ output/*.mp4 │  + .srt, .vtt, transcript.txt
                     └─────────────┘
```

Every stage is its own module with a single responsibility (SOLID-style),
so backends are swappable — e.g. switch the translation engine or add a
new TTS provider without touching anything else.

---

## Project structure

```
video-dubber/
├── main.py                    # CLI entry point, orchestrates the pipeline
├── config.py                  # all tunables in one place (dataclasses)
├── requirements.txt
├── README.md
├── .env.example                # copy to .env to override defaults
│
├── downloader/
│   └── download.py            # yt-dlp wrapper + error handling
├── audio/
│   ├── extract.py              # video → WAV (ffmpeg)
│   └── replace.py              # mux dubbed audio into video (ffmpeg, -c:v copy)
├── transcription/
│   └── whisper_engine.py       # faster-whisper, GPU/CPU auto-select
├── translation/
│   └── translator.py           # Google / NLLB / IndicTrans2 backends
├── tts/
│   ├── edge_engine.py          # Microsoft Edge TTS synthesis
│   └── voice_clone.py          # optional Coqui XTTS voice cloning
├── diarization/
│   └── diarize.py              # optional pyannote.audio speaker labels
├── sync/
│   └── align.py                # silence trim, speed adjust, timeline placement
├── subtitles/
│   └── subtitle_gen.py         # SRT / VTT / TXT generation
├── utils/
│   ├── logger.py                # Rich logging + per-stage timers
│   ├── progress.py              # Rich progress bars / spinners
│   └── helpers.py               # dependency checks, formatting, etc.
│
├── output/                     # final dubbed videos + subtitles land here
├── temp/                       # intermediate files (safe to delete)
├── models/                     # cached model weights (Whisper, etc.)
└── logs/                       # one timestamped log file per run
```

---

## Installation

### 1. Python

Requires **Python 3.10+** (developed against 3.12).

```bash
python3 --version
```

### 2. FFmpeg

FFmpeg and ffprobe must be on your `PATH`.

```bash
# Ubuntu / Debian
sudo apt update && sudo apt install ffmpeg

# macOS (Homebrew)
brew install ffmpeg

# Windows
# Download from https://ffmpeg.org/download.html and add the bin/ folder to PATH
```

Verify:

```bash
ffmpeg -version
```

### 3. Project dependencies

```bash
git clone <this-repo>
cd video-dubber
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

> **GPU note:** `faster-whisper` and voice cloning benefit hugely from a
> CUDA GPU, but the pipeline runs fine on CPU-only machines — it just
> automatically selects a smaller Whisper model (`medium` instead of
> `large-v3`) and skips voice cloning in favor of Edge TTS.

### 4. Optional configuration

```bash
cp .env.example .env
```

Edit `.env` if you want to:
- switch the translation backend (`TRANSLATION_ENGINE=indictrans2` for
  Hindi/Tamil/Telugu/etc., or `nllb` for broader language coverage)
- enable speaker diarization (`HUGGINGFACE_TOKEN=...`, after accepting
  the license for `pyannote/speaker-diarization-3.1` on huggingface.co)
- enable GPU voice cloning (`ENABLE_VOICE_CLONE=true`)

Nothing above is required to run the core pipeline — the defaults
(Google Translate + Edge TTS) work with zero configuration.

---

## Usage

```bash
python main.py
# Enter YouTube URL: <paste a link>
```

Or non-interactively:

```bash
python main.py "https://www.youtube.com/watch?v=XXXXXXXXXXX"

# Choose the default narrator voice for single-speaker videos
python main.py "<url>" --voice male

# Skip optional stages
python main.py "<url>" --no-diarization --no-subtitles
```

Output lands in `output/`:
- `<title>_dubbed_en.mp4` — the final dubbed video
- `<title>_dubbed_en.srt` / `.vtt` — subtitles
- `<title>_dubbed_en_transcript.txt` — plain translated transcript

---

## Error handling

The pipeline is built to fail *gracefully*, with a specific message
instead of a raw traceback, for every failure mode called out in the
spec:

| Failure | Behavior |
|---|---|
| Invalid URL | Rejected before any network call, with the expected format shown |
| Private / unavailable video | Clear error, no retry |
| Age-restricted video | Clear error explaining the `--cookies` workaround |
| No internet | Detected up front via a connectivity check |
| Interrupted download | Retried automatically (resumable, exponential backoff) |
| Missing ffmpeg | Detected at startup with install instructions |
| Disk full | Checked before downloading (configurable minimum free space) |
| Translation failure | Retried, then falls back to the original text for that segment |
| TTS failure | Retried per segment; the run aborts with a clear error only if a segment truly can't be synthesized after 3 attempts |
| No GPU | Whisper/voice cloning silently fall back to CPU-friendly / Edge TTS paths |
| Diarization unavailable | Falls back to single-speaker dubbing automatically |

Every run also writes a full debug log to `logs/run_<timestamp>.log`,
including timing for each stage — useful for diagnosing anything that
doesn't show up in the console summary.

---

## Performance & long videos

- Audio extraction and remuxing use `ffmpeg` directly (not `moviepy`),
  which keeps memory usage flat regardless of video length.
- The video stream is **never re-encoded** — only the audio stream is
  replaced — so a 2-hour video remuxes in seconds, not hours.
- `faster-whisper`'s CTranslate2 backend uses significantly less RAM
  than reference Whisper, with built-in VAD filtering to skip silence.
- TTS synthesis is done concurrently per segment (bounded semaphore) to
  keep total wall-clock time reasonable on long transcripts.
- For very long videos, `CONFIG.chunk_processing_threshold_minutes` /
  `chunk_length_minutes` are available for callers who want to process
  audio in bounded chunks (see `utils/helpers.split_into_chunks`).

---

## Troubleshooting

**`'ffmpeg' was not found on PATH`**
Install ffmpeg (see Installation §2) and restart your terminal.

**Whisper model download is slow / fails**
Models are cached in `models/` after the first run. If a download stalls,
delete the partial file under `models/` and re-run.

**Translation looks off for a specific language**
Try `TRANSLATION_ENGINE=indictrans2` (Indian languages) or `nllb`
(broader multilingual coverage) in `.env` instead of the default Google
backend.

**Diarization isn't detecting multiple speakers**
Confirm `HUGGINGFACE_TOKEN` is set and you've accepted the model license
for `pyannote/speaker-diarization-3.1` at huggingface.co. Without a
token, the pipeline silently dubs everything with one voice — check
`logs/` for the specific reason it fell back.

**Voice cloning isn't running**
It requires `ENABLE_VOICE_CLONE=true`, `coqui-tts` installed, and a CUDA
GPU. Without all three it falls back to Edge TTS automatically.

**Output audio feels out of sync**
Speed adjustment is capped (`SyncConfig.max_speed_up` /
`max_slow_down`) to avoid unnatural-sounding speech. Very dense/fast
source speech may still drift slightly — loosen those bounds in
`config.py` if you'd rather prioritize sync over natural pacing.

---

## Future improvements

- Word-level timestamp alignment (not just sentence-level) for tighter lip-sync
- Streaming/incremental processing so output starts appearing before the whole video finishes
- A lightweight web UI wrapping the same pipeline
- Automatic back-translation QA pass to catch translation drift
- Batch mode for dubbing an entire playlist in one run

---

## Screenshots

*(placeholders — replace with real captures from a sample run)*

- `docs/screenshot_cli_progress.png` — terminal progress during a run
- `docs/screenshot_output_folder.png` — contents of `output/` after a run
- `docs/screenshot_before_after.png` — side-by-side of source vs. dubbed video

---

## License

MIT — see [LICENSE](LICENSE).
