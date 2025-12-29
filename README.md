# audio-cloze

Mine Chinese vocabulary from YouTube and podcasts. Generate audio cloze cards for Anki.

## What it does

You give it a Chinese word. It finds that word spoken in real content, extracts the audio clip, and creates a version with the target word silenced (the "cloze"). You review the clips and export to Anki.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              ARCHITECTURE                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   sources.txt              vocab.db (SQLite)              audio_clips/      │
│   ┌──────────┐            ┌─────────────────┐            ┌──────────────┐   │
│   │ YouTube  │──index───▶ │ captions (FTS5) │──mine────▶ │ word_clip.mp3│   │
│   │ Podcasts │            │ episodes (FTS5) │            │ word_cloze.mp3│  │
│   └──────────┘            └─────────────────┘            │ review.html  │   │
│                                   │                      └──────────────┘   │
│                                   │                             │           │
│                           ┌───────▼───────┐                     │           │
│                           │  whisper.cpp  │◀── ASR for podcast  │           │
│                           │  (CoreML/M4)  │    timing only      │           │
│                           └───────────────┘                     │           │
│                                                                 │           │
│                                                          ┌──────▼──────┐    │
│                                                          │    Anki     │    │
│                                                          └─────────────┘    │
└─────────────────────────────────────────────────────────────────────────────┘
```

## The trick

YouTube videos have subtitles with timestamps. Podcasts have show notes but no timing. So:

1. **YouTube**: Search subtitles → instant timing → extract clip
2. **Podcasts**: Search show notes → if hit, run ASR on that episode → get timing → extract clip

ASR is expensive, so we only run it when we know the word is there.

## Current index

| Source | Count | Notes |
|--------|-------|-------|
| YouTube videos | 219 | 4 channels, full subtitle search |
| Podcast episodes | 587 | 12 feeds, show notes indexed |

## Setup

```bash
# Clone and install
pip install pydub opencc-python-reimplemented webvtt-py feedparser requests yt-dlp

# For podcast ASR (optional, only needed for podcast word timing)
brew install whisper-cpp
# Download large-v3 model (~3GB)
curl -L -o models/ggml-large-v3.bin \
  "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v3.bin"
```

For faster ASR on Apple Silicon, also grab the CoreML encoder:
```bash
curl -L -o models/ggml-large-v3-encoder.mlmodelc.zip \
  "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v3-encoder.mlmodelc.zip"
unzip models/ggml-large-v3-encoder.mlmodelc.zip -d models/
```

## Usage

```bash
# Index sources (YouTube + podcasts)
python audio_vocab_miner.py index --sources sources.txt --db vocab.db

# Check what's indexed
python audio_vocab_miner.py stats --db vocab.db

# Mine a word
python audio_vocab_miner.py mine 面交 --db vocab.db -o audio_clips

# Mine multiple words
python audio_vocab_miner.py mine 保鮮膜 托運 剪刀 --db vocab.db -o audio_clips

# Regenerate review page
python audio_vocab_miner.py review -o audio_clips
```

Open `audio_clips/review.html` to listen, approve/reject, then export to Anki.

## Output

For each word found:
```
audio_clips/
├── 面交_sxerWKZjaKs_clip.mp3   # Full sentence with context
├── 面交_sxerWKZjaKs_cloze.mp3  # Same clip, target word silenced
├── clips.json                   # Metadata for all clips
└── review.html                  # Web UI for reviewing
```

## ASR Performance (M4 Max)

| Backend | Speed | 5 min audio |
|---------|-------|-------------|
| mlx-whisper | 12-15x real-time | 20-25 sec |
| whisper.cpp + CoreML | 12-20x real-time | 15-25 sec |

Both use Apple Silicon acceleration. Performance varies with thermal state.

## Adding sources

Edit `sources.txt`:

```
# YouTube channels
youtube https://www.youtube.com/@shasha77 志祺七七

# Podcast RSS feeds
podcast https://feeds.buzzsprout.com/1974862.rss 百靈果News
```

Then re-run `index`.

## How clips.json works

Each mining run merges with existing clips. You won't lose previous work:

```json
{
  "word": "面交",
  "video_id": "sxerWKZjaKs",
  "transcript": "很多時候交朋友一定是要當面交談過才算數",
  "word_start": 8.0,
  "word_end": 10.97,
  "source_type": "youtube",
  "timing_confidence": "subtitle"
}
```

## Simplified ↔ Traditional

Searches both. If you search 护城河, it also searches 護城河.
