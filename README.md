# audio-cloze

Generate audio cloze deletion flashcards from Chinese YouTube content.

## How it works

1. **Index** - Crawl YouTube channels and build an inverted index of subtitle text
2. **Mine** - Search for vocabulary words and extract audio clips with cloze deletions
3. **Review** - Browse clips in HTML interface, export to Anki

## Usage

```bash
# Index channels (builds SQLite FTS database)
python audio_vocab_miner.py index --channels channels.txt --db vocab.db

# Check index stats
python audio_vocab_miner.py stats --db vocab.db

# Mine a word (generates clip + cloze audio)
python audio_vocab_miner.py mine 面交 --db vocab.db -o audio_clips
```

## Requirements

- Python 3.10+
- yt-dlp
- ffmpeg
- pydub

## Output

For each word, generates:
- `{word}_{video_id}_clip.mp3` - Full audio clip with context
- `{word}_{video_id}_cloze.mp3` - Audio with target word silenced
- `review.html` - Interactive review interface
