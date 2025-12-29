# Roadmap

What's done, what's next.

## Done

### Multi-source indexing
- [x] YouTube channel crawling with yt-dlp
- [x] Subtitle download and FTS5 indexing
- [x] RSS podcast feed parsing
- [x] Episode show notes indexing
- [x] Unified search across both sources
- [x] `sources.txt` config format

### ASR for podcasts
- [x] Two-phase search (text first, ASR only on hits)
- [x] whisper.cpp integration (replaces slower faster-whisper)
- [x] CoreML support for Apple Neural Engine
- [x] mlx-whisper benchmarked as alternative (similar speed)
- [x] Smart chunking for long episodes (5-min chunks, early exit)
- [x] Simplified/Traditional Chinese variant matching

### Clip generation
- [x] Audio extraction with ffmpeg
- [x] Cloze audio generation (silence target word)
- [x] Clip merging across multiple mining runs
- [x] HTML review interface with approve/reject

---

## Next up

### Review UI improvements
- [ ] Keyboard shortcuts (j/k navigate, a approve, r reject)
- [ ] Filter by status (pending/approved/rejected)
- [ ] Search within clips
- [ ] Dark mode

### Anki export
- [ ] Direct AnkiConnect integration
- [ ] Batch export approved clips
- [ ] Custom note type with audio fields

### Claude fallback
When a word isn't in the index:
- [ ] Ask Claude for synonyms and search suggestions
- [ ] Auto-search YouTube for suggested queries
- [ ] Index new videos on the fly
- [ ] Retry original word

### More sources
- [ ] WenetSpeech corpus support (pre-transcribed, huge)
- [ ] Bilibili videos (different subtitle format)
- [ ] Local audio files with manual transcripts

---

## Maybe later

- **Sentence difficulty scoring** - prioritize clips at user's level
- **Spaced repetition integration** - track which words are "learned"
- **Multi-language** - same approach works for Japanese, Korean, etc.
- **Web UI** - Flask/FastAPI instead of CLI
- **Pre-built index distribution** - share indexed databases

---

## Technical debt

- [ ] Better error handling for network failures
- [ ] Resume interrupted indexing
- [ ] Dedup clips with same audio (different videos, same sentence)
- [ ] Tests

---

## Benchmarks

Current numbers on M4 Max:

| Task | Time |
|------|------|
| Index 50 YouTube videos | ~10 min |
| Index 50 podcast episodes | ~2 min (show notes only) |
| Mine 1 word (YouTube hit) | instant |
| Mine 1 word (podcast, needs ASR) | ~20 sec per 5-min chunk |

ASR backends (5-min audio, large-v3):

| Backend | Speed |
|---------|-------|
| mlx-whisper | 12-15x realtime |
| whisper.cpp + CoreML | 12-20x realtime |
